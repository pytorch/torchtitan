# Supporting a grouped root/rest bucket with reshard_after_forward=True

## Goal

Support this user-facing bucket layout:

```python
BucketSpec(["layers.0.*"], reshard_after_forward=True)
BucketSpec(["layers.1.*"], reshard_after_forward=True)
BucketSpec(
    ["tok_embeddings.*", "pos_embeddings.*", "norm.*", "output.*"],
    reshard_after_forward=True,
)
```

This is the target. The plan below explores how to make it correct and
verifiable. Rejecting this layout is only the current baseline behavior, not a
candidate solution.

## Current baseline failure

Command used:

```bash
python - <<'PY'
from torchtitan.experiments.flex_shard import BucketSpec, flex_shard
from torchtitan.experiments.flex_shard.example.shard import per_param_placements
from torchtitan.experiments.flex_shard.tests.common import (
    make_transformer_model,
    single_rank_cuda_mesh,
)

with single_rank_cuda_mesh() as mesh:
    args, model = make_transformer_model(n_layers=2)
    buckets = [
        BucketSpec([f"layers.{idx}.*"], reshard_after_forward=True)
        for idx in range(args.n_layers)
    ]
    buckets.append(
        BucketSpec(
            ["tok_embeddings.*", "pos_embeddings.*", "norm.*", "output.*"],
            reshard_after_forward=True,
        )
    )
    try:
        flex_shard(
            model,
            mesh,
            shard_placement_fn=per_param_placements,
            buckets=buckets,
        )
    except Exception as exc:
        print(type(exc).__name__)
        print(exc)
    else:
        print("NO_ERROR")
PY
```

Observed result:

```text
RuntimeError
FlexShard eager reshard_after_forward could not register a recomputation-safe
batched all-gather hook for bucket 'norm, output, pos_embeddings, tok_embeddings'.
Bucket hooks must run both in the original forward and in activation checkpoint
recomputation. Split the bucket to match forward execution units, or set
reshard_after_forward=False for this bucket.
```

## Why the baseline rejects it

The grouped root/rest bucket has parameter uses in separate top-level execution
units:

```text
tok_embeddings / pos_embeddings -> before transformer layers
norm / output                   -> after transformer layers
```

The deepest common hook target for that bucket is the root model. Current
validation rejects a root hook when `reshard_after_forward=True` and the model
contains checkpoint-wrapped children, because activation checkpointing may
recompute a child directly and skip `model.__call__`.

For a layer bucket this is safe:

```text
AC recompute for layers.1:
  layers.1 pre-forward hook reruns
    all-gather layers.1 bucket
  layers.1.forward() recompute
```

For a root hook this is not replay-safe by default:

```text
AC recompute for output:
  output.forward() recompute
  root pre-forward hook does not run
```

So supporting the grouped root/rest bucket requires something stronger than a
single root hook.

## Required invariant

For every parameter access in the grouped bucket, the corresponding full
parameter must be materialized inside the activation-checkpoint replay region
that performs that access.

For the target bucket, this means:

```text
tok_embeddings AC region:
  materialize root/rest bucket full params, expose tok_embeddings params
  tok_embeddings.forward()

pos_embeddings AC region:
  materialize or reuse root/rest bucket full params, expose pos_embeddings params
  pos_embeddings.forward()

norm AC region:
  materialize or reuse root/rest bucket full params, expose norm params
  norm.forward()

output AC region:
  materialize or reuse root/rest bucket full params, expose output params
  output.forward()
```

The design problem is how to do this while preserving the user-facing single
`BucketSpec` for the root/rest group.

## Candidate support designs

### Option A: Fragmented hooks for one physical bucket

Keep one user-facing bucket and one physical bucket, but register multiple
execution-unit hooks for it:

```text
root/rest physical bucket:
  hook on tok_embeddings
  hook on pos_embeddings
  hook on norm
  hook on output
```

Each hook would expose only the entries used by that module, but the collective
would still operate on the whole physical bucket.

Open questions:

- Does each hook all-gather the whole bucket independently? That is correct but
  may duplicate all-gathers.
- Can the full bucket be kept alive across the root forward and reused by later
  hooks? That reduces duplicate all-gathers but weakens the memory benefit of
  `reshard_after_forward=True`.
- How does backward prefetch order work when one physical bucket has several
  hook points?

Initial judgment:

This supports the API shape directly, but the performance/memory model is
unclear. It needs profiler and memory-snapshot validation.

### Option B: Internal replay fragments, shared user BucketSpec

Keep the user-facing grouped `BucketSpec`, but internally split it into replay
fragments by execution unit:

```text
user bucket:
  ["tok_embeddings.*", "pos_embeddings.*", "norm.*", "output.*"]

internal replay fragments:
  tok_embeddings fragment
  pos_embeddings fragment
  norm fragment
  output fragment
```

Each fragment gets its own replay-safe hook and activation-checkpoint region.
The user still writes one bucket, but runtime scheduling behaves like several
smaller replay buckets.

#### Does this preserve one-bucket one-AG one-RS?

Not automatically. There are two different interpretations:

1. **Replay fragments also become communication fragments.**

   ```text
   tok_embeddings fragment -> AG/RS for tok_embeddings params
   pos_embeddings fragment -> AG/RS for pos_embeddings params
   norm fragment           -> AG/RS for norm params
   output fragment         -> AG/RS for output params
   ```

   This is the simplest replay-safe implementation, but it breaks the physical
   one-bucket one-AG one-RS contract. The user wrote one `BucketSpec`, but the
   runtime launches multiple all-gathers and reduce-scatters.

2. **Replay fragments share one physical communication bucket.**

   ```text
   one user bucket / one physical bucket:
     ["tok_embeddings.*", "pos_embeddings.*", "norm.*", "output.*"]

   multiple replay fragments:
     tok_embeddings hook
     pos_embeddings hook
     norm hook
     output hook
   ```

   This preserves the one-bucket one-AG one-RS contract only if the runtime
   ensures that the physical bucket's all-gather is launched once for the
   relevant forward/recompute window and the fragments consume views from that
   shared full-bucket result.

   That is significantly more complex because the all-gather result must stay
   alive across multiple child module forwards:

   ```text
   tok_embeddings hook consumes full bucket and exposes tok_embeddings params
   pos_embeddings hook reuses same full bucket and exposes pos_embeddings params
   ...
   output hook reuses same full bucket and exposes output params
   final fragment releases full bucket
   ```

   Backward has a matching issue: to keep one RS for the user bucket, gradients
   from all replay fragments must be accumulated until every fragment's
   full-param grads are available, then one reduce-scatter is launched for the
   whole bucket. If each fragment launches reduce-scatter independently, the
   contract is broken.

Open questions:

- Do we require one physical AG/RS per user `BucketSpec`, or is a user bucket
  allowed to expand into multiple replay collectives?
- If one physical AG/RS is required, where is the shared full-bucket result
  stored, and which fragment owns releasing it?
- How do we count fragment completion in backward so reduce-scatter waits for
  all full-param grads in the bucket without exposing the RS on the critical
  path?

Initial judgment:

Option B does not have to break one-bucket one-AG one-RS, but the simple version
does. Preserving the contract requires a shared physical bucket runtime plus
separate replay fragments that only control parameter exposure and AC replay.

### Option C: Root/rest persistent window across the root forward

Treat the grouped root/rest bucket as a special root-forward lifetime window:

```text
root pre-forward:
  all-gather root/rest bucket

tok_embeddings / pos_embeddings / norm / output:
  consume full params from the already-materialized root/rest bucket

root post-forward:
  release root/rest full params
```

For recompute, each child hook would need a way to rematerialize the root/rest
bucket on demand if its params are used by that child.

Open questions:

- This still needs child hooks for recompute, so it becomes close to Option A.
- The root/rest full params may stay live across the full forward, reducing
  `reshard_after_forward=True` memory savings.
- It needs careful stale-handle and prefetch interaction rules.

Initial judgment:

This may reduce duplicate all-gathers in the original forward, but it risks
memory regression and still requires per-child replay logic.

### Option D: Root-level activation checkpointing for root/rest bucket

Wrap the root model itself so the root hook reruns.

Open questions:

- Does this duplicate or interfere with per-layer activation checkpointing?
- Does it change recompute granularity enough to lose the expected memory/compute
  behavior?
- Can backward all-gather prefetch still be preserved?

Initial judgment:

This is the simplest replay story for the root hook, but it likely changes the
activation-checkpointing granularity too much. It should be a last resort.

## Explored direction: Option B iteration 1

Option B is still the right shape, but the first implementation explored the
wrong communication boundary.

The key abstraction to explore:

```text
BucketSpec = user grouping and policy
Replay fragment = minimal execution unit whose hook must rerun under AC
```

For layer buckets, the bucket has one replay fragment:

```text
BucketSpec(["layers.1.*"]) -> fragment on layers.1
```

For the grouped root/rest bucket, the bucket has multiple replay fragments:

```text
BucketSpec(["tok_embeddings.*", "pos_embeddings.*", "norm.*", "output.*"])
  -> fragment on tok_embeddings
  -> fragment on pos_embeddings
  -> fragment on norm
  -> fragment on output
```

### Iteration 1: correctness prototype

The first implementation chose the conservative communication strategy:

```text
one replay fragment all-gather per execution unit
```

This proved two useful things:

- The target user-facing grouped `BucketSpec` can be made replay-safe by
  splitting hook placement into top-level replay fragments.
- Prefetch order must be sorted by actual hook-module execution order, not by
  user `BucketSpec` declaration order.

It is not an acceptable final design because replay fragments also became
communication fragments:

```text
tok_embeddings fragment -> AG/RS for tok_embeddings params
pos_embeddings fragment -> AG/RS for pos_embeddings params
norm fragment           -> AG/RS for norm params
output fragment         -> AG/RS for output params
```

That breaks the required bucket contract:

```text
one user BucketSpec -> one physical all-gather and one physical reduce-scatter
```

Conclusion: iteration 1 is a correctness proof and should not be landed as the
root/rest bucket solution.

## Required next design: shared physical bucket plus replay fragments

The next implementation should keep the concepts separate:

```text
Physical bucket runtime:
  owns DStorage, all ParamEntry values, placement policy, mixed precision policy
  owns one physical all-gather per bucket window
  owns one physical reduce-scatter per backward bucket

Replay fragment:
  owns hook placement and activation-checkpoint replay boundary
  exposes only the params used by that hook module
  never launches its own bucket collective
```

For the target grouped bucket:

```text
Physical bucket:
  ["tok_embeddings.*", "pos_embeddings.*", "norm.*", "output.*"]

Replay fragments:
  tok_embeddings
  pos_embeddings
  norm
  output
```

The physical bucket must all-gather the whole grouped bucket once, then each
fragment consumes views or aliases for the params it owns.

### Forward lifetime

With one physical all-gather, the full root/rest bucket result has to stay live
from the first fragment access to the last fragment access:

```text
tok_embeddings hook:
  launch/finish one physical AG for root/rest bucket
  expose tok_embeddings params

pos_embeddings hook:
  reuse same physical full-bucket result
  expose pos_embeddings params

layers.0 / layers.1:
  no root/rest param access

norm hook:
  reuse same physical full-bucket result
  expose norm params

output hook:
  reuse same physical full-bucket result
  expose output params
  release physical full-bucket result after output forward
```

This preserves one physical all-gather. The tradeoff is that the grouped
root/rest full params stay live across the whole interval from embeddings to
output. That is the cost of grouping disjoint execution units into one physical
bucket. If that memory lifetime is unacceptable, the user should split the
bucket into contiguous execution-window buckets.

### Activation-checkpoint capture

Each replay fragment still needs an operation inside its checkpointed region
that produces the full-param tensor objects consumed by that module. Otherwise
views like `w.T` saved by autograd can escape the checkpoint boundary.

The shared physical all-gather result can be produced once, but each fragment
should expose its params through a fragment-local autograd-visible access op:

```text
fragment pre-forward hook:
  ensure shared physical full-bucket result exists
  select this fragment's full params from the shared result
  run fragment access autograd op inside this checkpoint region
  set _pre_gathered for this fragment's params
```

That keeps the checkpoint boundary local to the parameter use while avoiding a
new physical all-gather per fragment.

### Backward reduce-scatter

The physical bucket should launch one reduce-scatter after gradients for all
entries in the grouped bucket are available.

Do not assume a single shared `_BucketAllGather.backward()` will naturally
aggregate all fragment gradients. That is unsafe with separate replay fragments.
Non-reentrant activation checkpointing may replay fragments independently, and
autograd can run backward for an earlier fragment before later fragments have
attached or produced their gradients.

The risky naive design is:

```text
one physical bucket all-gather autograd node
  outputs all full params for the user bucket

fragment access autograd nodes
  consume selected full params
  return selected full params inside each checkpoint region
  pass grads back to the physical bucket outputs

physical bucket all-gather backward
  receives all full-param grads once all fragment users finish
  launches one physical reduce-scatter for the user bucket
```

This only works if autograd truly has one shared node whose backward cannot run
until every fragment use is connected and completed. That assumption is too
fragile for AC replay fragments.

The required model is explicit bucket-level gradient aggregation:

```text
fragment backward for tok_embeddings:
  report tok_embeddings full-param grads to physical bucket accumulator
  do not launch RS

fragment backward for pos_embeddings:
  report pos_embeddings full-param grads to physical bucket accumulator
  do not launch RS

fragment backward for norm:
  report norm full-param grads to physical bucket accumulator
  do not launch RS

fragment backward for output:
  report output full-param grads to physical bucket accumulator
  if all expected bucket grads have arrived:
    pack whole bucket and launch one physical RS
```

The accumulator must know the expected entries/fragments for the current
backward pass and launch reduce-scatter exactly once. If fragment access nodes
launch reduce-scatter independently, the design regresses to iteration 1 and is
not acceptable.

## Incremental implementation plan

### Step 1: Add a failing positive test

Add a test that constructs the target layout and expects `flex_shard()` to
succeed and train with reference parity.

Baseline before the runtime change:

```text
fails with recomputation-safe hook error
```

Observed command:

```bash
pytest -q \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_training.py::TestFlexShardTraining::test_reshard_after_forward_grouped_root_rest_bucket
```

Observed result:

```text
FAILED with RuntimeError:
FlexShard eager reshard_after_forward could not register a recomputation-safe
batched all-gather hook for bucket 'norm, output, pos_embeddings, tok_embeddings'.
```

### Step 2: Represent replay fragments

Introduce an internal structure like:

```python
@dataclass(frozen=True)
class BucketReplayFragment:
    physical_bucket: PhysicalBucketRuntime
    hook_module: nn.Module
    entries: list[ParamEntry]
```

Layer buckets produce one fragment. The grouped root/rest bucket produces one
fragment per top-level owner module.

### Step 3: Add a physical bucket runtime

Introduce a runtime that owns the original one-bucket communication contract:

```text
PhysicalBucketRuntime:
  storage
  all entries
  shared communication context
  current full-bucket result for the active forward/recompute window
  replay fragments
```

The physical runtime, not the fragments, owns:

```text
begin/finish all-gather
prefetch identity
full-bucket result lifetime
reduce-scatter launch
```

### Step 4: Install hooks per replay fragment

Instead of one hook target per bucket, install one hook per replay fragment. The
fragment hook:

```text
ensures the physical bucket full result exists
creates fragment-local autograd-visible param tensors from the shared result
sets _pre_gathered only for this fragment's entries
runs the module forward
clears only this fragment's entries
releases the physical result only if this is the last fragment in this window
```

Do not call `begin_all_gather_unshard()` from the fragment runtime.

### Step 5: Preserve one reduce-scatter

The backward path must aggregate all fragment grads into the physical bucket
and launch exactly one reduce-scatter for that bucket.

Acceptance rule:

```text
one grouped root/rest BucketSpec -> one reduce-scatter in backward
```

Do not rely on one shared `_BucketAllGather.backward()` to do this implicitly.
Use an explicit physical-bucket grad accumulator:

```text
PhysicalBucketGradAccumulator:
  expected entry ids or fragment ids for this backward window
  received full-param grads by entry id
  launches one RS when all expected grads for the bucket are present
  rejects duplicate launches
```

Fragment backward paths should only report grads into this accumulator. The
physical bucket owns packing and launching the single reduce-scatter.

### Step 6: Keep forward/backward prefetch order coherent

Once a user bucket can expand into multiple hook runtimes, storage order is no
longer the right prefetch order. For the target layout the user writes layer
buckets first and the root/rest bucket last:

```text
layers.0 bucket
layers.1 bucket
root/rest bucket -> tok_embeddings, pos_embeddings, norm, output fragments
```

Actual forward execution is:

```text
tok_embeddings -> pos_embeddings -> layers.0 -> layers.1 -> norm -> output
```

Therefore the runtime sorts the communication context's bucket runtimes by the
registered hook module order after all hooks are installed. This keeps forward
prefetch and backward recompute prefetch aligned with execution order instead
of user `BucketSpec` declaration order.

For the physical bucket, prefetch identity should be the user bucket, not the
fragment. A prefetched handle belongs to the physical bucket and may be consumed
by the first fragment that needs that bucket in the current direction.

### Step 7: Verification

Focused tests:

```bash
pytest -q \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_api.py::TestFlexShardAPI::test_reshard_after_forward_requires_replayable_bucket_hook

pytest -q \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_training.py::TestFlexShardTraining::test_reshard_after_forward_with_activation_checkpointing
```

Existing correctness tests:

```text
test_reshard_after_forward_grouped_root_rest_bucket
test_reshard_after_forward_requires_replayable_bucket_hook
test_reshard_after_forward_with_activation_checkpointing
```

Communication-contract verification:

```text
test_reshard_after_forward_grouped_root_rest_bucket
```

This test should verify the runtime call counts for the grouped bucket, not just
numerical parity. Expected behavior for one training step:

```text
original forward:
  one physical AG for grouped root/rest bucket

activation-checkpoint recompute:
  one physical AG for grouped root/rest bucket

backward:
  one physical RS for grouped root/rest bucket
```

Verification matrix:

| Case | Expected result | Why |
| --- | --- | --- |
| Layer buckets plus grouped root/rest bucket | Pass forward/backward/optimizer parity | This is the target layout. |
| Grouped root/rest bucket communication count | One physical AG per forward/recompute window and one physical RS | Required bucket contract. |
| Existing root catch-all bucket with checkpointed children | Still rejected | A root hook is not replayed by child checkpoint recompute. |
| Per-layer `reshard_after_forward=True` training | Still passes | Existing supported path must not regress. |
| Full flex_shard tests | Still pass | Fragment support should be isolated to hook installation/runtime scheduling. |

Commands:

```bash
pytest -q \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_training.py::TestFlexShardTraining::test_reshard_after_forward_grouped_root_rest_bucket

pytest -q \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_api.py::TestFlexShardAPI::test_reshard_after_forward_requires_replayable_bucket_hook

pytest -q \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_training.py::TestFlexShardTraining::test_reshard_after_forward_with_activation_checkpointing

pytest -q torchtitan/experiments/flex_shard/tests
```

Full suite:

```bash
pytest -q torchtitan/experiments/flex_shard/tests
```

Profiler/memory:

- Confirm the grouped root/rest bucket does not show one AG/RS per fragment.
- Compare memory against split top-level buckets. The grouped bucket is expected
  to have a longer full-param lifetime because preserving one physical AG
  requires holding the shared full-bucket result from first to last fragment.
- Check that all-gather and reduce-scatter overlap remain healthy.

## Iteration 1 results

Implemented:

- `BucketRuntime` resolves its hook module from runtime entries, not the whole
  storage, so replay fragments can use a subset of the user bucket.
- A root-level, top-level-only, `reshard_after_forward=True` bucket can expand
  into replay fragments by top-level owner module.
- Mixed root/layer buckets are still rejected, preserving the existing safety
  check for ambiguous checkpoint replay boundaries.
- The communication context is sorted by hook-module order after hook
  installation so prefetch follows execution order.

Validation:

```text
python -m py_compile ...                                  PASS
ruff check bucket_runtime.py test_flex_shard_training.py  PASS
test_reshard_after_forward_grouped_root_rest_bucket       PASS
test_reshard_after_forward_requires_replayable_bucket_hook PASS
test_reshard_after_forward_with_activation_checkpointing  PASS
pytest -q torchtitan/experiments/flex_shard/tests         36 passed
```

Iteration 1 limitation:

The implementation supports the exact user-facing grouped root/rest
`BucketSpec`, but the replay fragments are also communication fragments. This
means the grouped root/rest user bucket launches one all-gather/reduce-scatter
per top-level fragment today. Preserving one physical AG/RS for the grouped
user bucket requires the follow-up shared physical bucket design described
above.

## Recommendation

Do not land iteration 1 as-is. Use it only as evidence that fragment hook
placement and execution-order sorting are necessary.

Executed next step:

1. Refactor `BucketRuntime` into a physical bucket runtime plus replay fragment
   runtimes.
2. Make all fragments for one user `BucketSpec` consume one shared physical
   full-bucket result.
3. Add profiler/runtime assertions proving the grouped root/rest bucket has one
   physical AG per forward/recompute window and one physical RS in backward.
4. Re-run the existing correctness suite and compare memory/profiler traces
   against split buckets.

## Shared physical bucket implementation results

Implemented:

- `BucketRuntime` is now the physical user bucket runtime. It owns:
  - all entries for the user `BucketSpec`;
  - the physical all-gather;
  - the shared full-bucket result lifetime;
  - the physical reduce-scatter launch.
- `BucketReplayFragment` owns hook placement for one replay-safe execution
  unit. It never launches its own collective.
- A grouped root/rest bucket expands into replay fragments for
  `tok_embeddings`, `pos_embeddings`, `norm`, and `output`, but all fragments
  share the same physical bucket runtime.
- Fragment hooks expose params through `_BucketFragmentAccess`, which gives each
  checkpointed execution unit a local autograd edge while avoiding a new
  physical all-gather per fragment.
- Fragment backward reports full-param grads into `BucketGradAccumulator`.
  The accumulator launches exactly one reduce-scatter after all expected bucket
  grads arrive.

Important bug found during verification:

```text
layer0 recompute prefetch saw the root/rest bucket as not active because the
check used the candidate bucket's ContextVar state.
```

During `layers.0` recompute, the root/rest bucket is not in the active
ContextVar set, but it can still have a live full-bucket result for the overall
backward recompute window. The fix is to compare the candidate bucket's active
state against the current prefetch direction:

```text
candidate.active_recompute == current_is_recompute
```

instead of recomputing the candidate bucket's local ContextVar state.

Validation:

```text
python -m py_compile bucket_runtime.py test_flex_shard_training.py PASS
ruff check bucket_runtime.py test_flex_shard_training.py           PASS
test_reshard_after_forward_grouped_root_rest_bucket                PASS
  root/rest AG count: 2
    original forward: 1
    activation-checkpoint recompute: 1
  root/rest RS count: 1
test_reshard_after_forward_with_activation_checkpointing           PASS
test_reshard_after_forward_requires_replayable_bucket_hook         PASS
pytest -q torchtitan/experiments/flex_shard/tests                  36 passed
```

Remaining follow-up:

- Dump profiler/memory traces against split top-level buckets. The grouped
  physical bucket intentionally has a longer full-param lifetime because one
  physical AG requires holding the shared result from first fragment to last
  fragment.
