# RaggedShard DBuffer Layout Plan

## Goal

Redesign RaggedShard bucket layout to support a DBuffer-style, param-major
layout where:

1. local sharded parameters are views into bucket storage (`view-in`);
2. unsharded full parameters are views into gathered bucket storage (`view-out`);
3. the runtime issues one unshard collective per bucket;
4. gradient reduction keeps one reduce collective per bucket;
5. the contract is explicit enough to support padding and permutation.

The target is closer to veScale grouped RaggedShard than the current per-param
RaggedShard implementation.

## Current Layout

Current `RaggedShard` is per-param:

```text
rank-local bucket storage:
  [param0 local shard][param1 local shard]...[paramN local shard]

all_gather result:
  rank0 bucket, rank1 bucket, ..., rankW bucket
```

That makes the gathered storage rank-major. A full param needs pieces from every
rank:

```text
param0 full = [rank0 param0][rank1 param0]...[rankW param0]
```

Those pieces are separated by the other params inside each rank-major bucket.
Therefore, for a multi-param bucket, a dense full param cannot generally be
represented as a single `torch.Tensor` view into the rank-major gathered output.
The current `torch.cat()` in `finish_prepared_unshard()` is the copy-out that
repairs that layout mismatch.

Current copy sites:

1. `ShardedBucketStorage.copy_params_from()` copies initial local shards into
   bucket storage.
2. `RaggedShard.prepare_unshard_bucket()` builds a flat send buffer with
   `torch.cat()`.
3. `RaggedShard.finish_prepared_unshard()` builds each full param with
   `torch.cat()`.
4. `RaggedShard.prepare_reduce_grad()` packs full grads into reduce-scatter
   segments.

Initial materialization can keep a one-time copy, but steady-state unshard and
reshard should avoid copy-in/copy-out on the forward runtime path.

## Constraint Check

With standard NCCL all-gather, the output order is the concatenation of each
rank's input buffer. If each rank's input buffer is `[p0 local][p1 local]`, then
the output is rank-major. If full params must be ordinary dense tensors, each
param's storage must be contiguous in param-major order.

For a bucket containing more than one non-empty param, these requirements
conflict unless the bucket layout itself changes:

1. one rank sends one contiguous bucket buffer;
2. all-gather writes rank-major output;
3. every full param is a dense view;
4. every rank keeps per-param local shards in per-param order.

A DBuffer-style solution must plan the bucket as one global logical buffer first,
then derive rank-local views and param views from that plan. It cannot be added
only inside `RaggedShard.prepare_unshard_bucket()`.

## Proposed Contract

Add a bucket-level layout contract beside the existing per-param `Placement`
contract.

```python
@dataclass(frozen=True)
class BucketParamLayout:
    fqn: str
    global_shape: torch.Size
    global_numel: int
    global_offset: int
    padded_global_numel: int
    local_shape: torch.Size
    local_offset: int
    local_numel: int


@dataclass(frozen=True)
class BucketLayout:
    param_layouts: dict[str, BucketParamLayout]
    local_numel: int
    local_padded_numel: int
    global_numel: int
    global_padded_numel: int
    dtype: torch.dtype
```

The layout planner owns:

1. param-major global order;
2. padding between params and/or ranks;
3. rank-local contiguous bucket ranges;
4. per-param local views into local bucket storage;
5. per-param full views into gathered bucket storage.

The runtime owns:

1. bucket storage allocation;
2. one all-gather or reduce-scatter per bucket;
3. lifetime of gathered and reduction buffers;
4. installing sharded/full param views.

## DBuffer Layout Candidate

Plan one logical param-major global buffer:

```text
global gathered bucket:
  [param0 full][pad][param1 full][pad]...[paramN full][pad]
```

Then partition that logical buffer into rank-local contiguous ranges:

```text
rank0 local range = global[rank0_start:rank0_end]
rank1 local range = global[rank1_start:rank1_end]
...
```

If every rank sends exactly its local range and the collective gathers ranges in
rank order, the gathered output is the global bucket buffer. Each full parameter
is then a view:

```python
full_param = gathered_bucket[param.global_offset:param.global_offset + param.numel()]
full_param = full_param.view(param.global_shape)
```

This achieves one collective and view-out.

The tradeoff is semantic: local ownership is now a bucket-global range, not
independent per-param `local_units`. A rank's range may include the tail of one
param, whole middle params, and the head of another param. This is acceptable for
a grouped DBuffer placement, but it is not the current per-param RaggedShard
contract.

## Open Design Point

To keep the existing per-param `RaggedShard(local_units=...)` semantics exactly,
the rank-local payload for a bucket is generally:

```text
[param0 rank shard][param1 rank shard]...
```

That cannot become param-major dense full-param views after one plain all-gather
without either:

1. copy-out after the collective;
2. one collective per param or a coalesced set of per-param collectives;
3. a collective API that scatters each rank's input into non-contiguous output
   locations;
4. changing semantics to a grouped bucket-global layout.

The implementation should therefore introduce a new grouped placement or an
explicit bucket layout mode instead of silently changing `RaggedShard`.

## Iteration Plan

### Iteration 1: Prove and Guard the Existing Limitation

Add focused unit coverage that documents why current per-param RaggedShard cannot
provide view-out for multi-param buckets with one standard all-gather.

Checks:

1. current prepared unshard send buffer does not alias bucket storage;
2. current full params do not alias gathered buffers;
3. a single-param bucket can be made view-out with the existing rank-major output;
4. a multi-param bucket requires either copy-out or a new layout contract.

No production behavior change in this iteration.

### Iteration 2: Add Bucket Layout Contract

Introduce `BucketLayout` metadata without changing default placements:

1. `ShardedBucketStorage` asks a placement whether it supports bucket-level
   planning for the whole bucket;
2. default placements continue to use the current per-param path;
3. a grouped RaggedShard layout can reserve one local bucket range and install
   local param views from that range;
4. tests assert that local param tensors alias the bucket storage.

### Iteration 3: Grouped RaggedShard DBuffer Placement

Add a grouped placement, tentatively `GroupedRaggedShard`, that:

1. plans a param-major global bucket;
2. inserts padding so rank ranges and param slices can be represented as views;
3. partitions the global bucket into rank-local ranges using `local_units`;
4. exposes per-param local views where possible;
5. rejects shapes/layouts that would require non-view local shards.

The placement should fail loudly for unsupported layouts instead of falling back
to hidden copies.

### Iteration 4: One-Collective View-In/View-Out Unshard

Use the grouped layout for unshard:

1. send buffer is a typed view of the bucket's local storage;
2. all-gather writes into one gathered bucket buffer;
3. full params are views into the gathered bucket buffer;
4. tests assert `data_ptr()` and storage offsets share backing storage;
5. profiler scopes no longer show RaggedShard `all_gather_copy_in` or
   `all_gather_copy_out` on the grouped path.

### Iteration 5: One-Collective Reduce Path

Mirror the DBuffer layout for gradient reduction:

1. full grads are viewed from the param-major bucket when possible;
2. reduce-scatter writes directly into local bucket grad storage;
3. local grad tensors are views into the reduce output;
4. unsupported mixed/padded layouts produce a `ValueError`.

### Iteration 6: Runtime and Compile Validation

Run:

```bash
python -m pytest -q torchtitan/experiments/flex_shard/tests/test_flex_shard_ragged_shard.py
python -m pytest -q torchtitan/experiments/flex_shard/tests/test_flex_shard_buckets.py
```

Then run the 2-rank CUDA runtime test and a profiler trace. The trace should show
one NCCL unshard collective per bucket and no steady-state copy-in/copy-out scopes
for grouped RaggedShard.

## Acceptance Criteria

1. `view-in`: unshard send buffer aliases bucket local storage.
2. `view-out`: full params alias one gathered bucket buffer.
3. exactly one unshard collective per bucket.
4. exactly one gradient reduce collective per bucket.
5. unsupported layouts fail with actionable `ValueError`s.
6. existing `RaggedShard` tests continue passing.
7. new grouped DBuffer tests cover CPU aliasing and 2-rank CUDA runtime.

## Current Decision

Do not try to remove copy-out from current per-param RaggedShard directly. That
would either keep hidden copies or violate the one-collective requirement for
multi-param buckets. The next implementation step is to add the bucket-level
layout contract and prototype a grouped DBuffer placement behind explicit tests.

## Execution Notes

Implemented prototype:

1. added a private bucket-level planner protocol in `bucket_storage.py`;
2. added bucket-global offsets and rank range metadata to `ParamInfo`;
3. added `GroupedRaggedShard`, which plans one param-major global bucket and
   partitions it into rank-local ranges;
4. made grouped unshard use a send buffer that aliases local bucket storage;
5. made grouped unshard return full params as views into one gathered bucket;
6. kept exactly one `dist.all_gather()` per grouped unshard bucket;
7. kept exactly one `dist.reduce_scatter_tensor()` per grouped reduce bucket;
8. added CPU aliasing tests and a 2-rank CUDA runtime test.

Validated commands:

```bash
python -m ruff check torchtitan/experiments/flex_shard/flex_shard/placement_contract.py torchtitan/experiments/flex_shard/flex_shard/bucket_storage.py torchtitan/experiments/flex_shard/flex_shard/utils.py torchtitan/experiments/flex_shard/example/__init__.py torchtitan/experiments/flex_shard/example/ragged_shard.py torchtitan/experiments/flex_shard/tests/test_flex_shard_ragged_shard.py
pre-commit run ufmt --files torchtitan/experiments/flex_shard/flex_shard/placement_contract.py torchtitan/experiments/flex_shard/flex_shard/bucket_storage.py torchtitan/experiments/flex_shard/flex_shard/utils.py torchtitan/experiments/flex_shard/example/__init__.py torchtitan/experiments/flex_shard/example/ragged_shard.py torchtitan/experiments/flex_shard/tests/test_flex_shard_ragged_shard.py
pre-commit run flake8 --files torchtitan/experiments/flex_shard/flex_shard/placement_contract.py torchtitan/experiments/flex_shard/flex_shard/bucket_storage.py torchtitan/experiments/flex_shard/flex_shard/utils.py torchtitan/experiments/flex_shard/example/__init__.py torchtitan/experiments/flex_shard/example/ragged_shard.py torchtitan/experiments/flex_shard/tests/test_flex_shard_ragged_shard.py
pre-commit run pydoclint --files torchtitan/experiments/flex_shard/flex_shard/placement_contract.py torchtitan/experiments/flex_shard/flex_shard/bucket_storage.py torchtitan/experiments/flex_shard/flex_shard/utils.py torchtitan/experiments/flex_shard/example/__init__.py torchtitan/experiments/flex_shard/example/ragged_shard.py torchtitan/experiments/flex_shard/tests/test_flex_shard_ragged_shard.py
python -m pytest -q torchtitan/experiments/flex_shard/tests/test_flex_shard_ragged_shard.py
python -m pytest -q torchtitan/experiments/flex_shard/tests/test_flex_shard_buckets.py
```
