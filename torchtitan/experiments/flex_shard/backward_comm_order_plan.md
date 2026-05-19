# Backward all-gather priority over reduce-scatter

## Goal

Address the reviewer concern that FlexShard can enqueue a backward
reduce-scatter before a later critical-path backward all-gather.

The target scheduling rule is:

- backward all-gather prefetch is on the critical path and should be enqueued
  before a competing reduce-scatter;
- reduce-scatter should still launch early enough to overlap with later backward
  compute;
- the fix should preserve the current `_BucketAllGather` autograd visibility;
- the non-reshard-after-forward path should not regress;
- the implementation should stay traceable by Dynamo and avoid bringing back
  `torch.autograd.graph.register_multi_grad_hook()` in the traced path.

## Verified problem

Profiled command:

```bash
TRACE_DIR=/tmp/flex_shard_rs_ag_order_20260513_100758
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  torchtitan/experiments/flex_shard/compile_repro.py \
  --device cuda --model transformer --depth 8 --dim 1024 --heads 16 \
  --seq-len 512 --vocab-size 32768 --batch-size 12 \
  --reshard-after-forward --profile-eager --profile-dir "$TRACE_DIR" \
  --allow-compile-failure --backend eager
```

Rank0 Perfetto trace:

```text
https://fburl.com/fiz9c5w5
```

Rank0 local trace:

```text
/tmp/flex_shard_rs_ag_order_20260513_100758/flex_shard_eager_rank0.json
```

Observed CPU launch order:

```text
75.389 ms  all_gather      layers.2
79.407 ms  reduce_scatter  layers.3
82.083 ms  all_gather      layers.1
85.985 ms  reduce_scatter  layers.2
88.583 ms  all_gather      layers.0
```

Observed CUDA range order around the reviewer example:

```text
134.520 ms  reduce_scatter_copy_in(layers.2)
134.579 ms  all_gather_copy_out(layers.1)
134.582 ms  reduce_scatter(layers.2)
134.679 ms  all_gather_copy_in(layers.0)
134.873 ms  all_gather(layers.0)
```

This confirms the reviewer concern: `reduce_scatter(layers.2)` is enqueued
before the later critical-path `all_gather(layers.0)`.

## Current scheduling model

For `reshard_after_forward=True`, backward recompute runs bucket pre-forward
hooks in reverse bucket order.

For a bucket `B`:

1. `pre_forward_hook(B)` consumes the pending all-gather for `B`.
2. `pre_forward_hook(B)` calls `prefetch_next()`, which launches the next
   all-gather in the recompute order.
3. Recompute forward for `B` runs.
4. Backward compute for `B` runs.
5. `_BucketAllGather.backward(B)` launches `reduce_scatter(B)`.

This means `reduce_scatter(B)` runs after the all-gather for the immediately
next bucket, but before the next bucket's own pre-forward hook can launch the
following all-gather.

Example:

```text
pre_forward(layers.2) launches AG(layers.1)
backward(layers.2) launches RS(layers.2)
pre_forward(layers.1) launches AG(layers.0)
```

So `RS(layers.2)` can be ahead of `AG(layers.0)`.

## Design options

### Option 1: rely on stream priority

Keep the code structure and make the all-gather stream higher priority than the
reduce-scatter stream.

Rejected as the primary fix:

- stream priority does not reliably reorder already-enqueued NCCL work;
- the trace problem is enqueue order, not only GPU scheduling priority;
- this would not encode the intended FSDP-style ordering rule in FlexShard.

### Option 2: prefetch two backward all-gathers before reduce-scatter

When `_BucketAllGather.backward(B)` is about to launch `RS(B)`, ensure the
pending all-gather queue already contains both the next bucket and the bucket
after that.

Potential implementation:

- replace `BucketCommContext.pending` with a deque of pending all-gather
  handles;
- allow `prefetch_next(target_depth=2)` during recompute;
- launch the second pending all-gather before `RS(B)`.

Pros:

- directly puts `AG(layers.0)` before `RS(layers.2)` in the example;
- keeps reduce-scatter launch in `_BucketAllGather.backward()`.

Cons:

- increases peak all-gather buffer memory by allowing two pending full-param
  buckets;
- complicates stale pending handling;
- deeper prefetch can be too aggressive for large buckets.

This is viable, but it is not the cleanest first implementation.

### Option 3: defer reduce-scatter launch until after the next pre-forward prefetch

Split reduce-scatter into two phases:

1. `_BucketAllGather.backward(B)` records or prepares `RS(B)`, but does not
   enqueue the NCCL reduce-scatter yet.
2. The next bucket's `pre_forward_hook()` first launches its next all-gather
   prefetch, then flushes one deferred reduce-scatter.

Example target order:

```text
pre_forward(layers.2) launches AG(layers.1)
backward(layers.2) defers RS(layers.2)
pre_forward(layers.1) consumes AG(layers.1)
pre_forward(layers.1) launches AG(layers.0)
pre_forward(layers.1) flushes deferred RS(layers.2)
```

This gives:

```text
AG(layers.1) -> AG(layers.0) -> RS(layers.2)
```

Pros:

- preserves a single pending all-gather depth;
- encodes all-gather priority explicitly;
- avoids extra full-param all-gather buffers;
- reduce-scatter can still overlap with `layers.1` backward compute.

Cons:

- needs a small scheduler queue for deferred reduce-scatter work;
- if full grads are stored directly in the queue, memory may increase until the
  next pre-forward hook flushes the queue;
- the tail bucket still has no later compute to overlap with and must flush at
  the end of backward.

This is the recommended first design.

### Option 4: full communication scheduler

Introduce a central bucket communication scheduler with explicit task types:

- all-gather task;
- reduce-scatter task;
- wait/release task.

The scheduler always drains all-gather tasks before reduce-scatter tasks when
both are ready.

Pros:

- most general and easiest to reason about long term;
- naturally supports configurable prefetch depth and bucket tuning.

Cons:

- larger refactor than needed for the first PR;
- higher risk for a landable experimental runtime.

This should be considered only if Option 3 becomes too ad hoc.

## Recommended implementation

Implement Option 3 in two iterations.

### Iteration 1: minimal deferred reduce-scatter queue

Add a small queue to `BucketCommContext`:

```python
@dataclass
class PendingReduceScatterLaunch:
    bucket: BucketRuntime
    prepared: PreparedReduceScatterGrad
    param_refs: list[ParamModuleInfo]
```

Add methods:

```python
def queue_reduce_scatter_launch(
    self,
    bucket: BucketRuntime,
    grads: list[torch.Tensor],
    infos: list[ParamInfo],
    param_refs: list[ParamModuleInfo],
) -> None:
    ...

def flush_pending_reduce_scatter_launches(
    self,
    *,
    max_to_flush: int | None,
) -> None:
    ...
```

Change `_BucketAllGather.forward()` to save whether this instance was created
during reshard-after-forward recompute:

```python
ctx.is_reshard_after_forward_recompute = (
    runtime.bucket.in_reshard_after_forward_recompute()
)
```

Change `_BucketAllGather.backward()`:

- collect full-param grads as it does today;
- if `ctx.is_reshard_after_forward_recompute` is true, enqueue deferred
  reduce-scatter instead of calling `bucket.reduce_grads(...)`;
- otherwise keep the current immediate async `bucket.reduce_grads(...)`;
- continue returning `None` for local shard input grads.

Change `BucketRuntime.pre_forward_hook()`:

```python
full_params = list(_BucketAllGather.apply(runtime, *local_shards))
self.prefetch_next()
self.context.flush_pending_reduce_scatter_launches(max_to_flush=1)
```

Flush only one deferred reduce-scatter after each all-gather prefetch. Flushing
all pending reduce-scatters at once could recreate the same problem for the next
critical all-gather.

Change the queued post-backward callback:

- flush all remaining deferred reduce-scatters;
- wait for all pending reduce-scatter handles;
- release buffers.

This handles tail buckets that have no later pre-forward hook.

### Iteration 2: reduce memory by preparing the send buffer immediately

If Iteration 1 fixes ordering but memory is worse, avoid storing full grad
tensors in the deferred queue.

Refactor `begin_reduce_scatter_grad(...)` into:

```python
prepare_reduce_scatter_grad(...)
launch_reduce_scatter_grad(prepared, reduce_scatter_stream, ...)
```

`prepare_reduce_scatter_grad(...)` should:

- run `placement.pack_reduce_grad(...)` on the autograd current stream;
- record the copy-in event;
- return a prepared request holding `send_buf`, layout, placement, group,
  rank, and world size.

`launch_reduce_scatter_grad(...)` should:

- enqueue the NCCL reduce-scatter on the reduce-scatter stream after the saved
  copy-in event;
- run copy-out and sharded-grad accumulation on the reduce-scatter stream;
- store the `ReduceScatterGradHandle` for post-backward wait/release.

This keeps full grads live only until the packed reduce-scatter send buffer is
created, but still delays NCCL enqueue until after the next all-gather prefetch.

## Correctness details

### Scope of deferral

Defer only for `_BucketAllGather` instances created during
reshard-after-forward recompute.

Do not defer:

- normal forward instances;
- `reshard_after_forward=False` instances;
- compile current-stream paths unless the same ordering issue is visible and the
  implementation remains traceable.

### Queue ownership

The deferred queue belongs to the per-`flex_shard()` `BucketCommContext`, not to
module globals.

Each queue item must carry enough bucket identity to call the same
bucket-owned reduce-scatter path used today.

### Multiple outstanding items

The steady-state expected queue length is one. The implementation should still
support more than one item and flush them FIFO.

If the queue grows beyond one in normal transformer traces, inspect why before
landing. It may indicate a missing pre-forward flush or an unexpected
activation-checkpoint execution order.

### Gradient accumulation

The final accumulation should keep using the existing async side-effect
contract:

- reduce-scatter produces local sharded grads;
- sharded grads are accumulated into original sharded parameters on the
  reduce-scatter stream;
- the post-backward callback waits and releases temporary buffers.

Do not return sharded grads from `_BucketAllGather.backward()`, since that puts
reduce-scatter completion back on the autograd critical path.

## Verification plan

### 1. Unit tests

Run:

```bash
pytest -q torchtitan/experiments/flex_shard/tests
```

At minimum, confirm:

- `reshard_after_forward=True` training parity still passes;
- `reshard_after_forward=False` training parity still passes;
- `param.grad` remains populated on original sharded parameters.

### 2. Trace ordering check

Run the same trace command used to reproduce the issue.

Expected ordering after the fix:

```text
all_gather(layers.1)
all_gather(layers.0)
reduce_scatter(layers.2)
```

Failure pattern:

```text
all_gather(layers.1)
reduce_scatter(layers.2)
all_gather(layers.0)
```

Use a small parser over the rank0 trace to print the CPU launch and CUDA range
order for:

- `all_gather(layers.1)`;
- `reduce_scatter(layers.2)`;
- `all_gather(layers.0)`.

### 3. Overlap check

Run the larger non-CPU-bound transformer profile:

```bash
TRACE_DIR=/tmp/flex_shard_rs_ag_order_fixed_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  torchtitan/experiments/flex_shard/compile_repro.py \
  --device cuda --model transformer --depth 12 --dim 1024 --heads 16 \
  --seq-len 512 --vocab-size 32768 --batch-size 16 \
  --reshard-after-forward --profile-eager --profile-dir "$TRACE_DIR" \
  --allow-compile-failure --backend eager
```

Acceptance:

- reduce-scatter should remain substantially overlapped with later compute;
- both ranks should not regress to fully exposed reduce-scatter;
- compare against the previous async implementation traces, which had about
  70% to 78% reduce-scatter overlap on the larger transformer profile.

### 4. Memory check

Profile with memory enabled:

```bash
--profile-memory --memory-snapshot-dir "$TRACE_DIR/memory"
```

Acceptance:

- Iteration 1 should not show unbounded growth in deferred full-grad tensors;
- if peak memory moves materially, implement Iteration 2 before landing.

### 5. Compile check

After eager ordering is fixed, run the compile trace and tlparse check from the
existing compile workflow.

Acceptance:

- no new graph breaks;
- compile trace still contains real FlexShard bucket all-gather and
  reduce-scatter logic;
- no reintroduction of `register_multi_grad_hook()` graph breaks.

## Open questions

1. Is flushing one deferred reduce-scatter per pre-forward hook enough for all
   expected bucket layouts?
2. Does delaying reduce-scatter until after the next all-gather prefetch reduce
   overlap enough to require a configurable policy?
3. Should the deferral policy be only for reshard-after-forward recompute, or
   should it be a general backward communication scheduler rule?
4. Does the minimal queue of full grad tensors cause a measurable memory
   regression on large buckets?
5. If Iteration 2 is needed, can the reduce-scatter prepare/launch split be
   expressed cleanly without duplicating collective packing logic?

## Implementation status

Implemented both iterations.

### Iteration 1 result

The first implementation added a deferred reduce-scatter queue and flushed one
queued reduce-scatter after each pre-forward all-gather prefetch. The code ran
correctly, but the predicate used `_BucketAllGather.forward()`'s
`in_reshard_after_forward_recompute()` state. The trace showed that this was too
narrow: the recompute marker is not active at the `_BucketAllGather.backward()`
launch point, so the code still took the immediate reduce-scatter path.

### Iteration 2 result

The final implementation uses the memory-friendly prepare/launch split:

- `prepare_reduce_scatter_grad(...)` packs full grads into the reduce-scatter
  send buffer and records the copy-in readiness event;
- `launch_reduce_scatter_grad(...)` later enqueues the NCCL reduce-scatter on
  the reduce-scatter stream;
- `_BucketAllGather.backward()` defers the launch for
  `reshard_after_forward=True` buckets;
- `BucketRuntime.pre_forward_hook()` launches the next all-gather prefetch first
  and then flushes one deferred reduce-scatter;
- the post-backward callback flushes any tail deferred reduce-scatter before
  waiting and releasing buffers.

The deferral predicate is bucket-level `storage._reshard_after_forward`, not the
dynamic recompute marker. The dynamic marker is useful for choosing the
all-gather prefetch order, but it is not reliable at the later autograd
backward launch point.

### Verified after implementation

Focused ordering run:

```text
/tmp/flex_shard_rs_ag_order_fixed2_20260513_103557/flex_shard_eager_rank0.json
```

The CUDA range order around the reviewer example is now:

```text
132.461 ms  reduce_scatter_copy_in(layers.2)
132.521 ms  all_gather_copy_out(layers.1)
132.615 ms  all_gather_copy_in(layers.0)
132.661 ms  all_gather(layers.0)
133.008 ms  reduce_scatter(layers.2)
```

The packed reduce-scatter copy-in still happens early, but the NCCL
`reduce_scatter(layers.2)` launch is now after the critical-path
`all_gather(layers.0)`.

Larger non-CPU-bound run:

```text
/tmp/flex_shard_rs_ag_order_large_20260513_103632
```

Rank0 eager Perfetto:

```text
https://fburl.com/386f0kqb
```

Overlap result:

| Run | Rank | RS kernels | RS time | Overlap | Kernels overlapped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 0 | 16 | 2.689 ms | 2.010 ms, 74.8% | 15/16 |
| Eager | 1 | 16 | 3.558 ms | 2.802 ms, 78.7% | 15/16 |
| Compile trace | 0 | 16 | 2.790 ms | 2.111 ms, 75.7% | 15/16 |
| Compile trace | 1 | 16 | 3.251 ms | 2.517 ms, 77.4% | 15/16 |

The same run reported:

```text
loss=0.333566, grad_norm=0.020426
max_allocated=7452532224, max_reserved=9070182400
```

Unit tests:

```text
pytest -q torchtitan/experiments/flex_shard/tests
35 passed, 14 warnings
```

## Proposed reviewer reply after implementation

After the fix is verified, the review reply should say:

```text
Good point. I reproduced this in the profiler: the old schedule could enqueue
reduce_scatter(layers.2) before all_gather(layers.0). I changed the backward
communication schedule so _BucketAllGather.backward() defers the reduce-scatter
launch during reshard-after-forward recompute. The next bucket pre-forward hook
now launches the next all-gather prefetch first, then flushes one deferred
reduce-scatter. This gives all-gather priority without losing reduce-scatter
overlap with later backward compute. I verified the new rank0 trace has
all_gather(layers.0) before reduce_scatter(layers.2), and the larger transformer
profile still shows reduce-scatter overlap rather than fully exposed
reduce-scatter.
```
