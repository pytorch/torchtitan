# Preserve async reduce-scatter overlap in `_BucketAllGather.backward()`

## Goal

Redesign the reshard-after-forward bucket autograd path so it keeps the same
async reduce-scatter overlap property as the eager non-reshard-after-forward
path.

The target behavior is:

- Bucket all-gather remains visible to autograd so full parameters can be
  recomputed under activation checkpointing.
- Full-parameter grads are packed and reduce-scattered per bucket.
- Reduce-scatter launches on the bucket reduce-scatter stream.
- The autograd thread does not wait for reduce-scatter before it can run later
  backward compute.
- Temporary reduce-scatter buffers are retained until a queued post-backward
  wait releases them.
- The design remains compatible with torch.compile tracing, so it should avoid
  reintroducing `torch.autograd.graph.register_multi_grad_hook()` in the
  traced path.

## Current implementation

### Non-reshard-after-forward path

`BucketRuntime.post_forward_hook()` registers
`torch.autograd.graph.register_multi_grad_hook()` on the detached full-param
leaves created by parameter access.

When all grads for a bucket are available, `_reduce_collected_grads()` calls:

```python
bucket.reduce_grads(grads, infos, param_refs)
```

`reduce_grads()`:

1. waits for any older pending reduce-scatter result;
2. calls `begin_reduce_scatter_grad(...)`;
3. finishes the handle on the reduce-scatter stream;
4. accumulates sharded grads into the original sharded parameters on that stream;
5. stores the handle in `context.reduce_scatter_states`;
6. queues `BucketCommContext.queue_reduce_scatter_wait()` to wait and release
   buffers after backward.

This is the path that currently gives reduce-scatter/backward overlap.

### Reshard-after-forward path

`BucketRuntime.pre_forward_hook()` calls:

```python
full_params = list(_BucketAllGather.apply(runtime, *local_shards))
```

`_BucketAllGather.backward()` currently calls:

```python
sharded_grads = bucket.reduce_grads_to_shards(grads, valid_infos)
return (None, *input_grads)
```

`reduce_grads_to_shards()` calls `result.finish()` on the current stream and
returns sharded grads to autograd. That wait makes autograd schedule later
backward work after the reduce-scatter result is ready, which removes overlap.

## Verified profiler evidence

All runs used the untracked local repro:

```bash
torchrun --standalone --nproc_per_node=2 \
  torchtitan/experiments/flex_shard/compile_repro.py \
  --device cuda --model transformer --depth 8 --dim 512 --heads 8 \
  --seq-len 256 --vocab-size 8192 --batch-size 16 \
  --profile-dir "$TRACE_DIR" --profile-eager --profile-memory \
  --allow-compile-failure
```

For reshard-after-forward runs, the command also passed
`--reshard-after-forward`.

The overlap metric below is reduce-scatter NCCL kernel time overlapped with
non-communication GPU kernels in the same trace.

| Run | Trace dir | Rank | RS kernels | RS time | Overlap | Kernels overlapped |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Current non-reshard-after-forward hook path | `/tmp/flex_shard_rs_overlap_baseline_20260513_005129` | 0 | 12 | 8.456 ms | 2.386 ms, 28.2% | 11/12 |
| Current non-reshard-after-forward hook path | `/tmp/flex_shard_rs_overlap_baseline_20260513_005129` | 1 | 12 | 0.607 ms | 0.033 ms, 5.4% | 1/12 |
| Forced non-reshard path through current `_BucketAllGather.backward()` | `/tmp/flex_shard_rs_overlap_unified_20260513_005205` | 0 | 12 | 4.835 ms | 0.000 ms, 0.0% | 0/12 |
| Forced non-reshard path through current `_BucketAllGather.backward()` | `/tmp/flex_shard_rs_overlap_unified_20260513_005205` | 1 | 12 | 0.519 ms | 0.000 ms, 0.0% | 0/12 |
| Current reshard-after-forward path | `/tmp/flex_shard_rs_overlap_reshard_after_forward_current_20260513_005802` | 0 | 12 | 0.529 ms | 0.000 ms, 0.0% | 0/12 |
| Current reshard-after-forward path | `/tmp/flex_shard_rs_overlap_reshard_after_forward_current_20260513_005802` | 1 | 12 | 28.417 ms | 0.000 ms, 0.0% | 0/12 |
| Current reshard-after-forward compile trace | `/tmp/flex_shard_rs_overlap_reshard_after_forward_current_20260513_005802` | 0 | 12 | 1.875 ms | 0.000 ms, 0.0% | 0/12 |
| Current reshard-after-forward compile trace | `/tmp/flex_shard_rs_overlap_reshard_after_forward_current_20260513_005802` | 1 | 12 | 4.009 ms | 0.000 ms, 0.0% | 0/12 |

The loss and grad norm matched in these runs:

```text
loss=0.333171, grad_norm=0.027899
```

### Larger Transformer confirmation

The smaller runs above were useful for root cause, but the final mitigation
must be checked on a less CPU-bound model. The larger run used:

```bash
torchrun --standalone --nproc_per_node=2 \
  torchtitan/experiments/flex_shard/compile_repro.py \
  --device cuda --model transformer --depth 12 --dim 1024 --heads 16 \
  --seq-len 512 --vocab-size 32768 --batch-size 16 \
  --reshard-after-forward --profile-dir "$TRACE_DIR" \
  --profile-eager --profile-memory --allow-compile-failure
```

Current `reshard_after_forward=True` still showed fully exposed reduce-scatter:

| Run | Trace dir | Rank | RS kernels | RS time | Overlap | Kernels overlapped |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Current reshard-after-forward eager | `/tmp/flex_shard_transformer_overlap_reshard_after_forward_20260513_011358` | 0 | 16 | 2.619 ms | 0.000 ms, 0.0% | 0/16 |
| Current reshard-after-forward eager | `/tmp/flex_shard_transformer_overlap_reshard_after_forward_20260513_011358` | 1 | 16 | 2.495 ms | 0.000 ms, 0.0% | 0/16 |
| Current reshard-after-forward compile | `/tmp/flex_shard_transformer_overlap_reshard_after_forward_20260513_011358` | 0 | 16 | 2.346 ms | 0.000 ms, 0.0% | 0/16 |
| Current reshard-after-forward compile | `/tmp/flex_shard_transformer_overlap_reshard_after_forward_20260513_011358` | 1 | 16 | 3.617 ms | 0.000 ms, 0.0% | 0/16 |

The async side-effect accumulation prototype recovered overlap at this scale:

| Run | Trace dir | Rank | RS kernels | RS time | Overlap | Kernels overlapped |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Async `_BucketAllGather.backward()` prototype eager | `/tmp/flex_shard_transformer_overlap_async_backward_proto_20260513_011936` | 0 | 16 | 3.079 ms | 2.239 ms, 72.7% | 13/16 |
| Async `_BucketAllGather.backward()` prototype eager | `/tmp/flex_shard_transformer_overlap_async_backward_proto_20260513_011936` | 1 | 16 | 5.524 ms | 4.223 ms, 76.4% | 13/16 |
| Async `_BucketAllGather.backward()` prototype compile | `/tmp/flex_shard_transformer_overlap_async_backward_proto_20260513_011936` | 0 | 16 | 3.067 ms | 2.234 ms, 72.8% | 14/16 |
| Async `_BucketAllGather.backward()` prototype compile | `/tmp/flex_shard_transformer_overlap_async_backward_proto_20260513_011936` | 1 | 16 | 5.797 ms | 4.434 ms, 76.5% | 14/16 |

Rank0 Perfetto links:

- Current reshard-after-forward eager: https://fburl.com/x63ofwin
- Current reshard-after-forward compile: https://fburl.com/x8sht3kg
- Async prototype eager: https://fburl.com/tx1lrqtd
- Async prototype compile: https://fburl.com/cueso6zi

The larger run changes the recommendation: async side-effect accumulation in
`_BucketAllGather.backward()` is a viable primary mitigation. The earlier small
run showed that it is not enough when there is too little remaining GPU compute,
but it fixes the unacceptable 100% exposed reduce-scatter case for a compute-heavy
Transformer profile.

Final implementation verification used the same large Transformer command after
changing `_BucketAllGather.backward()` to call `bucket.reduce_grads(...)` and
return `None` for local shard grads:

| Run | Trace dir | Rank | RS kernels | RS time | Overlap | Kernels overlapped |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Final implementation eager | `/tmp/flex_shard_transformer_overlap_async_backward_impl_20260513_012836` | 0 | 16 | 3.641 ms | 2.569 ms, 70.6% | 13/16 |
| Final implementation eager | `/tmp/flex_shard_transformer_overlap_async_backward_impl_20260513_012836` | 1 | 16 | 3.171 ms | 2.332 ms, 73.5% | 14/16 |
| Final implementation compile | `/tmp/flex_shard_transformer_overlap_async_backward_impl_20260513_012836` | 0 | 16 | 3.140 ms | 2.276 ms, 72.5% | 14/16 |
| Final implementation compile | `/tmp/flex_shard_transformer_overlap_async_backward_impl_20260513_012836` | 1 | 16 | 8.363 ms | 6.556 ms, 78.4% | 14/16 |

Final implementation rank0 Perfetto links:

- Eager: https://fburl.com/zxalra7z
- Compile: https://fburl.com/dn69q27m

## Rejected prototypes

The following prototypes were applied locally, profiled, and then reverted.
No prototype code remains in the working tree.

### Prototype 1: async side-effect accumulation in `_BucketAllGather.backward()`

Change tested:

- Replace `reduce_grads_to_shards()` with `reduce_grads()`.
- Accumulate sharded grads into original sharded params from the
  reduce-scatter stream.
- Return `None` for all local shard input grads.

Result:

| Run | Trace dir | Rank | RS kernels | RS time | Overlap | Kernels overlapped |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Forced non-reshard path through async prototype | `/tmp/flex_shard_rs_overlap_forced_autograd_async_proto_20260513_005952` | 0 | 12 | 0.533 ms | 0.040 ms, 7.4% | 2/12 |
| Forced non-reshard path through async prototype | `/tmp/flex_shard_rs_overlap_forced_autograd_async_proto_20260513_005952` | 1 | 12 | 13.404 ms | 7.088 ms, 52.9% | 11/12 |
| Reshard-after-forward async prototype | `/tmp/flex_shard_rs_overlap_reshard_after_forward_async_proto_20260513_005852` | 0 | 12 | 0.523 ms | 0.004 ms, 0.7% | 1/12 |
| Reshard-after-forward async prototype | `/tmp/flex_shard_rs_overlap_reshard_after_forward_async_proto_20260513_005852` | 1 | 12 | 10.463 ms | 0.152 ms, 1.5% | 8/12 |

Conclusion:

- Returning unresolved sharded grads to autograd is not required for correctness.
- Manual async accumulation is a necessary primitive.
- The first smaller run did not recover enough overlap, but the larger
  Transformer run above showed this is sufficient for the target non-CPU-bound
  case.

### Prototype 2: per-full-param custom autograd collector

Change tested:

- Wrap each full parameter returned by `_BucketAllGather.forward()` in a tiny
  custom autograd function.
- Each wrapper records that parameter's full grad.
- The last wrapper for the bucket calls `bucket.reduce_grads(...)`.
- `_BucketAllGather.backward()` receives `None` grads and does not launch
  reduce-scatter.

Result:

| Run | Trace dir | Rank | RS kernels | RS time | Overlap | Kernels overlapped |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Reshard-after-forward collector prototype | `/tmp/flex_shard_rs_overlap_reshard_after_forward_collector_proto_20260513_010136` | 0 | 12 | 6.683 ms | 0.000 ms, 0.0% | 0/12 |
| Reshard-after-forward collector prototype | `/tmp/flex_shard_rs_overlap_reshard_after_forward_collector_proto_20260513_010136` | 1 | 12 | 6.784 ms | 0.000 ms, 0.0% | 0/12 |
| Reshard-after-forward collector compile trace | `/tmp/flex_shard_rs_overlap_reshard_after_forward_collector_proto_20260513_010136` | 0 | 12 | 0.538 ms | 0.000 ms, 0.0% | 0/12 |
| Reshard-after-forward collector compile trace | `/tmp/flex_shard_rs_overlap_reshard_after_forward_collector_proto_20260513_010136` | 1 | 12 | 15.172 ms | 0.000 ms, 0.0% | 0/12 |

Conclusion:

- Splitting the multi-output `_BucketAllGather.backward()` node is not enough.
- In the measured checkpointed schedule, the full-param grad collection still
  runs after useful backward compute for the bucket, so the reduce-scatter has
  no later compute to overlap with.

## Root cause hypothesis

There are two separate issues:

1. `_BucketAllGather.backward()` currently performs a real current-stream wait
   through `reduce_grads_to_shards()`. This is definitely wrong for overlap.
2. For `reshard_after_forward=True`, activation checkpointing changes where the
   full-param gradient path is scheduled. Even after removing the synchronous
   current-stream wait, the measured reduce-scatter kernels still land after the
   useful backward compute. That means the launch point is too late, not just
   the stream wait.

FSDP2 avoids relying only on the unshard autograd node for gradient reduction.
It registers post-backward logic on the parameter `AccumulateGrad` object and
launches gradient reduction on a post-backward stream once the parameter's grad
is finalized. FlexShard needs an equivalent bucket-level launch point that is
early enough to overlap with remaining backward compute and is still traceable.

## Mitigation proposals

### Proposal 1: make `_BucketAllGather.backward()` async and side-effect based

This is the primary recommendation.

Change `_BucketAllGather.backward()` to:

1. collect non-`None` full-param grads;
2. call the existing async `bucket.reduce_grads(...)` path with the matching
   `ParamModuleInfo` references;
3. manually accumulate reduced sharded grads into the original sharded
   parameters on the reduce-scatter stream;
4. retain the `ReduceScatterGradHandle` in `context.reduce_scatter_states`;
5. queue the existing post-backward callback to wait and release buffers;
6. return `None` for all local shard input grads.

This keeps `_BucketAllGather` as the autograd-visible all-gather boundary, but
removes reduce-scatter from the autograd critical path. The large prototype
recovered about 73% to 76% reduce-scatter overlap in both eager and compile
traces.

Risks to check before landing:

- It changes the sharded parameter gradient edge from normal autograd returned
  grads to FlexShard-owned manual `.grad` accumulation. This matches the
  current non-reshard-after-forward bucket path, but any supported parameter
  hook behavior should be made explicit.
- `ctx.local_shard_dtypes` becomes unnecessary in `_BucketAllGather.forward()`
  once backward no longer returns sharded grads.
- `reduce_grads_to_shards()` may become unused and should be deleted or kept
  only if another callsite still needs synchronous returned shard grads.
- Need a focused test that `param.grad` is populated and numerically identical
  after backward for `reshard_after_forward=True`.

### Proposal 2: keep `reshard_after_forward=False` for performance-critical buckets

This is a configuration-level mitigation, not the core fix.

The current non-reshard-after-forward hook path already overlaps reduce-scatter:

- rank0 large run: 70.9% overlap, 15/16 kernels overlapped;
- rank1 large run: 71.9% overlap, 15/16 kernels overlapped.

If memory allows, users can temporarily set `reshard_after_forward=False` for
hot buckets while keeping `reshard_after_forward=True` only where full-param
memory pressure matters most. This avoids the exposed reduce-scatter path but
does not solve the compiler-friendly reshard-after-forward design.

### Proposal 3: bucket and schedule tuning after Proposal 1

After async side-effect accumulation, a few tail reduce-scatter kernels can
remain exposed because no later backward compute exists. Mitigations:

- merge very small tail buckets to reduce launch overhead;
- avoid putting the earliest-forward parameters in large buckets if their
  reduce-scatter lands at the very end of backward;
- inspect per-bucket exposed time and tune `BucketSpec` order/layout.

This should be treated as a second-order tuning pass. It cannot fix the current
100% exposed reduce-scatter by itself.

### Proposal 4: FSDP2-style post-accumulate launch point as fallback

If Proposal 1 breaks unsupported-but-important parameter hook semantics, the
fallback is a bucket-level post-accumulate launch point similar in spirit to
FSDP2:

- attach to the point where the original sharded parameter grad is finalized;
- collect bucket grads in per-backward state;
- launch the same async reduce-scatter accumulation helper once the bucket is
  complete.

This is more complex and should not be the first implementation now that the
large profile shows Proposal 1 restores overlap.

## Proposed direction

Do not land a unification that routes non-reshard-after-forward through the
current `_BucketAllGather.backward()` implementation. The forced run proves that
this regresses overlap to zero.

The implementation should happen in two steps.

### Step 1: make reduce-scatter completion side-effect based

Introduce a bucket-level helper, for example:

```python
def reduce_grads_async_accumulate(
    self,
    grads: list[torch.Tensor],
    infos: list[ParamInfo],
    param_refs: list[ParamModuleInfo],
) -> None:
    ...
```

This should be the existing `reduce_grads()` contract made explicit:

- launch `begin_reduce_scatter_grad(...)`;
- finish and accumulate on `context.reduce_scatter_stream`;
- retain the handle in `context.reduce_scatter_states`;
- queue the existing post-backward wait and buffer release.

Then `_BucketAllGather.backward()` should not call
`reduce_grads_to_shards()` and should not return sharded grad tensors to
autograd. It should either call this async helper directly or delegate to the
final launch mechanism below.

This removes the known blocking bug and makes the ownership model match the
non-reshard-after-forward hook path.

### Step 2: verify whether the async launch point is sufficient

The large Transformer profile shows that async side-effect accumulation is
sufficient for the non-CPU-bound target case. The implementation should still
verify this in the final code shape and check for any remaining tail exposure.
Only if the final code regresses should we prototype a bucket-level equivalent
of FSDP2's post-accumulate launch point.

Important constraint: do not reintroduce `register_multi_grad_hook()` into the
torch.compile-traced path unless Dynamo support for it is confirmed. It was one
of the previous graph-break sources.

## Open questions to answer during implementation

1. What is the earliest traceable autograd point where the original sharded
   parameter grad is finalized for a bucket under activation checkpointing?
2. Can `register_post_accumulate_grad_hook()` on original sharded parameters
   give the right launch timing without hiding the bucket logic from Dynamo?
3. If a hook API is not traceable, can we express the same launch point with a
   custom autograd function that Dynamo/AOTAutograd can trace?
4. How should per-backward bucket state be keyed so multiple forwards before one
   backward cannot mix gradients between forward instances?
5. Does manual `.grad` accumulation change any supported parameter hook
   behavior compared with the existing non-reshard-after-forward path?
6. How should the final design handle gradient accumulation when `param.grad`
   is already populated before the current backward?

## Acceptance criteria

The redesign is not ready until all of these pass:

1. Numerics match the current run for the transformer repro:

   ```text
   loss=0.333171, grad_norm=0.027899
   ```

2. `reshard_after_forward=True` eager trace shows reduce-scatter kernels
   interleaved with later non-communication GPU kernels. A result where both
   ranks have `0/12` overlapped kernels is a failure.

3. `reshard_after_forward=True` compile trace still has no graph breaks and
   still traces the real bucket all-gather and reduce-scatter logic.

4. Non-reshard-after-forward eager trace does not regress from the current hook
   path. It must not look like the forced synchronous `_BucketAllGather.backward()`
   run, where both ranks had zero overlapped kernels.

5. FlexShard tests pass:

   ```bash
   pytest -q torchtitan/experiments/flex_shard/tests
   ```

6. The temporary debug script remains uncommitted:

   ```text
   torchtitan/experiments/flex_shard/compile_repro.py
   ```

## Current recommendation

The best current path is:

1. Keep the existing non-reshard-after-forward tensor-hook path for now.
2. Do not land a patch that simply returns sharded grads from
   `_BucketAllGather.backward()` after `result.finish()`.
3. Change `reshard_after_forward=True` `_BucketAllGather.backward()` to call
   async side-effect accumulation and return `None` for local shard grads.
4. Delete or isolate the synchronous `reduce_grads_to_shards()` path.
5. Re-run the large Transformer eager and compile profiles and require
   reduce-scatter overlap close to the async prototype result before landing.
6. Only consider a FSDP2-style post-accumulate trigger if Proposal 1 breaks
   required hook semantics or loses overlap in the final code shape.
