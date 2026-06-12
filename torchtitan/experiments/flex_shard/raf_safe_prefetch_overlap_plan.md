# RAF-safe prefetch and overlap plan

## Goal

Recover all-gather prefetch overlap for FlexShard buckets with
`reshard_after_forward=True` without violating selective activation checkpoint
replay.

The conservative correctness fix is to not prefetch RAF buckets. That is safe,
but it is not the final performance design. The formal fix should preserve this
invariant:

```text
For each checkpointed execution unit, the replay-visible op stream in the
original forward must match the replay-visible op stream in backward recompute.
```

At the same time, the runtime should still be able to launch the next bucket's
all-gather early enough to overlap with useful compute.

## Current failure

The failing pattern is:

```text
forward layers.4 checkpoint region:
  consume prefetched layers.4 all-gather
  launch prefetch for layers.5

backward recompute layers.4 checkpoint region:
  no matching prefetch is available
  launch layers.4 all-gather in this region
```

Selective activation checkpointing then sees dispatch ops during recompute that
were not present in the original forward region, for example:

```text
profiler._record_function_enter_new.default encountered during backward but not
found in storage

aten.empty.memory_format invocation index 2 encountered during backward but not
found in storage
```

There is a second unsafe path: flushing deferred reduce-scatter work from a
checkpoint recompute hook inserts backward-only ops into the replay stream.

## Required invariants

1. A bucket's full parameters must be exposed inside the execution unit that
   reads them.
2. The dispatch ops visible to selective activation checkpointing must be stable
   for that execution unit, independent of whether a physical prefetch is ready.
3. Physical communication may be launched earlier, but the early launch must not
   appear as extra replay-visible ops in another checkpoint region.
4. Backward reduce-scatter launch must not run inside a checkpoint recompute
   context unless the same launch is replay-visible in the original forward,
   which is not true for reduce-scatter.
5. The design must preserve the one-collective-per-physical-bucket contract.

## Proposed design

Split bucket unshard into two concepts:

- **Physical prefetch**: asynchronous allocation and collective launch on the
  FlexShard unshard stream. This is a performance optimization and can happen
  before the bucket's execution unit.
- **Replay-visible consume**: the logical operation inside the bucket's own hook
  that returns the full parameters to parameter accessors. This operation must
  appear in both original forward and recompute for the same execution unit.

The replay-visible consume always runs from the bucket's own pre-forward hook.
It uses an already-launched physical prefetch handle when available; otherwise
it launches the physical unshard synchronously as a fallback. Selective
activation checkpointing should see the same logical consume op in either case.

```text
previous bucket hook:
  launch physical prefetch for next bucket under SAC-transparent boundary

target bucket hook:
  replay-visible consume(next bucket)
    if physical prefetch exists:
      wait and return full params
    else:
      launch hidden physical unshard, wait, return full params
```

### Replay-visible unshard op

Introduce a single logical bucket-unshard op, for example:

```text
flex_shard::consume_bucket_unshard(Tensor[] local_shards, int bucket_id)
    -> Tensor[] full_params
```

This op is called from `_BucketUnshard.forward()` for every bucket hook. The
autograd edge still belongs to `_BucketUnshard`, so `_BucketUnshard.backward()`
continues to own gradient reduction.

The op's implementation should:

1. resolve the live `BucketRuntime` from `bucket_id`;
2. take a matching prefetched handle if present;
3. otherwise launch the physical unshard on demand;
4. wait for the handle and return full parameter tensors.

Add the logical op to the RAF checkpoint policy as `MUST_RECOMPUTE`. The policy
should key off this logical op, not the internal c10d/aten ops used by physical
prefetch.

### SAC-transparent physical prefetch

Physical prefetch should be hidden from selective activation checkpoint's
storage bookkeeping. The first prototype should prove one of these mechanisms:

1. a real `torch.library` or C++ custom op boundary whose internal allocation
   and c10d work are not seen as separate SAC-dispatched ops; or
2. a narrowly scoped internal helper that suspends SAC dispatch tracking only
   around the physical prefetch implementation.

This boundary must be limited to physical communication. It must not hide the
logical consume op, because that op is the replay contract for the bucket.

Physical prefetch must preserve enough trace visibility to diagnose overlap. A
Python `record_function` range is also a dispatched profiler op, so if the first
implementation uses a dispatch-suppression guard it may need to suppress those
Python ranges and rely on c10d/NCCL events instead. Restoring stable
`FlexShard::all_gather` ranges for hidden RAF prefetch likely needs either a
logical consume marker that appears in both forward and recompute, or a lower
level profiler annotation that does not enter SAC storage.

### Prefetch state

Replace the single pending slot with explicit keyed state:

```python
PendingKey = tuple[int, str, int]  # bucket id, direction, backward generation
pending_unshards: dict[PendingKey, UnshardHandle]
```

The direction is:

- `forward` for original forward prefetch;
- `backward_recompute` for RAF backward recompute prefetch.

The generation increments once per train step/backward pass so stale handles
cannot be consumed by a later iteration.

`consume_bucket_unshard()` only consumes a handle for its own bucket, current
direction, and current generation. If the handle is missing or stale, it falls
back to launching the physical unshard in place. That fallback is correctness
safe because the replay-visible consume op is still the same.

### Forward scheduling

Original forward keeps one-bucket-ahead prefetch:

```text
hook(B_i):
  consume logical unshard for B_i
  launch physical prefetch for B_{i+1}
  expose B_i params
```

SAC only sees `consume(B_i)` in `B_i`'s region. The physical prefetch for
`B_{i+1}` is a hidden side effect and no longer pollutes `B_i`'s replay stream.

### Backward recompute scheduling

Backward recompute uses reverse execution order:

```text
recompute hook(B_i):
  consume logical unshard for B_i
  launch physical prefetch for B_{i-1}
  expose B_i params
```

The first recomputed bucket has no previous hook to seed its prefetch. It simply
falls back to launching from `consume(B_last)`. This loses overlap for one tail
bucket but keeps the rest of the backward recompute chain overlapped.

### Reduce-scatter scheduling

Do not flush deferred reduce-scatter work from inside RAF recompute hooks.

Instead:

1. `_BucketUnshard.backward(B_i)` prepares the bucket reduce-scatter input;
2. if the next critical all-gather prefetch has not launched, enqueue the
   reduce-scatter request;
3. after the relevant prefetch is launched, flush at a point outside the SAC
   recompute context;
4. keep the post-backward callback as the final drain and buffer release point.

If there is no non-SAC launch point early enough, the fallback is to launch
reduce-scatter directly from `_BucketUnshard.backward()` after recompute exits.
That is still safer than flushing from a recompute hook and should preserve the
current async gradient accumulation behavior.

## Implementation steps

1. Add the logical `consume_bucket_unshard` op and route `_BucketUnshard.forward`
   through it.
2. Add a fake/meta implementation for compile tests, even if the first target is
   eager, so this does not create a new compile blocker.
3. Add the logical op to `_FLEX_SHARD_COLLECTIVE_OPS` or replace that set with a
   FlexShard-specific replay policy keyed on the logical op.
4. Implement SAC-transparent physical prefetch and prove that internal
   `aten.empty`, profiler, and c10d ops do not create SAC storage entries.
5. Replace `BucketCommContext.pending` with keyed pending state and stale-handle
   cleanup.
6. Re-enable RAF bucket prefetch through the logical consume path.
7. Keep reduce-scatter flushing out of RAF recompute contexts.
8. Delete the temporary `not bucket_storage._reshard_after_forward` prefetch
   guard once the above path is validated.

## Validation

Unit and integration coverage:

- `test_reshard_after_forward_with_activation_checkpointing` passes with RAF
  prefetch enabled.
- A new test forces one bucket to consume a prefetched handle in original
  forward and to fallback-launch during recompute; it should still pass because
  SAC sees only the logical consume op.
- A new test asserts no reduce-scatter launch occurs while
  `in_reshard_after_forward_recompute()` is true.
- Existing non-RAF prefetch behavior and overlap tests continue to pass.
- Grouped root/rest bucket tests still reject unsafe replay boundaries unless
  replay fragments are explicitly enabled.

Trace validation:

- 8-GPU DeepSeek V3 debug model with `dp_shard=8`, `ep=4`,
  `reshard_after_forward=default`, and `training.steps >= 10` reaches the final
  step.
- Trace files exist for all ranks.
- c10d/NCCL all-gather events for RAF buckets start before that bucket's compute
  often enough to recover overlap compared with the conservative no-prefetch
  path. If stable SAC-safe FlexShard profiler ranges are added, use those labels
  instead.
- The first reverse recompute bucket may remain exposed; later buckets should
  show prefetch overlap.
- `FlexShard::reduce_scatter` launches outside checkpoint recompute and overlaps
  with later backward compute.

Numerics:

- Compare loss and grad norm with `--debug.seed=42` and
  `--debug.deterministic` against the conservative no-prefetch RAF path.
- Do not use `--debug.deterministic_warn_only`.

## Risks and open questions

- Python `torch.library` custom ops may still redispatch internal torch ops in a
  way that SAC can see. If so, the physical prefetch boundary needs a C++ op or
  a targeted internal dispatch guard.
- Returning `Tensor[]` from a logical op needs compile/fake coverage before this
  can support the compile path.
- Hiding physical ops from SAC also hides Python FlexShard profiler ranges for
  RAF all-gathers in the first implementation. Trace analysis currently uses
  c10d/NCCL all-gather kernel events for RAF overlap, while non-RAF buckets still
  show `FlexShard::all_gather` ranges.
- The pending-handle generation must be robust to gradient accumulation,
  backward exceptions, and repeated backward attempts.
- RAF-safe prefetch for grouped root/rest physical buckets should compose with
  replay fragments, but that is a follow-up to the single-fragment bucket design.

## Executed iteration

The first implementation keeps `_BucketUnshard` as the eager logical consume
boundary and makes the physical unshard SAC-transparent:

- `BucketRuntime.begin_unshard_from_tensors(..., sac_transparent=...)` hides the
  physical unshard with `torch._C._DisableTorchDispatch()` and suppresses
  FlexShard eager profiler ranges while the hidden work is launched.
- `prefetch_next()` enables RAF prefetch again in original forward order and RAF
  recompute reverse order.
- Pending unshards are keyed by bucket id and execution phase, with stale handles
  drained before falling back to an in-place launch.
- Deferred reduce-grad launches are still not flushed from RAF recompute hooks.
  If the next backward all-gather prefetch is already in flight, reduce-grad
  launches immediately from `_BucketUnshard.backward()` instead of waiting until
  post-backward.
- The RAF checkpoint parity test now runs forward/backward under
  `torch.profiler.profile(...)` so profiler-dispatch mismatches are covered by a
  focused test.

Validation run:

```text
python -m pytest -q -s \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_training.py::\
TestFlexShardTraining::test_reshard_after_forward_with_activation_checkpointing
PASS

python -m pytest -q -s \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_training.py
4 passed

python -m pytest -q \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_runtime.py \
  torchtitan/experiments/flex_shard/tests/test_flex_shard_buckets.py::\
TestMultiMeshBuckets
10 passed, 3 warnings
```

8-GPU profile run:

```text
NGPU=8 MODULE=flex_shard.deepseek_v3 \
CONFIG=flex_shard_deepseek_v3_debugmodel_dp8_ep4 ./run_train.sh \
  --training.steps=10 \
  --activation_checkpoint.mode=none \
  --profiler.enable_profiling \
  --profiler.profile_freq=4 \
  --profiler.profiler_warmup=1 \
  --profiler.profiler_active=1 \
  --profiler.save_traces_folder=profiling/traces_flex_shard_raf_prefetch \
  --dump_folder=outputs/profile_flex_shard_dp8_ep4_raf_prefetch_hidden3
```

Result: completed 10 steps with `reshard_after_forward=default`; rank0 final
loss was `4.47239`, final grad norm was `0.3420`, and 16 trace files were
written for iterations 4 and 8 across ranks 0-7.

Rank0 iteration-8 trace comparison against the earlier conservative
no-RAF-prefetch trace:

```text
conservative_no_raf_prefetch:
  AllGather kernels: 34
  kernels with compute overlap: 1
  capped compute-overlapped time: 44.3 us

raf_prefetch_hidden_final:
  AllGather kernels: 53
  kernels with compute overlap: 26
  capped compute-overlapped time: 5639.5 us
```

Apple-to-apple DeepSeek V3 profile commands:

Both runs use data parallel shard degree 8, expert parallel degree 4, CE loss
instead of chunked CE loss, and no activation checkpointing. The FSDP baseline
also splits `norm` and `lm_head` so its forward all-gather count matches the
FlexShard bucket surface.

```bash
NGPU=8 MODULE=deepseek_v3 CONFIG=deepseek_v3_debugmodel_ep_ce_loss \
./run_train.sh \
  --parallelism.data_parallel_shard_degree=8 \
  --parallelism.expert_parallel_degree=4 \
  --training.steps=10 \
  --activation_checkpoint.mode=none \
  --profiler.enable_profiling \
  --profiler.profile_freq=4 \
  --profiler.profiler_warmup=1 \
  --profiler.profiler_active=1 \
  --profiler.save_traces_folder=profiling/traces_fully_shard_ce_loss \
  --dump_folder=outputs/profile_compare_20260610_153704_fully_shard_dp8_ep4_ce_loss_split_norm_lm
```

```bash
NGPU=8 MODULE=flex_shard.deepseek_v3 \
CONFIG=flex_shard_deepseek_v3_debugmodel_dp8_ep4_ce_loss \
./run_train.sh \
  --training.steps=10 \
  --activation_checkpoint.mode=none \
  --profiler.enable_profiling \
  --profiler.profile_freq=4 \
  --profiler.profiler_warmup=1 \
  --profiler.profiler_active=1 \
  --profiler.save_traces_folder=profiling/traces_flex_shard_ce_loss \
  --dump_folder=outputs/profile_compare_20260610_214924_flex_shard_dp8_ep4_ce_loss_dtensor_ep
```

Iteration-8 AllGather comparison:

```text
fully_shard:
  c10d AllGather per rank: 26 total = 14 forward + 12 backward
  NCCL AllGather kernels per rank: 26 total = 14 forward + 12 backward

flex_shard:
  c10d AllGather per rank: 26 total = 14 forward + 12 backward
  NCCL AllGather kernels per rank: 26 total = 14 forward + 12 backward
```

DTensor ExpertParallel iteration:

- FlexShard DeepSeek V3 now calls the production `apply_moe_ep_tp()` path so
  routed experts are first partitioned by DTensor `ExpertParallel()`.
- FlexShard unwraps DTensor parameters to their local EP payload before
  applying bucket sharding over the FSDP/EFSDP mesh.
- RAF selective checkpointing saves MoE token-dispatch all-to-all outputs so
  parameter unshard recompute does not replay non-FlexShard collectives in
  backward.

Iteration-8 communication launch comparison:

```text
fully_shard:
  c10d AllGather per rank: 26 total = 14 forward + 12 backward
  c10d ReduceScatter per rank: 14 backward
  c10d AllToAll per rank: 25 total = 15 forward + 10 backward

flex_shard with DTensor ExpertParallel:
  c10d AllGather per rank: 26 total = 14 forward + 12 backward
  c10d ReduceScatter per rank: 14 backward
  c10d AllToAll per rank: 25 total = 15 forward + 10 backward
```

Previous non-parity: backward MoE AllToAll.

The earlier FlexShard DeepSeek V3 path manually sliced routed expert parameters
across EP ranks before bucket sharding. That is not the same graph as the
production FSDP path, which applies DTensor `ExpertParallel()` to the routed
experts. To make the traces apple-to-apple, make FlexShard compose with DTensor
`ExpertParallel()`: teach FlexShard to accept or unwrap DTensor parameters while
preserving local shard and gradient placement semantics. This is more work than
manual expert slicing, but it compares FlexShard against the real fully_shard
production EP path. The DTensor ExpertParallel iteration above removes the
extra backward-window AllToAll launches; remaining trace differences are runtime
duration/overhead rather than collective launch-count mismatches.

Remaining follow-ups:

- Add stable SAC-safe FlexShard profiler labels for hidden RAF all-gathers, or
  document that RAF overlap analysis should use c10d/NCCL all-gather events.
- Add an explicit backward generation to the pending-unshard key if the runtime
  grows beyond one in-flight one-bucket-ahead prefetch.
- Run a deterministic loss/grad-norm comparison with `--debug.seed=42` and
  `--debug.deterministic` before landing outside the experiment.

Runtime fragmentation follow-up:

The apple-to-apple traces had matching communication launch counts, but
FlexShard still showed host/runtime fragmentation:

```text
fully_shard:
  step: 108.7 ms
  GPU busy: 72.5 ms (66.7%)
  cudaLaunchKernel: 9,137
  cudaMemcpyAsync: 736
  small GPU gaps: 8.5 ms/rank

flex_shard before pack/split changes:
  step: 204.9 ms
  GPU busy: 93.9 ms (45.8%)
  cudaLaunchKernel: 15,783
  cudaMemcpyAsync: 2,024
  small GPU gaps: 46.2 ms/rank
```

Trace inspection showed two independent sources:

1. `Shard.prepare_unshard_bucket()` packed all-gather inputs with generic
   `aten::cat`, even though materialized FlexShard bucket params are normally
   already contiguous views into one bucket storage.
2. RAF-only selective checkpointing used `PREFER_RECOMPUTE` for all
   non-collective ops, so backward replay recomputed large parts of DeepSeek V3
   model compute, not just parameter unshards.

The current mitigation:

- Use a zero-copy flat view for all-gather input when bucket params are
  contiguous in storage.
- Use `all_gather_single()` with one flat output buffer when per-rank bucket
  sizes are uniform.
- For uniform `Shard(0)` buckets, split the rank-major all-gather output with
  `torch.split_with_sizes_copy(..., dim=1, out=...)` instead of per-parameter
  `cat`.
- Tag eager `_allgather_base_` as a RAF collective so the policy still catches
  the uniform all-gather path.
- In RAF-only selective checkpointing, prefer saving selected heavy pure compute
  ops (`mm`, grouped mm, SDPA, RMSNorm, simple elementwise outputs) while leaving
  factory/indexing/mutating ops recomputed to avoid cached-tensor mutation
  errors.
- Replace the RAF-only module-wide selective-checkpoint wrapper with
  `saved_tensors_hooks`: forward saves lightweight handles for FlexShard full
  params, and backward unpacks those handles by physically re-unsharding only
  the needed bucket parameter values. The original `_BucketUnshard` autograd
  edge still owns reduce-grad, so the unpacked value tensors are value-only.
  Existing activation-checkpointed modules keep the composed selective policy.

Fresh FlexShard DSV3 profile after those changes:

```text
dump:
  outputs/profile_compare_20260610_233830_flex_shard_dp8_ep4_ce_loss_raf_saved_tensor_hooks

flex_shard after changes:
  step: 117.2 ms
  GPU busy: 76.3 ms (65.1%)
  cudaLaunchKernel: 10,418
  cudaMemcpyAsync: 736
  small GPU gaps: 13.2 ms/rank
  PythonDispatchMode: 0
```

This removes about 5.4k kernel launches and eliminates about 26k
`PythonDispatchMode` events versus the previous FlexShard trace. The remaining
gap to FSDP is much smaller: FlexShard is now 117.2 ms/step versus FSDP's
108.7 ms/step on this trace, with the same `cudaMemcpyAsync` count and similar
GPU busy percentage. Remaining overhead is mostly FlexShard-specific bucket
autograd/cast/copy bookkeeping (`_BucketUnshard`, `_BucketUnshardBackward`,
`_UnshardedParamMixedPrecisionCast`, and extra `aten::_to_copy`/`aten::copy_`),
not module-wide selective-checkpoint dispatch.

Follow-up trace after moving mixed-precision casts into bucket unshard/reduce
packing:

```text
dump:
  outputs/profile_compare_20260610_235952_flex_shard_dp8_ep4_ce_loss_bucket_mp_cast

flex_shard after bucket-level mixed precision:
  step: 108.8 ms
  GPU busy: 70.0 ms (64.3%)
  cudaLaunchKernel: 8,654
  cudaMemcpyAsync: 736
  small GPU gaps: 11.3 ms/rank
  _UnshardedParamMixedPrecisionCast: 0
  aten::_to_copy: 1,472
  aten::copy_: 1,824
```

Relative to the saved-tensor-hook trace, this removes the 1,264 per-parameter
`_UnshardedParamMixedPrecisionCast` nodes and about 1,720 `aten::_to_copy` /
`aten::copy_` events. Relative to the matched FSDP trace above, FlexShard now
has the same collective counts and `cudaMemcpyAsync` count, fewer
`cudaLaunchKernel` events, but still more small GPU gaps. The remaining visible
FlexShard-specific surface is the one `_BucketUnshard` / `_BucketUnshardBackward`
autograd pair per bucket.

Follow-up trace after trimming `_BucketUnshardBackward` Python bookkeeping:

```text
dumps:
  flex_shard before:
    outputs/profile_compare_20260611_000536_flex_shard_dp8_ep4_ce_loss_latest_code_after_bucket_mp_cast
  flex_shard after:
    outputs/profile_compare_20260611_004537_flex_shard_dp8_ep4_ce_loss_backward_trim_final
  fresh fully_shard baseline:
    outputs/profile_compare_20260611_004031_fully_shard_dp8_ep4_ce_loss_after_sched_cache

flex_shard before:
  step: 106.4 ms
  GPU busy: 65.4 ms (61.5%)
  cudaLaunchKernel: 8,676
  cudaMemcpyAsync: 736
  small GPU gaps: 12.7 ms/rank
  _BucketUnshardBackward: 112 events, 71.4 ms inclusive, 39.9 ms self

flex_shard after final:
  step: 114.3 ms
  GPU busy: 72.4 ms (63.4%)
  cudaLaunchKernel: 8,720
  cudaMemcpyAsync: 736
  small GPU gaps: 12.1 ms/rank
  _BucketUnshardBackward: 112 events, 52.8 ms inclusive, 20.9 ms self

fully_shard fresh baseline:
  step: 125.2 ms
  GPU busy: 83.3 ms (66.6%)
  cudaLaunchKernel: 9,144
  cudaMemcpyAsync: 736
  small GPU gaps: 10.3 ms/rank
```

The change keeps the existing per-bucket autograd boundary, but removes avoidable
Python work around it:

- Cache forward bucket indices and the RAF recompute prefetch order instead of
  rebuilding and searching that schedule from every bucket backward.
- Reuse the valid gradient/info/owner lists already built in
  `_BucketUnshard.backward()` instead of scanning `full_param_grads` a second
  time before scheduling reduce-grad.
- Stop forcing `grad.contiguous()` at the autograd boundary. Placement packing
  already owns contiguity, dtype conversion, and buffer layout for reduce-grad.
- Avoid maintaining compile-only input-gradient index state in eager backward,
  where reduced grads are accumulated directly into the sharded parameters.

This is a measured improvement, not the full FSDP-style design. The remaining
extra `aten::_to_copy` / `aten::copy_` counts versus FSDP did not change in this
trace because those copies are now the real bucket communication packing path:
fp32 local storage to bf16 all-gather send buffers, and bf16/full grad values to
the configured reduce dtype. Further reducing that path needs a storage/comm
dtype change or a fused/multi-tensor copy-in path, not another per-parameter cast
wrapper. Total step time still has run-to-run noise; the stable signal in these
traces is the `_BucketUnshardBackward` self-time drop from about 39.9 ms to
about 20.9 ms across ranks.

Follow-up trace after FSDP-style foreach copy-in for bucket packing:

```text
dumps:
  flex_shard before foreach copy-in:
    outputs/profile_compare_20260611_004537_flex_shard_dp8_ep4_ce_loss_backward_trim_final
  flex_shard after foreach copy-in:
    outputs/profile_compare_20260611_125232_flex_shard_dp8_ep4_ce_loss_foreach_copy_in
  fully_shard baseline:
    outputs/profile_compare_20260611_004031_fully_shard_dp8_ep4_ce_loss_after_sched_cache

fully_shard:
  step: 125.2 ms
  GPU busy: 83.3 ms (66.6%)
  cudaLaunchKernel: 9,144
  cudaMemcpyAsync: 736
  aten::_to_copy: 1,248
  aten::copy_: 1,616
  aten::_foreach_copy_: 832

flex_shard before foreach copy-in:
  step: 114.3 ms
  GPU busy: 72.4 ms (63.4%)
  cudaLaunchKernel: 8,720
  cudaMemcpyAsync: 736
  aten::_to_copy: 1,472
  aten::copy_: 1,824
  aten::_foreach_copy_: 0

flex_shard after foreach copy-in:
  step: 103.7 ms
  GPU busy: 65.7 ms (63.4%)
  cudaLaunchKernel: 8,742
  cudaMemcpyAsync: 736
  aten::_to_copy: 1,264
  aten::copy_: 1,616
  aten::_foreach_copy_: 416
```

The copy-in change keeps fp32 local storage and allocates communication buffers
directly in the desired dtype. It then copies/casts bucket tensors into those
buffers with `torch._foreach_copy_` instead of materializing intermediate
`tensor.to(dtype)` outputs. This removes the remaining `aten::copy_` delta
against FSDP and reduces the `aten::_to_copy` delta from +224 to +16 total
events across ranks. The remaining +16 `_to_copy` events are outside the
placement copy-in path.

Follow-up trace after batching temporary buffer release handoffs:

```text
dumps:
  flex_shard before batched release:
    outputs/profile_compare_20260611_133738_flex_shard_dp8_ep4_ce_loss_latest_actual
  flex_shard after batched release:
    outputs/profile_compare_20260611_135608_flex_shard_dp8_ep4_ce_loss_batched_release

flex_shard before batched release:
  step: 117.1 ms
  GPU busy: 77.4 ms (66.0%)
  cudaStreamWaitEvent: 3,656 events, 3.3 ms
  _BucketUnshard: 112 events, 16.7 ms inclusive, 13.5 ms self
  _BucketUnshardBackward: 112 events, 54.9 ms inclusive, 26.7 ms self

flex_shard after batched release:
  step: 99.8 ms
  GPU busy: 66.2 ms (66.3%)
  cudaStreamWaitEvent: 2,672 events, 2.6 ms
  _BucketUnshard: 112 events, 13.0 ms inclusive, 10.3 ms self
  _BucketUnshardBackward: 112 events, 46.8 ms inclusive, 23.6 ms self
```

The batched release change preserves the same lifetime rule but holds all
temporary tensors from one async result in a single handoff object. That records
one consumer-stream event and waits on it once per result instead of once per
temporary tensor. The stable signal is the 984 fewer `cudaStreamWaitEvent`
runtime calls and lower bucket self time; total step time remains subject to
normal run-to-run noise.

## Landing criteria

Do not replace the conservative correctness patch with this design until all of
the following are true:

- RAF prefetch is enabled in tests and no SAC mismatch occurs.
- Traces show recovered all-gather overlap for RAF buckets.
- Reduce-scatter is not launched from checkpoint recompute hooks.
- Loss and grad norm match the conservative RAF path under deterministic debug
  settings.
- The implementation does not regress the non-RAF path.
