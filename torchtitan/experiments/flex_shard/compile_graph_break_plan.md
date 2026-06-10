# FlexShard Torch Compile Graph-Break Plan

## Goal

Compile FlexShard while Dynamo traces the same parameter bucketing path used by
eager execution.

The trace should include:

- bucket pre-forward hooks;
- one bucket all-gather producing full parameters for that bucket;
- parameter access consuming hook-provided `_pre_gathered` tensors;
- bucketed gradient reduction through the bucket runtime.

The compile path may use simpler current-stream or functional collective
execution. Preserving the eager CUDA side-stream schedule is not required for
this investigation.

## Constraints

| Constraint | Status |
| --- | --- |
| Do not use the rejected per-parameter compile fallback from the parameter accessor. | Removed. |
| Do not skip hooks or bypass bucket state during compile. | Current trace uses the real bucket hook/runtime path. |
| `dist.record_comm(...)` is profiler metadata only. | Safe to gate during compile. |
| reshard-after-forward checkpoint side effects must be handled without bypassing eager bucketing. | Requires Dynamo checkpoint side-effect handling. |

## Change Summary

| File | Change |
| --- | --- |
| `flex_shard/param_access.py` | Removed the eager-only graph-capture rejection path. Parameter access now relies on hook-provided `_pre_gathered` tensors during compile, the same as eager. |
| `flex_shard/bucket_runtime.py` | Kept Dynamo on the real bucket pre-forward hook path. The hook builds `_BucketAllGather.apply(...)` from bucket local shards and stores the resulting full params in each parameter access state. |
| `flex_shard/bucket_runtime.py` | Changed `_BucketAllGather.backward()` to return local sharded gradients for the original sharded parameter inputs instead of mutating `param.grad` from custom backward. |
| `flex_shard/bucket_runtime.py` | Generalized the bucket autograd all-gather path to CUDA non-reshard-after-forward buckets as well as reshard-after-forward buckets. This avoids the synthetic detached leaf and Python multi-grad-hook path for CUDA. |
| `flex_shard/bucket_runtime.py` | Replaced direct reshard-after-forward `ContextVar.get()` reads with `_is_reshard_after_forward_recompute()`, so Dynamo can specialize on the recompute state instead of tracing `ContextVar.get()`. |
| `flex_shard/bucket_collectives.py` | Added already-finished sync handles for compile-time all-gather and reduce-scatter. Compile still enters the bucket collective functions, but uses current-stream collective execution instead of eager side-stream/event scheduling. |
| `flex_shard/bucket_collectives.py` | Kept `dist.record_comm(...)` only on the eager collective path. The compile path skips it because it is profiler metadata and caused graph breaks. |
| `flex_shard/reshard_after_forward.py` | Added `_is_reshard_after_forward_recompute()` with `torch.compiler.assume_constant_result`. |
| `flex_shard/reshard_after_forward.py` | Replaced the plain recompute context manager with a `TorchDispatchMode` wrapper so checkpoint `context_fn` remains compile-compatible. |
| `tests/test_flex_shard_runtime.py` | Replaced the old “graph capture raises” expectation with a CPU `torch.compile(..., backend="eager")` forward/backward smoke test. |
| `compile_repro.py` | Added a standalone `torchrun` repro for FlexShard compile, profiler dumping, `--reshard-after-forward`, `--fullgraph`, reshard-after-forward checkpoint side-effect config, and the FlexShard unit-test Transformer model. |
| `compile_graph_break_plan.md` and `.html` | Documented the root causes, open questions, verification commands, and trace results for the Dynamo tracing work. |

The important behavioral distinction is that compile no longer bypasses
parameter bucketing. The only compile-only branch is in collective execution:
compiled all-gather/reduce-scatter uses simpler current-stream execution, while
the bucket hook, bucket runtime, `_pre_gathered` parameter access, and bucket
autograd gradient path are still traced.

## Current Direction

| Area | Decision |
| --- | --- |
| Compile collectives | Use synchronous current-stream collective execution during compile. Inductor lowers traced c10d collectives to `_c10d_functional` ops. |
| CUDA bucket unshard | Use the bucket autograd path for both `reshard_after_forward=True` and `reshard_after_forward=False`. |
| Non-reshard-after-forward synthetic leaf | Avoid `pre.detach().requires_grad_(...)` for CUDA buckets by using bucket autograd in eager and compile. |
| Non-reshard-after-forward gradient hook | Avoid `torch.autograd.graph.register_multi_grad_hook(...)` for CUDA buckets by using bucket autograd in eager and compile. |
| reshard-after-forward recompute state | Read `_reshard_after_forward_recompute` through `_is_reshard_after_forward_recompute()` decorated with `torch.compiler.assume_constant_result`. |
| Bucket all-gather backward | Return sharded input gradients from `_BucketAllGather.backward()` instead of mutating `param.grad`. |
| reshard-after-forward checkpoint side effects | Enable `torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint=True` for exact reshard-after-forward hook bucketing. |

## reshard-after-forward Checkpoint Setting

Exact reshard-after-forward hook bucketing currently needs this compiler setting:

```python
torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
```

Without it, Dynamo rejects the real eager hook mutations as checkpoint HOP side
effects. The rejected mutations are `BucketCommContext.pending` updates from
`prefetch_next()` and `take_pending()`. After those are permitted, the trace
continues through the bucket hook path and has no graph breaks in the repro.

Avoiding this setting locally would require either bypassing eager hook state or
redesigning reshard-after-forward bucket state to be functional and graph-visible.

## Root-Cause Status

| Break | Root Cause | Status |
| --- | --- | --- |
| `dist.record_comm(...)` | Profiler metadata is not traceable. | Gated during compile. |
| `pre.detach().requires_grad_(...)` | `requires_grad_()` creates an intermediate that escapes as output. | Resolved for CUDA by using bucket autograd in eager and compile. |
| `torch.autograd.graph.register_multi_grad_hook(...)` | Dynamo cannot proxy the nested Python hook/list arguments. | Resolved for CUDA by using bucket autograd in eager and compile. |
| `ContextVar.get()` | Dynamo cannot trace `ContextVar.get()` directly. | Resolved with `torch.compiler.assume_constant_result`. |
| `_BucketAllGather.backward()` mutating `param.grad` | Checkpoint HOP treats the mutation as an escaping side effect. | Resolved by returning local shard gradients. |
| reshard-after-forward hook state mutations | Checkpoint HOP rejects Python state writes inside the checkpointed region. | Requires `skip_fwd_side_effects_in_bwd_under_checkpoint=True` unless reshard-after-forward state is redesigned. |

## Verified Results

| Scenario | Result |
| --- | --- |
| Single-rank CUDA non-reshard-after-forward compile | Passes. `eager loss=0.418410, grad_norm=1.891607`; `compiled loss=0.418410, grad_norm=1.891607`. |
| Single-rank CUDA non-reshard-after-forward tlparse | Zero graph-break files: `/tmp/flex_shard_nonraf_tlparse_flag_baseline_20260510_174453`. |
| Single-rank CUDA reshard-after-forward without checkpoint side-effect config | Compile succeeds, but tlparse reports four graph breaks from `BucketCommContext.pending` writes. |
| Single-rank CUDA reshard-after-forward with checkpoint side-effect config | Passes. `eager loss=0.418410, grad_norm=1.891607`; `compiled loss=0.418410, grad_norm=1.891606`. |
| Single-rank CUDA reshard-after-forward tlparse with checkpoint side-effect config | Zero graph-break files: `/tmp/flex_shard_raf_tlparse_flag_20260510_174432`. |
| Two-rank CUDA non-reshard-after-forward `--fullgraph` | Passes. `eager loss=0.418410, grad_norm=1.396019`; `compiled loss=0.418410, grad_norm=1.396019`. |
| Two-rank CUDA reshard-after-forward `--fullgraph --skip-fwd-side-effects-under-checkpoint` | Passes. `eager loss=0.418410, grad_norm=1.396019`; `compiled loss=0.418410, grad_norm=1.396019`. |
| Two-rank CUDA reshard-after-forward tlparse with checkpoint side-effect config | Zero graph-break files on both ranks: `/tmp/flex_shard_raf_tlparse_2rank_flag_rank0_20260510_174558` and `/tmp/flex_shard_raf_tlparse_2rank_flag_rank1_20260510_174558`. |
| FlexShard unit tests | `pytest -q torchtitan/experiments/flex_shard/tests` passes: `32 passed`. |

The zero-break reshard-after-forward traces contain bucket collective ops including
`_c10d_functional.all_gather_into_tensor`, `_c10d_functional.wait_tensor`, and
`_c10d_functional.reduce_scatter_tensor`.

## Verification Checklist

- [x] Add standalone repro flag `--skip-fwd-side-effects-under-checkpoint`.
- [x] Re-run single-rank CUDA non-reshard-after-forward compile and tlparse.
- [x] Re-run single-rank CUDA reshard-after-forward compile and tlparse with checkpoint
      side-effect config.
- [x] Re-run two-rank CUDA non-reshard-after-forward `--fullgraph`.
- [x] Re-run two-rank CUDA reshard-after-forward
      `--fullgraph --skip-fwd-side-effects-under-checkpoint`.
- [x] Inspect two-rank reshard-after-forward traces for graph-break files.
- [x] Run FlexShard unit tests.
- [ ] If larger multi-rank runs expose uneven all-gather limitations, replace
      the compile branch with explicit functional collectives that preserve the
      bucket packed-buffer layout.

## Open Questions

1. Is requiring
   `torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint=True`
   acceptable for reshard-after-forward compile? It preserves exact eager hook bucketing, but opts
   into checkpoint behavior where forward side effects are not replayed in
   backward.

2. If that config is not acceptable, should the fix be upstream compiler support
   for these checkpoint hook side effects, or a TorchTitan redesign that makes
   reshard-after-forward bucket state functional and graph-visible?

3. Is changing CUDA non-reshard-after-forward eager to the bucket autograd path acceptable as the
   long-term unification? It removes two compile breaks without a compile-only
   parameter accessor fallback.

4. Should the compile collective branch call functional collectives directly, or
   keep c10d calls and rely on compiler lowering?

5. What numerical proof is sufficient beyond the tiny repro: deterministic loss
   comparison on the debug model, or a larger multi-rank FlexShard config?
