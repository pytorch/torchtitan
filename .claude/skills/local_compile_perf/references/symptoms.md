# Symptom -> fix

Read the kernel table from the profiler (see harness.md), then look up what you see.

| Trace symptom | Likely cause | Try (pattern #) |
|---|---|---|
| `CatArrayBatchedCopy` / cat or stack kernels in eager glue, incl. a stack in the MoE backward | a cat outside a region, split-backward cats, or an eager unbind of a packed gate/up projection | put producer + cat + consumer in one region; unbind inside it; try both cat lowerings (2, 5) |
| `triton_red_*` / `triton_per_*` over a dim of size 2-16, or a reduction with `XBLOCK=64` far off ideal | small reduction not unrolled; the dim may be symbolic | static: Python loop, `unroll_reductions_threshold`, `mark_static` inside the region (4, 5); varying width: `autograd.Function` or a GEMM (4, 17) |
| fp32 SIMT GEMM (`sgemm`, cutlass simt), or a large fp32 copy before a GEMM | fp32 matmul precision differs from the trainer's, or an upcast materialized | match the trainer's fp32 matmul precision, then 13 |
| Region kernel with many masked loads, slow vs ideal around a cat | pointwise-cat lowering on a bad shape | `max_pointwise_cat_inputs=0` (2) |
| `gemv`/`gemvx` or `extern_kernels.bmm` with a tiny inner dim, plus a separate fp32 copy | extern GEMV | explicit weighted sum (3) |
| `elementwise_kernel<...BinaryFunctor...>` non-vectorized broadcast mul on a wide tensor | per-row scale in eager on the widest tensor | move the scale to a narrower tensor inside a region (7) |
| `aten` complex mul / `view_as_complex` kernels surviving under compile | no complex codegen | real arithmetic (1) |
| `index_put`, `scatter_add`, `indexing_backward_kernel`, atomics | scatter combine or gather backward | gather-sum `autograd.Function` (6); measure Inductor's compiled scatter first |
| Same region compiled twice: static then dynamic graph | a small dim became symbolic | `mark_static` on fixed dims, `maybe_mark_dynamic` on T in eager (5) |
| `recompiles` log: guard on `4096 <= size` | mix-order heuristic | `triton.mix_order_reduction_non_strict_mode` |
| `recompiles` log: `GLOBAL_STATE changed: grad_mode` | no_grad pass | budget; fewer call signatures |
| `FailOnRecompileLimitHit` | too many signatures for one code object | fewer layouts per region, non-strict mix-order, split regions |
| `cudaStreamSynchronize`, DtoH memcpy, CPU gaps between kernels | `.item()`, `.tolist()`, boolean-mask indexing, blocking copies | use returned counts, `non_blocking=True` from pinned memory, sentinel sort + host-int slice (11) |
| Many tiny kernels from a loop over a tiny matrix (`[T, 4, 4]`) | each iteration realized | per-entry unrolling (9) |
| Library kernel runs differently under compile (e.g. Triton instead of CuTe) | library branches on `is_compiling()` | check the library; wrap its fast path as a custom op (14) |
| Graph break at a library call | non-traceable kernel | custom op with a correct fake (14) |
| Region slower at small T only | host time per region call | merge regions, or accept under CUDA graphs |
| `direct_copy_kernel` on a region-output-shaped tensor in the backward | the region returns a strided view; its gradient arrives in another layout and AOTAutograd inserts a copy | return a dense (contiguous) tensor under compile |
| Kernel over `[T, N, D]` with a symbolic small N, index math dividing by N, far off ideal | 1-D tiling folds the symbolic N into every index | `triton.prefer_nd_tiling=True` on the region (gating-and-options.md) |
