# Autoresearch Kernel Experiment Log

## Baseline — keep (460d8d3)

- **Idea**: Establish baseline with autoresearch_kernel_pass enabled as a no-op placeholder.
- **Changes**: None — ran the existing code as-is.
- **Result**: tps=6621, MFU=38.77%, memory=30.9 GiB. (Second run with GPU contention: tps=6040-6464, ~7% variance.)
- **Analysis**: This is the reference point for all subsequent kernel optimizations.
- **Lessons**: The pass runs after regional Inductor, so many fusible patterns may already be compiled. Need to dump the graph to see what's left unfused.

## Annotate silu+mul for regional Inductor — crash (xxxxxxx)

- **Idea**: Tag the 32 silu+mul (SwiGLU) pairs in the forward graph with `compile_with_inductor` so regional_inductor fuses them into single Triton kernels. Moved the autoresearch_kernel_pass before regional_inductor_pass.
- **Changes**: Modified `autoresearch_kernel_pass.py` to annotate silu nodes and their mul consumers. Moved pass ordering in `passes.py`.
- **Result**: OOM during Inductor compilation (SIGKILL by Linux OOM killer). Never reached training.
- **Analysis**: Regional Inductor compilation of 32 silu+mul regions requires significant memory for tracing/codegen. Combined with an existing 10 GiB process on GPU 0, this exceeded system memory limits.
- **Lessons**: Annotating many small regions for Inductor has high compilation memory overhead. Future approaches should either: (1) minimize the number of annotated regions, (2) work post-Inductor with direct graph transformations instead, or (3) use custom Triton kernels registered as ATen ops.

## In-place silu_ replacement — discard (xxxxxxx)

- **Idea**: Replace `silu.default` with `silu_.default` (in-place) where the input tensor has no other consumers, eliminating the intermediate buffer allocation.
- **Changes**: Modified `autoresearch_kernel_pass.py` to find silu ops with single-consumer inputs and replace their target with `aten.silu_.default`. 32 replacements made.
- **Result**: tps=2504 (60% regression from ~6250 baseline), memory=30.35 GiB (slight improvement from 30.92).
- **Analysis**: In-place silu_ causes massive performance degradation despite saving a small amount of memory. Likely causes: (1) CUDA graphs may handle in-place ops differently, causing suboptimal replay, (2) in-place mutation may prevent kernel fusion or parallelism within the graph, (3) the PyTorch runtime may add synchronization barriers around in-place ops.
- **Lessons**: Do NOT use in-place ops within CUDA-graphed regions. The performance cost far outweighs any memory savings. CUDA graphs pre-allocate all buffers anyway, so in-place doesn't help.
