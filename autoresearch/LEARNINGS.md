# Autoresearch Kernel Learnings

## Methodology
- Always establish a clean baseline before experimenting.
- Dump and inspect the FX graph to understand op patterns before designing passes.
- The autoresearch_kernel_pass runs AFTER regional Inductor compilation, so we're operating on a graph that already has Triton kernels for annotated regions (FlexAttention). Focus on ops that Inductor didn't fuse.
- Verify numerics after every change — kernel fusions can introduce subtle floating-point differences.
- GPU contention from other users can cause 7%+ tps variance. Always re-run baselines before/after to account for this.

## Key Observations

### Graph Structure (Llama3 8B, FSDP4+TP2, 8xH100)
- Total graph: ~23K lines, dominated by reshapes (2313), mm (675), _to_copy (679), clones (646).
- 32 silu+mul (SwiGLU) pairs in forward, each operating on [1, 8192, 7168] bf16 tensors.
- 130 _fused_rms_norm calls (already fused).
- 192 view_as_real + 192 view_as_complex (RoPE, metadata-only ops).
- Existing noop removal passes are thorough — no consecutive reshape chains remain.

### What Doesn't Work
- **In-place ops (silu_) under CUDA graphs**: 60% tps regression. CUDA graph replay handles in-place mutations very poorly. Never use in-place ops in CUDA-graphed regions.
- **Annotating many regions for regional Inductor**: 32 silu+mul annotations caused compilation OOM (SIGKILL). Regional Inductor compilation memory scales with number of annotated regions.
- **Moving the kernel pass before regional_inductor**: Required for annotation-based approaches, but increases compilation memory.

### Architecture Constraints
- The kernel pass runs post-Inductor, post-bucketing. Most metadata ops (view, reshape, transpose, t) are zero-cost.
- Actual kernel count: mm (675) + attention (32) + silu (64) + mul (321) + add (258) + rmsnorm (130) + collectives (~500) = ~2000 real kernels per step.
- Under CUDA graphs, kernel launch overhead is negligible. The main bottlenecks are memory bandwidth and compute intensity.
- Baseline: tps ~6250, memory 30.9 GiB, MFU ~37%.
