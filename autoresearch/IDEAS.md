# Autoresearch Kernel Optimization Ideas

- [x] **Annotate silu+mul for regional Inductor**: Tag SwiGLU silu+mul pairs with compile_with_inductor so Inductor fuses them into single Triton kernels.
  - @agent, 2026-05-12 16:48 — Crashed with OOM during Inductor compilation. 32 silu+mul regions require too much compilation memory.
- [x] **In-place silu_ replacement**: Replace silu.default with silu_.default where input has no other consumers.
  - @agent, 2026-05-12 16:54 — Massive 60% tps regression (2504 vs ~6250). In-place ops break CUDA graph performance.
- [ ] **Fuse residual add + RMSNorm**: The common pattern `add(residual, x)` followed by `rmsnorm(result)` can be fused into a single kernel to reduce memory traffic.
- [ ] **Custom Triton silu_mul kernel**: Register a custom ATen op backed by a Triton kernel that computes `silu(x) * y` in one pass, avoiding the intermediate write.
- [ ] **Fuse embedding scale**: If there's a multiply-by-constant after embedding lookup, fuse it into the embedding.
- [ ] **Eliminate redundant _to_copy in RoPE**: The bf16->f32->complex->f32->bf16 chain in RoPE has multiple dtype conversions. Explore keeping more computation in bf16 or reducing conversion count.
- [ ] **Reduce CUDA graph pool size**: The 31.75 GiB CUDA graph private pool may be over-allocated. Investigate if tighter memory planning reduces waste.
