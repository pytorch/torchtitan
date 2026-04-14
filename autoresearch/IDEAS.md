# Research Ideas

Optimization ideas for the `aot_fx_trace` graph pass pipeline. The flat
fwd+bwd graph from `make_fx` currently has no optimization passes applied
(`apply_default_graph_passes` only does tlparse logging). All ideas target
passes added to `apply_default_graph_passes` in `passes.py`.

- [x] **Comm/compute overlap reordering**: Reorder FSDP collectives (all-gather, reduce-scatter) relative to compute ops in the flat graph to maximize overlap. The flat graph sees both fwd and bwd collectives, enabling cross-boundary scheduling impossible in partitioned AOT mode.

- [x] **Operator fusion**: Fuse adjacent elementwise ops and small kernels (e.g., RMSNorm components, bias+activation, pointwise chains) to reduce kernel launch overhead and memory traffic.

- [x] **CUDA graph wrapping**: Wrap the entire flat traced graph (or static subregions) in CUDA graphs to eliminate per-step kernel launch overhead. Needs careful handling of dynamic shapes and collective ops.

- [~] **Regional Inductor compilation — tag more nodes**: The `regional_inductor_pass` infrastructure already works but only compiles flex attention HOPs (via `annotate_flex_attention_for_regional_inductor_pass`). Write a new annotation pass that tags additional node regions with `compile_with_inductor` so regional inductor compiles them too. Candidates include:
  - Compute-heavy regions: matmul clusters, MLP blocks (linear + activation + linear), attention projections.
  - Repeated small-compute patterns: RMSNorm components (pow + mean + rsqrt + mul), pointwise chains (bias + activation, residual add + norm), embedding lookups + scaling.
  - The key insight: even small ops benefit from Inductor fusion when they repeat hundreds of times across layers. Inductor can fuse elementwise chains into single kernels, reducing launch overhead and memory traffic.
  - Leave collectives (`_c10d_functional.*`) and their `wait_tensor` consumers untagged — these must stay as eager ops.
  - Use `annotate_flex_attention_for_regional_inductor_pass` as a reference for the tagging pattern: set `node.meta["custom"]["compile_with_inductor"]` on target nodes.
  - Start simple: tag all non-collective `call_function` nodes and measure. Then selectively exclude patterns that hurt numerics or perf.
  - @claude, 2026-04-13 18:30 — Extensively explored. Tagging all compute ops: no TPS improvement (cudagraph makes launch overhead zero). Tagging elementwise-only: collectives fragment graph into 2-3 node regions, too small for useful fusion. Tagging silu+mul: +1.1% TPS but breaks DSv3 numerics (Triton sigmoid != eager CUDA sigmoid). Tagging RoPE (complex ops): breaks numerics (complex mul decomposed differently). **Conclusion: regional_inductor cannot improve TPS without breaking bitwise determinism for transcendental/complex ops. Only purely algebraic ops (add, mul) are safe but regions are too small.**

- [ ] **Selective activation checkpointing**: Apply SAC-style memory/compute tradeoffs on the flat graph. Since fwd and bwd are unified, the pass can directly see recomputation costs and make global decisions rather than per-partition heuristics.

- [x] **Dead code elimination**: Extend beyond `_remove_cpu_shadow_chains` to remove any ops whose outputs are unused. Standard FX DCE on the traced graph.

- [ ] **In-place op conversion**: Convert out-of-place ops (e.g., `add`, `mul`) to in-place variants (`add_`, `mul_`) where liveness analysis proves the input is not used after the op. Reduces peak memory.

- [ ] **Constant folding**: Fold constant subexpressions (ops whose inputs are all constants or graph inputs that never change) at trace time to reduce per-step compute.

- [ ] **Operator scheduling for memory**: Reorder ops to minimize peak memory by scheduling consumers closer to producers (reduce live tensor lifetimes), independent of comm overlap concerns.
