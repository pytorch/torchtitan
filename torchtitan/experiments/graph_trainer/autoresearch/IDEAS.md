# Research Ideas

Optimization ideas for the `aot_fx_trace` graph pass pipeline. The flat
fwd+bwd graph from `make_fx` currently has no optimization passes applied
(`apply_default_graph_passes` only does tlparse logging). All ideas target
passes added to `apply_default_graph_passes` in `passes.py`.

- [ ] **Comm/compute overlap reordering**: Reorder FSDP collectives (all-gather, reduce-scatter) relative to compute ops in the flat graph to maximize overlap. The flat graph sees both fwd and bwd collectives, enabling cross-boundary scheduling impossible in partitioned AOT mode.

- [ ] **Operator fusion**: Fuse adjacent elementwise ops and small kernels (e.g., RMSNorm components, bias+activation, pointwise chains) to reduce kernel launch overhead and memory traffic.

- [ ] **CUDA graph wrapping**: Wrap the entire flat traced graph (or static subregions) in CUDA graphs to eliminate per-step kernel launch overhead. Needs careful handling of dynamic shapes and collective ops.

- [ ] **Regional Inductor compilation**: Apply torch Inductor to compute-heavy subgraphs (matmuls, attention, MLPs) while leaving collectives as eager. Similar to `regional_inductor_pass` in AOT mode but adapted for the flat graph.

- [ ] **Selective activation checkpointing**: Apply SAC-style memory/compute tradeoffs on the flat graph. Since fwd and bwd are unified, the pass can directly see recomputation costs and make global decisions rather than per-partition heuristics.

- [ ] **Dead code elimination**: Extend beyond `_remove_cpu_shadow_chains` to remove any ops whose outputs are unused. Standard FX DCE on the traced graph.

- [ ] **In-place op conversion**: Convert out-of-place ops (e.g., `add`, `mul`) to in-place variants (`add_`, `mul_`) where liveness analysis proves the input is not used after the op. Reduces peak memory.

- [ ] **Constant folding**: Fold constant subexpressions (ops whose inputs are all constants or graph inputs that never change) at trace time to reduce per-step compute.

- [ ] **Operator scheduling for memory**: Reorder ops to minimize peak memory by scheduling consumers closer to producers (reduce live tensor lifetimes), independent of comm overlap concerns.
