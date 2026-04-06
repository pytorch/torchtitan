# Optimization Ideas

- [x] **Graph inspection**: Dump and study the FX graph structure to understand ops, collectives, and patterns before attempting optimizations.
  - @claude, 2026-04-05 22:25 — Graph has 11381 nodes, 421 AG, 421 RS, 68 AR, 675 mm, 842 _to_copy, 32 SDPA, 65 rmsnorm. Heavy collective traffic.
- [ ] **Dead code elimination**: Remove unused nodes from the traced graph (e.g. CPU shadow chains, unused outputs).
- [x] **Comm/compute overlap via autobucketing**: Reorder ops to overlap collective communication (all-gather, reduce-scatter) with compute.
  - @claude, 2026-04-05 22:30 — Applied schedule_overlap_bucketing to full traced graph. +3.3% tps (4247→4387 avg). Kept.
- [ ] **Op fusion**: Fuse sequences of element-wise ops (e.g. rmsnorm components, activation functions) to reduce kernel launch overhead.
- [~] **Bucketing collectives**: Bucket small all-gather/reduce-scatter ops to reduce collective launch overhead.
  - @claude, 2026-04-05 22:30 — Autobucketing includes collective bucketing. Could try manual transformer-block-level bucketing for more targeted control.
- [ ] **Selective recomputation tuning**: Tune SAC policy to find better save/recompute tradeoffs for this specific model size and parallelism config.
- [ ] **Memory planning**: Optimize tensor lifetime and memory allocation patterns to reduce peak memory and improve cache locality.
- [ ] **Remove redundant _to_copy ops**: 842 dtype conversion ops in graph — investigate if some are unnecessary.
- [ ] **Transformer block bucketing**: Use manual_overlap_bucketing with per-transformer-block bucket plans for more structured overlap.
- [ ] **Regional Inductor compilation**: Apply Inductor kernel codegen (triton) to fuse element-wise ops within regions.
- [ ] **CUDAGraph wrapping**: Wrap the traced graph with CUDAGraph to eliminate kernel launch overhead.
