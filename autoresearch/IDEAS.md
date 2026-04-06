# Optimization Ideas

- [ ] **Graph inspection**: Dump and study the FX graph structure to understand ops, collectives, and patterns before attempting optimizations.
- [ ] **Dead code elimination**: Remove unused nodes from the traced graph (e.g. CPU shadow chains, unused outputs).
- [ ] **Comm/compute overlap**: Reorder ops to overlap collective communication (all-gather, reduce-scatter) with compute.
- [ ] **Op fusion**: Fuse sequences of element-wise ops (e.g. rmsnorm components, activation functions) to reduce kernel launch overhead.
- [ ] **Bucketing collectives**: Bucket small all-gather/reduce-scatter ops to reduce collective launch overhead.
- [ ] **Selective recomputation tuning**: Tune SAC policy to find better save/recompute tradeoffs for this specific model size and parallelism config.
- [ ] **Memory planning**: Optimize tensor lifetime and memory allocation patterns to reduce peak memory and improve cache locality.
