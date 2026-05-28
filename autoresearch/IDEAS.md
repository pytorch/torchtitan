# Autoresearch Ideas

Curated seed ideas for Run 3. Directional only — investigate the graph
to figure out what's actually there and how to exploit it.

## Format

Each idea is a top-level bullet with a status checkbox:
- `[ ]` — open, not yet explored
- `[~]` — partially explored, more work possible
- `[x]` — fully explored or no further opportunity

Comments and findings go as **indented sub-bullets** with a timestamp at
the beginning:

```
- [ ] **Idea name**: Description.
  - YYYY-MM-DD HH:MM — Finding or comment.
```

## Ideas

- [~] **Graph cleanup**: Remove no-ops from the graph to make it cleaner and easier for future optimizations.
  - 2026-05-28 00:15 — Upstream `joint_graph_passes` (CSE, noop removal, constant folding) inserted after `normalize_view_ops_as_reshape` was bitwise-safe but +0.11% (noise). Production no-op cleanup is already adequate at the joint-graph stage. Any future cleanup wins likely live in the *post-bucketing* / *post-regional-Inductor* graph (new patterns appear there) or in fusing kernels Inductor never touches (RMSNorm, residual+norm, embedding+norm).
- [x] **CUDA graphs**: If the model is CPU-bound, CUDA graphs remove CPU overhead.
  - 2026-05-28 00:07 — Already in the production starting graph (`cudagraph_pass`). Nothing further to do at this layer.
- [ ] **Computation/communication overlap**: If there are exposed communications, see if they can be overlapped with computations.
- [ ] **Kernel fusions**: Find regions worth fusing and generate fused kernels. torch.compile, Triton kernels, or custom kernels could all help.
- [ ] **Collective coalescing**: Bucket many small NCCL launches into fewer large ones to reduce launch overhead.
- [ ] **Profile-driven optimization**: Profile the model, analyze the trace, and look for opportunities to optimize.
- [ ] **Graph inspection**: Dump and study the FX graph to find optimization opportunities not covered by the ideas above.
- [ ] **Study other frameworks**: Look at other pretraining frameworks (e.g. https://github.com/apple/axlearn, https://github.com/openxla/xla, Megatron-LM, DeepSpeed, etc.) for optimization ideas.
- [ ] **Literature research**: When the above ideas are exhausted, search online for recent papers and blog posts on LLM training optimization to find new directions.
