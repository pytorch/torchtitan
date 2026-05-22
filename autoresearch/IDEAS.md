# Autoresearch Ideas

Curated seed ideas for Run 2. Directional only — investigate the graph
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
  - 2026-05-22 14:52 — Removed 196 `aten.detach.default` nodes (autograd no-op under runtime no_grad). TPS delta within noise but memory dropped ~2 GiB. Still TODO: investigate 842 `_to_copy.default` round-trips and any redundant view/transpose chains.
- [ ] **CUDA graphs**: If the model is CPU-bound, CUDA graphs remove CPU overhead.
- [ ] **Computation/communication overlap**: If there are exposed communications, see if they can be overlapped with computations.
- [ ] **Kernel fusions**: Find regions worth fusing and generate fused kernels. torch.compile, Triton kernels, or custom kernels could all help.
- [ ] **Collective coalescing**: Bucket many small NCCL launches into fewer large ones to reduce launch overhead.
- [ ] **Profile-driven optimization**: Profile the model, analyze the trace, and look for opportunities to optimize.
- [~] **Graph inspection**: Dump and study the FX graph to find optimization opportunities not covered by the ideas above.
  - 2026-05-22 14:44 — Counted FX nodes by op (recon pass; see LEARNINGS.md snapshot). 11.9k nodes total. Largest collective categories: 421 all_gather_into_tensor, 421 reduce_scatter_tensor, 68 all_reduce — bucketing/overlap is the biggest single target. 196 detach.default and 842 _to_copy look like cheap cleanup targets.
- [ ] **Study other frameworks**: Look at other pretraining frameworks (e.g. https://github.com/apple/axlearn, https://github.com/openxla/xla, Megatron-LM, DeepSpeed, etc.) for optimization ideas.
- [ ] **Literature research**: When the above ideas are exhausted, search online for recent papers and blog posts on LLM training optimization to find new directions.
