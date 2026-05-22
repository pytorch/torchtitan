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
  - 2026-05-22 14:58 — `_to_copy.default` same-dtype/device elimination: 0 of 841 qualified — they are all real fp32↔bf16 mixed-precision casts. Plain elimination won't help; need bf16→fp32→bf16 round-trip collapse or fuse-into-producer/consumer.
- [ ] **CUDA graphs**: If the model is CPU-bound, CUDA graphs remove CPU overhead.
- [~] **Computation/communication overlap**: If there are exposed communications, see if they can be overlapped with computations.
  - 2026-05-22 15:50 — FX-level prefetch of AGs (move node up to earliest valid input dependency) only buys 7-position max move for 65/421 AGs. `make_fx` already places AGs right after their input shards. To make this work we'd need to also hoist the AG input ops (`_to_copy`, `view`) earlier, OR move waits LATER toward their consumers, OR rely on Inductor compile to do scheduling.
- [~] **Kernel fusions**: Find regions worth fusing and generate fused kernels. torch.compile, Triton kernels, or custom kernels could all help.
  - 2026-05-22 16:00 — Applied Inductor `joint_graph_passes` + `post_grad_passes`: 11,538 → 9,101 nodes (−21%), **tps 4,161 → 4,572 (+9.9%), mfu 24.4% → 26.8%**, numerics bitwise identical. First real TPS win. Next: route the simplified graph through Inductor codegen (`compile_fx`) to get actual Triton kernels — the current win is from FX-level pattern rewrites alone, no kernel codegen.
- [~] **Collective coalescing**: Bucket many small NCCL launches into fewer large ones to reduce launch overhead.
  - 2026-05-22 15:25 — Dim-0 shape-grouped bucketing of all_gathers: 421→64+261 (~23% fewer launches) but tps unchanged. FSDP AG shapes are heterogeneous so only adjacent same-shape pairs/triples bucket together; and slice→reshape after the bucket likely costs a memcpy that wipes the savings. Try (a) flatten-to-1D bucketing to enable broader coverage and contiguous-view recovery, and (b) prefetch AGs to overlap with intervening compute (probably the bigger win).
- [ ] **Profile-driven optimization**: Profile the model, analyze the trace, and look for opportunities to optimize.
- [~] **Graph inspection**: Dump and study the FX graph to find optimization opportunities not covered by the ideas above.
  - 2026-05-22 14:44 — Counted FX nodes by op (recon pass; see LEARNINGS.md snapshot). 11.9k nodes total. Largest collective categories: 421 all_gather_into_tensor, 421 reduce_scatter_tensor, 68 all_reduce — bucketing/overlap is the biggest single target. 196 detach.default and 842 _to_copy look like cheap cleanup targets.
- [ ] **Study other frameworks**: Look at other pretraining frameworks (e.g. https://github.com/apple/axlearn, https://github.com/openxla/xla, Megatron-LM, DeepSpeed, etc.) for optimization ideas.
- [ ] **Literature research**: When the above ideas are exhausted, search online for recent papers and blog posts on LLM training optimization to find new directions.
