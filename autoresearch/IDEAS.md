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

- [x] **Graph cleanup**: Remove no-ops from the graph to make it cleaner and easier for future optimizations.
  - 2026-05-28 00:15 — Upstream `joint_graph_passes` after no-op cleanup: bitwise-safe but +0.11% noise (graph already clean).
  - 2026-05-28 04:15 — `remove_redundant_contiguous_clone_pass` removes 68 stride-already-contiguous `aten.clone(memory_format=contiguous_format)` calls per step (+0.34%, KEEP). Subgraph descent (Exp 22) and post-bucketing rerun (Exp 23) find 0 more. The other ~290 FSDP-weight-unpack clones from initial profile get consumed by auto-bucketing.
  - 2026-05-28 08:00 — `remove_noop_to_copy_pass` finds 0 matches; no `aten._to_copy.default` in the graph is a kwargs-only no-op.

- [x] **CUDA graphs**: If the model is CPU-bound, CUDA graphs remove CPU overhead.
  - 2026-05-28 00:07 — Already in production starting graph (`cudagraph_pass`).

- [x] **Computation/communication overlap**: If there are exposed communications, see if they can be overlapped with computations.
  - 2026-05-28 02:15 — `auto_overlap_bucketing_pass` (upstream `schedule_overlap_bucketing(collective_bucketing=True, max_memory_increase_gb=2.0, ratio=0.05)`) replaces manual N-layer plan (+0.90%, KEEP). 131 AG / 163 RS buckets.
  - 2026-05-28 02:25 — `overlap_fsdp_ag_rs_pass` adds 0% (auto-bucketer already overlaps enough).
  - 2026-05-28 02:35-11:30 — Parameter tuning (`compute_overlap_multipler`, `max_memory_increase_gb=16`, `max_in_flight_gb=32`, `max_compute_pre_fetch=20`, `pre_bucketing_fsdp_collectives_bucket_cap_mb=8/128`, `bucket_mode="custom_ops"/"coalesced"`, second post-Inductor call) — all noise or worse. Bucketer is at a local optimum; further wins blocked by the roofline cost model not seeing Triton-compiled compute times.

- [x] **Kernel fusions**: Find regions worth fusing and generate fused kernels.
  - 2026-05-28 01:30 — Initial SwiGLU annotation (silu + immediate-mul user, 128 nodes): +0.14% noise. Established bitwise-safety of pointwise Inductor fusion.
  - 2026-05-28 01:50 — Broad pointwise tagging (1328 nodes incl. complex ops): -0.51% — Inductor can't codegen complex ops, fragments graph.
  - 2026-05-28 05:00 — Custom `mm → reshape → reduce_scatter` fusion via `symm_mem.fused_matmul_reduce_scatter` (looser matcher than upstream's `micro_pipeline_tp_pass`): 65 fwd TP fusions, +1.17% (KEEP). 225 bwd weight-grad RS skipped on dtype filter (fp32 upcast).
  - 2026-05-28 05:50 — Extended mm-RS to bwd dx-RS add-tree chains: -0.74% (linearity rewrite fragments into smaller RS ops).
  - 2026-05-28 06:30 — Shape-fix in fuse_mm_reduce_scatter + residual-add Inductor (224 nodes): +0.26% (KEEP).
  - 2026-05-28 06:50 — Extended SwiGLU chain (silu/silu_backward seeds, iterative expand to bf16 mul neighbors): 160 nodes, +1.36% (KEEP). Big win — connected-component tagging > op-by-op.
  - 2026-05-28 07:10/09:20 — Unified pointwise pass (no improvement over disjoint SwiGLU+residual): noise.
  - 2026-05-28 08:20 — Extend SwiGLU chain to absorb `_to_copy`: 0 new nodes (no adjacent `_to_copy` to chains).
  - 2026-05-28 08:50 — RoPE complex→real-valued rewrite: CRASH on bitwise (FMA rounding differs). Hard floor under strict bitwise (~9 ms).

- [x] **Collective coalescing**: Bucket many small NCCL launches into fewer large ones.
  - 2026-05-28 00:40/00:50 — Upstream `bucket_all_reduce` for 67 TP weight-grad ARs: bitwise-safe but topology CRASH (`merge_all_reduce_bucket` inserts merged collective at `bucket_nodes[-1].next`, but consumers of earlier-bucketed ARs were before that → graph.lint() fail). Position-independent upstream bug. Fixing requires custom AR coalescer with topology awareness — out of scope.
  - FSDP collectives ARE bucketed (via auto-bucketing keep).
  - TP RS collectives ARE collapsed via custom mm-RS fusion keep.

- [x] **Profile-driven optimization**: Profile the model, analyze the trace, and look for opportunities to optimize.
  - 2026-05-28 00:55, 02:50, 06:30+ — Multiple profile rounds drove the experiments. Key findings logged in LEARNINGS.md.
  - Profile-after-keep showed bucketer overlap regression post-Inductor (94%→11% AG overlap), drove the bucketer reordering / parameter experiments.

- [x] **Graph inspection**: Dump and study the FX graph to find optimization opportunities.
  - 2026-05-28 00:20 — Discovery experiment dumped post-production graph, identified key structural facts: regional Inductor was dormant, TP collectives unbucketed, FSDP collectives healthy, 358 clones with ~290 from FSDP unpack, hot RoPE/SwiGLU/residual unfused patterns.

- [~] **Study other frameworks**: Look at other pretraining frameworks for optimization ideas.
  - Not directly studied this run. Async-TP (micro_pipeline_tp) from upstream PyTorch was the most relevant comparable — blocked by matcher; we wrote our own.

- [~] **Literature research**: Search online for recent papers and blog posts.
  - Considered: gradient compression (numerics-changing), sequence parallelism (model change), CPU optimizer offload (latency cost), FP8 matmul (numerics-changing). None applicable under strict bitwise constraint with `passes.py`-only scope.

## Final keep stack (cumulative +5.07% over Run 3 production 7,162 → 7,525 tps)

1. `remove_redundant_contiguous_clone_pass` (68 clones/step)
2. `fuse_mm_reduce_scatter_pass` w/ shape-fix reshape (65 fwd TP mm→RS fusions)
3. `auto_overlap_bucketing_pass` (auto-bucketing, replaces manual bucket plan)
4. `annotate_swiglu_chains_for_regional_inductor_pass` (160 nodes; iterative seed+expand from silu/silu_backward to bf16 mul neighbors)
5. `annotate_residual_add_for_regional_inductor_pass` (224 bf16 adds)
6. `regional_inductor_pass` (compiles all tagged regions)

## Remaining hard limits (probably unreachable from `passes.py` alone)

- ~9 ms RoPE complex multiplication — bitwise blocked by aten FMA usage
- ~13-18 ms FSDP RS f32 exposure — auto-bucketer at local optimum
- ~17 ms FSDP AG exposure (regressed post-Inductor) — roofline doesn't see Triton kernel times
- ~17 ms bwd TP RS bf16 (add-tree pattern) — linearity rewrite hurts
- 67 TP weight-grad AR coalescing — upstream `bucket_all_reduce` topology bug
