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

- [x] **Graph cleanup**: Remove no-ops from the graph to make it cleaner and easier for future optimizations.
  - 2026-05-22 14:52 — Removed 196 `aten.detach.default` nodes (autograd no-op under runtime no_grad). TPS delta within noise but memory dropped ~2 GiB.
  - 2026-05-22 14:58 — `_to_copy.default` same-dtype/device elimination: 0 of 841 qualified — they are all real fp32↔bf16 mixed-precision casts.
  - 2026-05-22 19:50 — bf16→fp32→shape-op→bf16 round-trip collapse: 0 of 358 collapsible (joint_graph_passes already did the easy ones; remaining feed `cat` with fp32 siblings or `view_as_complex` for RoPE).
  - 2026-05-22 22:00 — General cleanup batch: 450 double-transpose removed but +0.05% (downstream codegen already collapses); 354 clone-removal + 257 view-of-view collapse both break runtime via non-contiguous strides.
- [x] **CUDA graphs**: If the model is CPU-bound, CUDA graphs remove CPU overhead.
  - 2026-05-22 18:20 — Wrapped gm.forward in `torch.cuda.CUDAGraph`. **+16.4% TPS on top of iter 7, +33.8% vs baseline. MFU 24.4% → 32.6%.** Critical fixes during implementation: replay-after-capture before returning, use `num_static_inputs` to skip copying stable FSDP params (memory stays flat at 49 GiB), monkey-patch `destroy_process_group` for clean NCCL teardown.
- [x] **Computation/communication overlap**: If there are exposed communications, see if they can be overlapped with computations.
  - 2026-05-22 15:50 — FX-level prefetch of AGs (move node up to earliest valid input dependency) only buys 7-position max move for 65/421 AGs. `make_fx` already places AGs right after their input shards. To make this work we'd need to also hoist the AG input ops (`_to_copy`, `view`) earlier, OR move waits LATER toward their consumers, OR rely on Inductor compile to do scheduling.
  - 2026-05-22 17:00 — Inductor's `overlap_scheduling.schedule_overlap_bucketing` does the right thing: ~+2% TPS on its own, no node-count change. It performs proper dependency analysis that hand-rolled FX reordering missed. Together with bucketing, +4.6% on iter 5.
  - 2026-05-23 11:55 — Tuning `aten_distributed_optimizations` knobs (`max_memory_increase_gb=20`, `max_in_flight_gb=10`, `max_compute_pre_fetch=200`, `compute_estimator="benchmark"`) for `schedule_overlap_bucketing`: 0 effect. Peak mem unchanged → budget wasn't binding; `compute_estimator` already defaults to "benchmark". Overlap surface saturated at defaults.
- [x] **Kernel fusions**: Find regions worth fusing and generate fused kernels. torch.compile, Triton kernels, or custom kernels could all help.
  - 2026-05-22 16:00 — Applied Inductor `joint_graph_passes` + `post_grad_passes`: 11,538 → 9,101 nodes (−21%), **tps 4,161 → 4,572 (+9.9%), mfu 24.4% → 26.8%**, numerics bitwise identical. First real TPS win.
  - 2026-05-22 16:25 — Tried whole-graph `compile_fx` / `compile_fx_inner` for Triton codegen; both blocked (in-place `_c10d_functional.all_reduce_` rejected by re-functionalization, and `_to_copy.default registered twice` after iter 5's patterns).
  - 2026-05-22 19:05 — Swapped 133 in-place collectives to functional+wait, drained 10 decomp/fallback registrations from `L.decompositions`, but `compile_fx_inner` then hit `AssertionError: device_mesh._get_submesh is not an OpOverload` — DTensor's redistribute emits a non-ATen Python op that Inductor refuses. `compile_fx`'s reinplace pass resurrects the in-place collectives anyway.
  - 2026-05-22 23:50 — Removed the single `_get_submesh` node (its sole user was `output`; rewired to mesh input). Drained decomps + cleared `fast_random_decomps` cache. `compile_fx_inner` SUCCEEDED, tps=6,042 (+2.8% on iter 12) — but **numerics fail bitwise** on the 8B model (debugmodel test passes but masks the drift). **Hard structural ceiling**: Inductor codegen produces numerically-correct-but-not-bitwise-equivalent kernels. Cannot proceed under the bitwise constraint.
- [x] **Collective coalescing**: Bucket many small NCCL launches into fewer large ones to reduce launch overhead.
  - 2026-05-22 15:25 — Dim-0 shape-grouped bucketing of all_gathers: 421→64+261 (~23% fewer launches) but tps unchanged. FSDP AG shapes are heterogeneous so only adjacent same-shape pairs/triples bucket together; and slice→reshape after the bucket likely costs a memcpy that wipes the savings.
  - 2026-05-22 17:00 — Upstream `bucketing.bucket_all_gather` + `bucketing.bucket_reduce_scatter` succeed where the manual iter-3 didn't — they use a layout strategy that recovers per-original outputs without an extra copy. Combined with `overlap_scheduling.schedule_overlap_bucketing`, this delivered **+4.6% TPS on top of iter 5**. Reuse upstream when available.
- [x] **Profile-driven optimization**: Profile the model, analyze the trace, and look for opportunities to optimize.
  - 2026-05-22 19:30 — Profile under `--profiler.enable_profiling`. After iter 8 the kernel mix is: GEMM 52.8% (irreducible), FlashAttn 23.1% (irreducible), NCCL 22.6% (fp32 RS dominant at 12.7%), FillFunctor 5.9% (4,061 zero-init kernels), `_to_copy` 4.6% (842 casts). GPU 98% busy. Biggest single non-irreducible lever is **bf16 RS (-9% TPS potential) but BLOCKED by bitwise-numerics constraint**.
  - 2026-05-22 19:42 — Realized the 4,061 FillFunctor kernels are caused by `torch.utils.deterministic.fill_uninitialized_memory=True` (set by `--debug.deterministic`). Audited buffers; safely disabled the flag for the captured graph. **+5.5% TPS (5,566 → 5,876), MFU 32.6% → 34.4%.**
  - 2026-05-23 00:30 — Re-profile after iter 12: 88 residual FillFunctor (1.35% / 22 ms). Categorization: 64 from FlashAttn-internal scratch (opaque, 91%), 2 from CE backward `zeros_like → index_put_` (semantically required), 8 negligible scalars. Confirmed iter-12 was the only `FillFunctor` lever reachable.
- [x] **Graph inspection**: Dump and study the FX graph to find optimization opportunities not covered by the ideas above.
  - 2026-05-22 14:44 — Counted FX nodes by op (recon pass; see LEARNINGS.md snapshot). 11.9k nodes total. Largest collective categories: 421 all_gather_into_tensor, 421 reduce_scatter_tensor, 68 all_reduce — bucketing/overlap is the biggest single target. 196 detach.default and 842 _to_copy look like cheap cleanup targets.
- [~] **Study other frameworks**: Look at other pretraining frameworks (e.g. https://github.com/apple/axlearn, https://github.com/openxla/xla, Megatron-LM, DeepSpeed, etc.) for optimization ideas.
  - 2026-05-23 00:00 — Quick web search for FSDP+TP Llama H100 optimization specifics surfaced the PyTorch blog "Maximizing Training Throughput Using FSDP and torch.compile". Their key levers for Llama bf16 on H100: torch.compile (we tried — bitwise-blocked), fp8 (out of scope), TP at ≥34B (we already have TP=2 at 8B), larger batch (parallelism fixed). Their reported MFU for 7B on H100 was 45% with compile, vs our 34.4% without compile (under bitwise constraint). Confirms our perf is in the right ballpark.
- [ ] **Literature research**: When the above ideas are exhausted, search online for recent papers and blog posts on LLM training optimization to find new directions.
