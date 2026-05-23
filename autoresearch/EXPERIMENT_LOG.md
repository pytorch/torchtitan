# Autoresearch Experiment Log

Cumulative log of every experiment in this run. Append-only — never
overwrite previous entries. Re-read at the start of each loop iteration to
learn from past experiments and avoid repeating failed approaches.

## Format

```markdown
## <short title> — <status> (<commit hash>)

- **Idea**: What optimization was attempted and why it was expected to help.
- **Changes**: What was actually modified (brief summary, not a full diff).
- **Result**: Perf numbers (tps, MFU, memory_gib, wall_time_s) or crash/error description.
- **Analysis**: Why it worked, why it didn't help, or why it broke.
- **Lessons**: Key takeaways — what to build on, what to avoid, what to try.
```

---

## Baseline — keep (baseline)

- **Idea**: Establish the Run 1 baseline. Llama3 8B with FSDP=4 TP=2 bs=1, `construct_default_graph_passes` returns an empty list. The agent will iterate from this raw aten-op graph.
- **Changes**: None to `passes.py`. Infrastructure-only commit.
- **Result**: tps=4,161, mfu=24.36%, memory=49.0 GiB (52% of H100), wall_time=75s; loss=9.21808, grad_norm=4.5867 (last step of 20, `c4_test` dataset).
- **Analysis**: Bitwise-identical loss/grad_norm across runs with `--debug.seed 42 --debug.deterministic`. Memory is comfortable (31 GiB headroom). There is meaningful headroom above this floor — the agent's job is to find it. Wall time ~75s per benchmark → ~1100 iterations in 24h.
- **Lessons**: FSDP=4 TP=2 bs=1 chosen over bs=2 because of larger optimization headroom and safer memory budget (31 GiB free vs 12 GiB).

---

## coalesce_independent_adds (recon-only) — discard (xxxxxxx)

- **Idea**: Profile shows 435 `elementwise_add` kernels (2.5% / 41 ms). If many are independent same-shape adds, coalesce into `aten._foreach_add` to cut kernel count.
- **Recon**: 225 FX `aten.add.Tensor` nodes (≠435 kernels — discrepancy from cuda-graph internal binding). Distribution:
  - ~64 forward residual connections (`x = x + sublayer(x)`): each LHS = prior add's output, **strict sequential chain** across 32 layers × 2 adds/layer. Inherently uncoalescable.
  - ~161 backward grad accumulators (`"Gradient addition node due to multiple use of tensor"` stack_trace): also chained — each extends the prior accumulator.
  - 128/225 have multi-user LHS (skipped anyway).
  - Independent-group analysis (greedy, same shape/dtype, distinct LHS): **0 groups of size ≥ 2**.
- **Result**: discard. No worthwhile coalescing pattern; pass not implemented.
- **Lessons**:
  - **`_foreach_add` requires INDEPENDENT inputs.** Sequential residual / grad accumulator chains can't be parallelized this way regardless of multi-tensor packing.
  - **To reduce the elementwise_add cost would require either (a) fusing each add with its surrounding compute (e.g. residual-add into attention output projection's epilogue, grad-accum into the preceding view/mm) — Inductor pattern-matching territory, OR (b) rewiring backward to a tree-reduction structure** — substantial complexity, out of scope for this experiment.

---

## iter17 exploration batch — discard (xxxxxxx)

- **Idea**: Try a batch of cheap exploratory tweaks: apply `joint_graph_passes` twice, add `post_grad_passes(is_inference=True)`, toggle `epilogue_fusion`/`coordinate_descent_tuning` etc., find `dedupe_symint_uses` / `binary_folding` under alternate module paths, call `joint_graph_passes` after bucketing.
- **Result**: discard. All attempts no-op (deletes 0 nodes after the first fixed-point call) or noise-level TPS. `dedupe_symints` exists under the same module but is a no-op on our graph. `binary_folding_pass` doesn't exist as a callable in this PyTorch build (only `register_binary_folding_pattern` / `binary_folding_init`).
- **Lessons**:
  - **`joint_graph_passes` reaches fixed point in one call** on this graph. Repeated application doesn't help.
  - **`post_grad_passes(is_inference=True)` is equivalent to `is_inference=False` here** — both no-ops on the iter-12 graph (already fully reduced).
  - **Inductor config flags don't change FX-level pass output**; they gate codegen behavior which we bypass via CUDA graphs.
  - **The FX-pass surface for further wins is exhausted at this configuration.** Remaining levers are at the kernel level (bf16 RS — blocked by bitwise numerics) or scheduling level — not reachable via FX pattern passes.

---

## cleanup_redundant_ops — discard (xxxxxxx)

- **Idea**: Look for survivors after iter-5 passes: redundant double-transpose, redundant clone, view-of-view collapse, identity permute/squeeze/unsqueeze/expand.
- **Recon counts** (iter-12 graph): double_transpose=450, redundant_clone=354, view_of_view=257; permute_identity / squeeze_unsqueeze / dead_to_copy / expand_identity = 0 each.
- **Result**: discard. Only double_transpose was safely applicable (numerics pass after 450 removals). tps=5,879 ≈ iter-12 5,876 (+0.05%, noise). Clone and view-of-view removal **broke runtime** with "view size is not compatible with input tensor's size and stride" — surviving clones serve as contiguity canonicalizations, surviving views depend on inner-view strides. Inductor preserves both intentionally.
- **Lessons**:
  - **450 double-transpose removals = 0 TPS win** because Inductor's downstream codegen already collapses them during lowering. FX-level removal is redundant work.
  - The clone / view-of-view "survivors" are NOT bugs — they encode stride/layout requirements that downstream consumers depend on. Removing them silently breaks numerics or runtime.
  - **The graph after iter-5 passes is essentially fully optimized at the FX level for low-risk pattern rewrites.** Future wins require either deeper transforms (kernel-level fusion, codegen) or attacking different categories (numerics-blocked or out-of-graph items).

---

## collapse_cat_of_upcasts — discard (xxxxxxx)

- **Idea**: Iter-11 recon (on the bare make_fx graph) found 290 `_to_copy(bf16→fp32) → shape_op → cat.default(...)` chains. Collapse: `cat([_to_copy(x_i, fp32) for x_i])` → `_to_copy(cat([x_i]), fp32)` when all cat inputs share an upcast and source dtype.
- **Result**: discard. **0 patterns found** in the post-iter-5 graph (260 cat nodes and 551 `_to_copy` nodes exist but no `_to_copy → cat` adjacency remains). `joint_graph_passes` already collapsed them upstream.
- **Lessons**: The iter-11 recon was on the bare make_fx graph; the iter-12 graph is post-iter-5 and post-iter-7. Inductor's pattern matchers run early in our pipeline and have already done this work. **Recon counts at one pipeline stage do not predict opportunities at later stages.**

---

## regional_inductor_compile — discard (xxxxxxx)

- **Idea**: Whole-graph `compile_fx` is blocked by DTensor's `_get_submesh` and `compile_fx`'s reinplace pass. Compile only **pure-ATen connected subgraphs** (no collectives, no `wait_tensor`, no `_get_submesh`, no in-place) via `compile_fx_inner`, replace nodes with `call_function(compiled_fn, ...)`.
- **Changes**: Added `regional_inductor_compile` pass. Found 97 regions by maximal consecutive-run topo-sort grouping (convex by construction). Drained decomp conflicts: `_to_copy`, `t`, `transpose.int`, `silu`, `arange`, `zeros_like`, `ones_like`, `_fused_rms_norm_backward`, `silu_backward`, `embedding_dense_backward`. Cleared `fast_random_decomps`'s `@functools.cache` (otherwise pop doesn't propagate). All 97 regions compiled successfully.
- **Result**: discard. tps=5,890 vs iter-12 5,876 → **+0.24%** (within noise). Numerics PASS bitwise. Wall time 161s (compile +11s for 97 regions).
- **Analysis**:
  - Regions are small (20-39 nodes, avg 31). The graph is fragmented by collectives every ~30 nodes; Triton fusion can't span a useful fraction of the graph.
  - CUDA graphs (iter 8) already amortize per-launch overhead. Compiling small pointwise runs recovers very little arithmetic intensity on top.
  - **Mechanism works** (compile_fx_inner succeeds on cleaned regions, replaced cleanly in outer graph, numerics survive). It just doesn't pay back when individual regions are too small.
- **Lessons**:
  - **Region size matters**: <40-node pointwise regions don't recoup compile overhead post-CUDA-graphs.
  - **Decomp drain + cache clear** unblocks `compile_fx_inner` on subgraphs. Future iterations can reuse this technique.
  - **The graph's collective density (~30 nodes between collectives) is the structural limit** for Triton coverage. Larger regions would require bypassing collectives — e.g., compile across collective boundaries by treating collectives as opaque calls inside Triton (probably needs custom inductor lowerings, out of scope).

---

## expanded apply_inductor_pattern_passes attempt #2 — discard (xxxxxxx)

- **Idea**: Try untried `torch._inductor.fx_passes.*` submodules (`fsdp.dedup_fsdp_reduce_scatter`, `reduced_atomic_contention.partitioned_scatter_optimization_pass`, `ddp_fusion.schedule_comm_wait`, plus 8 others).
- **Result**: discard. All 3 callable rewrites ran cleanly but were node-count no-ops. tps=5,883 ≈ iter-12 5,876 (within noise).
- **Lessons**: After iter-7's `bucket_*` + `schedule_overlap_bucketing` the graph is already in these passes' "post-pass" form. Other untried modules are utilities (memory_estimator, node_runtime_estimation, etc.), require driver context we don't have (overlap_manual_scheduling needs module_bucket_plans; overlap_preserving_bucketer needs scheduled set; control_dependencies needs additional_deps_map), or are in the wrong domain (apply_gumbel_max_trick=RL, efficient_conv_bn_eval=CNN, quantization=N/A, mkldnn_fusion=CPU, freezing_patterns=inference). Inductor's fx_passes surface for this workload is fully explored.

---

## disable_uninitialized_memory_fill — keep (d99dce8)

- **Idea**: Profile shows 4,061 FillFunctor zero-init kernels / step (~95 ms, 5.9%). Investigate the source and eliminate where safe.
- **Recon finding**: Only **75** zero-fill *FX nodes* in the entire 10,947-node graph (not 4,061). Distribution:
  - **65 × `aten.empty.memory_format`** — large bf16 buffers (20M–88M elts), each backing an `_c10d_functional.all_gather_into_tensor_out`. Each `empty` has exactly two users: a `slice.Tensor` (read view, safely no-op until written) and the `all_gather_into_tensor_out` (FULL overwrite).
  - **9 × `aten.full.default`** — tiny scalars feeding `where` / `index_put_`.
  - **1 × `aten.zeros_like.default`** — cross-entropy backward grad buffer (intentional zero, must keep).
  - **4,061 FillFunctor CUDA kernels ≈ 65 empties × ~62 launches** per step. They come from `torch.utils.deterministic.fill_uninitialized_memory = True`, which `--debug.deterministic` enables. Every `aten.empty.*` emits a FillFunctor zero-init even when the consuming op overwrites the buffer immediately.
- **Changes**: Added `disable_uninitialized_memory_fill` pass. Audits every `aten.empty.memory_format` / `aten.empty_strided.default` node; requires every user be either a full-overwrite op (`all_gather_into_tensor_out`, `copy_`, `fill_`, `zero_`) or a read-only view AND at least one full-overwrite user. If all safe, sets `torch.utils.deterministic.fill_uninitialized_memory = False` so subsequent `install_cuda_graph` capture records the graph WITHOUT the FillFunctor zero-init kernels. Registered between `apply_inductor_pattern_passes` and `install_cuda_graph`. Falls through on any exception.
- **Result**: keep. 2 runs: tps=5,875 / 5,877, mfu=34.40% / 34.42%, memory=49.08 GiB. Loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics PASS. Wall time 141s.
  - **+5.5% over iter 8, +41.2% over baseline.**
- **Lessons**:
  - **`--debug.deterministic` quietly costs ~5.5% TPS via `fill_uninitialized_memory=True`.** The flag's intent (reproducibility through clean uninitialized memory) is achieved here by other means (`--debug.seed`), so we can safely disable the fill once we've audited that every uninitialized buffer is fully overwritten before any non-view read.
  - **Profile-driven optimization works**: iter-10's profile flagged the bottleneck, this iter realized the actual root cause was a global PyTorch flag, not the FX graph itself.
  - **FillFunctor kernel count ≠ FillFunctor FX node count.** A single `empty.memory_format` can generate many FillFunctor launches when deterministic mode is on — count the kernels not the nodes.

---

## collapse_dtype_roundtrips — discard (xxxxxxx)

- **Idea**: Among 842 `_to_copy.default` casts, find bf16→fp32→shape-op→bf16 round-trips where the intermediate ops are dtype-trivial (view/transpose/reshape/etc.). Collapse to a single bf16 path; bitwise identical since shape ops don't compute.
- **Changes**: Added `collapse_dtype_roundtrips` pass between `apply_inductor_pattern_passes` and `install_cuda_graph`. Whitelisted shape ops; required single-user chains; required matching upcast/downcast dtypes.
- **Result**: discard. tps=5,567 (≡ iter 8), numerics pass. **0 collapses applied** out of 358 candidates.
- **Analysis**: Iter 5's `joint_graph_passes` already collapsed the easy round-trips. The remaining 358 candidates fall into two non-collapsible patterns:
  - **290 (~81%)**: `_to_copy(bf16→fp32) → shape_op → cat.default(...)`. `cat` is dtype-trivial in isolation but the other inputs to `cat` are fp32, so keeping our branch in bf16 would dtype-mismatch.
  - **64 (~18%)**: `_to_copy(bf16→fp32) → shape_op → view_as_complex.default → ...`. `view_as_complex` needs fp32 input (RoPE precompute); not a round-trip.
- **Lessons**:
  - **Joint_graph_passes is thorough at the simple cases** — we don't get to re-do its work.
  - The remaining `_to_copy` overhead (74 ms / 4.6%) needs a different attack: e.g. `_to_copy + cat` fusion into a typed cat, or restructuring RoPE precompute, or earlier-pipeline elimination before inductor introduces the `cat` consumers.

---

## profile-driven analysis (iter 10) — info-only

- **Idea**: After iter 8 (CUDA graphs) we're at tps=5,566, mfu=32.6%. Profile to find the next bottleneck.
- **Findings** (steady-state, 8 ranks, FSDP=4 TP=2):
  - GEMM: 52.8% of kernel time (851 ms / step, mostly irreducible)
  - FlashAttn fwd+bwd: 23.1% (372 ms, irreducible)
  - NCCL collectives: 22.6% (365 ms). Of which: **204 ms fp32 ReduceScatter (132 calls)** + 63 ms bf16 RS + 92 ms AllGather + ~5 ms AllReduce. fp32 RS dominates because 3× costlier per call than bf16.
  - **FillFunctor zero-init: 4,061 kernels / 95 ms (5.9%)** — one stream alone has 592 of these (median 2 µs each).
  - direct_copy + casts: ~74 ms (4.6%) from 842 `_to_copy.default` nodes.
  - GEMM/FlashAttn/Collectives total 98%. GPU idle only 1.9%; NCCL stream 70% overlapped with compute, exposed NCCL ~60 ms.
- **Suggested levers** (ordered by realistic upside without breaking bitwise numerics):
  1. **bf16 RS**: best lever (~140 ms / +9% TPS) but **blocked — would change numerics.**
  2. **FillFunctor coalesce**: ~50-95 ms savings if grouped into multi-tensor zero or DCE'd. Most likely 3-6% TPS.
  3. **`_to_copy + cat` typed-fusion**: ~30-50 ms. Tricky (cat needs same dtype across operands).
  4. **Regional Inductor compile** on ATen-only subgraphs between collectives: potentially 5-15% TPS if achievable.
  5. **Async loss/grad_norm**: pure host-time win; GPU at 98% busy so no MFU gain.
- **Lessons**:
  - **Bitwise constraint blocks the biggest single lever (bf16 RS).** Worth flagging this to the experimenter — relaxing to loss-equivalent rather than bitwise opens up ~9% TPS.
  - **GPU is essentially never idle** (1.9%); remaining wins must come from reducing kernel count/size, not from filling bubbles.
  - **fp32 ReduceScatter (132 calls / 204 ms) is the single largest non-irreducible category**, and `simple_fsdp` uses fp32 for grad reduction even when params/activations are bf16.

---

## functionalize_collectives_and_compile_fx — discard (xxxxxxx)

- **Idea**: The blocker for Inductor codegen (iter 6) was the in-place `_c10d_functional.all_reduce_.default` collective. Swap all in-place functional collectives for their functional+wait_tensor equivalents, then try `compile_fx_inner` again to unlock Triton kernel codegen.
- **Changes**: Added `functionalize_collectives_and_compile_fx` pass with helpers `_swap_inplace_collective_node` and `_try_compile_fx_codegen`. Registered AFTER `apply_inductor_pattern_passes` and BEFORE `install_cuda_graph`.
- **Result**: discard. 3-run avg tps=5,480 (mfu=32.08%) vs iter-8 5,566 → **−1.5% regression**. Loss/grad_norm bitwise identical. Numerics PASS.
- **Outcomes per phase**:
  - **Collective swap**: 133 in-place ops swapped (68 `all_reduce_` + 65 `all_gather_into_tensor_out`; 0 `reduce_scatter_tensor_out`). Numerics OK on swap alone, but adds 133 extra `wait_tensor` dispatches (one per swap) → small TPS regression.
  - **`compile_fx_inner`**: drained 10 "both a fallback and a decomp" assertions by deleting offending ops from `L.decompositions` (`_to_copy`, `t`, `transpose.int`, `silu`, `arange`, `zeros_like`, `ones_like`, `_fused_rms_norm_backward`, `silu_backward`, `embedding_dense_backward`). Then hit unrecoverable `AssertionError: device_mesh._get_submesh is not an OpOverload` — a non-ATen DTensor compile-time helper in the joint graph.
  - **`fx_codegen_and_compile`**: same root cause.
  - **`compile_fx`**: Inductor's `post_grad` reinplace pass resurrects the in-place collectives → fails again. Re-introduced after our swap.
  - **Net codegen impact**: zero (every entry point blocked).
- **Lessons**:
  - **`compile_fx`'s reinplace pass re-introduces in-place collectives** even after we explicitly swapped them. The reinplace machinery treats functional+wait as redundant.
  - **`compile_fx_inner` decomp/fallback conflicts can be drained** by deleting offending entries from `torch._inductor.lowering.decompositions`, but you only delay the wall, you don't move it.
  - **The real structural blocker is `device_mesh._get_submesh`** — DTensor's redistribute emits this non-ATen Python op into the joint graph. Inductor refuses any non-`OpOverload` op. **To unlock codegen, either (a) re-trace with a graph that doesn't emit `_get_submesh`, or (b) replace `_get_submesh` nodes with their resolved literal value (a sub-mesh constant) before codegen.**
  - **Free `wait_tensor` dispatches aren't free** — adding 133 of them cost −1.5% TPS. The fewer the collective sync points, the better. So the iter-7 bucketing's reduction of total collectives is doubly valuable.

---

## install_cuda_graph — keep (63c54ff)

- **Idea**: After iter 7's pattern passes, the graph is largely static — same ops, same shapes every step. Wrap the joint fwd+bwd graph in a `torch.cuda.CUDAGraph` and replay each step to eliminate CPU launch overhead. Functional collectives in recent PyTorch are cuda-graph-friendly.
- **Changes**: Added `install_cuda_graph(gm, example_inputs, *, num_static_inputs)` registered last in `construct_default_graph_passes`. Uses `functools.partial` to plumb `traced_result.num_static_inputs` so the wrapper can skip persistent buffers for the leading FSDP-stable inputs (avoids OOM). Standard PyTorch capture pattern: non-default stream, ≥1 warmup, capture inside `with torch.cuda.graph(g):`. **Crucial detail**: after capture, also call `g.replay()` once so `state["outputs"]` reflects executed values before they're returned (otherwise the first real call returns warmup leftovers and training diverges). Self-aliasing skip via `data_ptr()` for callers that already pass the static buffer. Failure paths fall through and restore the iter-7 baseline.
- **Result**: keep. 2 runs: tps=5,566 / 5,565 → **avg 5,566, mfu=32.59%**, memory=49.08 GiB. Loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics PASS. Wall time 141s, clean shutdown.
  - **+16.4% over iter 7, +33.8% over baseline.**
- **Implementation gotchas resolved**:
  - **First-call replay**: capture only records ops, doesn't execute them. The first post-capture call must `g.replay()` before returning, else outputs are warmup-stale and training diverges.
  - **Static input copying**: naively allocating a persistent buffer for every flat input doubles model-state memory (was +8 GiB for 295 buffers). With `num_static_inputs` we keep only the varying inputs (3 persistent buffers), restoring memory to iter-7 levels.
  - **Shutdown hang**: captured graph + buffers held NCCL state alive past process teardown for ~22 min until SIGTERM. Monkey-patched `torch.distributed.destroy_process_group` to sync, drop graph, drop buffers, `gc.collect`, `empty_cache` before NCCL teardown. Plus an `atexit` fallback.
- **Lessons**:
  - **CUDA graphs deliver large wins (+16%) on this workload** because steady-state CPU launch overhead is non-trivial (~10-15% of step time for ~9-11k node graphs).
  - **`num_static_inputs` from `TracedResult` is essential** to keep memory flat when wrapping in CUDA graphs. Plumb traced_result fields into passes via `functools.partial` from `construct_default_graph_passes`.
  - **Always replay-after-capture** before returning to the caller. Otherwise outputs are recorded references with no fresh execution.
  - **NCCL + CUDA graphs need explicit teardown** to avoid shutdown hangs.

---

## expanded apply_inductor_pattern_passes (bucketing + overlap scheduling) — keep (b9c27f1)

- **Idea**: `torch._inductor.fx_passes` exposes 30+ submodules. Iter 5 only used `joint_graph_passes` + `post_grad_passes`. The submodules `bucketing`, `overlap_scheduling`, `micro_pipeline_tp`, `low_contention_collectives`, `fsdp`, `fuse_attention`, `pad_mm`, `b2b_gemm`, `decompose_mem_bound_mm`, `group_batch_fusion`, etc. should give additional gains specific to distributed and matmul-heavy graphs.
- **Changes**: Expanded `apply_inductor_pattern_passes`. After iter 5's pattern matchers, added (in order): config-flag setup, then `bucketing.bucket_all_gather(gm)`, `bucketing.bucket_reduce_scatter(gm)`, `stable_topological_sort(gm)`, `overlap_scheduling.schedule_overlap_bucketing(gm)`. Discovered candidates via `dir(module)`, tried plausible entry points with try/except, kept only ones that changed node count OR improved TPS, dropped all no-ops.
- **Result**: keep. 4 runs: tps=4,781 / 4,777 / 4,793 / 4,800 → **avg 4,788, mfu=28.0%**, memory=49.1 GiB. Loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics test passes. Wall time 140s (compile got ~63s longer than iter 5 due to `schedule_overlap_bucketing`'s analysis — one-time, not per-step).
- **TPS gain attribution**:
  - Without `schedule_overlap_bucketing`: tps drops to ~4,690 (just bucketing alone gives ~+2-3% over iter 5).
  - With `schedule_overlap_bucketing`: tps reaches ~4,780 (extra +2% from reorder).
  - Together: **+4.6% on top of iter 5, +14.9% vs baseline.**
- **Modules tried that did nothing useful here**:
  - `micro_pipeline_tp.micro_pipeline_tp_pass`: ran but no-op. Theory: the 68 TP `all_reduce` nodes don't match its AG+mm / mm+RS patterns — it expects `_c10d_functional.all_gather`/`reduce_scatter`, not `all_reduce`.
  - `fsdp.dedup_fsdp_reduce_scatter`, `bucketing.bucket_all_reduce`, `low_contention_collectives.replace_collectives_with_low_contention`, `reinplace.*`, `group_batch_fusion.group_batch_fusion_passes(pre_grad=False)`, `misc_patterns.numpy_compat_normalization`: all no-ops on this graph.
  - `pad_mm`, `b2b_gemm`, `decompose_mem_bound_mm`, `fuse_attention`: no directly-callable pass entry points in this PyTorch — they're handler/registration modules driven via `joint_graph_passes` config flags. We set the config flags but didn't see additional node changes beyond what joint_graph already did.
- **Lessons**:
  - **`schedule_overlap_bucketing` is the real win** — node-count is unchanged but reordering for comm/compute overlap moved TPS by ~+2%. Iter 3/4's manual FX reordering couldn't find this; the upstream pass does proper dependency analysis.
  - **Upstream `bucketing.*` works where iter 3 didn't**: it knows how to bucket AGs into contiguous-recoverable layouts (no slow reshape-copy).
  - **Iter 5's `binary_folding_pass` and `dedupe_symint_uses_pass` are silent ImportError no-ops** on this PyTorch build — kept the try/except so they activate if a future PyTorch upgrade exposes them.
  - **The `+810` and `+615` node deltas from the bucketing passes are not "more work" but graph restructure**: they add cat / wait / slice / reshape nodes around the bucketed collectives. Net effect on runtime is positive because the per-launch fixed cost goes down.

---

## compile_fx_inductor (whole-graph) — discard (xxxxxxx)

- **Idea**: Route the cleaned joint fwd+bwd graph through Inductor's full pipeline (`compile_fx_inner` / `compile_fx`) to get **Triton kernel codegen**. Iter 5's pattern passes did FX-level rewrites; the next lever is full kernel codegen.
- **Changes**: Added `compile_fx_inductor(gm, example_inputs)` that tries 3 entry points and overrides `gm.forward` with the compiled callable. Registered AFTER `apply_inductor_pattern_passes`. Failures fall through to no-op so iter 5 gains persist.
- **Result**: discard. step 20: tps=4,577 mfu=26.80% (basically iter 5's number, since pass falls through). Numerics PASS. The pass never installed a compiled function.
- **Failures (all 3 variants)**:
  1. `compile_fx_inner(gm, list(example_inputs))` → `InductorError(AssertionError: both a fallback and a decomp for same op: aten._to_copy.default)`. The decomp registry sees `_to_copy.default` twice. Suspected: iter 5's `joint_graph_passes` primed Inductor state, and the second call hits a duplicate.
  2. `fx_codegen_and_compile(gm, list(example_inputs))` → `TypeError: missing 1 required positional argument: 'inputs_to_check'`. API needs args we can't infer without reading the source.
  3. `compile_fx(gm, list(example_inputs))` → `RuntimeError: Found a custom (non-ATen) operator whose output has alias annotations: _c10d_functional::all_reduce_`. `compile_fx` re-runs functionalization; the joint graph already contains the in-place `_c10d_functional.all_reduce_.default` (DTensor redistribute baked it in). This is structural: graphs with mutable functional collectives can't pass through `compile_fx`.
- **Lessons**:
  - **Whole-graph Inductor codegen is blocked on (a) in-place functional collectives, and (b) decomposition registry conflicts after iter 5's pattern passes.** Two distinct walls; both need workarounds.
  - To use Inductor codegen on this workload we must either (i) compile **regions** that exclude collectives, (ii) reset Inductor's registry between calls, or (iii) re-trace the graph fresh and avoid emitting the in-place all_reduce_.
  - First-step compile attempts only added ~7s wall time even when they all failed, so safe try/except wrappers are cheap.

---

## apply_inductor_pattern_passes — keep (76d3f9e)

- **Idea**: PyTorch's Inductor ships several FX-level pattern matchers (`joint_graph_passes`, `post_grad_passes`, etc.) that fuse common ATen sequences (mm+bias+activation, SDPA epilogues, `_to_copy` round-trips, …) by rewriting the GraphModule in place. They were designed for the same kind of joint fwd+bwd graph that `make_fx` produces, so they should apply directly.
- **Changes**: Added `apply_inductor_pattern_passes(gm, example_inputs)` that tries 5 upstream entry points in sequence, each wrapped in its own `try/except`: `joint_graph_passes`, `post_grad_passes(is_inference=False)`, `pre_grad_passes`, `binary_folding_pass`, `dedupe_symint_uses_pass`. Each logs availability and before/after node counts. Registered AFTER `remove_detach_nodes`.
- **Result**: keep. Two consecutive runs: tps=4,569 (mfu=26.75%) and tps=4,576 (mfu=26.80%); avg **tps=4,572 (+9.9% vs baseline 4,161)**. Loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics test passes. Wall time 77s.
- **Node-count trace**:
  - initial: 11,538
  - after `joint_graph_passes`: 9,106 (−2,432, −21%)
  - after `post_grad_passes(is_inference=False)`: 9,101 (−5)
  - `pre_grad_passes`: ran no-op (pre-grad targets a different graph form)
  - `binary_folding_pass`, `dedupe_symint_uses_pass`: not available in this PyTorch build
- **Analysis**: This is the first iteration to move TPS meaningfully — the −21% node reduction translates to a clean +10% TPS. Earlier topology-only FX edits (bucket/prefetch) failed to translate to TPS, but Inductor's pattern matchers are doing real *semantic* fusion (combining producer/consumer ATen ops into single fused ops, removing entire chains of casts, etc.), not just reordering. The pattern matchers also know how to fuse the `_to_copy` round-trips that iter 2 couldn't touch, which likely accounts for a large chunk of the gain.
- **Lessons**:
  - **Reuse upstream Inductor passes before writing your own.** PyTorch already ships pattern matchers that took years to tune; rewriting from scratch is wasteful.
  - The next big wins likely come from running additional Inductor pipeline stages — full **codegen via `compile_fx`** (Triton kernels for pointwise), and **CUDA graphs** for static replay.
  - `joint_graph_passes` is the right entry point for joint fwd+bwd graphs out of `make_fx`. `post_grad_passes` does small extra cleanup on top.
  - The two unavailable passes (`binary_folding_pass`, `dedupe_symint_uses_pass`) suggest the installed PyTorch version is older than the ones that exposed those names. Future iterations should be careful about API drift.

---

## prefetch_all_gathers (FX-level move) — discard (xxxxxxx)

- **Idea**: Move each `all_gather_into_tensor` node to its earliest valid FX position (right after its inputs). The launch happens on a separate NCCL stream, so intervening compute should overlap with the AG transfer; the `wait_tensor` stays at its original position.
- **Changes**: `prefetch_all_gathers(gm, example_inputs)` walks the graph, computes each AG's earliest-valid position from `A.all_input_nodes`, and moves the AG via the FX doubly-linked-list API. Distance capped at 200 to bound memory peak. Registered after `remove_detach_nodes`.
- **Result**: discard. step 20: tps=4,177 mfu=24.46% memory=47.03GiB; loss=9.21808, grad_norm=4.5867. Pass moved only **65 of 421 AGs**, average distance **7** positions, max **8**. The 200-cap never triggered. Numerics test pass.
- **Analysis**: `make_fx` already places AGs immediately after their input shards are computed. Without also hoisting the AG's **input ops** (`_to_copy(view(param))` patterns) earlier in the graph, the AG cannot move. The 7-position headroom is essentially insignificant.
- **Lessons**:
  - **FX-level reordering alone cannot prefetch FSDP all_gathers**: the AG input chain (cast + view of a placeholder param) is what pins the AG in place. To prefetch effectively we must (a) move the input ops too, OR (b) reorder at a different level (NCCL stream priorities, separate compile step), OR (c) attack the **wait_tensor side** instead — move waits LATE toward consumers so AG↔wait gap widens.
  - This also explains iter 3's bucket result: even after hoisting input ops, the shape-heterogeneity meant most AGs stayed singletons. Together they suggest FX-level scheduling of FSDP collectives has limited slack — the dominant gains live in **fusion of the compute** (kernel side) or in **whole-graph compile via inductor** (which does its own scheduling), not in shuffling existing nodes.

---

## bucket_all_gathers (dim-0 shape-grouped) — discard (xxxxxxx)

- **Idea**: Bucket consecutive `_c10d_functional.all_gather_into_tensor.default` calls (421 of them) to amortize launch overhead. Combine inputs via `aten.cat(..., 0)`, do one all_gather, then `view → slice along dim 1 → reshape` to recover per-original outputs preserving rank-major layout.
- **Changes**: Added `bucket_all_gathers` pass after `remove_detach_nodes`. Bucket size cap = 8; grouping by `(group_size, group_name, dtype, device, len(shape), shape[1:])`. Hoisting strategy: input ops (`_to_copy`, `view`, `_unsafe_view`, `t`, `transpose`, `clone`, `reshape`) of later bucket entries get moved to before the first AG so the cat input is computable in-order. Non-pure-op input chains cause flush. Slice→reshape returns non-contiguous → used `aten.reshape.default` (may copy) instead of `view`.
- **Result**: discard. step 20: tps=4,169 mfu=24.41% memory=47.03GiB; loss=9.21808, grad_norm=4.5867 (bitwise identical). Numerics test pass. 421 AGs → 64 bucketed calls (160 AGs bucketed, 261 left as singletons).
- **Analysis**: Shape heterogeneity is the limiter. FSDP unsharding emits AGs with widely varying `shape[1:]` (per-layer: `(16032,4096), (1024,), (1,4096,4096), (512,4096), (128,4096), (128,4096), (1024,2048)` etc.), so the shape-compatible grouping rule rarely finds 3+ neighbors. Average bucket size 2.5. The reduction from 421 → 64+261 = 325 collective launches (~23% fewer launches) is real but does NOT show up as tps. Two likely reasons: (i) the reshape after slice copies data (non-trivial extra work), wiping out the savings; (ii) the unbucketed AGs still dominate runtime — bucketing 38% of nodes leaves 62% untouched.
- **Lessons**:
  - **Shape-grouped dim-0 bucketing has structural limits.** To bucket aggressively across all AGs we must FLATTEN inputs to 1D before cat, then carefully reshape back. The flat path also dodges the non-contiguous slice problem (a 1D slice is contiguous).
  - **Per-original reshape can cost as much as the launch savings.** If a bucketed approach forces a copy per output, the net is ~zero. Future bucketing should arrange so each per-original output is recoverable via a contiguous view (no copy).
  - **The wins from reducing launches require also overlapping with compute.** Without prefetching, a synchronous AG+wait pair still blocks on data arrival. **Bucketing without prefetch ≠ free perf.**
  - **Next ideas to pursue:** (a) AG prefetch — move AG nodes earlier so communication overlaps with intervening compute, leaving the wait at the original position; (b) inductor regional compile on pointwise hot paths; (c) CUDA graphs.

---

## remove_redundant_to_copy — discard (xxxxxxx)

- **Idea**: 842 `aten._to_copy.default` nodes appear in the graph. If any are no-op casts (target dtype/device equals source dtype/device), removing them is safe and cheap.
- **Changes**: Added `remove_redundant_to_copy(gm, example_inputs)` that walks all `_to_copy.default` nodes, compares requested kwargs against the input's `meta["val"]`, and erases only when dtype/device/layout all match AND no user is in-place. Registered after `remove_detach_nodes`.
- **Result**: discard. 0 of 841 candidate `_to_copy` nodes qualified. tps/loss/grad_norm bitwise identical to iteration 1 (4180 tps, loss 9.21808, grad_norm 4.5867). Numerics test passes.
- **Analysis**: All `_to_copy` nodes in this graph are genuine fp32 → bf16 mixed-precision casts (or vice versa), not redundant. The conservative criterion correctly rejected all of them.
- **Lessons**: The 842 `_to_copy` count cannot be reduced by pure elimination; it must be attacked by **(a)** detecting bf16→fp32→bf16 round-trips and collapsing them, **(b)** fusing the cast into producer/consumer kernels (e.g., via inductor regional compile), or **(c)** restructuring the graph so casts happen once instead of repeatedly. Future iterations should investigate the round-trip pattern.

---

## remove_detach_nodes — keep (ae6e425)

- **Idea**: The recon counted 196 `aten.detach.default` nodes. The runtime executes the traced graph under `torch.no_grad()` (see `trainer.py: _make_fx_forward_backward_step`), so detach has no autograd effect at all. Removing them should reduce dispatcher work and free held tensor references.
- **Changes**: Added `remove_detach_nodes(gm, example_inputs)` to `passes.py`. Collects all `aten.detach.default` call_function nodes, rewires users via `replace_all_uses_with(node.args[0])`, erases, then runs `eliminate_dead_code()` / `lint()` / `recompile()`. Registered as the first pass in `construct_default_graph_passes`.
- **Result**: 196 detach nodes removed. Numerics test passes (`aot_fx_trace_vs_eager`). Two benchmark runs: tps=4,202 (mfu=24.60%) and tps=4,174 (mfu=24.44%). Both produce loss=9.21808, grad_norm=4.5867 — bitwise identical to baseline. Memory drops from 49.0 → 47.0 GiB consistently.
- **Analysis**: TPS delta is within the 1-3% noise band, so the speedup is at best marginal. The robust effect is the ~2 GiB memory drop: each detach node held a Python reference to its input via the FX graph, preventing the underlying tensor from being released during graph execution. Numerics are unaffected because detach shares storage with its input, so users that read the same data after removal are reading identical bytes.
- **Lessons**: (1) Removing FX nodes that hold references can free real memory even when the op itself is "free." (2) detach.default in joint fwd+bwd graphs is essentially always dead under run_traced's no_grad. (3) Future cleanup passes should focus on bigger node categories (e.g. 842 `_to_copy.default`, 196 (now 0) detach was the smallest cleanup target).

---

## Graph reconnaissance — discard (xxxxxxx)

- **Idea**: Before picking concrete optimizations, count FX node targets in the joint fwd+bwd graph to find the highest-leverage areas. No optimization is attempted in this iteration.
- **Changes**: Added a temporary `_recon_dump_graph_stats` pass that counts call_function targets and writes the top-40 ops, collective ops, and view-like ops to `/tmp/recon_graph_stats.txt`. Pass returns `gm` unchanged. Reverted after capturing the stats.
- **Result**: discard — no perf change attempted (single inspection run with `--training.steps 3`). Confirmed baseline still passes through unchanged.
- **Analysis**: See LEARNINGS.md "Graph shape snapshot" for the full table. Headline numbers: 11.9k nodes; 421 all_gather_into_tensor + 421 reduce_scatter_tensor + 68 all_reduce; 196 detach + 842 _to_copy; 32 SDPA flash fwd+bwd; 65 fused_rms_norm fwd+bwd. The collective counts are the obvious lever — about 13 unbucketed FSDP all-gathers per layer × 32 layers.
- **Lessons**: Pass list ordering matters: anything we add to `construct_default_graph_passes` runs after make_fx_tracer's `_remove_cpu_shadow_chains`, so the recon numbers above already exclude shadow chains. Future cleanup passes should not target CPU shadow nodes.
