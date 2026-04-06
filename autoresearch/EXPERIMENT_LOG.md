# Experiment Log

This file records every experiment attempted during the autoresearch loop.

## Baseline — keep (2231854b)

- **Idea**: Establish baseline performance for Llama3 8B with aot_fx_trace, FSDP4+TP2 on 8xH100.
- **Changes**: None — ran the benchmark as-is.
- **Result**: tps=4247, MFU=24.87%, memory=48.87GiB (51.44% of 95GB H100).
- **Analysis**: Steady-state at step 20. Step 1 shows ~92 tps due to compilation overhead.
- **Lessons**: Baseline established. ~49GB memory usage leaves headroom for passes that trade memory for speed.

## Autobucketing reordering pass — keep (abe376d5)

- **Idea**: Apply `schedule_overlap_bucketing` (autobucketing) to the full fwd+bwd traced graph to bucket collectives and reorder ops for comm/compute overlap. The graph has 421 all-gathers, 421 reduce-scatters, and 68 all-reduces — high collective count suggests bucketing/reordering could help.
- **Changes**: Added `autobucketing_reordering_pass(gm, example_inputs)` call in `apply_default_graph_passes` in `passes.py`.
- **Result**: tps=4404 (run1) / 4369 (run2), avg ~4387, MFU=25.79/25.58%, memory=50.96GiB. Numerics pass.
- **Analysis**: +3.3% tps improvement over baseline (4247). Memory increased by ~2.1GB (acceptable). The pass buckets small collectives together and reorders ops to overlap communication with computation.
- **Lessons**: `schedule_overlap_bucketing` works well on the full traced graph. It's a proven pass from Inductor and handles the complex collective patterns effectively. Good foundation to build on.

## DCE before autobucketing — discard (xxxxxxx)

- **Idea**: Apply dead code elimination (FX `eliminate_dead_code`) before autobucketing to remove unused nodes. Found 167 dead nodes in the graph (11381 → 11214).
- **Changes**: Added `dce_pass` before `autobucketing_reordering_pass` in `apply_default_graph_passes`.
- **Result**: tps=4292, MFU=25.13%, memory=49.00GiB. Single run due to shared GPU contention.
- **Analysis**: 4292 tps is worse than autobucketing alone (4404 avg). DCE removed 167 nodes but likely disrupted the bucketing pass's scheduling heuristics. Memory improved slightly (49.0 vs 51.0 GiB).
- **Lessons**: DCE before bucketing hurts performance — the dead nodes may serve as scheduling anchors for the overlap pass. Don't assume fewer nodes = better performance when reordering passes follow. DCE alone (without bucketing) may still be worth testing, and DCE after bucketing might not cause the same issue.

## DCE after autobucketing — discard (xxxxxxx)

- **Idea**: Apply DCE after autobucketing instead of before, to clean up dead nodes introduced by the bucketing pass without disrupting its scheduling decisions.
- **Changes**: Added `dce_pass` after `autobucketing_reordering_pass`. Bucketing increased nodes from 11381→12334, DCE removed 167 dead nodes.
- **Result**: tps=4480 (run1) / 4245 (run2), avg ~4362. Wide variance due to GPU contention from other users.
- **Analysis**: Average 4362 tps vs autobucketing alone 4387 tps — within noise range. No clear benefit. The 5% spread between runs (4480 vs 4245) makes it impossible to distinguish signal from noise.
- **Lessons**: High variance on shared GPUs makes small improvements hard to measure. Need larger effects (>5%) to be confident. DCE of 167 nodes (out of 12K) is too small to matter.

## Regional Inductor after autobucketing — discard (xxxxxxx)

- **Idea**: Apply regional_inductor after autobucketing to compile compute-heavy subgraphs through Inductor (triton) for kernel fusion.
- **Changes**: Added `regional_inductor_pass(gm, example_inputs)` after autobucketing.
- **Result**: tps=4463, MFU=26.13%, memory=50.96GiB. Single run.
- **Analysis**: Regional inductor was a no-op — the aot_fx_trace graph has no `compile_with_inductor` annotations. In AOT mode, flex_attention ops are annotated via `annotate_flex_attention_for_regional_inductor` context manager, but aot_fx_trace mode doesn't use this. 4463 tps is noise variation from autobucketing alone.
- **Lessons**: Regional inductor needs explicit annotations on graph nodes. It won't auto-detect regions to compile. To use it in aot_fx_trace mode, we'd need to either (a) manually annotate regions, or (b) use full_inductor_compilation_pass instead (which compiles everything, not just regions).

## Full Inductor compilation — crash (xxxxxxx)

- **Idea**: Apply `compile_fx_inner` (full Inductor with triton codegen) after autobucketing to fuse ops aggressively.
- **Changes**: Added `full_inductor_compilation_pass(gm, example_inputs)` after autobucketing.
- **Result**: Crash — `InductorError: AssertionError: both a fallback and a decomp for same op: aten._to_copy.default`
- **Analysis**: Full Inductor compilation requires `inductor_decomposition_pass` first to decompose ops into Inductor-lowerable forms. In AOT mode, this is done as a joint pass. In aot_fx_trace mode, we'd need to apply it manually. However, `inductor_decomposition_pass` requires `JointWithDescriptors` context (primals/tangents split info) which is not available in `apply_default_graph_passes`.
- **Lessons**: Inductor compilation requires its own decomposition pipeline. Can't just drop `compile_fx_inner` onto an arbitrary FX graph. Would need significant plumbing to make this work in aot_fx_trace mode.

## Remove detach nodes before autobucketing — keep (c6f0bef6)

- **Idea**: Remove 196 `aten.detach.default` nodes from the traced graph. In a traced graph, detach is semantically a no-op (no autograd context). Removing them reduces graph size and may improve scheduling/memory.
- **Changes**: Added `remove_detach_pass` before autobucketing in `apply_default_graph_passes`.
- **Result**: tps=4466/4427/4388 (avg 4427), MFU=26.15/25.92/25.70%, memory=49.0GiB. Numerics pass.
- **Analysis**: +0.9% tps over autobucketing alone (marginal, within noise). But -2 GiB memory (49.0 vs 51.0), which is consistent across runs. The detach removal simplifies the graph and reduces memory pressure from unnecessary tensor references.
- **Lessons**: Removing identity ops is always a good idea even if tps doesn't change — it simplifies the graph and can save memory. Apply identity removal early (before other passes) so downstream passes work on a cleaner graph.

## Regional Inductor with element-wise annotations — crash (xxxxxxx)

- **Idea**: Annotate 1598 element-wise/reduction ops with `compile_with_inductor` metadata, then apply regional_inductor to fuse them into triton kernels.
- **Changes**: Added `annotate_elementwise_for_inductor_pass` and called regional_inductor_pass after it.
- **Result**: Crash — `AssertionError: Invalid partition, found dependency cycles` during `fuse_by_partitions`.
- **Analysis**: Annotating scattered element-wise ops creates dependency cycles: annotated node A depends on unannotated node B which depends on annotated node C. The partitioner can't group them without creating cycles. Would need to annotate ALL nodes in contiguous regions (including views, transposes) to avoid this.
- **Lessons**: Regional inductor's partition algorithm requires contiguous annotated regions without internal dependencies on unannotated nodes. Can't just sprinkle annotations on arbitrary ops. Need to annotate complete subgraphs between partition boundaries (e.g. between collectives).

## Remove identity view/reshape nodes — keep (6406b785)

- **Idea**: Remove view/reshape/_unsafe_view ops where output shape matches input shape. These are identity ops in the traced graph — no-op reshapes that add graph complexity.
- **Changes**: Added `remove_identity_view_pass` before autobucketing. Checks `aten._unsafe_view.default`, `aten.view.default`, `aten.reshape.default` — compares input/output shapes via fake tensor metadata.
- **Result**: tps=4818/4879 (avg 4849), MFU=28.21/28.57%, memory=49.0GiB. Numerics pass.
- **Analysis**: Removed **1522** identity view/reshape nodes (out of ~11K total) — 13.4% of all nodes! This is ~6.5x more than expected (225 _unsafe_view) because view.default and reshape.default also had many identity instances. The +9.5% tps improvement over previous best (4427→4849) comes from: (a) fewer nodes means autobucketing can produce better schedules, (b) reduced graph execution overhead from fewer kernel dispatches, (c) simpler data flow for the runtime to optimize.
- **Lessons**: Identity view/reshape removal is a high-impact graph cleanup. Unlike DCE (which removed 167 nodes and hurt bucketing), removing identity views actually helps bucketing because it removes noise nodes that dilute the scheduler's view of actual compute/comm patterns. Graph simplification passes that remove genuinely unnecessary ops (not just dead code) can have outsized impact.

## Remove identity slice nodes — keep (85446291)

- **Idea**: Remove `aten.slice.Tensor` ops that select the full dimension (start=0, end>=dim_size, step=1). These are no-ops that return the input tensor unchanged.
- **Changes**: Added `remove_identity_slice_pass` after identity view removal, before autobucketing.
- **Result**: tps=4955/4962 (avg 4959), MFU=29.01/29.06%, memory=48.98GiB. Numerics pass.
- **Analysis**: All 453 slice ops are identity — every single one selects the full dimension. +2.2% tps over previous best (4849→4959). Combined with view removal, we've now eliminated 2171 identity ops (196 detach + 1522 view + 453 slice = 2171) from the original ~11.4K nodes — 19% of the graph.
- **Lessons**: Identity slices are very common in traced graphs from PyTorch's eager dispatch. The full-dimension slice pattern occurs when code like `x[:]` or `x[..., :]` is traced. Cumulative effect of graph simplification is strong: each round of identity removal makes the graph cleaner for autobucketing.

## Collapse consecutive view chains — keep (ab0e0a82)

- **Idea**: Collapse chains of consecutive view/reshape ops into a single op. When view(view(x, s1), s2), the intermediate view with single use is redundant.
- **Changes**: Added `collapse_view_chains_pass` after identity removal, before autobucketing. Only collapses when intermediate result has a single user.
- **Result**: tps=5032/5004 (avg 5018), MFU=29.47/29.30%, memory=49.0GiB. Numerics pass.
- **Analysis**: Collapsed 323 view chains. +1.5% tps over previous (4959→5018). Total graph simplification now: 2494 nodes removed/collapsed (22% of original graph). We've broken the **5000 tps barrier** (baseline was 4247).
- **Lessons**: View chain collapsing is a modest but additive win. The single-use constraint is important — if intermediate shapes are needed by other nodes, collapsing would change semantics. Combined with identity removal, graph simplification is the dominant optimization strategy so far.

## Remove canceling transpose pairs — keep (58900f23)

- **Idea**: Remove t(t(x)) → x and transpose(transpose(x, d0, d1), d0, d1) → x pairs where the inner transpose has single use.
- **Changes**: Added `remove_transpose_pairs_pass` after view chain collapse, before autobucketing.
- **Result**: tps=5094/5064 (avg 5079), MFU=29.83/29.65%, memory=49.0GiB. Numerics pass.
- **Analysis**: Removed 225 pairs (450 nodes = 40% of 1125 transposes). +1.2% tps. These come from forward + backward cancellation: in forward, weight is transposed for mm; in backward, the transpose is reversed. Total nodes removed/collapsed: 2944 (26% of original 11,381).
- **Lessons**: Transpose pairs are common in fwd+bwd traced graphs. The forward adds transpose for mm weight, backward reverses it. Diminishing returns on graph simplification — each successive pass finds fewer redundancies.
