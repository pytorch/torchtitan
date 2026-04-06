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
