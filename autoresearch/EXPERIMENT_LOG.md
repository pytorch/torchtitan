# Experiment Log

Cumulative log of all experiments. Never overwrite previous entries.

## Baseline — keep (4463e48)

- **Idea**: Establish baseline performance for Llama3 8B with aot_fx_trace, FSDP(4)+TP(2) on 8 GPUs.
- **Changes**: No changes. Default graph passes: tlparse logging, remove_detach, remove_identity_view, remove_identity_slice, annotate_flex_attention_for_regional_inductor, regional_inductor, cudagraph.
- **Result**: TPS=6938, MFU=40.63%, Memory=46.90GiB
- **Analysis**: This is the reference point. The graph already has regional_inductor compiling flex attention regions, and cudagraph wrapping the entire graph.
- **Lessons**: Starting point for all future experiments. Need to profile to understand where time is spent.

## Regional Inductor for all compute ops — discard (xxxxxxx)

- **Idea**: Tag all non-collective call_function nodes with `compile_with_inductor` so regional_inductor compiles them into fused Inductor kernels. The baseline had no inductor compilation for the 8B model (it uses SDPA, not flex_attention, so the flex_attn annotation pass is a no-op). Expected benefit: reduced kernel count and memory bandwidth through op fusion.
- **Changes**: Added `annotate_all_compute_for_regional_inductor_pass` that tags all `torch.ops.aten.*` nodes (excluding `_c10d_functional`, `_dtensor`, `device_mesh` ops) with `compile_with_inductor`. Inserted before `regional_inductor_pass` in the pass list.
- **Result**: TPS=6927 vs baseline 6938 (-0.16%), Memory=46.90GiB, MFU=40.57%. Numerics pass (all 4 aot_fx_trace_vs_eager tests pass).
- **Analysis**: No meaningful improvement. cudagraph already eliminates kernel launch overhead, so fusing ops into fewer kernels doesn't help the launch path. The fused kernels might save memory bandwidth for small ops, but the overall computation is dominated by large matmuls and SDPA which inductor doesn't improve.
- **Lessons**: With cudagraph enabled, kernel fusion via regional_inductor provides minimal benefit since launch overhead is already amortized. The bottleneck must be elsewhere — likely in the compute/communication overlap (or lack thereof). Need to profile to identify actual bottlenecks. Pre-existing test failure: `test_eager_self_deterministic` for flex_attn variants is broken regardless of our changes (hash mismatch in expected values).

## Autobucketing reordering pass — keep (5b1ae19)

- **Idea**: Add `autobucketing_reordering_pass` (which calls `schedule_overlap_bucketing` with `collective_bucketing=True`) to improve comm/compute overlap. The flat fwd+bwd graph has 421 all-gathers + 421 reduce-scatters + 68 all-reduces — substantial communication that could overlap with compute.
- **Changes**: Added `autobucketing_reordering_pass` to the default pass list after `regional_inductor_pass` and before `cudagraph_pass`.
- **Result**: TPS=7214/7198 (two runs, avg ~7206) vs baseline 6938 (+3.9%), Memory=48.95GiB (+2.05GiB), MFU=42.24%. Numerics: all 4 aot_fx_trace_vs_eager tests pass.
- **Analysis**: The bucketing pass reorders collectives and compute ops to maximize overlap. The +3.9% improvement confirms that comm/compute overlap was a significant bottleneck in the baseline. The memory increase is modest and acceptable.
- **Lessons**: Comm/compute overlap is the primary optimization lever for this workload. The autobucketing pass from inductor's overlap scheduling infrastructure works well on the make_fx flat graph. Future experiments should explore whether more aggressive overlap strategies (manual bucketing, separate PGs for AG/RS) can stack on top.

## Regional inductor + autobucketing — discard (xxxxxxx)

- **Idea**: Combine regional inductor compilation with autobucketing. Annotate all compute ops → autobucketing reorder → regional inductor compile → cudagraph. Hypothesis: fused compute kernels execute faster between collectives.
- **Changes**: Added `annotate_all_compute_for_regional_inductor_pass` before autobucketing, moved `regional_inductor_pass` after autobucketing.
- **Result**: TPS=7223 vs current best 7214 (+0.1%). No meaningful improvement.
- **Analysis**: Even with better kernel fusion, the performance is the same. Confirms inductor fusion is redundant with cudagraph.
- **Lessons**: Kernel fusion + cudagraph is a dead end for this workload. Don't revisit.

## Dead Code Elimination — discard (xxxxxxx)

- **Idea**: Add FX's built-in `eliminate_dead_code()` to remove unused nodes, reducing graph size and execution time.
- **Changes**: Added `dead_code_elimination_pass` (calls `gm.graph.eliminate_dead_code()`) after the cleanup passes but before autobucketing.
- **Result**: TPS=7215 vs current best 7214 (±0.01%). No change.
- **Analysis**: The existing cleanup passes (remove_detach, remove_identity_view, remove_identity_slice) already handle the main sources of dead code. FX DCE finds no additional dead nodes.
- **Lessons**: Not worth adding. The graph is already clean after existing passes.

## Autobucketing without cudagraph — discard (xxxxxxx)

- **Idea**: Remove cudagraph to allow more flexible dynamic scheduling. Hypothesis: per-step scheduling might find better overlaps.
- **Changes**: Disabled cudagraph pass in `construct_default_graph_passes`, kept autobucketing.
- **Result**: TPS=5002 vs current best 7214 (-30.7%). Catastrophic regression.
- **Analysis**: Without cudagraph, kernel launch overhead for ~11K+ operations dominates. Cudagraph is not optional.
- **Lessons**: Cudagraph is essential. Never remove it.

## Separate AG PG + autobucketing — discard (xxxxxxx)

- **Idea**: Create a separate NCCL process group for FSDP all-gathers so AG and RS use different CUDA streams, enabling AG/RS overlap in addition to compute/comm overlap.
- **Changes**: Added `create_separate_ag_pg_pass` that finds FSDP PG from graph, creates extra PG, reassigns AGs. Applied before autobucketing.
- **Result**: TPS=7199 vs current best 7214 (-0.2%). Numerics pass. No improvement.
- **Analysis**: The separate PG doesn't help because autobucketing already handles scheduling. Adding a second communicator doesn't create useful overlap — the compute between collectives is the same.
- **Lessons**: Separate PGs are only useful when there's untapped overlap potential. With autobucketing already scheduling well, there's no additional benefit.

## Aggressive autobucketing parameters — discard (xxxxxxx)

- **Idea**: Tune `schedule_overlap_bucketing` parameters: `max_memory_increase_gb=None`, `max_memory_increase_ratio=None`, `max_in_flight_gb=10`, `compute_overlap_multipler=2.0`.
- **Changes**: Modified `autobucketing_reordering_pass` to use aggressive parameters.
- **Result**: TPS=7159 vs current best 7214 (-0.8%). Worse than default params.
- **Analysis**: Over-aggressive scheduling causes contention. Default parameters are already well-tuned.
- **Lessons**: Don't over-tune autobucketing parameters. The defaults work well for this workload.

## Elementwise-only regional inductor — discard (xxxxxxx)

- **Idea**: Tag only elementwise ops (skip matmuls, SDPA, fused RMSNorm) for regional_inductor. Targets the small chains BETWEEN large compute kernels.
- **Changes**: Added `annotate_elementwise_for_regional_inductor_pass` skipping `_SKIP_INDUCTOR_TARGETS`. Had a bug: `str(node.target)` returns `'aten.xyz'` not `'torch.ops.aten.xyz'`, so _is_taggable rejected ALL nodes.
- **Result**: TPS=7200 vs 7214 (-0.2%). Bug meant 0 nodes were actually tagged. After fixing the bug (next experiment), still no improvement.
- **Lessons**: The `str(node.target)` representation in FX graphs is `'aten.op.default'` not `'torch.ops.aten.op.default'`. Must use substring matching, not prefix matching.

## Targeted RoPE + MLP activation inductor (fixed BFS) — crash (xxxxxxx)

- **Idea**: Tag RoPE regions (view_as_complex/real, _conj) and MLP activation (silu, silu_backward) with BFS expansion to find contiguous regions. After fixing the `_is_taggable` bug, tagged 1920 nodes.
- **Changes**: BFS expansion from 384 seeds through all taggable connected nodes.
- **Result**: TPS=7282 (+0.94% vs autobucketing), but **numerics FAILED** for both Llama3 (loss diff ~5e-7) and DSv3. Inductor's complex multiplication (RoPE) and silu/sigmoid produce different rounding than eager.
- **Analysis**: The RoPE chain involves complex multiplication which inductor decomposes differently. The silu function's sigmoid computation has different FMA chains in Triton vs CUDA.
- **Lessons**: **Inductor does NOT guarantee bitwise identity for transcendental functions** (exp, sigmoid) or complex arithmetic. `eager_numerics.division_rounding=True` only fixes division rounding, not all ops. `emulate_precision_casts`, `use_pytorch_libdevice` also insufficient.

## SiLU+mul only inductor (no RoPE) — crash (xxxxxxx)

- **Idea**: Tag only silu + its immediate mul user (SwiGLU gate pattern). No BFS expansion. Excludes RoPE entirely.
- **Changes**: Direct tagging of silu nodes and their mul users (160 nodes total).
- **Result**: TPS=7292/7287 (+1.1%), Llama3 numerics **PASS**, but DSv3 numerics **FAIL** (loss diff ~1e-6). The Triton silu kernel doesn't match eager on DSv3's tensor shapes.
- **Analysis**: Inductor's silu kernel works for Llama3 shapes but fails for DSv3 MoE shapes. Tried `eager_numerics.use_pytorch_libdevice=True` — still fails. The issue is fundamental to how Triton computes sigmoid vs eager CUDA.
- **Lessons**: **Cannot use regional_inductor for silu/sigmoid if bitwise determinism is required across all models.** The +1.1% TPS gain for Llama3 is real but not achievable without breaking DSv3.

## Constant folding — discard (xxxxxxx)

- **Idea**: Use `torch._inductor.constant_folding.constant_fold` to fold constant subexpressions at compile time.
- **Result**: TPS=7221 vs 7214 (±0.1%). No improvement.
- **Analysis**: With cudagraph, constants are computed once during graph capture and reused. Folding them in the FX graph just moves the work from capture time to pass time.

## In-place op conversion — discard (xxxxxxx)

- **Idea**: Convert `add.Tensor` and `mul.Tensor` to in-place variants where the first input has no other users.
- **Changes**: Added `convert_to_inplace_pass`. Converted 226 ops. Memory dropped 40 MB (48.91 vs 48.95 GiB).
- **Result**: TPS=7225 vs 7214 (±0.15%). No meaningful TPS improvement.
- **Analysis**: With cudagraph, all memory is pre-allocated at capture time. In-place ops don't avoid allocation since the memory pool is fixed. The 40 MB memory savings is minor.

---

## Final Summary

**Stopping after 10 consecutive non-keeps following the autobucketing result.**

### What worked
- **Autobucketing reordering** (`schedule_overlap_bucketing` with `collective_bucketing=True`): **+3.9% TPS** (6938 → 7214). This is the only successful optimization. It reorders FSDP/TP collectives relative to compute to maximize comm/compute overlap, and buckets small collectives into larger ones (reduced from 1820 to 1382 collective ops).

### What was tried but didn't work
1. **Regional inductor (all compute ops)**: No improvement — cudagraph makes kernel launch overhead zero
2. **Regional inductor (elementwise only)**: No improvement — collectives fragment graph into tiny 2-3 node regions
3. **Regional inductor (RoPE/complex ops)**: Breaks numerics — inductor decomposes complex multiplication differently
4. **Regional inductor (silu+mul fusion)**: +1.1% TPS but breaks DSv3 numerics — Triton sigmoid != eager
5. **Dead code elimination**: No improvement — existing passes already clean the graph
6. **Removing cudagraph**: -30% TPS catastrophic regression
7. **Separate FSDP PG for AG/RS**: No improvement — autobucketing already schedules well
8. **Aggressive autobucketing params**: Worse than default
9. **Constant folding**: No improvement — cudagraph already caches constants
10. **In-place op conversion**: -40 MB memory but no TPS gain

### Avenues that remain unexplored
- **Manual transformer block bucketing**: Requires `nn_module_stack` metadata (currently disabled in make_fx). Could provide better structured overlap than autobucketing.
- **Full inductor compilation** (compile_fx_inner on entire graph): Potentially faster than eager+cudagraph but requires `inductor_decomposition` joint pass and careful numerics validation.
- **Selective activation checkpointing**: Could trade memory for compute to enable larger batch sizes (if batch size were variable).
- **Model-specific optimizations**: e.g., fusing RoPE complex multiplication as a custom kernel that guarantees bitwise identity.

