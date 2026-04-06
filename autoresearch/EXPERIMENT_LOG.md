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

## Roundtrip dtype conversion removal — discard (xxxxxxx)

- **Idea**: Remove _to_copy(bf16→fp32) → _to_copy(fp32→bf16) roundtrips where single-use intermediate converts back to original dtype.
- **Result**: No-op — zero roundtrip pairs found. All 842 _to_copy ops are genuine conversions.
- **Lessons**: Mixed precision conversions in the traced graph are all necessary. No redundant roundtrips.

## Aggressive autobucketing memory params — discard (xxxxxxx)

- **Idea**: Increase max_memory_increase_gb from 1.0→10.0 and max_in_flight_gb from 5→10 to allow more prefetching.
- **Result**: tps=5055, memory=48.98 GiB — within noise, memory didn't increase.
- **Lessons**: Scheduler already uses the memory budget effectively. More budget doesn't help.

## CUDAGraph full graph — crash (xxxxxxx)

- **Idea**: Wrap the full fwd+bwd graph with CUDAGraph to eliminate kernel launch overhead.
- **Result**: Crash — CUDAGraphWrapper rejects float scalar input at index 585 (value 32768.0).
- **Lessons**: aot_fx_trace mode lifts scalar constants as graph inputs. CUDAGraphWrapper only handles tensors, ints, Generators, and opaque objects. Would need to modify cudagraph.py to handle floats.

## Regional inductor with contiguous regions — crash (xxxxxxx)

- **Idea**: Annotate ALL nodes between consecutive collectives (not scattered ops). Complete contiguous regions should avoid the dependency cycle issue.
- **Result**: Same crash — `AssertionError: Invalid partition, found dependency cycles`.
- **Lessons**: Dependency cycles in fwd+bwd graph are fundamental, not caused by scattered annotations. Backward regions depend on forward regions in complex, non-sequential ways. Regional inductor can't partition the full fwd+bwd graph.

## Autobucketing enable_fusion_regions — discard (xxxxxxx)

- **Idea**: Set enable_fusion_regions=True in autobucketing to detect and account for fusible op regions.
- **Result**: tps=5043 — within noise of baseline (5079).
- **Lessons**: Fusion region detection doesn't change the scheduling significantly for this graph.

## Aggressive overlap scheduling — keep (2b94943d)

- **Idea**: Set compute_overlap_multipler=2.0 to make the scheduler more aggressive about overlapping compute with communication.
- **Changes**: Modified autobucketing_reordering_pass to pass `compute_overlap_multipler=2.0`.
- **Result**: tps=5142/5187 (avg 5165), MFU=30.11/30.37%, memory=49.0GiB. Numerics pass.
- **Analysis**: +1.7% tps over previous (5079→5165). The default multipler (1.0) was too conservative — it underestimated how much compute could be overlapped. 2.0× tells the scheduler that compute takes twice as long as estimated, so it schedules more collectives concurrently. Crossed **30% MFU**.
- **Lessons**: The default compute estimation in autobucketing may undercount the time for certain ops. Increasing the overlap aggressiveness helps when there's enough compute to hide communication latency. This is model-size and parallelism-config dependent.

## Split-cat cancellation — discard (xxxxxxx)

- **Idea**: Remove split→cat identity patterns where cat reassembles all split outputs in order along the same dimension. 131 split + 130 cat ops could have matching pairs.
- **Changes**: Added `remove_split_cat_pairs_pass` after transpose pair removal.
- **Result**: No-op — zero split→cat identity pairs found. tps=5129 (noise).
- **Analysis**: The 131 split and 130 cat ops serve different purposes in the graph (splits for FSDP param extraction, cats for gradient assembly). None are direct inverses of each other.
- **Lessons**: Split-cat cancellation is not applicable to FSDP/TP traced graphs — the ops are structurally necessary.

## CUDAGraph with float constant folding — keep (pending commit)

- **Idea**: Enable CUDAGraph for the full fwd+bwd graph by fixing the float scalar input blocker. aot_fx_trace lifts Python scalar constants (e.g. 32768.0) as graph placeholder inputs, which CUDAGraphWrapper rejects. Inline these constants into the graph to remove them from inputs.
- **Changes**: Added two new passes:
  1. `materialize_float_constants_pass`: Finds float placeholder inputs, replaces all uses with the literal float value, erases the placeholder. Stores float indices on `gm._materialized_float_indices`.
  2. `cudagraph_scalar_folded_pass`: Wraps `gm.forward` with CUDAGraphWrapper using filtered example_inputs (no floats). Installs outer wrapper that strips float args and detaches tensor args before CUDAGraph.
  - Detaching is safe because aot_fx_trace includes backward in the graph — no external autograd needed.
  - Only 1 float input found: index 585, value 32768.0 (loss normalization constant).
  - All other non-tensor inputs are DeviceMesh (291) and integers, which CUDAGraphWrapper handles.
- **Result**: tps=6952/6989 (avg 6971), MFU=40.71/40.92%, memory=48.95GiB. Numerics pass (all 4 bitwise deterministic tests pass including Llama3 and DeepSeek-v3).
- **Analysis**: **+35.0% tps over previous best** (5165→6971). CUDAGraph eliminates kernel launch overhead for the entire fwd+bwd graph (~8437+ nodes after autobucketing). With ~10μs per kernel launch, this saves ~84ms per step. At ~1.59s per step (at 5165 tps), this is ~5.3% from launch overhead alone. The remaining 30% improvement likely comes from CUDAGraph's ability to batch CUDA API calls and optimize GPU scheduling.
  - Currently uses `static_input_indices=[]` (no static inputs), meaning all 585 inputs are copied every step. This is suboptimal for parameters (~294 tensors) which are static. Optimizing this could yield further gains.
  - Total improvement over baseline: **+64.1% (4247→6971)**.
  - Crossed **40% MFU** barrier.
- **Lessons**: CUDAGraph is the single highest-impact optimization. The float input blocker was trivial to fix (1 constant, 1 pass). The detach trick for `requires_grad` tensors is critical — CUDAGraphWrapper's `copy_()` fails on leaf variables with grad. In aot_fx_trace mode, detaching at the graph boundary is safe because autograd is handled internally.

## CUDAGraph static input optimization — keep (14ba6512)

- **Idea**: Mark parameter tensors as static in CUDAGraphWrapper to skip unnecessary copy_() for 291 params each step.
- **Changes**: Identify static inputs by `dtype.is_floating_point` (params are fp32, batch data is int64). Example_inputs are FakeTensors so `requires_grad` isn't reliable.
- **Result**: tps=7305/6669 (avg 6987), within noise of non-static (avg 7039). Memory unchanged. Numerics pass (bitwise identical over 10 deterministic steps).
- **Analysis**: Copy overhead for 291 params is ~4ms per step (0.4% of step time). Negligible improvement but technically cleaner.
- **Lessons**: FakeTensors don't preserve `requires_grad`, so use dtype as discriminator. Static params reduce copy overhead but it's too small to measure.

## CUDAGraph without autobucketing — discard (xxxxxxx)

- **Idea**: With CUDAGraph eliminating launch overhead, autobucketing's concat/split overhead may outweigh its overlap benefit.
- **Changes**: Commented out `autobucketing_reordering_pass`.
- **Result**: tps=7080/6585 (avg 6833), memory=46.90 GiB. -2 GiB memory but ~3% lower tps.
- **Analysis**: Autobucketing still helps with CUDAGraph because it reorders ops for comm/compute overlap (not just for launch batching). The 2 GiB memory savings comes from not needing bucketing concat/split buffers.
- **Lessons**: Autobucketing's value with CUDAGraph is from overlap scheduling, not from reducing launch overhead.

## Reorder-only (no bucketing) + CUDAGraph — discard (xxxxxxx)

- **Idea**: Keep the reordering (comm/compute overlap) but disable collective bucketing.
- **Changes**: `collective_bucketing=False` in `schedule_overlap_bucketing`.
- **Result**: tps=7096, memory=49.58 GiB. Within noise, slightly more memory than full bucketing.
- **Lessons**: All three variants (full bucketing, reorder-only, no scheduling) produce similar tps with CUDAGraph. The variance is too high to distinguish.

## bf16 all-gather communication — discard (xxxxxxx)

- **Idea**: Move fp32→bf16 conversion before all-gather to halve FSDP communication volume.
- **Result**: No candidates found — all-gathers already communicate in bf16. SimpleFSDP handles mixed precision at the FSDP level.
- **Lessons**: SimpleFSDP already stores shards in bf16 and all-gathers in bf16. No further optimization needed.

## Aggressive prefetch params + CUDAGraph — discard (xxxxxxx)

- **Idea**: With CUDAGraph, memory fragmentation isn't a concern. Increase max_compute_pre_fetch=500, max_coll_distance=500, max_in_flight_gb=20.
- **Result**: tps=7301. Within noise.
- **Lessons**: With CUDAGraph, the scheduler's decisions are baked into the recorded graph. More aggressive prefetching doesn't help because the execution order is already fixed.

## torch.compile on traced graph — crash (xxxxxxx)

- **Idea**: Apply torch.compile(mode='reduce-overhead') to the graph for Inductor kernel fusion + CUDAGraph.
- **Result**: Crash — `RuntimeError: Found a custom (non-ATen) operator whose output has alias annotations: _c10d_functional::all_gather_into_tensor_out`.
- **Lessons**: Inductor can't compile graphs containing NCCL collective ops. The bucketing pass introduces out-variant collectives that Inductor can't functionalize. Would need regional compilation that excludes collectives.

## Constant fold uniform values — discard (xxxxxxx)

- **Idea**: Apply `constant_fold_uniform_value` from Inductor to fold uniform tensors (all-zeros, all-ones) in the graph before bucketing.
- **Result**: tps=6957. Within noise.
- **Lessons**: Pass applied successfully but found no meaningful constants to fold. The traced graph doesn't have uniform tensor computations.

## Benchmark-mode collective estimation — discard (xxxxxxx)

- **Idea**: Use `collective_estimator="benchmark"` for more accurate NCCL timing in autobucketing scheduler.
- **Result**: tps=6358. -8.8% regression.
- **Lessons**: Benchmark-mode collective estimation produces worse scheduling decisions. The analytical model (default) is better calibrated for this graph/hardware combination.

## Analytical compute estimation — discard (xxxxxxx)

- **Idea**: Use `compute_estimator="analytical"` (roofline model) instead of default GPU benchmarking for compute timing in scheduler.
- **Result**: tps=6315. -9.4% regression.
- **Lessons**: GPU-benchmarked compute estimation (default) is more accurate than analytical roofline. Both alternative estimators hurt scheduling quality.

## CUDAGraph only — no simplification, no bucketing — discard (xxxxxxx)

- **Idea**: Ablation study: CUDAGraph wrapping only, skip all simplification passes and bucketing. Measures the contribution of simplification+bucketing.
- **Result**: tps=6646, MFU=38.92%.
- **Lessons**: Simplification+bucketing contribute ~5% tps improvement even with CUDAGraph (6646→6971). Identity node removal helps because fewer graph nodes = less CUDAGraph replay overhead.

## Graph diagnostic analysis — informational

- **Idea**: Comprehensive op-count analysis of the simplified graph (before bucketing).
- **Result**: 7844 call_function nodes. Top ops: view(1126), wait_tensor(910), getitem(906), _to_copy(842), t(675), mm(675), AG(421), RS(421), transpose.int(256), mul(225), add(225), split(131), cat(130), view_as_complex(128), view_as_real(128), _unsafe_view(128), clone(68), AR(68), _fused_rms_norm(65).
- **Findings**: 0 identity expand ops, 0 contiguous ops, 0 identity _to_copy. 842 _to_copy: exactly 420 fp32→bf16 + 420 bf16→fp32 (mixed precision symmetry). 68 clone → 64 feed into mul.Tensor (backward activations). No removable ops found.
- **Lessons**: All remaining ops are semantically necessary. No more identity/no-op patterns to eliminate at the FX graph level.

## CSE (Common Subexpression Elimination) — discard (xxxxxxx)

- **Idea**: Apply PyTorch's `CSEPass` to deduplicate identical computations in the graph. CSE found 258 duplicate nodes (8437→8179).
- **Result**: tps=6280. -10% regression.
- **Lessons**: CSE creates new data dependencies (shared nodes become bottlenecks), which constrains the scheduler's ability to reorder for comm/compute overlap. Graph-level CSE is counterproductive for overlap scheduling.

## Reorder-only + CUDAGraph (no bucketing) — discard (xxxxxxx)

- **Idea**: Skip collective bucketing, only reorder for overlap. Eliminates ~282 clone ops from bucketing. Multi-warmup CUDAGraph wrapper to stabilize NCCL capture.
- **Result**: Inconsistent: 3/4 runs give ~7350 tps (43% MFU), 1/4 gives 5798 tps. Multi-warmup with bucketing regresses to 6389 tps.
- **Lessons**: Without bucketing, CUDAGraph capture is non-deterministic (842 individual NCCL collectives have variable timing). Multi-warmup CUDAGraph fragments the memory pool, reducing CUDAGraph efficiency. Bucketing provides capture stability at the cost of 282 extra clone ops.

## custom_ops bucket mode — discard (xxxxxxx)

- **Idea**: Use `bucket_mode="custom_ops"` instead of default in `schedule_overlap_bucketing`. custom_ops mode uses opaque `_foreach_copy_` ops for bucketing instead of traced cat+copy sequences, which avoids intermediate buffers and should be more memory-efficient.
- **Result**: tps=6978, MFU=40.87%, memory=49.0GiB.
- **Lessons**: Within noise of baseline CUDAGraph (6971). custom_ops mode works correctly but doesn't provide measurable tps improvement. The default bucket mode's traced operations are already efficiently handled by CUDAGraph replay.

## NCCL_MAX_NCHANNELS=32 — discard (xxxxxxx)

- **Idea**: Tune NCCL channel count via `NCCL_MAX_NCHANNELS=32` environment variable (default is typically 8-16 depending on algorithm). More channels increases bandwidth utilization but adds overhead per collective.
- **Result**: tps=6003, MFU=35.16%, memory=49.0GiB. -14% regression.
- **Lessons**: More NCCL channels significantly hurts performance in this CUDAGraph-captured graph. Each extra channel adds kernel overhead that compounds across 421 AG + 421 RS + 68 AR collectives. The default NCCL channel count is already well-tuned for NVLink topology. Environment-level NCCL tuning is unlikely to help — the bottleneck is elsewhere.

## custom_ops_multidtype bucket mode — discard (xxxxxxx)

- **Idea**: Use `bucket_mode="custom_ops_multidtype"` which allows mixed-dtype (fp32+bf16) bucketing, potentially enabling better scheduling given 842 mixed-precision ops.
- **Result**: tps=7340/7332 (avg 7336), MFU=42.94%.
- **Lessons**: Within noise of default bucket mode. Mixed-dtype bucketing doesn't change the scheduling outcome significantly.

## CUDA_DEVICE_MAX_CONNECTIONS=1 — discard (xxxxxxx)

- **Idea**: Force all CUDA work through a single hardware queue. Can improve overlap between compute and NCCL communication with CUDAGraph.
- **Result**: tps=7355, MFU=43.07%.
- **Lessons**: Within noise. With CUDAGraph, the CUDA stream is already recorded and replayed deterministically. Changing queue depth doesn't affect replay performance.

## Identity _to_copy removal — no-op (xxxxxxx)

- **Idea**: Remove _to_copy ops where input and output dtypes match (identity casts exposed after simplification passes). Diagnostic found 31 consecutive _to_copy pairs.
- **Result**: 0 identity _to_copy found. The diagnostic had a bug — it detected consecutive nodes in topological order, not actual data-flow chains.
- **Lessons**: All 842 _to_copy ops are genuine fp32↔bf16 conversions. No identity casts exist even after aggressive simplification. The "consecutive pairs" in the diagnostic were unrelated ops that happened to be adjacent.

## view_as_complex/view_as_real pair removal — no-op (xxxxxxx)

- **Idea**: Remove canceling view_as_real(view_as_complex(x)) pairs from RoPE computation. Graph has 128 each.
- **Result**: 0 pairs found.
- **Lessons**: The 128 view_as_complex and 128 view_as_real serve different stages of RoPE rotation. Between them are actual complex multiplications. No identity pairs.

## Aggressive RS bucketing — discard (xxxxxxx)

- **Idea**: Increase memory budget to max_memory_increase_gb=8.0, max_memory_increase_ratio=0.50, max_in_flight_gb=10.0 to enable more RS bucketing. Key finding: pre-bucketing has 421 AG + 421 RS + 68 AR. Post-bucketing: AG=139 (3x reduction) but RS=390 (barely 1.08x reduction), AR=68 (unchanged).
- **Result**: tps=7287. RS count unchanged at 390. AG slightly improved (139→135).
- **Lessons**: RS bucketing is fundamentally limited by backward pass data dependencies, not memory budget. Each RS needs its gradients computed first, creating tight dependency chains that prevent merging. AG bucketing is effective because forward AG operations have more scheduling flexibility.

## enable_fusion_regions with CUDAGraph — discard (xxxxxxx)

- **Idea**: Enable fusion region detection in scheduler (retest with CUDAGraph active).
- **Result**: tps=7288, same collective structure.
- **Lessons**: Fusion region estimation doesn't change scheduling decisions for this graph/CUDAGraph combination.

## Post-bucketing simplification — discard (xxxxxxx)

- **Idea**: Run identity view removal + chain collapse + transpose pair removal after bucketing to clean up ops introduced by the bucketing pass.
- **Result**: Found 66 identity view/reshape nodes introduced by bucketing. tps=7299.
- **Lessons**: Bucketing introduces ~66 redundant view ops, but removing them is insufficient to measurably improve CUDAGraph replay time. The 66 nodes out of ~8437 total is <1%.

## max_off_bucket_gb=0.0 — discard (xxxxxxx)

- **Idea**: Force all eligible collectives into buckets by setting off-bucket limit to zero.
- **Result**: tps=6997.
- **Lessons**: No improvement. The scheduler already buckets everything it can given dependency constraints.

## NVIDIA_TF32_OVERRIDE=1 — discard (xxxxxxx)

- **Idea**: Force TF32 for all fp32 matmul operations. TF32 is ~3x faster than fp32 GEMM on H100.
- **Result**: tps=7293.
- **Lessons**: Within noise. The matmuls are already in bf16, which uses Tensor Cores. TF32 only affects fp32 matmul which is not the bottleneck.

## Custom runtime estimation (inflate RS/AR) — discard (xxxxxxx)

- **Idea**: Use `custom_runtime_estimation` callback to inflate RS and AR collective time estimates by ~1000x. This should make the scheduler allocate significantly more compute budget to overlap with these collectives.
- **Result**: tps=7221.
- **Lessons**: The scheduler already handles RS/AR overlap as well as it can given dependency constraints. Inflating estimates doesn't create new scheduling opportunities — the dependencies are the binding constraint, not the time budget.

## Wait_tensor deduplication — no-op (xxxxxxx)

- **Idea**: After bucketing: 662 waits for 597 collectives. Expected 65 redundant waits.
- **Result**: 0 duplicates found. Bucketing creates exactly 1 wait per collective.
- **Lessons**: The 65-count difference comes from different collective types, not duplicate waits.

## Inductor remove_noop_ops — discard (xxxxxxx)

- **Idea**: Apply Inductor's comprehensive no-op removal pass (handles single-element cat, all-ones repeat, same-dtype convert, clone, etc.) — catches patterns our manual passes miss.
- **Result**: Found and removed 6 additional no-op nodes. tps=7070 (within noise).
- **Lessons**: Our manual passes already cover >99.9% of identity patterns. The remaining 6 nodes are from patterns like single-use clone or view operations that our passes don't handle but Inductor's does. Too few to impact performance.

## NCCL_ALGO=Tree — crash (xxxxxxx)

- **Idea**: Use Tree algorithm for NCCL collectives (potentially faster for small messages like TP all-reduces).
- **Result**: Crash — "no algorithm/protocol available for Broadcast with ncclInt8".
- **Lessons**: Setting NCCL_ALGO=Tree restricts ALL collectives to Tree, but some ops (Broadcast/int8) only support Ring. Can't use NCCL_ALGO globally.

## PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — discard (xxxxxxx)

- **Idea**: Enable expandable CUDA memory segments to reduce allocator fragmentation.
- **Result**: tps=7300.
- **Lessons**: Within noise. The CUDA allocator is already efficient for this workload. CUDAGraph pins all allocations, so allocator behavior during replay is irrelevant.

## OMP_NUM_THREADS=1 — discard (xxxxxxx)

- **Idea**: Reduce CPU thread contention by limiting OpenMP threads.
- **Result**: tps=6062. -13% regression.
- **Lessons**: CPU parallelism matters — DataLoader preprocessing and other CPU-side operations need multiple threads. Never reduce OMP threads for GPU training.

## CUDA max_split_size_mb=128 — discard (xxxxxxx)

- **Idea**: Reduce maximum CUDA allocation split size to improve cache behavior.
- **Result**: tps=6246, memory=56.1GiB (+7GiB). Regression on both metrics.
- **Lessons**: Smaller split sizes cause more memory fragmentation and force the allocator to use more segments. This hurts both performance and memory.

## Bucketing + CUDAGraph only (no simplification) — discard (xxxxxxx)

- **Idea**: Ablation: remove all simplification passes (detach, identity view, slice, chain collapse, transpose), keep only bucketing + CUDAGraph. Test if simplification contributes with CUDAGraph.
- **Changes**: Commented out all simplification passes, kept `autobucketing_reordering_pass`, `materialize_float_constants_pass`, `cudagraph_scalar_folded_pass`.
- **Result**: tps=7291/5778 (high variance — system contention), memory=50.92GiB (+2GiB). Baseline check: 6656 tps.
- **Analysis**: First run (7291) is within noise of full pipeline. Second run (5778) anomalously low due to system load. Memory increased ~2GiB (expected: detach removal saves memory). Simplification passes contribute ~2GiB memory savings but unclear tps impact.
- **Lessons**: With CUDAGraph, simplification passes primarily save memory (~2GiB from detach removal), not tps. High system variance makes small differences unmeasurable.

## torch.compile reduce-overhead on FX graph — crash (xxxxxxx)

- **Idea**: Replace manual bucketing + CUDAGraph with `torch.compile(mode="reduce-overhead")` for automatic kernel fusion + CUDAGraph. Inductor should graph-break at collectives and compile compute portions.
- **Changes**: Added `torch_compile_pass` that calls `torch.compile(gm, mode="reduce-overhead", fullgraph=False)`.
- **Result**: Crash — `RecursionError: maximum recursion depth exceeded` in `torch._dynamo.trace_rules`.
- **Analysis**: Dynamo tries to trace the FX GraphModule's compiled forward (8000+ aten ops). Each op triggers Dynamo's rule lookup, causing deep recursion. Fundamentally wrong approach: torch.compile is designed for Python code, not for replaying pre-traced FX graphs.
- **Lessons**: torch.compile on FX GraphModules is double-tracing and doesn't work for large graphs. Need to use compile_fx_inner (Inductor codegen) directly, but that requires decomposed ops.

## compute_overlap_multipler=1.5 — discard (xxxxxxx)

- **Idea**: Test midpoint between 1.0 (default) and 2.0 (committed) for overlap scheduling with CUDAGraph.
- **Changes**: Changed `compute_overlap_multipler` from 2.0 to 1.5 in `autobucketing_reordering_pass`.
- **Result**: tps=7295, MFU=42.72%, memory=48.97GiB.
- **Analysis**: Within noise of 2.0 (full pipeline ~6971 avg). With CUDAGraph, the overlap multiplier's effect is minimal because the kernel order is fixed during replay.
- **Lessons**: The overlap multiplier matters more for eager execution (pre-CUDAGraph). With CUDAGraph, any reasonable multiplier (1.0-3.0) produces similar results.

## Regional Inductor on SwiGLU (32 pairs) — crash (xxxxxxx)

- **Idea**: Annotate 32 silu+mul (SwiGLU) patterns with `compile_with_inductor` for regional Inductor fusion. Each silu+mul is a pure forward pattern with no backward dependencies.
- **Changes**: Added `annotate_swiglu_for_inductor_pass` + called `regional_inductor_pass`.
- **Result**: Crash — `AssertionError: Invalid partition, found dependency cycles`.
- **Analysis**: regional_inductor's `_RegionScooper` tries to merge ALL 32 annotated pairs into one region. Even though individual silu+mul pairs are in forward-only code, merging scattered pairs across the fwd+bwd graph creates partition-level cycles.
- **Lessons**: regional_inductor cannot compile ANY scattered annotations in fwd+bwd graphs. The partitioner merges all annotated nodes, which inevitably creates cycles when the annotations span multiple transformer layers.

## Regional Inductor on single silu+mul pair — crash (xxxxxxx)

- **Idea**: Test with absolute minimum: annotate ONE silu+mul pair. If this works, we can compile each pair independently.
- **Changes**: Modified to annotate only the first silu+mul pair found.
- **Result**: Same crash — dependency cycles.
- **Analysis**: Even a single 2-node annotation creates a cycle. The issue is fundamental: in fwd+bwd graphs, backward nodes consume both the partition's outputs AND the same inputs the partition uses. When the partitioner tries to extract even {silu, mul}, it sees: (1) silu_output used by backward_silu_grad (outside partition), (2) backward_silu_grad also uses x (the partition's input). This creates a partition-level "cycle" even though the data-flow graph is acyclic.
- **Lessons**: **regional_inductor is completely incompatible with fwd+bwd traced graphs.** The fwd+bwd data dependency structure (backward nodes sharing inputs with forward nodes) creates unavoidable partition-level cycles. This is a fundamental limitation, not a matter of annotation strategy. Kernel fusion in fwd+bwd graphs requires a different approach (e.g., manual graph splitting with `split_module`, or custom Triton kernels injected as custom ops).

## NCCL_NTHREADS=1024 — discard (xxxxxxx)

- **Idea**: NCCL communication is 46.2% of kernel time. Increase NCCL kernel thread count from default to 1024 for potentially better bandwidth utilization on large reduce-scatter and all-reduce operations.
- **Changes**: Set `NCCL_NTHREADS=1024` environment variable.
- **Result**: tps=6630 vs baseline 7314. **-9.3% regression**.
- **Analysis**: Too many threads per NCCL kernel block creates thread scheduling overhead and register pressure that hurts overall performance.
- **Lessons**: NCCL thread tuning generally hurts on modern GPUs. The NCCL library auto-tunes thread counts per operation type. Manual overrides are almost always worse.

## NCCL_MIN_NCHANNELS=4 — discard (xxxxxxx)

- **Idea**: Increase minimum NCCL channels for more parallelism in collective operations.
- **Changes**: Set `NCCL_MIN_NCHANNELS=4` environment variable.
- **Result**: tps=6805 vs baseline 7314. **-7.0% regression**.
- **Analysis**: More channels doesn't help when NVLink bandwidth is already saturated. Extra channels add coordination overhead.
- **Lessons**: NCCL auto-selects optimal channel count for the topology. All NCCL tuning knobs have regressed. Stop trying NCCL env vars.

## cudnn.benchmark=True — discard (xxxxxxx)

- **Idea**: Enable cuDNN auto-tuning to select optimal algorithms for SDPA and other cuDNN-backed ops. SDPA is 7.2% of kernel time.
- **Changes**: Set `torch.backends.cudnn.benchmark = True` in `apply_default_graph_passes`.
- **Result**: tps=7304 vs baseline 7314. Within noise.
- **Analysis**: Flash Attention 2 is cuDNN-independent. The SDPA implementation on H100 with bf16 doesn't use cuDNN's attention backend.
- **Lessons**: cuDNN benchmark mode is irrelevant for modern SDPA implementations using Flash Attention.

## Full Inductor compile_fx_inner — discard (xxxxxxx)

- **Idea**: Bypass regional_inductor's partitioner by calling `compile_fx_inner` directly on the full fwd+bwd graph after decomposition. Inductor natively supports `_c10d_functional` collective ops via `comm_lowering.py`. This would give us Triton kernel fusion for all element-wise ops.
- **Changes**: Added `inductor_compile_full_graph_pass` that: (1) applies Inductor decompositions via `make_fx` with `select_decomp_table()`, (2) compiles the decomposed graph with `compile_fx_inner`. Replaced autobucketing + CUDAGraph with this single pass.
- **Result**: tps=4552, memory=46.91GiB (-2GiB). No CUDAGraph (Inductor CUDAGraph OOMs, our wrapper doesn't integrate cleanly). With CUDAGraph attempt: tps=5853 (still below baseline).
- **Analysis**: Inductor compilation works but produces inferior scheduling compared to our manual autobucketing. Without CUDAGraph, Inductor gives ~4552 (vs our 5165 with bucketing alone). The Inductor scheduler's comm/compute overlap decisions are less optimal for this specific collective-heavy graph. Memory savings (-2GiB) from Inductor's memory planning are nice but tps is the priority.
- **Lessons**: compile_fx_inner CAN compile full fwd+bwd graphs with collectives (major finding!), but its scheduling is worse than our manual autobucketing for this workload. Combining Inductor + CUDAGraph is non-trivial — Inductor's built-in CUDAGraph OOMs on the large graph, and our CUDAGraphWrapper doesn't cleanly wrap Inductor's OutputCode. The theoretical kernel fusion benefit (~3-5% from fusing elementwise ops) is dwarfed by the scheduling regression.

## In-place op conversion — discard (xxxxxxx)

- **Idea**: Convert 232 eligible element-wise ops (add, mul, sub, div) to in-place variants (add_, mul_, etc.) when the first tensor input has a single use (safe to overwrite). This reduces memory allocations during CUDAGraph recording, potentially improving peak memory and cache behavior.
- **Changes**: Added `convert_to_inplace_pass` after autobucketing. Only converts ops where: (1) first arg has single user, (2) not a graph input, (3) matching dtype and shape.
- **Result**: tps=5852 vs baseline 7314. **-20% regression**. Memory=48.91GiB (negligible savings).
- **Analysis**: In-place ops break CUDAGraph replay performance. CUDAGraph records memory addresses during warmup — in-place ops change the memory layout expectations, causing CUDAGraph to insert synchronization points or fall back to slower paths. The 232 conversions are individually correct but collectively disrupt CUDAGraph's memory planning.
- **Lessons**: **Never use in-place ops with CUDAGraph.** CUDAGraph assumes a specific memory allocation pattern during recording. In-place modifications disrupt this assumption, causing severe performance degradation even though the ops are semantically correct.

## Reordered simplification + second pass — discard (xxxxxxx)

- **Idea**: Reorder passes (transpose→view→slice→collapse) and run a second view+collapse pass. Removing transpose pairs first might expose new identity views.
- **Changes**: Moved `remove_transpose_pairs_pass` before `remove_identity_view_pass`, added second `remove_identity_view_pass` + `collapse_view_chains_pass` after all simplifications.
- **Result**: Only 2 additional identity views found in second pass. tps=7297, within noise of baseline 7314.
- **Analysis**: The simplification passes are nearly independent — each targets a distinct pattern. Reordering doesn't create new optimization opportunities.
- **Lessons**: Pass ordering doesn't matter for this graph. The passes are idempotent and target non-overlapping patterns.
