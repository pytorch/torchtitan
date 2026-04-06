# Learnings

High-level guide on what works, what doesn't, and how to approach graph optimization effectively. Updated after meaningful experiments.

## Methodology

1. **Always understand before optimizing**: Dump the graph, study the ops, understand the bottleneck before trying a fix.
2. **Measure carefully**: TPS can vary ~1-3% between runs. Re-run to confirm before declaring victory.
3. **Verify numerics**: Every change must pass the bitwise deterministic test. No exceptions.
4. **Simpler is better**: A 1% speedup that adds 100 lines of complex pass code is not worth it.

## Graph Structure (Llama3 8B, FSDP4+TP2, 8xH100)

- 11,381 nodes total in fwd+bwd traced graph
- 421 all-gathers, 421 reduce-scatters (FSDP), 68 all-reduces (TP)
- 675 mm, 842 _to_copy, 32 SDPA, 65 rmsnorm, 32 silu
- Very collective-heavy — bucketing and overlap are high-value

## What Works

- **Autobucketing** (`schedule_overlap_bucketing`): +3.3% tps on the full traced graph. Simple one-liner with `collective_bucketing=True`. Good default foundation.
- **Remove detach nodes**: -2 GiB memory, neutral-to-positive on tps. Simple graph cleanup. Apply before bucketing.
- **Remove identity view/reshape ops**: +9.5% tps (4427→4849). Removed 1522 nodes where shape_in==shape_out. Unlike DCE (which removes dead nodes and disrupts bucketing), identity view removal removes noise nodes that dilute the scheduler's view of real compute/comm patterns. Apply before bucketing.
- **Remove identity slice ops**: +2.2% tps (4849→4959). Removed 453 full-dimension slices. All slices in the traced graph were identity. Cumulative identity removal (detach+view+slice = 2171 nodes = 19% of graph) makes autobucketing much more effective.
- **Collapse view chains**: +1.5% tps (4959→5018). Collapsed 323 single-use intermediate view/reshape chains. Combined with identity removal, 22% of original graph simplified.
- **Remove transpose pairs**: +1.2% tps (5018→5079). Removed 225 canceling t(t(x)) pairs (450 nodes). Common in fwd+bwd traced graphs from weight transpose patterns.
- **Aggressive overlap scheduling**: +1.7% tps (5079→5165). `compute_overlap_multipler=2.0` in autobucketing. Default is too conservative for this workload.
- **CUDAGraph with float constant folding**: **+35% tps (5165→6971).** Single highest-impact optimization. aot_fx_trace lifts 1 float scalar (32768.0) as a graph input, which CUDAGraphWrapper rejects. Inlining it as a literal + detaching tensor args fixes both blockers. Eliminates kernel launch overhead for ~8437+ node graph.

## What Doesn't Work

- **DCE before bucketing**: Removing dead nodes disrupts bucketing heuristics, reducing performance.
- **DCE after bucketing**: Removes ~167 nodes but no measurable impact on tps.
- **Regional Inductor (no annotations)**: No-op without explicit `compile_with_inductor` annotations on graph nodes. aot_fx_trace mode doesn't annotate.
- **Full Inductor**: Crashes without inductor decomposition pass. Would need significant plumbing.
- **Regional Inductor on fwd+bwd graph**: Dependency cycles are fundamental — backward regions depend on forward regions in complex, non-sequential ways. Can't partition the full fwd+bwd graph.
- **CUDAGraph on full graph (without float folding)**: Float scalar inputs from aot_fx_trace mode are not supported by CUDAGraphWrapper. Fixed by `materialize_float_constants_pass`.
- **Autobucketing memory params**: Scheduler already uses memory budget effectively — more budget doesn't help.
- **enable_fusion_regions**: Doesn't change scheduling significantly for this graph.
- **Roundtrip dtype removal**: No roundtrips exist — all 842 _to_copy ops are genuine conversions.

## Bucketing Analysis

After simplification and autobucketing:
- Pre-bucketing: AG=421, RS=421, AR=68, wait=910 (910 non-wait collectives)
- Post-bucketing: AG=139, RS=390, AR=68, wait=662 (597 non-wait collectives)
- **AG bucketing is effective**: 421→139 (3x reduction). Forward pass has scheduling flexibility.
- **RS bucketing is near-zero**: 421→390 (31 merged). Backward pass dependencies prevent merging.
- **AR is never bucketed**: 68→68. Different PG (tensor parallel), tight per-layer dependencies.
- **Memory budget is not the limiter**: Increasing max_memory_increase_gb from 1.0 to 8.0 doesn't improve RS bucketing.
- **Total CUDA ops in CUDAGraph**: ~8437+ nodes. CUDAGraph replay overhead ~12ms (<1% of step time).

## What's Left to Explore (and Why It's Hard)

- **Kernel fusion**: Inductor compilation crashes with collective ops in the graph. Regional Inductor has dependency cycles in fwd+bwd graph. The remaining memory-bound ops (842 _to_copy, normalization) can't be fused at the FX level.
- **RS bucketing**: Fundamentally limited by backward pass data dependencies. Each RS needs its layer's gradients computed first.
- **Communication volume**: FSDP already communicates in bf16. Reducing parallelism would reduce communication but requires config changes.
- **MFU ceiling**: At 42% MFU, ~58% overhead is split between communication (~5%), memory bandwidth (~10-15%), and non-matmul compute time.

## Benchmark Notes

- Use `--dataloader.dataset c4_test` for local data (avoids network dependency).
- First step includes compilation overhead (~92 tps vs ~4400 tps steady-state). Always look at step 20.
- Confirm improvements with a second run. Noise is ~1-3%.
