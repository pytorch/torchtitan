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

## What Doesn't Work

- **DCE before bucketing**: Removing dead nodes disrupts bucketing heuristics, reducing performance.
- **DCE after bucketing**: Removes ~167 nodes but no measurable impact on tps.
- **Regional Inductor (no annotations)**: No-op without explicit `compile_with_inductor` annotations on graph nodes. aot_fx_trace mode doesn't annotate.
- **Full Inductor**: Crashes without inductor decomposition pass. Would need significant plumbing.

## Benchmark Notes

- Use `--dataloader.dataset c4_test` for local data (avoids network dependency).
- First step includes compilation overhead (~92 tps vs ~4400 tps steady-state). Always look at step 20.
- Confirm improvements with a second run. Noise is ~1-3%.
