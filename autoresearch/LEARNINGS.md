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

## Benchmark Notes

- Use `--dataloader.dataset c4_test` for local data (avoids network dependency).
- First step includes compilation overhead (~92 tps vs ~4400 tps steady-state). Always look at step 20.
- Confirm improvements with a second run. Noise is ~1-3%.
