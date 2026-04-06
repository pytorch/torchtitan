# Learnings

High-level guide on what works, what doesn't, and how to approach graph optimization effectively. Updated after meaningful experiments.

## Methodology

1. **Always understand before optimizing**: Dump the graph, study the ops, understand the bottleneck before trying a fix.
2. **Measure carefully**: TPS can vary ~1-3% between runs. Re-run to confirm before declaring victory.
3. **Verify numerics**: Every change must pass the bitwise deterministic test. No exceptions.
4. **Simpler is better**: A 1% speedup that adds 100 lines of complex pass code is not worth it.
