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
