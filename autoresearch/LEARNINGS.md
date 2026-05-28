# Autoresearch Learnings

Living document of what works, what doesn't, and how to approach kernel
fusion optimization effectively. The agent owns this file.

Re-read at the start of each loop iteration alongside `IDEAS.md` and
`EXPERIMENT_LOG.md`. Update after meaningful experiments (especially
surprising results, both positive and negative). Keep concise and
actionable — per-experiment details belong in `EXPERIMENT_LOG.md`.

## Methodology

- Production starting graph baseline = 7,162 tps. Treat ±1% (≈±70 tps) as
  noise. A single benchmark within ±1% should be re-run before keeping or
  discarding if the change is otherwise promising.
- Each experiment edits only `passes.py`. Revert with
  `git checkout -- torchtitan/experiments/graph_trainer/passes.py` before
  starting a new experiment so the baseline graph is consistent.
- Numerics gate FIRST, perf SECOND: a perf win that fails
  `aot_fx_trace_vs_eager` is `crash`, not `keep`.

## Patterns that worked

(none yet)

## Patterns that didn't work

- Adding upstream Inductor `joint_graph_passes` at the joint-graph stage
  (after `normalize_view_ops_as_reshape`, before bucketing): bitwise-safe
  but no perf win on this graph. The cleanup it performs (CSE, noop
  removal, constant folding) finds nothing material because existing
  no-op passes have already cleaned the graph and Llama3 8B compute is
  dominated by large matmul/RMSNorm/SDPA/collective kernels with little
  algebraic redundancy.

## Tooling tips

- `bash autoresearch/scripts/run_benchmark.sh` produces `run.log`.
- `grep "step:" run.log | tail -1` → steady-state tps/mfu/memory line.
- `grep "benchmark_wall_time_s" run.log` → total wall time.
- Numerics check (must pass): `pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -k "TestLlama3BitwiseDeterministic and aot_fx_trace_vs_eager" -x > numerics.log 2>&1`.
  Failure → `crash` regardless of perf.
- Inspect graph: add a one-line `gm.print_readable(...)` inside a temp pass,
  run once, then remove before benchmarking.
