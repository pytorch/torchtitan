# Autoresearch Harness

An autonomous experiment loop for graph-pass / kernel optimization in
TorchTitan's `graph_trainer`. An LLM agent iteratively edits a single
file (`torchtitan/experiments/graph_trainer/passes.py`), benchmarks each
change, and keeps it only if training-step time improves **and** numerics
stay bitwise-identical to the eager reference. It runs indefinitely until
manually stopped.

The harness lives at `torchtitan/experiments/graph_trainer/autoresearch/`.
File paths in this README are relative to that directory; commands are run
from the repo root.

## Files

| File | Owner | Purpose |
|---|---|---|
| `autoresearch.md` | experimenter + agent | The agent's operating manual. Fill in the `[SETUP]` sections before starting a run. |
| `ideas.md` | experimenter + agent | Optimization ideas. Seed with curated directions, or leave empty for pure-autonomy. |
| `experiment_log.md` | agent | Append-only journal: idea / changes / result / analysis / lessons per experiment. |
| `learnings.md` | agent | Distilled, cross-experiment learnings. |
| `results.tsv` | agent | Machine-readable performance tracking (one row per experiment). |
| `scripts/run_benchmark.sh` | experimenter | Benchmark command — edit the model/config/parallelism for your run. |

## Starting a run

1. Edit `scripts/run_benchmark.sh` for your model, config, and
   parallelism.
2. Fill in the `[SETUP]` sections of `autoresearch.md`: the target,
   starting graph, scaffolding level (curated ideas / reference access /
   web), the reading-scope restrictions, and the numerics-check command.
3. Optionally seed `ideas.md` with curated optimization directions.
4. Record the baseline as the first row of `results.tsv` and the first
   entry of `experiment_log.md`.
5. Point the agent at `autoresearch.md` and let the loop run.

## Design properties

- **Asynchronous**: the shared files are the only communication channel
  between human and agent; neither needs to be online at the same time.
- **Auditable**: every experiment is logged with full context, and the
  commit tree preserves all code states including failures.
- **Safe**: numerics correctness is a hard gate. Failed experiments
  revert `passes.py` only; the best-so-far state always remains HEAD.
- **Compounding**: each `keep` builds on all previous keeps.
