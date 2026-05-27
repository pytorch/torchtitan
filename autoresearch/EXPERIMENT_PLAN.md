# Autoresearch Experiment Plan

A comparative study of LLM-driven autonomous kernel fusion under different
scaffolding conditions. Each run is a long-running autoresearch loop on a
separate branch; the differences across runs answer specific questions about
what the agent needs to be effective.

## Goal

Showcase what an LLM agent can autonomously discover when given the
custom graph passes (added to `construct_default_graph_passes` and `compile_time_passes` in `passes.py`)
on a Llama3 8B training graph, and quantify how much human scaffolding
(curated ideas, reference implementations, starting-graph optimizations)
changes the outcome.

## Study design: 3-run progressive layering

Each subsequent run adds capability over the previous one.
Consecutive pairs form clean single-axis A/Bs; the cumulative shape across
all 3 runs tells the headline story.

| # | IDEAS | References | Graph passes | Added vs prior | Tests question |
|---|---|---|---|---|---|
| 1 | empty | only FX graph + `autoresearch.md` in-scope files | none | (baseline) | Can the LLM find useful fusions purely from graph inspection? |
| 2 | curated | only FX graph + `autoresearch.md` in-scope files | none | + curated IDEAS | Do human-curated ideas (without reference implementations) help the agent find better fusions? |
| 3 | curated | + in-repo reference implementations | production set | + curated IDEAS + reference access + production starting graph | Can autoresearch still add value on top of the already-optimized production graph? |

"No graph passes" means `construct_default_graph_passes` returns the empty
list and the agent's custom passes are the only ones running. "Production
set" means the full set of in-repo numerics-preserving passes runs first
and the agent layers on top.

## Branches

Run 1 lives on `ar_base`. Runs 2–3 each fork from `main` so they inherit
the shared scaffolding but iterate independently. Each has its own
`autoresearch.md` customized with the run-specific constraints.

| # | Branch |
|---|---|
| 1 | `ar_base` |
| 2 | `ar_ideas` |
| 3 | `ar_hand_opt` |

## Per-run constraints

Each branch's `autoresearch.md` is the authoritative description of that
run's constraints; this section gives a high-level summary only.

### Run 1 — Pure Autonomy Baseline (Floor)

- **IDEAS.md**: empty. Agent generates ideas from FX graph inspection alone.
- **References**: the agent may read only the files explicitly enumerated
  in `autoresearch.md`. External sources (upstream PyTorch source, torchao,
  external kernel libraries, web content, papers, blog posts) are
  forbidden, as are in-repo reference implementations of prior
  optimization work.
- **Numerics guardrail**: bitwise vs eager via
  `test_bitwise_deterministic.py::TestLlama3*` cases (see `autoresearch.md`).
- **Result**: 47 experiments, 5 keeps + 1 config change. TPS 4,481 → 6,492
  (+44.9%) under non-deterministic measurement.

### Run 2 — + Curated IDEAS

- **IDEAS.md**: curated by the experimenter. Ideas provide optimization
  directions but no implementation details.
- **References**: same as Run 1 (only FX graph + `autoresearch.md`
  in-scope files). In-repo reference implementations remain forbidden.
- **Numerics guardrail**: same as Run 1.
- **Result**: 42 experiments, 6 keeps. TPS 4,482 → 7,378 (+64.7%) under
  non-deterministic measurement. Key lever: CUDA graphs (+30.5% single step).

### Run 3 — + Production Starting Graph + Reference Access (Ceiling)

- **IDEAS.md**: curated, same as Run 2.
- **References**: in-repo reference implementations are now **allowed**
  as design inspiration. The agent may read all files in
  `torchtitan/experiments/graph_trainer/` including the production passes.
- **Graph passes**: production pass set runs before the agent's passes.
  The agent sees an already-optimized graph and must find further wins on
  top.
- **Numerics guardrail**: bitwise vs eager.
- **Production baseline**: 7,156 tps (no SAC, non-deterministic).

## Execution

Sequential on a single 8×H100 node. Each run:

1. Initial setup (branch + scaffolding) — minutes.
2. Baseline benchmark + bitwise reproducibility check — ~20 min.
3. Autoresearch loop — runs indefinitely until manually stopped
   (`autoresearch.md` says NEVER STOP). Expect ~10–15 min per experiment.
   Target ~24h+ per run for a robust signal.

## Showcase artifact

`autoresearch/scripts/plot_comparison.py` produces
`autoresearch/comparison.png` — an overlay chart of incremental TPS by
keep experiment for all runs + a production reference line.

`autoresearch/scripts/benchmark_all_keeps.sh` re-measures every keep
commit from all branches under uniform conditions (no `--debug.deterministic`,
no SAC) for apples-to-apples comparison.

## Results summary (non-deterministic, no SAC)

| Run | Branch | Best TPS | Best MFU | # keep | # experiments |
|---|---|---|---|---|---|
| 1 | `ar_base` | 6,492 | 38.0% | 5 (+1 config) | 47 |
| 2 | `ar_ideas` | 7,378 | 43.2% | 6 | 42 |
| — | Production | 7,156 | 41.9% | — | — |
| 3 | `ar_hand_opt` | TBD | TBD | TBD | TBD |

## Status

- [x] Run 1 complete — branch `ar_base`, 47 experiments, best 6,492 tps.
- [x] Run 2 complete — branch `ar_ideas`, 42 experiments, best 7,378 tps.
- [x] Uniform re-measurement of all keeps (benchmark_all_keeps.tsv).
- [x] Comparison chart (comparison.png).
- [ ] Run 3 setup + loop — branch `ar_hand_opt`.
- [ ] Final showcase synthesis.
