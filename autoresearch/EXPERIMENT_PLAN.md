# Autoresearch Experiment Plan

A comparative study of LLM-driven autonomous kernel fusion under different
scaffolding conditions. Each run is a long-running autoresearch loop on a
separate branch; the differences across runs answer specific questions about
what the agent needs to be effective.

## Goal

Showcase what an LLM agent can autonomously discover when given the
custom graph passes (added to `construct_default_graph_passes` in `passes.py`)
on a Llama3 8B training graph, and quantify how much human scaffolding
(curated ideas, reference implementations, starting-graph optimizations)
changes the outcome.

## Study design: 4-run progressive layering

Each subsequent run adds **exactly one capability** over the previous one.
Consecutive pairs form clean single-axis A/Bs; the cumulative shape across
all 4 runs tells the headline story.

| # | IDEAS | References | Graph passes | Added vs prior | Tests question |
|---|---|---|---|---|---|
| 1 | empty | only FX graph + `autoresearch.md` in-scope files | none | (baseline) | Can the LLM find useful fusions purely from graph inspection? |
| 2 | empty | + in-repo reference implementations | none | + reference access | Does reading what humans built help the agent find better fusions? |
| 3 | curated (orthogonal) | + in-repo reference implementations | none | + curated IDEAS | Do human-curated ideas *outside what the repo implements* add headroom on top of reference access? |
| 4 | curated (orthogonal) | + in-repo reference implementations | production set | + production starting graph | Does autoresearch still add value on top of the already-optimized graph? |

"No graph passes" means `construct_default_graph_passes` returns the empty
list and the agent's custom passes are the only ones running. "Production
set" means the full set of in-repo numerics-preserving passes runs first
and the agent layers on top.

## Branches

Run 1 lives on `ar_base`. Runs 2–4 each fork from `ar_base` so they inherit
the shared scaffolding (scripts, plan, EXPERIMENT_LOG format, results.tsv
header, baseline numerics reference) but iterate independently. Each has
its own `autoresearch.md` customized with the run-specific constraints.

| # | Branch |
|---|---|
| 1 | `ar_base` |
| 2 | `ar_refs` |
| 3 | `ar_refs_ideas` |
| 4 | `ar_refs_ideas_opt` |

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

### Run 2 — + Reference Access

- **IDEAS.md**: empty (same as Run 1).
- **References**: in-repo reference implementations are now **allowed**
  as design inspiration. External sources still forbidden.
- **Numerics guardrail**: same as Run 1.

### Run 3 — + Curated IDEAS (orthogonal to refs)

- **IDEAS.md**: curated by the experimenter. Items are chosen to be
  orthogonal to what the in-repo references already implement, so the
  Run 2 → Run 3 comparison has a real delta. The specific curated items
  are held outside this file (and outside the autoresearch agent's
  reading scope) to avoid leaking them across runs.
- **References**: same as Run 2.
- **Numerics guardrail**: same as Run 1/2.

### Run 4 — + Production Starting Graph (Ceiling)

- **IDEAS.md**: curated, same as Run 3.
- **References**: same as Run 2/3.
- **Graph passes**: production pass set runs before the agent's passes.
  The agent sees an already-optimized graph and must find further wins on
  top.
- **Numerics guardrail**: bitwise vs eager.

## Execution

Sequential on a single 8×H100 node. Each run:

1. Initial setup (branch + scaffolding) — minutes.
2. Baseline benchmark + bitwise reproducibility check — ~20 min.
3. Autoresearch loop — runs indefinitely until manually stopped
   (`autoresearch.md` says NEVER STOP). Expect ~10–15 min per experiment.
   Target ~24h+ per run for a robust signal.

## Showcase artifact

`autoresearch/scripts/summarize_progress.py` produces
`autoresearch/progress.png` — a side-by-side TPS / MFU chart from
`results.tsv`. For the final showcase we render the 4 runs' cumulative-best
curves on one figure (overlay or 2×2 grid) plus a summary table:

| Run | Best TPS | Best MFU | Best memory_gib | # keep | # discard | # crash |
|---|---|---|---|---|---|---|

The headline narrative is the *shape* of the progression — which capability
addition shifted the curve the most.

## Status

- [x] Run 1 set up — branch `ar_base`, infrastructure files, baseline
  recorded (tps=4,161, mfu=24.36%, memory=49.0 GiB).
- [ ] Run 1 autoresearch loop — pending kickoff.
- [ ] Run 2 setup + loop.
- [ ] Run 3 IDEAS curated, set up + loop.
- [ ] Run 4 setup + loop.
- [ ] Final showcase synthesis (combined chart + summary).
