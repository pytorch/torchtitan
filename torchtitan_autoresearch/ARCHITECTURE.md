# Hillclimbing Architecture

An agentic compute-efficiency optimizer built on TorchTitan for model
pretraining. The system has **three actors** — **Human**, **Harness**, **Agent**
— each internally pluggable (any actor can be implemented many ways) but
interacting only through fixed contracts. This document defines the actors, their
contracts, and the formats of the two files the Human authors: `constitution.md`
(binding) and `ideas.md` (advisory).

---

## 1. Goal and key assumption

The starting recipe already produces a **good model** — the modeling research is
done. We are not improving the model; we are making it **cheaper to train without
degrading it**. So this is an **efficiency search under a quality-preservation
constraint**, and we are **degradation-sensitive**: shipping a quietly-worse
recipe is far worse than missing a speedup.

## 2. Objective: climb throughput, floor quality

```
maximize   throughput            (tokens/sec — the axis we hill-climb)
subject to quality >= golden - epsilon   (an absolute, one-sided floor; epsilon tight)
```

- **Throughput — relative, climbed.** Each candidate must beat the current best.
- **Quality — absolute, floored.** Each candidate is checked against the frozen
  good model. Degradation up to a tight `epsilon` is tolerated; **improvement is
  always free**; falling below the floor disqualifies a candidate no matter how
  fast it is.

There is **no cumulative quality bookkeeping**: because quality is checked
absolutely against the fixed golden every time, the floor is self-enforcing.

## 3. The three actors

```
            constitution.md (binding rules)        amend_constitution()
  HUMAN  ───────────────────────────────────▶  HARNESS  ◀──────────────  HUMAN
    │        ideas.md (advisory guidance)          │  ▲
    └──────────────────────────────────────────▶  │  │ submit(candidate)
                                                   │  │
                              observe() ───────────┘  │
                              {rules, ledger, champion,│
                               golden, traces, ideas}  │
                                     │                 │
                                     ▼                 │
                                   AGENT ──────────────┘
```

- **Human — legislator + creative steerer.** Authors `constitution.md` (the
  binding rules the harness enforces) and `ideas.md` (advisory guidance that
  steers the agent's search). Defines what "good" means and the risk tolerance.
  Steers, but never judges.
- **Harness — judge + executor.** The code that runs commands, measures
  throughput, validates quality, **enforces the constitution**, maintains the
  reference state and the ledger, and **defines the Agent API**. Non-bypassable.
- **Agent — searcher.** A pluggable policy that sees the system *only* through the
  Harness (including `ideas.md`), and produces candidate recipes. Cannot
  influence measurement or verdict.

Each actor is internally pluggable: the Human may be one person, a team, or an
automated meta-controller; the Harness may be implemented many ways but must
honor the API and enforce the constitution; the Agent may be a single LLM,
manager/worker, an ensemble, a classical optimizer, or a hybrid.

The organizing rule: **the agent proposes, the harness disposes, the human
legislates.** The load-bearing safety property is that the agent can influence
*what to try*, but never *the measurement, the verdict, or the definition of
good*. Everything gameable lives in the Harness (mechanism) or with the Human
(policy), never with the Agent.

---

## 4. The Human

The Human acts through exactly two files. The distinction between them is
load-bearing: one is binding and can affect a verdict; the other is advisory and
never can.

### 4.1 `constitution.md` — binding rules (enforced by the Harness)

The machine-read source of truth for the rules of this experiment. The Harness
reads it, enforces it, and exposes it to the Agent as `observe().rules`. The
Human amends it via `amend_constitution()`. If the prose in this architecture
doc and the constitution ever disagree, **the constitution wins**.

Required sections (this is the format; a populated instance lives in
`constitution.md`):

```
# Objective
  - what is maximized (throughput) and the quality floor form.

# Workload (LOCKED)
  - model flavor, dataset, seq_len, world size / hardware, launcher.

# Quality (LOCKED)
  - eval: how model quality is measured (held-out set, reference seq, metric).
  - golden: the frozen high-precision reference recipe (the quality bar).
  - epsilon: max one-sided degradation vs golden (relative eval loss).
  - quality-affecting classes: change kinds that require the eval.

# Editable scope
  - files/fields the Agent MAY change.
  - fixed fields the Agent may NOT change.

# Measurement protocol (LOCKED)
  - throughput: pinned metrics.log_freq, step caps, steady-state window.

# Significance policy
  - promotion test (tail-aware), z / band, noise calibration procedure.

# Substrate policy
  - family-budget thresholds, run timeouts.
```

### 4.2 `ideas.md` — advisory guidance (consumed by the Agent)

The Human's channel to steer the Agent's creativity: hints, priors, hypotheses,
soft focus, cautions, and references. It is **advisory** — the Agent may use,
reprioritize, or ignore any item, and nothing in it can change rules,
measurement, or verdicts. The Agent reads it (via the Harness) as
`observe().ideas` and acknowledges items by referencing their `id` in
`candidate.addresses`; outcomes are then visible by joining the ledger to those
ids. The Human owns this file; the Agent never writes it.

Format — a list of items, each:

```
- id:      short stable slug
  kind:    hint | prior | soft_constraint | reference
  target:  (optional) lever or bottleneck it pertains to
  weight:  (optional) emphasis, default 1.0
  text:    the guidance, in plain language
```

`kind` semantics: **hint** = a concrete thing to try; **prior** = a belief about
where value/bottlenecks are; **soft_constraint** = deprioritize/avoid (not
binding — binding bans go in the constitution); **reference** = a fact, doc, or
hardware property to ground reasoning.

---

## 5. The Harness

The Harness enforces the constitution and is the sole judge. It implements the
mechanics below and exposes the Agent API.

### 5.1 Reference anchors

- **Golden** — the good starting model, frozen forever. The **quality bar**
  (absolute) and the **numerical-faithfulness anchor**. Everything is measured
  against it, so nothing drifts.
- **Champion** — current best recipe. Two narrow jobs only: the **throughput
  target to beat**, and the **verify increment audit**. Never the quality
  reference.

### 5.2 Quality is one quantity, estimated by a cost/fidelity ladder

| Change kind | Quality check | Cost |
|---|---|---|
| **quality-neutral** (compile, fusion, FSDP overlap, parallelism, faithful kernels, graph-invariant flags) | **verify** — numerical faithfulness **vs golden** (two-sided; any deviation beyond rounding noise means not neutral). Faithful implies quality `>= floor`, no eval. | cheap, every candidate |
| **quality-affecting** (precision/FP8/MXFP8, batch, LR, weight decay, optimizer) | **held-out eval** at the reference seq, **absolute and one-sided** (reject only if eval quality `< golden - epsilon`). | expensive, rare |

- Faithfulness is two-sided; quality is one-sided (improvement is welcome).
- Verify is anchored to the **golden**, so numerical drift cannot accumulate.
- **Compile is verify-audited**: a few-batch *bias test* separates unbiased
  fusion noise from a systematic precision bias; precision-touching sub-knobs
  (TF32, max-autotune, reduced-precision reductions) are forced to the eval.

### 5.3 The per-candidate pipeline

```
1. Admissible?  (no locked-invariant edit; idea's family not deferred) -> else reject, no run.
2. Run once, measuring steady-state throughput; early-abort on crash/NaN.
   (A crash is classified substrate-vs-logic and updates the family budget.)
3. Quality - verify routes:
     verify faithfulness vs GOLDEN (cheap: few-batch + compile bias test)
       faithful     -> quality preserved (no eval)
       not faithful -> quality-affecting -> held-out eval >= golden - epsilon
                        (the run is long enough to train; eval its result).
4. Promote iff throughput beats champion (significance, tail-aware) AND quality
   holds. Else restore source. Append ledger row.
```

The Agent never declares a change's class — **verify routes by measured
faithfulness** (`faithful <=> quality-preserved`).

### 5.4 Measurement, significance, conservatism

- **Throughput** is harness-owned (`metrics.log_freq=1` pinned, fixed window,
  mean/cv) — ungameable. Noise is heavy-tailed, so promotion uses a **tail-aware**
  test (a single sample must clear the tail, not merely 2-sigma).
- **Quality** is the absolute eval vs golden; `epsilon` must clear the eval-noise
  floor, which sets the eval-run length. Both noise models are calibrated by
  repeating the reference.
- **Conservatism**: tight `epsilon`; ties go to reject; prefer provably-safe
  quality-neutral wins.

### 5.5 State and ledger (facts vs narrative)

The Harness **authors facts**: the ledger (`results.tsv`) — authoritative,
append-only measured outcomes — plus golden, champion, the two noise models, and
the family budget. It also emits raw signals (traces). It **never authors
interpretation**.

It does **store agent-authored narrative verbatim**, without interpreting it:
- the per-candidate `label`/`rationale` rides in the ledger (the micro trail);
- the latest `report()` snapshot is persisted and served to the Human (the macro
  synthesis).

The agent's *private memory* (its `learnings`, however represented) is **not**
harness state and is never read directly by the Human — only its `report()`
projection is. Because that projection cites the ledger and traces, and those are
harness-owned and durable, agent narrative is always **reconstructable** from the
facts: losing it (a restart, an agent swap) costs recomputation, not information.
This is what lets different agent architectures run against the same history.

### 5.6 The Agent API (Harness-defined)

Harness -> Agent (read-only):

```
observe() -> Observation:
    rules             # constitution: locked invariants, editable scope, objective, epsilon, sig policy
    ledger            # every prior candidate + outcome (authoritative search history)
    champion          # current best recipe + metrics
    golden            # the quality bar / reference model
    deferred_families # idea families time-boxed out by repeated substrate failures
    ideas             # the Human's advisory guidance items (from ideas.md)

get_traces(commit) -> Traces   # profiler + structured metrics for a run (lazy)
```

Agent -> Harness (the only write):

```
submit(candidate) -> Verdict
    candidate: { commit, label, rationale="", addresses=[] }   # addresses = ideas ids acted on
    Verdict:   { admitted, throughput_mean, throughput_cv,
                 quality{checked, passed, margin}, verdict, status, crash_class }
```

Agent must also expose (Harness-pulled, for Human evaluation):

```
report() -> Report:
    beliefs        # current hypotheses about bottlenecks, each with confidence
    conclusions    # what it has learned, EACH citing ledger commits / trace ids
    plan           # what it intends to try next and why
    open_questions # what it's uncertain about / wants to disambiguate
    ideas_usage    # which ideas.md ids it used / found useful / discarded, and why
```

This is the **public projection of the agent's private memory** (its `learnings`).
The Human never reads raw agent memory; it reads `report()` via the Harness.
Every `conclusion` must **cite** the ledger rows / trace ids / ideas ids it rests
on, so the Human can verify learnings against facts rather than trust prose. The
Harness pulls `report()` (on Human request and/or periodically), **persists the
latest snapshot verbatim** (attributed as agent-authored), and serves it to the
Human — so a snapshot survives the agent stopping or being swapped, and the
boundary holds (Human <-> Harness only). `report()` is mandatory but may be
trivial for a memoryless agent (e.g. a grid sweep returns empty beliefs and
"see ledger"). Because its shape is contracted and grounded, it also makes
*reasoning quality* comparable across agent architectures, not just outcomes.

Human -> Harness:

```
amend_constitution(...)   # binding: reconfigures enforced rules
post_idea(item)           # advisory: appends to ideas.md
read_report()             # serves the latest persisted agent report() snapshot
```

---

## 6. The Agent

A pluggable **search policy** mapping observations to candidates. Because the
oracle is objective and reproducible, agent architectures can be A/B'd on the
same problem and ranked by **GPU-hours to reach X% of the achievable
improvement** — the Harness doubles as a benchmark for agentic search.

- **Sees the system only through the Harness**, including the Human's `ideas.md`
  (as `observe().ideas`). No side channels.
- **Produces only committed candidate recipes** (within the editable scope) plus
  a label; cannot touch measurement, verdict, ledger, champion, golden, or rules.
- **Internal state is private** — memory, narrative, hypotheses, idea queues are
  not in the API; different agents represent them differently (files, a DB,
  nothing). But the agent must **expose** that state through the contracted
  `report()` projection (section 5.6) so the Human can evaluate its learnings
  without inspecting raw private memory.

The pluggable contract is therefore two methods: **`propose`** (turn an
observation into a candidate) and **`report`** (project current learnings,
grounded in ledger/trace citations). Everything else about the agent is free.

Topology axes (all behind the same contract): decomposition (single /
manager-worker / ensemble / hierarchical); idea-generation strategy (profiler-
roofline-driven, prior/playbook-driven, LLM-creative, classical optimizer,
retrieval, human-injected); space routing (code changes to an LLM, numeric knobs
to Bayesian opt / bandits); exploration policy; concurrency. A strong default is
a **portfolio of generators + a bandit meta-policy** that allocates the next slot
to whichever generator has the best recent promotion-rate-per-GPU-hour.

---

## 7. Productivity (reach the optimum in fewer GPU-hours)

Front-loaded EV/cost lever order; roofline-first profiling; batched factorial
sweeps for graph-invariant flags; warm-start Inductor cache; diminishing-returns
pivot; single-box unattended loop. These matter more now, because each
quality-affecting eval is expensive — keep them few. (These are Agent strategy +
Harness mechanics; they are not constitution.)

## 8. One-liner

> Lock what "good" means (the eval) and the good model (the golden) in the
> constitution. The Harness hill-climbs throughput over the recipe an Agent
> proposes, promoting a change only if it stays above an absolute one-sided
> quality floor vs the golden — proving quality cheaply with golden-anchored
> verify for the faithful majority, and paying for a real held-out eval only on
> the few changes that can move quality. The Human steers with advisory
> `ideas.md`; nothing the Agent does can touch the judge.

---

## Appendix: build status

**Built and CPU-validated:** `verify_main`, `compare`, `verify_config`,
`gradcheck_probe`, `trajectory_diff`, `measure`, `significance`,
`crash_classify`, `workload_guard`, `gate` (pipeline mock-tested), `run_verify.sh`.

**Designed, pending build (the C4 + quality reframe):** held-out **eval** quality
estimator (+ its noise model); `gate.py` realigned to the 4-step pipeline
(verify-routes-quality, one run on the affecting path, early-abort, no
cheap-screen / no cumulative-debt); few-batch **bias test** + compile precision
override in verify; `batch/lr/wd/optimizer` as quality-affecting (batch+LR
coupled); compile + `--debug.deterministic` fallback; the constitution loader and
the `observe()`/`submit()` Agent API; consume `ideas.md`.

**Supersession:** this API makes `program.md` (-> constitution + this doc +
agent-private playbook), the old `ideas.md` "Manager" queue (-> agent-private),
and `learnings.md` (-> agent-private memory, exposed to the Human only through
the contracted `report()` projection) obsolete. The Human-facing `ideas.md` is
retained, redefined as the advisory guidance channel above.
