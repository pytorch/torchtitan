# torchtitan_autoresearch

An agentic compute-efficiency optimizer for TorchTitan pretraining. The design is
in **`ARCHITECTURE.md`** (source of truth); the human's binding rules are in
**`Constitution.md`** and advisory guidance in **`Ideas.md`**. This README is the
code map.

The system has three actors: **Human** (authors `Constitution.md` + `Ideas.md`),
**Harness** (enforces the constitution, runs/measures/judges, defines the Agent
API), **Agent** (a pluggable search policy that only sees the system through the
Harness). Objective: **climb throughput, floor quality** — maximize tokens/sec
subject to an absolute one-sided quality floor vs a frozen golden.

## Module map

Harness core (pure orchestration — CPU-testable):
- `constitution.py` — load `Constitution.md`'s JSON rules block into `Rules`.
- `ideas.py` — parse `Ideas.md` advisory items.
- `types.py` — `Candidate`, `Verdict`, `Quality`, `Observation`, `Report`.
- `ledger.py` — authoritative append-only `results.tsv` (facts + verbatim agent words).
- `state.py` — golden, champion, noise models, family budget; JSON-persisted.
- `workload_guard.py` — admissibility: locked invariants + editable scope (step 1).
- `measure.py` — canonical steady-state throughput (pinned window; ungameable).
- `significance.py` — tail-aware promotion test.
- `crash_classify.py` — failure taxonomy + idea-family time-boxing.
- `quality.py` — verify-routes-quality: faithful ⇒ no eval; else eval ≥ golden−ε (step 3).
- `gate.py` — the 4-step pipeline: admit → run+measure → verify-routes-quality → decide.
- `executor.py` — the GPU-execution boundary (Protocol + `FakeExecutor` + `SubprocessExecutor`).
- `api.py` — `Harness`: `observe()`, `submit()`, `pull_report()`/`read_report()`,
  `amend_constitution()`, `post_idea()`.
- `agent.py` — `Agent` protocol (`propose` + `report`) + `ScriptedAgent`, `PlaybookAgent`.
- `loop.py` — the unattended single-box loop.

GPU entrypoints (torchrun; not runnable on CPU):
- `verify_main.py` + `compare.py` + `verify_config.py` + `gradcheck_probe.py` —
  the faithfulness/verify check.
- `eval_main.py` — the held-out eval (the quality measurement). Prints `EVAL: eval_loss=...`.
- `run_verify.sh`, `run_train.sh` (repo root) — launchers.

## The 4-step pipeline (`gate.py`)

```
1. admit       reject locked-invariant edits / deferred families (no run)
2. run+measure throughput; early-abort + classify crash + update family budget
3. quality     verify routes: faithful ⇒ preserved (no eval);
               not faithful ⇒ held-out eval ≥ golden − ε (one-sided)
4. decide      promote iff throughput beats champion (tail-aware significance)
               AND quality holds; else restore. Append one ledger row.
```

The agent never declares a change's class — verify routes by measured faithfulness.

## Running it

CPU smoke test of the whole loop (no GPUs), with a fake executor:

```python
from torchtitan_autoresearch.api import Harness
from torchtitan_autoresearch.executor import FakeExecutor
from torchtitan_autoresearch.agent import ScriptedAgent
from torchtitan_autoresearch.loop import run_loop
# build Harness(constitution_path, ideas_path, ledger_path, statefile, FakeExecutor(specs))
# run_loop(harness, ScriptedAgent([...]))
```

Real run (GPUs): construct `SubprocessExecutor(repo_root, log_freq, ngpu)` instead
of `FakeExecutor`, seed the golden + noise models into the state at setup
(calibration), and drive a real Agent. The executor shells to `run_train.sh`
(throughput, parsed by `measure`), `run_verify.sh` (faithfulness), and
`eval_main.py` (quality).

## Status

**Built and CPU-validated:** the entire harness core and the full
observe→propose→submit→verdict→ledger→report loop, including: seq_len/scope
admission, throughput significance (tail-aware), verify-routes-quality with a
one-sided eval floor (a faster-but-worse candidate is correctly rejected),
crash classification, and idea-family time-boxing.

**Pending (GPU, not runnable here):**
- `SubprocessExecutor` end-to-end against real TorchTitan runs;
- `verify_main.py`/`compare.py` realignment to emit golden-anchored
  **faithful/affecting** (the few-batch bias test + compile precision override
  from ARCHITECTURE.md s.5.2); they currently emit `pass/FAIL` (accepted as a
  synonym by the executor);
- `eval_main.py` wired to the live validator's eval-loss accessor;
- golden + noise-model calibration at setup.

**Superseded / to delete after the constitution loader is wired in:** the
repo-root `program.md`, `ideas.md`, `learnings.md` (their roles re-home to the
constitution, the advisory `Ideas.md`, the ledger, and agent-private memory — see
ARCHITECTURE.md "Supersession").
