# ATE-Bench Q&A results — TorchTitan

> **Status: IN PROGRESS.** A background 12×3 sweep is running; **1/36 runs**
> complete as of this commit. This file will be regenerated with the full medians
> (and `aggregate.py` output) once the sweep finishes.

## Setup
- **Framework under test:** TorchTitan @ `f854797fc` (this branch).
- **Fixed agent:** Claude Code headless, model `claude-opus-4-6` (the env default).
  The paper used **Opus 4.7 @ `xhigh` effort** — so absolute numbers will differ;
  trends/ratios are the comparable part. Keep the agent identical across frameworks
  for any real comparison.
- **Tools:** read-only `Read/Grep/Glob/Bash` (`Edit/Write/NotebookEdit` disabled),
  matching the paper's Q&A setup.
- **Protocol:** 12 Q&A tasks × 3 runs, **median** reported (lower = more efficient).
  Effort metrics taken from `result.modelUsage` (the authoritative billed usage).

## Ours vs. paper (paper Table 5, TorchTitan column)

| Task | Turns (paper) | Turns (ours) | Per-Turn Ctx (paper) | Per-Turn Ctx (ours) | Output Tok (paper) | Output Tok (ours) |
|---|---|---|---|---|---|---|
| Q1  | 18 | 16\* | 44.3K | 62.9K\* | 7.1K | 9.7K\* |
| Q2  | 28 | _running_ | 43.5K | _running_ | 7.8K | _running_ |
| Q3  | 14 | _running_ | 36.1K | _running_ | 3.5K | _running_ |
| Q4  | 26 | _running_ | 39.7K | _running_ | 6.9K | _running_ |
| Q5  | 19 | _running_ | 41.7K | _running_ | 4.8K | _running_ |
| Q6  | 4  | _running_ | 29.9K | _running_ | 2.3K | _running_ |
| Q7  | 12 | _running_ | 30.0K | _running_ | 3.4K | _running_ |
| Q8  | 11 | _running_ | 32.7K | _running_ | 3.5K | _running_ |
| Q9  | 16 | _running_ | 38.6K | _running_ | 6.0K | _running_ |
| Q10 | 17 | _running_ | 38.8K | _running_ | 5.1K | _running_ |
| Q11 | 8  | _running_ | 30.1K | _running_ | 2.5K | _running_ |
| Q12 | 5  | _running_ | 33.9K | _running_ | 2.2K | _running_ |

\* 1 of 3 runs so far (not yet a median). An earlier single smoke run of Q1 gave
19 turns / 39.5K / 8.9K — note the run-to-run spread (16–19 turns, 39.5–62.9K
context), which is exactly why the paper medians over 3 attempts.

**Early read:** our Q1 lands in the paper's TorchTitan ballpark on turns
(16 vs 18) and output tokens (9.7K vs 7.1K); per-turn context runs higher
(`opus-4-6` here vs `opus-4-7 xhigh` in the paper, different context budgeting).

## What this does and does not reproduce
- ✅ **TorchTitan Q&A column** (this file) — the CPU-only, fully runnable slice.
- ❌ **PithTrain / Megatron-LM columns** — not reproducible here (no checkouts), so
  the paper's *relative* headline (62% fewer turns / 64% less GPU time) cannot be
  reproduced — those are cross-framework deltas.
- ⏳ **Operate & Profile / New-Feature** — GPU-gated (8×H100 are available, but they
  need a real MoE checkpoint, vLLM+lm-eval, Nsight, and C4); harness is ready.

## Reproduce
```bash
python agent_tooling/ate_bench/runner/run_qa.py \
    --repo $(pwd) --tasks all --runs 3 \
    --out agent_tooling/ate_bench/results/titan_full
python agent_tooling/ate_bench/runner/aggregate.py \
    agent_tooling/ate_bench/results/titan_full
```
Correctness must be human-graded (`runner/grade_qa.md`) before drawing conclusions
— effort numbers only count for *correct* attempts.
