# ATE-Bench Q&A results — TorchTitan

> **Status: COMPLETE.** 12 tasks × 3 runs = **36/36 runs, 0 errors.** Medians below.

## Setup
- **Framework under test:** TorchTitan @ `f854797fc` (this branch).
- **Fixed agent:** Claude Code headless, model `claude-opus-4-6` (env default), 1M
  context window. The paper used **Opus 4.7 @ `xhigh` effort** — so absolute
  numbers differ; treat trends/ratios as the comparable part.
- **Tools:** read-only `Read/Grep/Glob/Bash` (`Edit/Write/NotebookEdit` disabled),
  matching the paper's Q&A setup.
- **Protocol:** median of 3 attempts (lower = more efficient). Effort metrics from
  `result.modelUsage` (authoritative billed usage).
- **Reproduce:** `run_qa.py --repo $(pwd) --tasks all --runs 3 --out <dir>` then
  `aggregate.py <dir>`.

## Ours vs. paper (paper Table 5, TorchTitan column)

| Task | Turns ours | Turns paper | Per-Turn Ctx ours | Per-Turn Ctx paper | Output ours | Output paper |
|---|---|---|---|---|---|---|
| Q1  | 11 | 18 | 73.1K  | 44.3K | 9.7K  | 7.1K |
| Q2  | 32 | 28 | 33.7K  | 43.5K | 7.2K  | 7.8K |
| Q3  | 9  | 14 | 73.2K  | 36.1K | 7.0K  | 3.5K |
| Q4  | 18 | 26 | 186.4K | 39.7K | 12.3K | 6.9K |
| Q5  | 7  | 19 | 90.8K  | 41.7K | 7.4K  | 4.8K |
| Q6  | 5  | 4  | 28.9K  | 29.9K | 1.7K  | 2.3K |
| Q7  | 10 | 12 | 27.5K  | 30.0K | 2.5K  | 3.4K |
| Q8  | 15 | 11 | 33.7K  | 32.7K | 3.1K  | 3.5K |
| Q9  | 9  | 16 | 294.8K | 38.6K | 14.1K | 6.0K |
| Q10 | 20 | 17 | 37.1K  | 38.8K | 4.8K  | 5.1K |
| Q11 | 9  | 8  | 24.9K  | 30.1K | 2.7K  | 2.5K |
| Q12 | 8  | 5  | 30.9K  | 33.9K | 2.0K  | 2.2K |
| **Σ/median** | **153 total** | **178 total** | median 35.4K | median 37.5K | **70.5K total** | **55.1K total** |

## What the numbers say
- **Agent turns track the paper well** — total 153 vs 178 (our agent is ~14% leaner
  on turns), per-task within a handful on most questions. The TorchTitan codebase
  imposes a similar number of agent steps on both agents.
- **Output tokens are comparable** (within ~1.3×), slightly higher for us.
- **Per-turn context is the big divergence**, driven by the *agent*, not the
  framework: on exploration-heavy questions our medians blow up — **Q9 (CP) 294.8K
  vs 38.6K, Q4 (seeds) 186.4K vs 39.7K, Q5 (attention) 90.8K vs 41.7K**. The env's
  `opus-4-6` with a 1M window reads broadly and keeps a large working context;
  `opus-4-7 @ xhigh` in the paper ran much tighter. On cheap, localized questions
  (Q6 RoPE, Q7 SwiGLU, Q11/Q12) the two agents match closely.
- **Takeaway:** the *turn-count* signal (how many agent steps the framework demands)
  reproduces the paper's TorchTitan column; the *context* metric is dominated by the
  agent model difference and isn't comparable across agents — exactly why ATE-Bench
  insists on holding the agent fixed across frameworks.

## Caveats
- **Correctness not yet graded.** Effort numbers only count for *correct* attempts
  (paper had all 108 Q&A attempts satisfied). Grade per `runner/grade_qa.md` (or an
  LLM-judge pass) before drawing firm conclusions; the `correct` column in
  `aggregate.py` is `-` until graded.
- **Agent ≠ paper's** (`opus-4-6` vs `opus-4-7 @ xhigh`) → absolute numbers differ.

## Scope (unchanged)
- ✅ **TorchTitan Q&A column** — this file (reproduced).
- ❌ **PithTrain / Megatron-LM columns** — no checkouts → the cross-framework
  headline (62% fewer turns / 64% less GPU time) is out of scope here.
- ⏳ **Operate & Profile / New-Feature** — mesh validated on 8×H100
  (`RESULTS_gpu_titan.md`); numeric repro blocked on a real MoE checkpoint + deps.
