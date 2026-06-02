# ATE-Bench (reproduction for TorchTitan)

A local reproduction of **ATE-Bench**, the *Agent-Task Efficiency* benchmark from
**PithTrain: A Compact and Agent-Native MoE Training System**
([arXiv:2605.31463](https://arxiv.org/abs/2605.31463),
[github.com/mlc-ai/pith-train](https://github.com/mlc-ai/pith-train)).

ATE-Bench is **not released as runnable code** by the authors — it is specified in
the paper (Appendix B: task descriptions + correctness checks; Appendix C:
per-attempt values). This folder reconstructs it so we can run it against
TorchTitan.

## The idea (paper §4)
Standard coding benchmarks (SWE-bench, MLE-bench, ...) **hold the codebase fixed
and vary the agent** to score agent capability. ATE-Bench **inverts this**: it
holds the *agent* fixed and varies the *framework*, so differences in agent cost
isolate **framework design**. Lower agent cost on the same tasks = a more
agent-friendly framework.

The paper's fixed agent was **Claude Code (Opus 4.7, `xhigh` effort)**; each task
was run **3 times** and **medians** reported across **PithTrain, Megatron-LM, and
TorchTitan** (TorchTitan at commit `d84e83d`).

## Effort metrics (lower is better)
Five metrics, reported independently (no single scalar):
| Metric | Source | Q&A? |
|---|---|---|
| Agent turns | `result.num_turns` | ✅ (Table 5) |
| Per-turn context | mean of `input + cache_read + cache_creation` per assistant turn | ✅ (Table 5) |
| Output tokens | sum of assistant `output_tokens` | ✅ (Table 5) |
| Session duration | `result.duration_ms` | extra |
| Active GPU time | training-run GPU time | ❌ N/A for Q&A (no training) |

## Task suite (20 tasks, paper §4 / Appendix B)
| Category | Count | Status here |
|---|---|---|
| **Q&A** — answer a code-property question, read-only, cite `file:line` | 12 | ✅ runnable (`tasks/qa/`) |
| **Operate & Profile** — run / train+eval / routing trace / Nsight | 4 | 📋 specified, deferred (`tasks/operate_profile/`) |
| **New Feature** — port Diff-attn / DynMoE / MoBA / MoE++ end-to-end | 4 | 📋 specified, deferred (`tasks/new_feature/`) |

Q&A is CPU-only and needs no checkpoints/data, so it's the first runnable slice.
The GPU categories need 8 GPUs, an MoE checkpoint, DCLM data, vLLM +
lm-evaluation-harness, and Nsight Systems — their full specs and correctness
checks are captured in the two category READMEs, ready to wire up later.

## Layout
```
agent_tooling/ate_bench/
  README.md                     <- this file
  tasks/
    qa/
      universal_instruction.txt <- prepended verbatim to every Q&A query
      q01_process_groups.md ... q12_checkpoint_serialization.md
    operate_profile/README.md   <- 4 tasks specified (deferred)
    new_feature/README.md       <- 4 tasks specified (deferred)
  runner/
    run_qa.py                   <- drives the fixed agent per task, N runs
    metrics.py                  <- stream-json transcript -> 5 metrics
    aggregate.py                <- medians -> Table-5-style report
    grade_qa.md                 <- human citation-grading protocol
  results/<label>/              <- transcripts + metrics (created on run)
```

## Run the Q&A suite
```bash
# one task, one run (smoke test)
python agent_tooling/ate_bench/runner/run_qa.py \
    --repo /home/bahuang/local/torchtitan_autodev \
    --tasks q01 --runs 1 \
    --out agent_tooling/ate_bench/results/titan

# full suite, 3 runs (paper protocol)
python agent_tooling/ate_bench/runner/run_qa.py \
    --repo /home/bahuang/local/torchtitan_autodev \
    --tasks all --runs 3 \
    --out agent_tooling/ate_bench/results/titan

# summarize (medians, Table-5 style)
python agent_tooling/ate_bench/runner/aggregate.py \
    agent_tooling/ate_bench/results/titan

# cross-framework comparison (two+ results dirs side by side)
python agent_tooling/ate_bench/runner/aggregate.py \
    agent_tooling/ate_bench/results/titan \
    agent_tooling/ate_bench/results/pithtrain
```

The agent is invoked headless and read-only:
```
claude -p --output-format stream-json --verbose \
  --allowedTools Read Grep Glob Bash \
  --disallowedTools Edit Write NotebookEdit
```
with the target `--repo` as the working directory.

## Reproducing the paper's comparison
To compare TorchTitan vs PithTrain (or Megatron-LM) fairly, run the **same**
`run_qa.py` with `--repo` pointed at each framework's checkout (each on the commit
you want), then `aggregate.py dir1 dir2`. Correctness is graded per
`runner/grade_qa.md` before comparing effort.

## Important caveats
- **Agent ≠ paper's.** The paper used **Opus 4.7 at `xhigh` effort**. Pass
  `--model` to pin a model; there is no verified CLI flag for reasoning effort, so
  absolute numbers will not match the paper. Keep the agent **identical across
  frameworks** in any one comparison — that's what makes the comparison valid.
- **`Bash` is pre-approved.** Faithful to the paper (the agent gets Read/Grep/Glob/
  Bash), but it means the headless agent can run arbitrary read-only shell
  commands without prompting. `Edit`/`Write`/`NotebookEdit` are disabled, so it
  cannot modify the working tree — but run only against repos you trust.
- **Correctness gates the metrics.** Effort numbers are only meaningful for
  *correct* attempts. Grade Q&A answers (`runner/grade_qa.md`) before reporting.
