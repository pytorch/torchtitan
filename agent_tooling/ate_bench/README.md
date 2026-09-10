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
| Active GPU time | sampled via `gpu_monitor.py` (nvidia-smi) | ❌ N/A for Q&A; ✅ for operate/feature |

## Task suite (20 tasks, paper §4 / Appendix B)
| Category | Count | Status here |
|---|---|---|
| **Q&A** — answer a code-property question, read-only, cite `file:line` | 12 | ✅ runnable (`run_qa.py`) |
| **Operate & Profile** — run / train+eval / routing trace / Nsight | 4 | ✅ harness ready, GPU-gated (`run_operate.py`) |
| **New Feature** — port Diff-attn / DynMoE / MoBA / MoE++ end-to-end | 4 | ✅ harness ready, GPU-gated (`run_feature.py`) |

Q&A is CPU-only (no checkpoints/data) and runs today. The operate/feature harness
is complete — prompts, runners, programmatic checks, GPU-time monitor, and the
LLM judge — but actually *passing* the tasks needs the GPU substrate (see
**Prerequisites** below). Without it the agent still runs and effort metrics are
logged; the artifact/loss checks just won't pass.

### TorchTitan mapping (fixed config, paper Appendix B)
`PP=4, EP=2, DP=1`, seq_len 2048, global batch 1024, BF16, 8 GPUs → TorchTitan's
`run_train.sh` with `--parallelism.{pipeline,expert}_parallel_degree` etc.

> **Mesh-notation gotcha (a framework-design difference ATE-Bench is about).** The
> paper's mesh is multiplicative (`PP*EP*DP = 8`). TorchTitan's mesh requires
> `dp_replicate*dp_shard*cp*tp*pp == world_size` with **EP carved from the data
> axis**, not multiplied on top. So we set `pp=4`, `dp_shard=-1` (auto → 2 on 8
> GPUs), `ep=2` (experts sharded across that data axis of 2). Setting `dp_shard=1`
> fails validation (`1*1*1*1*4 != 8`).

Model:
an MoE model (`deepseek_v3` default; also `qwen3` `debugmodel_moe`, `llama4`,
`gpt_oss`). Dataset: **C4** (TorchTitan native; the paper used DCLM). Checkpoint
export/import: `scripts/checkpoint_conversion/convert_{to,from}_hf.py`. All encoded
in `runner/repro_config.py` + `setup/train.sh`.

## Layout
```
agent_tooling/ate_bench/
  README.md
  tasks/
    qa/            universal_instruction.txt + q01..q12          (read-only Q&A)
    operate_profile/  preamble.md + op1..op4 + README            (run/instrument/profile)
    new_feature/      preamble.md + nf1..nf4 + references.md + rules/  (integrate arch)
  setup/
    train.sh       fixed-config TorchTitan launcher (the "provided training script")
    evaluate.sh    OP2 export-to-HF + lm-eval HellaSwag via vLLM
  runner/
    run_qa.py        read-only Q&A runner
    run_operate.py   operate/profile runner (full tools, GPU-time, artifact checks)
    run_feature.py   new-feature runner (worktree per attempt, loss check + judge)
    metrics.py       stream-json transcript -> effort metrics
    aggregate.py     medians -> Table-style report (+ GPU time, correctness)
    repro_config.py  fixed config + prompt placeholder rendering
    agent_session.py shared headless-agent runner + GPU monitor
    gpu_monitor.py   active GPU time via nvidia-smi sampling
    judge.py         independent LLM judge for new-feature rules
    worktrees.py     git worktree isolation for feature attempts
    grade_qa.md      human citation-grading protocol (Q&A)
    checks/          op1..op4 + nf_loss_curve programmatic checks
  workspace/<label>/  task artifacts (routing-traces/, heavy-kernels/, ...)  [gitignored]
  results/<label>/    transcripts + metrics                                   [gitignored]
  worktrees/          transient per-attempt worktrees                         [gitignored]
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

## Run the Operate & Profile suite (GPU)
```bash
python agent_tooling/ate_bench/runner/run_operate.py \
    --tasks all --runs 3 --label titan \
    --module deepseek_v3 --config deepseek_v3_debugmodel \
    --out agent_tooling/ate_bench/results/operate_titan
python agent_tooling/ate_bench/runner/aggregate.py agent_tooling/ate_bench/results/operate_titan
```
Full toolset (edits auto-accepted), runs in the main checkout, GPU time measured,
and after each attempt the task's programmatic artifact check runs (OP1 finite
loss at step 5, OP2 finite HellaSwag, OP3 routing-trace npz validation, OP4
top-kernels CSV schema). OP3 instruments code in the working tree — `git checkout`
afterward to reset.

## Run the New-Feature suite (GPU)
```bash
python agent_tooling/ate_bench/runner/run_feature.py \
    --tasks all --runs 3 --label titan \
    --judge-model claude-opus-4-7 \
    --out agent_tooling/ate_bench/results/feature_titan
python agent_tooling/ate_bench/runner/aggregate.py agent_tooling/ate_bench/results/feature_titan
```
Each attempt runs in an isolated git **worktree** off `main`. Correctness has two
axes (paper B.3): the 64-step CE loss must decrease + stay finite
(`checks/nf_loss_curve.py`), **and** an independent LLM **judge** must PASS the diff
against the 3 fixed per-feature rules (`runner/judge.py`, `tasks/new_feature/rules/`).
Both must hold.

## Prerequisites for the GPU tasks
- **8 GPUs** for the fixed mesh (PP=4 × EP=2). The `*_debugmodel` configs are tiny
  (harness validation); paper-scale numbers need a full MoE config + checkpoint.
- A **released MoE checkpoint** (OP3/OP4 resume from it; download via
  `scripts/checkpoint_conversion/` + `scripts/download_hf_assets.py`).
- **vLLM + lm-evaluation-harness** for OP2; **Nsight Systems** (`nsys`) for OP4.
- **C4** dataset access for training (TorchTitan's default loader).

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
- **Tool access differs by category.** Q&A is read-only (`Edit`/`Write`/
  `NotebookEdit` disabled). Operate and new-feature get the **full toolset with
  edits auto-accepted** (`--permission-mode acceptEdits`) since they must run and
  modify the framework — so the headless agent runs arbitrary shell/edits without
  prompting. New-feature attempts are isolated in throwaway git worktrees;
  operate runs in the main checkout (OP3 leaves instrumentation edits — reset with
  `git checkout`). Run only against repos you trust.
- **Correctness gates the metrics.** Effort numbers are only meaningful for
  *correct* attempts. Q&A is human-graded (`runner/grade_qa.md`); operate/feature
  correctness is computed automatically (artifact checks; loss curve + LLM judge).
- **`active_gpu_time_s` needs nvidia-smi.** Without a GPU it reports `null`
  (Q&A always does, by design).
