# Terminal-Bench with Verifiers and TitanRL

This example trains a Qwen3.5 terminal agent (9B or 35B-A3B) on AI2's TMax tasks
([allenai/tmax-15k-open-instruct](https://huggingface.co/datasets/allenai/tmax-15k-open-instruct))
and validates it on the 89 Terminal-Bench 2.1 tasks. TitanRL schedules rollout
groups, generates tokens and trains the model; Verifiers 0.3.1 runs the Terminus-2
agent and the Harbor verifier, through the Verifiers bridge in the parent directory.
Only the task adapter lives here. The recipes are in
`torchtitan_recipes/rl/verifiers_terminal_bench.py` and the tests in
`tests/rl/unit_tests/cpu/`; TitanRL's core is unchanged.

## Run

Install the TorchTitan RL dependencies and this directory's `requirements.txt` in one
Python 3.12 environment (Verifiers 0.3.1 needs MCP 1.x), and let the user run Docker
on the controller host. The agent runs **inside** the task's container, so do not use
`SubprocessConfig` for untrusted tasks.

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_terminal_bench \
  --config rl_grpo_qwen35_9b_terminal_bench \
  --hf_assets_path /path/to/Qwen3.5-9B
```

TitanRL's CLI defaults to a single host; placing the meshes across hosts needs a
caller-provided `HostMeshes` launcher.

## Datasets

Tasks come from Harbor datasets selected by id, the usual Verifiers way. Verifiers
looks for dataset `org/name@ref` in `~/.cache/harbor/org_name_ref` (`/` and `@`
replaced by `_`) and, if that directory is missing, runs `harbor download` to fill it.
Both recipes use:

| Role | Dataset id | Directory |
| --- | --- | --- |
| Train | `local/tmax@v1` | `~/.cache/harbor/local_tmax_v1` |
| Validation | `terminal-bench/terminal-bench-2-1` | `~/.cache/harbor/terminal-bench_terminal-bench-2-1` |

- Training: AI2's TMax tasks
  ([allenai/tmax-15k-open-instruct](https://huggingface.co/datasets/allenai/tmax-15k-open-instruct)),
  converted offline by `prepare_tmax.py`. `local/tmax@v1` is not on the Harbor Hub,
  so download the Hugging Face dataset and write the directory before the first run:

  ```bash
  python -m torchtitan.rl.examples.verifiers.terminal_bench.prepare_tmax \
    --parquet tmax/data --task-data tmax/task-data.tar.gz \
    --out ~/.cache/harbor/local_tmax_v1
  ```

- Validation: Terminal-Bench 2.1 (89 tasks), downloaded by the `harbor` CLI on first
  use and validated at the start and end of training. Pin `@ref` and record it with
  the checkpoint; the 2.0 and 2.1 benchmarks are not interchangeable.

To use other datasets, change the ids passed to `terminal_bench_rollouter_config` in
`torchtitan_recipes/rl/verifiers_terminal_bench.py`. They must differ, and the training task ids should be disjoint
from the benchmark's.

Each task must declare a pullable `[environment].docker_image`; a task with only a
Dockerfile is rejected rather than evaluated in the wrong environment. Not every
published image ships `tmux`; Terminus-2 installs it when it is missing.

**The task container needs outbound network at scoring time, or every reward is
silently 0.** Each Terminal-Bench 2.1 `tests/test.sh` installs its own toolchain
(`apt-get`, `uv`, then pytest from PyPI), and ends in `if ...; then echo 1; else echo
0; fi`, which exits 0 whether the tests passed, failed or were never installed.
Without egress the grader writes a `0` that looks like a real score. Docker's default
bridge network has egress; an isolated runtime or a host without it breaks scoring.

## Recipes

Both recipes share the datasets and these settings: 65,536-token context, 16,384
generated tokens per turn, 120 agent turns, 8 groups of 32 samples per step, 4 target
off-policy steps, 100 steps, constant learning rate 1e-6, fp32 master weights with
bf16 FSDP compute, full activation checkpointing and a DCP save every 20 steps.
Terminus-2 uses the XML parser without context summarization (`harness.py`). Timeouts
are 7,200 s for the agent and 12,000 s for scoring (sized for the installs above); the
tasks' own timeouts are ignored. The binary task reward goes through TitanRL's
ordinary advantage and GRPO path.

| Model | Config | Trainer | Generators | Generator CUDA graphs |
| --- | --- | --- | --- | --- |
| Qwen3.5-9B | `rl_grpo_qwen35_9b_terminal_bench` | FSDP=8 | 8 x 1 GPU | `FULL_DECODE_ONLY` |
| Qwen3.5-35B-A3B | `rl_grpo_qwen35_35b_a3b_terminal_bench` | FSDP=4, TP=2, EP=8 | 2 x 4 GPUs (DP=2, TP=2, EP=4) | off |

Each uses 16 GPUs: 8 for the trainer and 8 for generators.

- **35B-A3B layout.** The model has 2 KV heads and 256 experts, so TP is at most 2,
  trainer EP must be at least TP and divide both the experts and `dp_shard * tp`, and
  the generator's DP axis exists only to supply expert-parallel ranks (EP = DP x TP).
- **fp32 master weights** are the default. For the 35B-A3B that is about 70 GB of model
  states per trainer GPU before activations, so it needs GPUs with well over 80 GB.
  With `dtype="bfloat16"` and a 1e-6 learning rate most updates round away: in a quick
  check with weight standard deviation 0.02, about 2% of weights change per step.
- **CUDA graphs are off for the 35B-A3B generator.** The standard MoE token dispatcher
  copies all-to-all split sizes to the host, and graph capture fails on that copy.
  Re-enable capture together with a dispatcher that avoids the host read.
- **The 35B-A3B recipe has not been run end to end.** The CPU tests check that its
  layout fits the model's KV heads and experts, not that it fits in memory or that
  rollouts complete.

`TrainingSampleBuilder.drop_zero_std_reward_groups` is `True` and the reward is
binary, so a group whose 32 samples all score 0 is discarded. With a base model that
solves nothing every group is dropped, no batch forms, and the run waits in
`wait_for_training_batch` without reaching a second step. That looks like slow
rollouts; check the model's pass rate on a few tasks first.

## Known gaps

- Verifiers 0.3.1 has Docker and Prime runtimes but no Daytona.
- Verifiers does not build task Dockerfiles, so Dockerfile-only tasks need a prebuilt
  image.
- Datasets are fetched by Harbor id. A cluster without access to the Harbor Hub can
  pre-populate `~/.cache/harbor` with an exported task tree; loading from an arbitrary
  local path is not supported here.
- Harbor grading returns `0.0` when the grader never ran, which is indistinguishable
  from a task the agent failed: `HarborTask._graded` discards `test.sh`'s result and
  the reward-file readers swallow their exceptions. Treat a run of uniform zeros as
  unexplained, not as a capability measurement.
- TitanRL does not read `Trace.metrics`, so per-rollout metrics from a taskset cross the
  process boundary and are dropped without a warning.
- Verifiers' Harbor fixture staging restores host file ownership, which fails in a
  rootless container with unmapped host IDs.
- Verifiers' built-in Terminus-2 config does not expose the XML parser and summary
  policy, so `harness.py` patches them into the upstream program; exposing those knobs
  in Verifiers would remove the adapter.
- Mainline TitanRL supports GRPO, not DPPO. Validation reports greedy pass@1 only.
