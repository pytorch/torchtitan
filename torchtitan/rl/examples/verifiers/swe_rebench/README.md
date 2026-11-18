# SWE-rebench V2.1 with Verifiers and TitanRL

This example trains a coding agent with TitanRL on the Python GitHub issues of
[SWE-rebench V2.1](https://hub.harborframework.com/datasets/ibragim-badertdinov/swe-rebench-v2.1-rl).
Each rollout is one attempt at one issue: Verifiers' `bash` harness edits the
repository in the issue's Docker image, every model call goes to TitanRL's
generators, and the issue's held-out tests grade the result. Verifiers runs the
agent and the grading through the [Verifiers bridge](../README.md).

> **Toy sandbox.** Local Docker keeps this example self-contained; it is not a sandbox
> for real training. Containers run untrusted task code on the controller host and,
> by default, share its network. For real training, plug in a mature sandbox as
> the Verifiers runtime. Do not use `SubprocessConfig`: it runs the agent on the host.

## How it works

The tasks are [prime-envs' SWE-rebench V2 task](https://github.com/PrimeIntellect-ai/prime-envs/tree/5c9530ccef3d1be3d88b9be2c63d17a80d2ca13b/environments/swe/swerebench_v2),
filled from V2.1's Harbor task directories ([`taskset.py`](./taskset.py)).

- **Setup.** Deletes `/swebench_matterhorn`, where some images ship the reference
  solution, and records the repository's `HEAD` as the base commit. V2.1 has no base
  commit field, and the agent can move `HEAD` later.
- **Agent.** Verifiers' `bash` harness (bash and edit tools) runs in the task container.
  The prompt is the issue text plus prime-envs' rule against looking up the upstream fix.
- **Grading.** After the agent stops, the task restores the files the test patch touches
  to the base commit, applies the held-out test patch and runs the test command in the
  container. The env server parses the log with prime-envs' copy of SWE-rebench's
  log parsers, outside the container.
- **Reward.** `f2p_regression` (default) is the FAIL_TO_PASS pass fraction times
  `0.5 + 0.5 x` the PASS_TO_PASS pass fraction. `resolved` is V2.1's binary criterion:
  every FAIL_TO_PASS and PASS_TO_PASS test passes. Both are logged as metrics, with
  `f2p_fraction`, `p2p_fraction` and `answer_key_present`.

| What | Where |
| --- | --- |
| Recipes: model, parallelism, generator, training loop, and the Verifiers setup (taskset, harness, Docker runtime, timeouts) | `torchtitan_recipes/rl/verifiers_swe_rebench.py` |
| V2.1 task directories to prime-envs tasks, setup, grading and the train/validation split | [`taskset.py`](./taskset.py) |
| Bridge: task dataset, env server, generation server, trace to turns | [Verifiers integration](../README.md) |
| CPU tests | `tests/rl/unit_tests/cpu/test_verifiers_swe_rebench.py` |

## Install

Install the TorchTitan RL dependencies and this directory's `requirements.txt` in one
Python 3.12 environment. prime-envs' taskset is not on PyPI and declares a newer
Verifiers than this example pins, so install it without its dependencies:

```bash
pip install -r torchtitan/rl/examples/verifiers/swe_rebench/requirements.txt
GIT_LFS_SKIP_SMUDGE=1 pip install --no-deps \
  -r torchtitan/rl/examples/verifiers/swe_rebench/requirements-no-deps.txt
```

`GIT_LFS_SKIP_SMUDGE=1` skips the large files of other prime-envs environments. The
env server runs on the controller host, so that host needs Docker and pulls the task
images from Docker Hub.

## Run

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_swe_rebench \
  --config rl_grpo_qwen35_9b_swe_rebench
```

| Config | Model | GPUs |
| --- | --- | --- |
| `rl_grpo_qwen35_9b_swe_rebench` | Qwen3.5-9B, 64K context | 4 trainer, 4 generators x 1 |
| `rl_grpo_qwen38_27b_swe_rebench` | Qwen3.8-27B, 256K context | 8 trainer (FSDP=4, CP=2), 16 generators x 1 |

The recipes read the checkpoint from `./assets/hf/Qwen3.5-9B` and
`./assets/hf/Qwen3.8-27B`. `torchtitan.rl.train` places every process on the local host,
so the 27B recipe needs a multi-host launcher. Neither recipe has been run end to end.

## Datasets

The taskset loads the V2.1 Harbor dataset pinned in `taskset.py`. Verifiers downloads it
into `~/.cache/harbor` with the `harbor` CLI on first use; set the taskset's
`dataset_dir` to read a local copy instead.

- **Split.** V2.1 has no split. `partition="train"` and `"validation"` split the
  repositories by `validation_repository_fraction`, so no repository is in both.
- **Task lists.** To train on chosen tasks, set `partition="all"` and `task_list` to a
  file of instance ids, one per line. Every listed id must exist.
- **Quality flags.** Tasks V2.1 flags `weak_f2p` (FAIL_TO_PASS barely checks the fix)
  are dropped by default (`exclude_quality_flags`).
- **Resources.** The taskset asks for 4 CPUs, 7 GB memory and 10 GB disk per sandbox
  (Docker applies the CPU and memory limits), less than the 6 CPUs, 24 GB and 32 GB in
  every V2.1 `task.toml`, so more sandboxes fit on a host.

## Differences from V2.1's own grading

V2.1's `tests/test.sh` runs the test command under a login shell (`bash -lc`) and grades
with each task's `tests/grade.py`. Here prime-envs runs the command under a fixed `PATH`
in a non-login shell and parses the log with its parsers, so a task can score
differently. Check a task with Verifiers' gold-patch validation before trusting it.

## If every reward is 0

- **Check the gold patch first.** `SWERebenchV21Task.validate` applies the gold patch and
  scores it; it should return `True` for the tasks you train on.
- **The run fails after 10 untrainable batches.** A group whose samples all get the same
  reward is dropped (`TrainingSampleBuilder.drop_zero_std_reward_groups`), and the batcher
  raises "consecutive untrainable batches" after 10 in a row.
