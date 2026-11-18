# SWE-rebench V2.1 with Verifiers and TitanRL

This example trains a coding agent with TitanRL on the
[SWE-rebench V2.1 RL dataset](https://hub.harborframework.com/datasets/ibragim-badertdinov/swe-rebench-v2.1-rl),
whose tasks are all Python GitHub issues.
Each rollout is one attempt at one issue: the model edits the repository through
Verifiers' `bash` harness, in a container of the issue's Docker image; every model
call goes to TitanRL's generators, and the issue's held-out tests grade the result. Verifiers runs the
agent and the grading through the [Verifiers bridge](../README.md).

> **Toy sandbox.** Local Docker keeps this example self-contained; it is not a sandbox
> for real training. Containers run untrusted task code on the controller host and, by
> default, can reach any address the host can, including the upstream repository.
> For real training, plug in a mature sandbox as the Verifiers runtime.

## How it works

The tasks are [prime-envs' SWE-rebench V2 task](https://github.com/PrimeIntellect-ai/prime-envs/tree/5c9530ccef3d1be3d88b9be2c63d17a80d2ca13b/environments/swe/swerebench_v2),
filled from V2.1's Harbor task directories ([`taskset.py`](./taskset.py)).

- **Setup.** Deletes `/swebench_matterhorn`, where some images ship the reference
  solution, and records the repository's `HEAD` as the base commit. V2.1 has no base
  commit field, and the agent can move `HEAD` later.
- **Agent.** Verifiers' `bash` harness (bash and edit tools) runs in the task container.
  The user prompt is the task's `instruction.md`: the issue text, often followed by
  V2.1's added contract requirements. The system prompt adds prime-envs' rule against
  looking up the upstream fix; it is only a prompt, and the default Docker network lets
  the agent reach GitHub.
- **Grading.** After the agent stops, the task restores the files the test patch touches
  to the base commit, applies the held-out test patch and runs the test command in the
  container. The env server parses the log with prime-envs' copy of SWE-rebench's
  log parsers, outside the container.
- **Reward.** `f2p_regression` (default) is
  `fail_to_pass_fraction * (0.5 + 0.5 * pass_to_pass_fraction)`, the pass rates of the
  FAIL_TO_PASS and PASS_TO_PASS tests. `resolved` is V2.1's binary criterion: every
  FAIL_TO_PASS and PASS_TO_PASS test passes. The task also records `resolved`, both
  fractions, `parsed_tests` and `answer_key_present` on each Verifiers trace. TitanRL
  does not log or record them yet: of the trace's scores, the bridge passes on only the
  reward.

| What | Where |
| --- | --- |
| Recipe: model, parallelism, generator, training loop, and the Verifiers setup (taskset, harness, Docker runtime, timeouts) | `torchtitan_recipes/rl/verifiers_swe_rebench.py` |
| V2.1 task directories to prime-envs tasks, setup, grading and the train/validation split by project | [`taskset.py`](./taskset.py) |
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
images from Docker Hub. The recipe keeps up to 192 rollouts in flight (24 groups x 8),
each in its own container. Each task has its own image, 1-3.5 GB, and Verifiers removes
the containers but not the images, so prune them during long runs
(`docker image prune -a` removes every image no container uses, not only task images).

## Run

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_swe_rebench \
  --config rl_grpo_qwen35_9b_swe_rebench
```

The recipe trains Qwen3.5-9B at 64K context on one 8-GPU host: 4 trainer GPUs (FSDP=4)
and 4 one-GPU generators. It reads the checkpoint from `./assets/hf/Qwen3.5-9B`. It has
not been run end to end yet.

## Datasets

The taskset loads the V2.1 Harbor dataset pinned in `taskset.py`. Verifiers downloads it
into `~/.cache/harbor` with the `harbor` CLI on first use; set the taskset's
`dataset_dir` to read a local copy instead.

- **Split.** V2.1 has no split. `partition="train"` and `"validation"` split the projects
  by `validation_project_fraction`, so no project is in both, even when V2.1 lists it
  under two owners after a rename.
- **Quality flags.** Tasks that V2.1 flags `weak_f2p` (their FAIL_TO_PASS tests barely
  check the fix) are skipped by default (`exclude_quality_flags`).
- **Resources.** The taskset asks for 4 CPUs, 7 GB memory and 10 GB disk per sandbox
  (`resources`), less than the 6 CPUs, 24 GB and 32 GB in every V2.1 `task.toml`. Docker
  applies the CPU and memory values as per-container caps, not reservations, and ignores
  the disk size; a runtime that reserves resources fits more sandboxes per host with them.

## Differences from V2.1's own grading

A task can score differently here than under V2.1's own `tests/test.sh`:

- **Shell.** V2.1 runs the test command in a login shell (`bash -lc`). prime-envs runs it
  with a fixed `PATH` that puts `/opt/conda/bin` before `/usr/local/bin`, so in images
  that install the project under `/usr/local`, a test that starts `python` gets conda's
  interpreter instead.
- **Parser.** V2.1 grades with each task's `tests/grade.py` (parser audit of 2026-10-29),
  which also reads pytest's verbose `name PASSED` lines and fails a test if any line
  fails it. prime-envs uses its copy of SWE-rebench V2's log parsers from 2026-04, which
  reads only lines that start with a status, cuts the name at the first space and keeps
  the last status. An expected name with a space (`test_x[a b]`) never matches here: 78
  of the 3990 tasks have one, so they can never be fully resolved.
- **Patches.** V2.1 refreshes the git index, skips archives and files that already match,
  and finally applies what it can (`--reject`). prime-envs tries `git apply --3way`, then
  `patch --fuzz=5`.
- **Timeout.** V2.1 caps the test command at 1800 s and grades the partial log. Here the
  recipe's scoring timeout (2400 s, V2.1's verifier timeout) bounds restoring the test
  files, applying the test patch and running the tests together; a timeout errors the
  rollout, which the recipe scores 0 (`error_reward`).

A gold check of 100 tasks, one per repository, on one host: 91 valid. Of the other 9,
6 also fail under V2.1's own `test.sh` there (5 need direct internet access), 2 hit the
shell difference and 1 the parser difference. Gold-check the tasks before trusting them
(next section).

## If every reward is 0

- **Check the gold patch first.** Verifiers' `vf-validate` applies each task's gold patch
  and scores it (`--only-gold`); every task you train on should come out valid. The
  taskset is a local module, so register its alias before calling the CLI:

  ```bash
  python -c '
  import sys
  from torchtitan.rl.examples.verifiers.worker_setup import register_local_taskset_alias
  from verifiers.v1.cli.validate import main
  main([register_local_taskset_alias("torchtitan.rl.examples.verifiers.swe_rebench.taskset"), *sys.argv[1:]])
  ' --only-gold --runtime.type docker --select.include.names aio-libs__aiohttp-7889
  ```
- **The run fails after 10 untrainable batches.** A group whose samples all get the same
  reward is dropped (`drop_zero_std_reward_groups` in the training sample builder), and
  the batcher raises "consecutive untrainable batches" after 10 in a row. Gold-check the tasks, and
  check that the model solves some of them, before a long run.
