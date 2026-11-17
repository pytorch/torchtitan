# Terminal-Bench with Verifiers and TitanRL

This example trains a terminal agent with TitanRL on AI2's TMax tasks and validates it
on the 89 Terminal-Bench 2.1 tasks. Each rollout is one attempt at one task: Harbor's
Terminus-2 agent works in the task's Docker container, every model call goes to
TitanRL's generators, and the task's own tests grade the result. Verifiers runs the
agent and the grading through the [Verifiers bridge](../README.md).

> **Toy sandbox.** Local Docker keeps this example self-contained; it is not a sandbox
> for real training. Containers run untrusted task code on each host that runs rollout
> workers, with outbound network access by default. For real training, plug in a mature
> sandbox as the Verifiers runtime. Do not use `SubprocessConfig`: it runs the agent on the host.

## How it works

Each rollout takes the path in the Verifiers README's
[integration flow](../README.md#integration-flow). Here:

- **Agent.** The rollout worker's env starts a Docker container from the task's `docker_image` and runs
  Terminus-2 in it ([`harness.py`](./harness.py)). The agent types commands into a tmux shell and
  reads the screen back. Each agent turn is one model call.
- **Grading.** When the agent stops, Verifiers runs the task's `tests/test.sh` in the container
  and reads the reward file it writes.

Validation takes the same path on Terminal-Bench 2.1. A validation starts with training and
runs beside it (`ValidationConfig`).

| What | Where |
| --- | --- |
| Recipes: model, parallelism, generator, training loop, and the Verifiers setup (datasets, harness, Docker runtime, timeouts) | `torchtitan_recipes/rl/verifiers_terminal_bench.py` |
| Terminus-2 harness and the program it runs in the container, copied from Verifiers 0.3.1 to expose reasoning in the history and summarization | [`harness.py`](./harness.py), [`terminus_harness.py`](./terminus_harness.py) |
| Harbor taskset, declared here so each rollout worker imports it. Its tasks check `tests/` on the worker's host. | [`taskset.py`](./taskset.py) |
| Offline TMax to Harbor task conversion | [`prepare_tmax.py`](./prepare_tmax.py) |
| Bridge: task dataset, rollout worker, generation server, trace to turns | [Verifiers integration](../README.md) |
| CPU tests | `tests/rl/unit_tests/cpu/test_verifiers_terminal_bench*.py` |

## Run

Install the TorchTitan RL dependencies and this directory's `requirements.txt` in one
Python 3.12 environment, and let the job's OS user run Docker on every host that runs
rollout workers. Prepare the training tasks (see [Datasets](#datasets)), then:

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_terminal_bench \
  --config rl_grpo_qwen35_9b_terminal_bench
```

The 9B recipe sets `hf_assets_path` to `./assets/hf/Qwen3.5-9B`, and the 35B
recipe sets `./assets/hf/Qwen3.5-35B-A3B`. Change that assignment in
`torchtitan_recipes/rl/verifiers_terminal_bench.py` to use another directory.
There is no CLI flag for the path.

| Config | Model | GPUs |
| --- | --- | --- |
| `rl_grpo_qwen35_9b_terminal_bench` | Qwen3.5-9B | 8 trainer, 8 generators x 1 |
| `rl_grpo_qwen35_35b_a3b_terminal_bench` | Qwen3.5-35B-A3B | 8 trainer, 2 generators x 4 |

`torchtitan.rl.train` places every process on the local host, so either recipe needs a
16-GPU host as written. To use several hosts, see [More than one host](#more-than-one-host).
One 4-host run used remote sandboxes. No multi-host run has used local Docker yet. The 35B-A3B recipe has not been run end to end.

## Datasets

Verifiers loads a Harbor dataset `org/name@ref` from `~/.cache/harbor/org_name_ref` and
runs `harbor download` if that directory is missing.

| Role | Dataset id | Source |
| --- | --- | --- |
| Train | `local/tmax@v1` | AI2's [TMax](https://huggingface.co/datasets/allenai/tmax-15k-open-instruct), converted by `prepare_tmax.py` |
| Validation | `terminal-bench/terminal-bench-2-1` | Terminal-Bench 2.1 from the Harbor Hub |

`local/tmax@v1` is not on the Harbor Hub. Download the Hugging Face dataset and convert
it once:

```bash
python -m torchtitan.rl.examples.verifiers.terminal_bench.prepare_tmax \
  --parquet tmax/data --task-data tmax/task-data.tar.gz \
  --out ~/.cache/harbor/local_tmax_v1
```

- Each task must declare a pullable `[environment].docker_image`. Verifiers does not
  build Dockerfiles, and a single Dockerfile-only task makes the dataset fail to load.
  `prepare_tmax.py` writes each TMax task's published image.
- The tasks' own timeouts are ignored; the timeouts in the recipe file apply.
- The validation id has no `@ref`, so it uses whichever revision is in the cache. Pin a
  ref for numbers you want to compare; Terminal-Bench 2.0 and 2.1 are not
  interchangeable.
- Without Harbor Hub access, pre-populate `~/.cache/harbor` with an exported task tree.
- To use other datasets, change the ids passed to `_terminal_bench_rollouter_config` in the recipe file.
  The training and validation datasets must differ.

## More than one host

To run on several hosts, use a launcher, e.g. `SlurmLauncher` (see
[Launchers](../../../actors/monarch/README.md#launchers)). Rollout workers then run on the
generators' hosts by default. [Rollout workers on more than one host](../../../README.md#rollout-workers-on-more-than-one-host)
covers the shared folder. Each host that runs them also needs:

1. The Harbor cache at the same path as on the controller's host:
   - For example `/home/me/.cache/harbor/local_tmax_v1`, even on a host whose home directory
     is elsewhere.
   - A worker reads each task's whole directory there: `tests/` to grade it, and
     `environment/` to pick how its sandbox starts.
   - If a host mounts the shared folder at another path, a symlink from the controller's path
     to it also works.
   - Each worker checks the training dataset's dir when it starts.
   - The check skips the validation dataset: the controller downloads it into the same cache
     at the first validation.
   - With a copy on each host, first download the validation dataset into the controller's
     cache, then copy.
2. On that host, a Docker daemon that the job's OS user can run. Verifiers can't use a
   daemon on another machine. The daemon needs access to the tasks' image registry, and to
   Docker Hub for `python:3.11-alpine`, which Verifiers 0.4.1 pulls when a task image has no
   `python3`. Each host pulls each task's image on first use, so registry traffic and disk
   grow with the number of hosts.

A host without Docker or without the training dataset fails the job at start, and the error
names the host. A host with a partial copy, e.g. a task or the validation dataset missing,
fails those groups. Validation leaves its failed groups out of its reward, so check
`validation/group_failures`.

To keep every worker on the controller's host, set `rollouter.worker_roles.generator=None` and
`rollouter.worker_roles.controller=True`.

## If every reward is 0

- **A grader that never ran scores 0.** Verifiers ignores `tests/test.sh`'s exit status
  and scores a missing or unreadable reward file as 0, the same as a failed task.
  Terminal-Bench 2.1 test scripts install their test tools (`apt-get`, `uv`, pytest)
  when they run, so the container needs outbound network while grading. With its
  default network policy, Verifiers' Docker runtime allows any outbound address, so
  each host that runs rollout workers needs that access.
- **The run fails after 10 untrainable batches.** A group whose samples all get the
  same reward is dropped (`TrainingSampleBuilder.drop_zero_std_reward_groups`). The
  batcher warns after each batch's worth of such groups in a row and raises
  "consecutive untrainable batches" after 10. Check grading on a few tasks, and that
  the model solves some of them, before a long run.

## Known issues

- Verifiers stages each task's `tests/` with a plain `tar -x`, which restores host file
  ownership and fails in a rootless container with unmapped host IDs.
