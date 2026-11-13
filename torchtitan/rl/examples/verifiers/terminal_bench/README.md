# Terminal-Bench with Verifiers and TitanRL

This example trains a terminal agent with TitanRL on AI2's TMax tasks and validates it
on the 89 Terminal-Bench 2.1 tasks. Each rollout is one attempt at one task: Harbor's
Terminus-2 agent works in the task's Docker container, every model call goes to
TitanRL's generators, and the task's own tests grade the result. Verifiers runs the
agent and the grading through the [Verifiers bridge](../README.md).

> **Toy sandbox.** Local Docker keeps this example self-contained; it is not a sandbox
> for real training. Containers run untrusted task code on the controller host. They sit
> on Docker's shared bridge network and reach the internet through the host. For real
> training, plug in a mature sandbox as the Verifiers runtime. Do not use
> `SubprocessConfig`: it runs the agent on the host.
>
> The recipes set `workers_on_controller_host=True`, so a multi-host job still runs every
> rollout worker, and so every container, on the controller host. With it False and a launcher,
> the workers spread over the job's other hosts, and each of those hosts needs Docker, the
> task images and outbound network while grading.

## How it works

One group, top to bottom: rows 2-5 run for each rollout, rows 3-4 for each agent turn.
Each column is one kind of process, or the containers. The rollouter runs N =
`worker_pool_size` rollout workers. Row numbers match the steps below.

```text
   controller          rollout worker           container          generators        trainer
     (CPU)        (CPU, xN): Verifiers env    (per rollout)           (GPU)           (GPU)
        |                     |                     |                   |               |
 1      |--group------------->|                     |                   |               |
 2      |                     |--start------------->|                   |               |
 3      |                     |<-chat request-------|                   |               |
 4      |                     |--generate------------------------------>|               |
 5      |                     |--run test.sh------->|                   |               |
 6      |<-scored group-------|                     |                   |               |
 7      |--batch, packed by the batcher process---------------------------------------->|
        |                     |                     |                   |<-new weights--|
```

With the command in "Run", every column above runs on one host. With a launcher, the generators, and
any trainer ranks past host 0, can move to other hosts. The `VerifiersRollouter` docstring draws an
example.

1. **Task.** The controller sends each group to a rollout worker process
   (`VerifiersRolloutWorker`), which runs one Verifiers episode per rollout in the group.
2. **Agent.** The worker starts a container from the task's `docker_image` and runs
   Terminus-2 in it ([`harness.py`](./harness.py)). The agent types commands into a
   tmux shell and reads the screen back.
3. **Model call.** Each agent turn is a chat request to the worker's Verifiers
   interception server, which records the turn in the rollout's trace.
4. **Generation.** The interception server renders the chat to token ids and sends them
   to the worker's `GenerationServer`, which asks the router for a generator, then calls
   that generator itself.
5. **Grading.** When the agent stops, Verifiers runs the task's `tests/test.sh` in the
   container and reads the reward file it writes.
6. **Trace.** The rollout worker turns the trace into rollout turns, with the exact
   tokens the generators sampled, computes group advantages and returns the finished
   group to the controller.
7. **Training.** The batcher feeds the trainer, and the generators load the new weights
   after each step.

Validation takes the same path on Terminal-Bench 2.1, one greedy sample per task. A validation
starts with training and runs beside it (`ValidationConfig`).

| What | Where |
| --- | --- |
| Recipes: model, parallelism, generator, training loop, and the Verifiers setup (datasets, harness, Docker runtime, timeouts) | `torchtitan_recipes/rl/verifiers_terminal_bench.py` |
| Terminus-2 harness and the program it runs in the container, copied from Verifiers 0.3.1 to expose reasoning in the history and summarization | [`harness.py`](./harness.py), [`terminus_harness.py`](./terminus_harness.py) |
| Harbor taskset with each task's WORKDIR and verifier timeout; Verifiers imports it through `torchtitan_verifiers_terminal_bench/` | [`taskset.py`](./taskset.py) |
| Offline TMax to Harbor task conversion | [`prepare_tmax.py`](./prepare_tmax.py) |
| Bridge: task dataset, rollout worker, generation server, trace to turns | [Verifiers integration](../README.md) |
| CPU tests | `tests/rl/unit_tests/cpu/test_verifiers_terminal_bench*.py` |

## Run

Install the TorchTitan RL dependencies and this directory's `requirements.txt` in one
Python 3.12 environment, and let the user run Docker on the controller host. Prepare the
training tasks (see [Datasets](#datasets)), then:

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
| `rl_dapo_qwen35_35b_a3b_terminal_bench_dist_moe` | Qwen3.5-35B-A3B | 4 trainer, 4 generators x 4 (DP 4 x EP 4, Dist-MoE) |

`torchtitan.rl.train` places every process on the local host, so the 9B recipe needs a
16-GPU host as written. The 35B-A3B recipe uses Dist-MoE, which needs SM100+ GPUs. It needs
20 GPUs on several hosts, and its 4-GPU trainer holds 140 GB of model state per GPU, so it
needs 288 GB GPUs (GB300, B300). On B200 (180 GB), set the trainer's
`data_parallel_shard_degree=4` (8 GPUs). Wrap it in
`MonarchControllerWithLauncher.Config.from_controller_config` with a launcher, as
`rl_grpo_qwen3_0_6b_flex_slurm` in `torchtitan_recipes/rl/alphabet_sort.py` does. It has not
been run end to end.

With the Slurm launcher, host 0 runs the 4-GPU trainer and every CPU process. On 4-GPU hosts
(GB300), each 4-GPU generator gets its own host: 5 hosts. On 8-GPU hosts, generator_0 shares host 0
and the other three take two more hosts: 3 hosts, 4 GPUs idle.

With the default local Docker runtime, every rollout in flight is a container on the
controller host, and a task asks for at least 1 CPU and 2 GB (21 of the 89 Terminal-Bench
2.1 tasks ask for 4-8 GB). The 9B recipe keeps up to 1,536 training rollouts in flight, plus 89
while a validation runs: at least 1,625 CPUs and 3,250 GB of RAM on one host. The 35B-A3B recipe
keeps up to 1,280, plus the same 89: 1,369 CPUs and 2,738 GB. For runs this size, set a remote
sandbox as the `runtime` (below).

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
- Grading uses each task's verifier timeout from task.toml, or Harbor's 600 s when it
  declares none. The setup and agent timeouts in the recipe file apply to every task.
  Under training load each model call waits behind other rollouts, so the task's own
  agent timeout (600-12,000 s in Terminal-Bench 2.1) would end rollouts early. Because
  of this, validation scores are not directly comparable to `harbor run` results.
- A task without a `workdir` in task.toml starts in the `WORKDIR` of its
  `environment/Dockerfile`'s final build stage, as under `harbor run`.
- The validation id has no `@ref`, so it uses whichever revision is in the cache. Pin a
  ref for numbers you want to compare; Terminal-Bench 2.0 and 2.1 are not
  interchangeable.
- Without Harbor Hub access, pre-populate `~/.cache/harbor` with an exported task tree.
- To use other datasets, change the ids passed to `terminal_bench_rollouter_config` in the recipe file.
  The training and validation datasets must differ. For another sandbox or agent, set
  `worker.environment.agent.runtime` or `.harness` on the returned config.
  For another env, pass `taskset_id`: the name of your own top-level package that exports
  a `TerminalTaskset` subclass in `__all__` (and an Env, only if it needs its own).

## If every reward is 0

- **A grader that never ran scores 0.** Verifiers ignores `tests/test.sh`'s exit status
  and scores a missing or unreadable reward file as 0, the same as a failed task.
  Terminal-Bench 2.1 test scripts install their test tools (`apt-get`, `uv`, pytest)
  when they run, so the container needs outbound network while grading. Its traffic
  leaves through the controller host, so that host needs the access.
- **The run fails after 10 untrainable batches.** A group whose samples all get the
  same reward is dropped (`TrainingSampleBuilder.drop_zero_std_reward_groups`). The
  batcher warns after each batch's worth of such groups in a row and raises
  "consecutive untrainable batches" after 10. Check grading on a few tasks, and that
  the model solves some of them, before a long run.

## Known issues

- Verifiers stages each task's `tests/` with a plain `tar -x`, which restores host file
  ownership and fails in a rootless container with unmapped host IDs.
