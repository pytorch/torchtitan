# Verifiers integration

The optional bridge between TitanRL and [Verifiers](https://github.com/PrimeIntellect-ai/verifiers),
pinned to Verifiers 0.4.1 (`verifiers.v1` API). Other TitanRL recipes don't import Verifiers.

- TitanRL picks the tasks, serves the model calls and trains on the scored groups.
- Verifiers runs the env, the agent and its tools, the runtime and the task reward.
- The code here converts tasks and traces between the two.

Words used below:

```text
taskset   the list of tasks, e.g. Terminal-Bench's
harness   the program that runs the agent, e.g. Terminus
env       a taskset plus the agents that solve it
runtime   where an agent runs, e.g. a Docker container or a local process
trace     Verifiers' record of one agent's run: each message and model call
group     the rollouts of one task, scored together
```

Examples, each with its own README and dependencies. Start with [`dapo_math/`](./dapo_math):
single-turn math, no sandbox. [`terminal_bench/`](./terminal_bench) is a multi-turn terminal agent.

## Integration flow

One group, top to bottom. Each column is one process, or the agent in its runtime. Where each
column runs: the `VerifiersRollouter` docstring in [`rollouter.py`](./rollouter.py).

```text
    VerifiersRollouter  rollout worker      agent       router     generator
       (controller)                      its runtime               (rank 0)
             |                 |              |            |           |
 1           |--run_group----->|              |            |           |  Monarch, random worker
 2           |                 |--start------>|            |           |  one per rollout
 3           |                 |<-chat--------|            |           |  HTTP, each turn
 4           |                 |              |            |           |  render, POST to this
             |                 |              |            |           |  process's GenerationServer
 5           |                 |--reserve----------------->|           |  returns a generator
 6           |                 |--generate---------------------------->|  Monarch call
 7           |                 |<-Completion---------------------------|
 8           |                 |--release_reservation----->|           |  no reply
 9           |                 |--reply------>|            |           |  3-9 per turn
10           |                 * grade        |            |           |  by the worker, e.g. by
             |                 |              |            |           |  running the task's tests
11           |                 |--release_session--------->|           |  one per trace, after
             |                 |              |            |           |  run_slot returns
12           |<-RolloutGroup---|              |            |           |  scored
13           |--release_groups---------------------------->|           |  drops its cache salts
```

1. [`data.py`](./data.py) loads the taskset in the controller. The controller sends each task to a
   rollout worker as a plain dict (row 1). Workers never load the dataset.
2. The worker loads the Verifiers env once. For each rollout, it starts an agent in the configured
   runtime, e.g. `DockerConfig`, `SubprocessConfig` or `PrimeConfig` (row 2).
3. Rows 3-9 repeat for each agent turn. The `GenerationServer`
   ([`generation_server.py`](./generation_server.py)) also records what Verifiers' traces leave out:
   the policy versions and the generation metrics.
4. The worker grades each rollout and turns the trainable agent's trace into TitanRL
   `RolloutTurn`s (rows 10-12). The controller then releases the group at the router (row 13),
   and TitanRL trains on it as usual.

## Taskset ids

Verifiers loads each taskset, harness or env by id: the name of a Python package. It imports the
package and takes the one class of that kind in its `__all__`:

```text
id "torchtitan_verifiers_terminal_bench"
  import torchtitan_verifiers_terminal_bench
  __all__ = ["TerminalBenchTerminusHarness", "TerminalTaskset"]
  taskset -> TerminalTaskset, harness -> TerminalBenchTerminusHarness
```

Leave the harness or the env unset, and Verifiers takes the one in the taskset's package. If there
is none, it uses its defaults: the `bash` harness and `SingleAgentEnv`.

Outside this repo, any top-level package name works as the id. Don't reuse a built-in Verifiers
plugin's name (e.g. `harbor`, `bash`, `codex`): Verifiers looks for its own first.

Configs carry ids, not classes, so each of these processes imports the package:
the controller and each rollout worker.

`pyproject.toml` ships every `torchtitan_verifiers_*` package. An editable install
(`pip install -e . --no-deps`) sees only the packages that existed when it ran: rerun it after you
add or pull a new one.

## Add an experiment

For a task `my_task`:

```text
torchtitan/rl/examples/verifiers/my_task/
  __init__.py
  data.py                                  # your Taskset and its TasksetConfig
  requirements.txt                         # its own dependencies
torchtitan_verifiers_my_task/__init__.py   # __all__ re-exports your Taskset
torchtitan_recipes/rl/verifiers_my_task.py # recipe functions that return a Controller.Config
```

Keep the `torchtitan_verifiers_` prefix: `pyproject.toml`, the PR labeler and the reviewer routing
match it.

Start the recipe from a copy of
[`verifiers_dapo_math.py`](../../../../torchtitan_recipes/rl/verifiers_dapo_math.py). Change three
things:

1. In both `VerifiersTaskDataset.Config`s (training and validation), set your `TasksetConfig`, with
   `id="torchtitan_verifiers_my_task"`.
2. Set your agent, harness and runtime in `VerifiersRolloutWorker.Config.environment`. Leave its
   `taskset` unset: `VerifiersRollouter` copies it from the training dataset.
3. Set `ValidationConfig.steps` for your validation set: the recipe validates on 30 tasks and
   raises if there are fewer. -1 reads all of them.

Then run:

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_my_task \
  --config <recipe function name>
```

## Scaling

The rollout workers do the env's CPU work: rendering (turning messages into tokens) and decoding
traces. The agent itself runs in its runtime.

- Workers busy on CPU: raise the rollouter's `worker_pool_size` (default 4; the Terminal-Bench
  recipe uses 16).
- Renders waiting: lower the worker's `renderer_multiplex` (rollouts per renderer, default 256). A
  renderer renders one prompt at a time, behind a lock. More renderers (~75-95 MB each) render in
  parallel, up to `num_threads_per_worker` at once.
- Each worker is its own Verifiers run, so run-wide limits apply per worker. Example: 16 workers
  with `PrimeConfig.creates_per_min=10` start up to 160 sandboxes a minute.

## Which failed rollouts train

A failed rollout either trains (usually at reward 0) or is ERROR_INFRA. ERROR_INFRA means the
failure isn't the policy's fault. That rollout gets no reward and isn't trained. The rest of its
group still trains.

- ERROR_INFRA: a grading deadline passed, a model call failed (other than a prompt past the
  context), the rollout ended before the policy's first turn, or the sandbox, a tool or the task's
  code failed.
- Trains: the last prompt grew past the context, another deadline passed (e.g. the agent's), or
  the agent exited (e.g. the model ran `pkill` on it).
- A first prompt past the context has no tokens to train, so its whole group is dropped.

The exact rules, in order: `VerifiersRolloutWorker.rollout_status`. The metric
`rollout/status_rate/error_infra` tracks the share of ERROR_INFRA rollouts.
