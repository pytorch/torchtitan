# Verifiers integration

The optional bridge between TitanRL and [Verifiers](https://github.com/PrimeIntellect-ai/verifiers),
pinned to Verifiers 0.4.0 (`verifiers.v1` API). Other TitanRL recipes don't import Verifiers.

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

A diagram of one group: the `VerifiersRollouter` docstring in [`rollouter.py`](./rollouter.py).

1. [`data.py`](./data.py) loads the taskset in the controller, which sends each group's task to a
   rollout worker.
2. [`env_server.py`](./env_server.py) runs the Verifiers env server as a local process on the
   controller host. The server runs the agents in the configured runtime, e.g. `SubprocessConfig`
   or `DockerConfig`. With a pool, its pool workers run them. The Terminal-Bench recipe runs 16.
3. [`rollouter.py`](./rollouter.py) runs each group in a rollout worker process
   (`VerifiersRolloutWorker`), like TitanRL's own rollout groups. The worker sends the task to the
   env server (a plain dict, msgpack over ZMQ) and turns the trainable agent's trace of each
   rollout into TitanRL `RolloutTurn`s. The controller receives finished groups only.
4. [`generation_server.py`](./generation_server.py) serves Verifiers' model calls (token ids over
   HTTP, from the env server). For each call it gets a generator from `router.reserve`, then calls
   `generate` on it. Each rollout worker runs its own, on localhost. The server also records what
   Verifiers' traces leave out: the policy versions and the generation metrics.

For CPU-heavy envs, raise the env server's pool (`verifiers_env_server.serve.pool`, e.g.
`vf.StaticPoolConfig(num_workers=16)`). At high rollout rates, also raise the
rollouter's `worker_pool_size`: each rollout worker decodes one trace at a time.

## Taskset ids

Verifiers loads each taskset, harness or env by id: the name of a Python package. It imports the
package and takes the one class of that kind in its `__all__`:

```text
id "torchtitan_verifiers_terminal_bench"
  import torchtitan_verifiers_terminal_bench
  __all__ = ["HarborEnv", "TerminalBenchTerminusHarness", "TerminalTaskset"]
  taskset -> TerminalTaskset, harness -> TerminalBenchTerminusHarness,
  env -> HarborEnv (re-exported from Verifiers)
```

Leave the harness or the env unset, and Verifiers takes the one in the taskset's package. If there
is none, it uses its defaults: the `bash` harness and `SingleAgentEnv`.

Outside this repo, any top-level package name works as the id. Don't reuse a built-in Verifiers
plugin's name (e.g. `harbor`, `bash`, `codex`): Verifiers looks for its own first.

Configs carry ids, not classes, so each of these processes imports the package:
the controller, the env server and its pool workers.

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
2. Set your agent, harness and runtime in `verifiers_env_server.environment`. Leave its
   `taskset` unset: `VerifiersRollouter` copies it from the training dataset.
3. Set `ValidationConfig.steps` for your validation set: the recipe validates on 30 tasks.

Then run:

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_my_task \
  --config <recipe function name>
```
