# Verifiers integration

This directory contains the optional bridge between TitanRL and
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers). Other TitanRL
recipes do not import or require Verifiers.

The ownership boundary is:

- TitanRL loads samples, schedules rollout groups, routes generation, tracks
  policy versions, computes advantages, and trains the policy.
- Verifiers runs the environment, agent or tool loop, runtime, and task reward.
- The adapter converts samples and traces between the two systems.

Reusable bridge code lives alongside this README. Each experiment has its own
package and dependencies: [`dapo_math/`](./dapo_math) (single-turn math) and
[`terminal_bench/`](./terminal_bench) (multi-turn terminal agent on Harbor tasks).

## Integration flow

One group, top to bottom. Each column is a process, or the agent in its runtime.

```text
     controller      rollout worker            env server     agent        router     generator
     (Rollouter)     (VerifiersRolloutWorker   (Verifiers)    (runtime)
                     and its GenerationServer)
          |                  |                      |              |            |           |
 1        |--run_group------>|                      |              |            |           |
          |                  |                      |              |            |           |
 2        |                  |--task, per rollout-->|              |            |           |
          |                  |                      |              |            |           |
 3        |                  |                      |--start------>|            |           |
          |                  |                      |              |            |           |
 4        |                  |                      |<--chat-------|            |           |
          |                  |                      |              |            |           |
 5        |                  |<--token ids----------|              |            |           |
          |                  |                      |              |            |           |
 6        |                  |--reserve---------------------------------------->|           |
          |                  |--generate---------------------------------------------------->|
          |                  |                      |              |            |           |
 7        |                  |--tokens, logprobs--->|--reply------>|            |           |
          |                  |                      |              |            |           |
 8        |                  |                      |--grade------>|            |           |
          |                  |                      |              |            |           |
 9        |                  |<--trace, reward------|              |            |           |
          |                  |--release_session------------------------------->|           |
          |                  |                      |              |            |           |
10        |<--scored group---|                      |              |            |           |
```

1. The controller sends one task to a rollout worker. The task is plain data from
   [`data.py`](./data.py).
2. The worker sends the task to the env server once per rollout of the group: a plain dict,
   msgpack over ZMQ ([`env_server.py`](./env_server.py)).
3. The env server starts the agent in the env's runtime, e.g. a local process or a Docker
   container. With a pool, one of its pool workers runs the rollout.
4. Each agent turn, the agent sends a chat request to Verifiers' interception server, in the env
   server. Verifiers' train client renders the messages to token ids.
5. The token ids reach the rollout worker's [`GenerationServer`](./generation_server.py), over
   HTTP on the controller host.
6. The GenerationServer calls TitanRL's `generate_fn`, which reserves a generator at the router
   and then calls that generator.
7. The GenerationServer answers with the tokens and logprobs, and keeps the call's policy version
   and generator metrics: the trace has no field for them. The agent gets its reply. Steps 4 to
   7 repeat every turn.
8. When the agent stops, Verifiers grades the rollout, e.g. by running the task's tests.
9. The env server returns the rollout's traces. The worker releases each trace's routing session
   at the router.
10. The worker turns the rollout's trainable trace into TitanRL rollout turns
    ([`rollouter.py`](./rollouter.py)), scores the group, computes its advantages, and returns it.

How many processes: the controller, its rollout workers (one per `rollouts_per_worker` training
rollouts in flight), and one env server that all workers share. With a pool
(`verifiers_env_server.serve.pool`), the env server runs its rollouts in its pool workers. So 4
rollout workers and a pool of 8 add 4 + 1 + 8 processes to the controller, not 4 x 8.

The taskset is set once, in the training dataset (`VerifiersTaskDataset.Config.verifiers_taskset`).
`VerifiersRollouter.Config` fills the env server's taskset from it, so leave
`verifiers_env_server.environment.taskset` unset.

## Taskset ids

Verifiers loads a taskset, harness or env by id: the name of an installed package. It imports
the package's module ("-" becomes "_") and takes the one class of that kind in its `__all__`:

```text
id "torchtitan-verifiers-terminal-bench"
  import torchtitan_verifiers_terminal_bench
  __all__ = ["HarborEnv", "TerminalBenchTerminusHarness", "TerminalTaskset"]
  taskset -> TerminalTaskset, harness -> TerminalBenchTerminusHarness,
  env -> HarborEnv (re-exported from Verifiers)
```

Leave the harness or the env unset, and Verifiers takes the one in the taskset's package, or its
defaults: the `bash` harness and `SingleAgentEnv`. Don't reuse a built-in plugin's name (e.g.
`harbor`, `bash`): Verifiers looks for its own first. A `Taskset` subclass needs its own package
id: Verifiers picks the `Taskset` by id alone.

Each example folder is such a package: its `pyproject.toml` names it, and its
`torchtitan_verifiers_<example>/` module re-exports the example's classes. Install it, with its
dependencies, by `pip install -e torchtitan/rl/examples/verifiers/<example>`.

## Add an experiment

For a task `my_task`:

```text
torchtitan/rl/examples/verifiers/my_task/
  __init__.py
  data.py                                  # your Taskset and its TasksetConfig
  pyproject.toml                           # package "torchtitan-verifiers-my-task", its dependencies
  torchtitan_verifiers_my_task/__init__.py # __all__ re-exports your Taskset
torchtitan_recipes/rl/verifiers_my_task.py # recipe functions that return a Controller.Config
```

Install it with `pip install -e torchtitan/rl/examples/verifiers/my_task`. Start the recipe from a
copy of [`verifiers_dapo_math.py`](../../../../torchtitan_recipes/rl/verifiers_dapo_math.py), and
change its `VerifiersRollouter.Config`: your `TasksetConfig` with
`id="torchtitan-verifiers-my-task"` (see "Taskset ids") in its training and validation datasets,
and your agent, harness and runtime in `verifiers_env_server.environment`. Then run:

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_my_task \
  --config <recipe function name>
```

See [DAPO Math](./dapo_math) for a complete single-turn example and
[Terminal-Bench](./terminal_bench) for a multi-turn agent example. This
integration pins Verifiers 0.4.0 and uses its `verifiers.v1` API.
