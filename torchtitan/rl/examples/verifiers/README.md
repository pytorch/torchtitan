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
     controller      rollout worker                    agent        router     generator
     (Rollouter)     (VerifiersRolloutWorker: env,     (runtime)
                     interception, GenerationServer)
          |                  |                            |            |           |
 1        |--run_group------>|                            |            |           |
          |                  |                            |            |           |
 2        |                  |--start, per rollout------->|            |           |
          |                  |                            |            |           |
 3        |                  |<--chat---------------------|            |           |
          |                  |                            |            |           |
 4        |                  |--reserve------------------------------->|           |
          |                  |--generate------------------------------------------->|
          |                  |                            |            |           |
 5        |                  |--reply-------------------->|            |           |
          |                  |                            |            |           |
 6        |                  |--grade-------------------->|            |           |
          |                  |                            |            |           |
 7        |                  |--release_session----------------------->|           |
          |                  |                            |            |           |
 8        |<--scored group---|                            |            |           |
          |                  |                            |            |           |
 9        |--release_groups--------------------------------------------->|           |
```

1. The controller sends one task to a rollout worker. The task is plain data from
   [`data.py`](./data.py).
2. The worker's Verifiers env runs the task once per rollout of the group (`env.run_slot`). It
   starts the agent in the env's runtime, e.g. a local process or a Docker container. Verifiers
   reports each trace it starts (`on_trace`), and the worker registers that trace id in its
   GenerationServer as a session of this group.
3. Each agent turn, the agent sends a chat request to Verifiers' interception server, in the
   worker. Verifiers' train client renders the messages to token ids and sends them to the worker's
   [`GenerationServer`](./generation_server.py), over HTTP on the loopback address.
4. The GenerationServer finds the call's group by its session id, the trace id (a session it
   doesn't know gets 422). It calls TitanRL's `generate_fn`, which reserves a generator at the
   router and then calls that generator.
5. The GenerationServer answers with the tokens and logprobs, and keeps the call's policy version
   and generator metrics: the trace has no field for them. The agent gets its reply. Steps 3 to
   5 repeat every turn.
6. When the agent stops, Verifiers grades the rollout, e.g. by running the task's tests.
7. The worker releases the routing session of each trace Verifiers made, a retried attempt's
   included.
8. The worker turns the rollout's trainable trace into TitanRL rollout turns
   ([`rollouter.py`](./rollouter.py)), scores the group, computes its advantages, and returns it.
9. The group makes no more model calls. Through the router, the controller tells each generator to
   forget the group: the policy version its prompts are cached under, and any KV cache it still
   holds for the group.

Shutdown:

1. Stop admitting groups.
2. Close the generators, so model calls still waiting fail fast.
3. Drain the running groups.
4. Each worker's `close` cancels its episodes, which stops their sandboxes. Then it stops serving
   the env, then the GenerationServer.
5. Stop the worker processes.

A killed worker skips `close`, so its sandboxes stay up.

How many processes: the controller and the rollout workers (one per `rollouts_per_worker` training
rollouts in flight, rounded up). Each worker loads the env and runs its episodes; an agent runs in the
env's runtime.

## Add an experiment

For a task `my_task`:

```text
torchtitan/rl/examples/verifiers/my_task/
  __init__.py
  data.py                                  # your Taskset and its TasksetConfig
  requirements.txt                         # its dependencies
torchtitan_recipes/rl/verifiers_my_task.py # recipe functions that return a Controller.Config
```

Install its dependencies with `pip install -r torchtitan/rl/examples/verifiers/my_task/requirements.txt`.
Start the recipe from a copy of
[`verifiers_dapo_math.py`](../../../../torchtitan_recipes/rl/verifiers_dapo_math.py), and change its
`Rollouter.Config`: your `TasksetConfig` in its training and validation datasets, and your agent,
harness and runtime in `worker.environment`. Pass the same training taskset object to the
training dataset and to `worker.environment`: the worker builds each task with the env's taskset.
So the validation dataset's taskset needs the same task class and task config.
Then run:

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_my_task \
  --config <recipe function name>
```

See [DAPO Math](./dapo_math) for a complete single-turn example and
[Terminal-Bench](./terminal_bench) for a multi-turn agent example. This
integration pins Verifiers 0.4.0 and uses its `verifiers.v1` API.
