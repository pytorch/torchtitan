# Verifiers integration

Runs [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) envs in TitanRL, pinned to
Verifiers 0.4.0 (the `verifiers.v1` API). Other TitanRL recipes don't import Verifiers.

- TitanRL picks the tasks, serves the model calls and trains on the scored groups.
- Verifiers runs the env: the agent, its sandbox, its tools and the grade.

The two examples, [`dapo_math/`](./dapo_math) and [`terminal_bench/`](./terminal_bench), each
have their own README and dependencies. Start with DAPO math: single-turn, no sandbox.

Each task carries its env's config, and a worker serves every env it is sent. So one worker can
run DAPO and Terminal-Bench groups side by side, and a run can validate on another env than it
trains on. One training stream that mixes two envs isn't supported yet (a TODO in
[`data.py`](./data.py)).

## How it works

One group, top to bottom. Each column is a process, or the agent in its runtime.

```text
     controller      rollout worker                                          agent      router  generator
     (Rollouter)     Verifiers env  interception  train client  Generation   (runtime)
                                    server                      Server
          |               |              |              |           |            |          |        |
 1        |--run_group--->|              |              |           |            |          |        |
          |               |              |              |           |            |          |        |
 2        |               |--start agent------------------------------------->|          |        |
          |               |              |              |           |            |          |        |
 3        |               |              |<--chat-------------------------------|          |        |
          |               |              |              |           |            |          |        |
 4        |               |              |--messages--->|           |            |          |        |
          |               |              |              |           |            |          |        |
 5        |               |              |              |--tokens-->|            |          |        |
          |               |              |              |           |            |          |        |
 6        |               |              |              |           |--reserve------------>|        |
          |               |              |              |           |            |          |        |
 7        |               |              |              |           |--generate------------------->|
          |               |              |              |           |            |          |        |
 8        |               |              |              |<--tokens--|            |          |        |
          |               |              |              |           |            |          |        |
 9        |               |              |--reply------------------------------>|          |        |
          |               |              |              |           |            |          |        |
10        |               |--grade------------------------------------------->|          |        |
          |               |              |              |           |            |          |        |
11        |               |--release_session------------------------------------------->|        |
          |               |              |              |           |            |          |        |
12        |<--group-------|              |              |           |            |          |        |
```

1. The controller sends one task to the rollout worker with the fewest rollouts in flight. The
   task is plain data from [`data.py`](./data.py), with the config of the env that runs it.
   A worker runs many groups at once.
2. The worker runs the task `group_size` times with Verifiers' `env.run_slot`. For each rollout,
   Verifiers starts the agent in the env's runtime.
3. Each agent turn, the agent sends a chat request to the interception server: Verifiers' HTTP
   server in the worker that stands in for the model. A local Docker agent on Linux reaches it
   through Verifiers' egress proxy, one more hop for the same request.
4. The interception server hands the messages to Verifiers' train client.
5. The train client renders them to tokens and posts them to the worker's
   [`GenerationServer`](./generation_server.py), on loopback.
6. The GenerationServer calls TitanRL's `generate_fn`, which reserves a generator at the router.
7. `generate_fn` calls that generator.
8. The GenerationServer answers with the tokens and logprobs, and keeps the call's policy
   version and generator metrics.
9. The agent gets its reply. Steps 3 to 9 repeat every turn.
10. When the agent stops, Verifiers grades the rollout, e.g. by running the task's tests.
11. The worker releases each trace's routing session at the router.
12. The worker turns the trainable agent's trace into a TitanRL rollout
    ([`rollouter.py`](./rollouter.py)). Other agents' traces, e.g. a judge's, don't train. The
    worker returns the scored group. The controller frees the group's KV cache and trains on it.

Shutdown, at the end of training:

1. The controller closes the generators, so pending `generate_fn` calls fail fast.
2. It waits a short time for the groups still running (`Rollouter.close`).
3. It calls each worker's `close`. The worker cancels the rollouts still running: each one's
   `run_slot` stops its own runtime, e.g. removes its container. Then the worker stops serving its
   envs (their interception servers and shared tools), then its GenerationServer. A `close` that
   raises, or that still runs after 60 s, is logged, and step 4 runs anyway.
4. The controller stops the worker processes. Step 3 runs before this stop because Monarch gives
   an actor's own cleanup 3 s, then fails the job.

## What crosses each hop

```text
hop                                                      what crosses
agent (local Docker on Linux) -> egress proxy (worker)   the whole chat history, as JSON
egress proxy -> interception server                      the same bytes, relayed
agent (any other runtime) -> interception server         the whole chat history, as JSON
train client -> GenerationServer (loopback)              the whole prompt's token ids
generate_fn -> generator (Monarch)                       the whole prompt's token ids
generate_fn -> router                                    the request and session ids
each reply, back to the agent                            the new completion only
```

- Each call carries the whole history: the agent speaks OpenAI chat, and the generator gets the
  full prompt. So the bytes a rollout moves grow with turns x context. They cost worker CPU to
  encode and decode, more than network time.
- Memory grows only with the rollout's own length. The trace keeps each message once, with
  only the tokens it adds; the GenerationServer keeps counters per session; a `Rollout` keeps each
  turn's new prompt tokens and its completion.
- Size limits. The GenerationServer reads request bodies up to 1 GiB; replies have no limit.
  The largest Monarch message here is a worker's scored group, G rollouts. With router replay or
  top-k logprobs on, the batcher's packed step becomes the largest message. A recipe that turns
  either on also sets `async_loop.batcher.rdma_min_tensor_bytes`, so those tensors go by RDMA:
  Monarch caps a message at 10 GiB and fails the whole actor mesh above it.

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
change its env configs: your `TasksetConfig` with `id="torchtitan-verifiers-my-task"`, and your
agent, harness and runtime. Training and validation each name their env, so one run can validate on
another taskset. Set `ValidationConfig.steps` to your validation set's size, or to -1 to read it
once. Then run:

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_my_task \
  --config <recipe function name>
```
