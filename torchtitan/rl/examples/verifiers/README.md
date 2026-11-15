# Verifiers integration

Runs [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) envs in TitanRL, pinned to
Verifiers 0.4.1 (the `verifiers.v1` API). Other TitanRL recipes don't import Verifiers.

- TitanRL picks the tasks, serves the model calls and trains on the scored groups.
- Verifiers runs the env: the agent, its sandbox, its tools and the grade.

Words used below:

```text
taskset    the list of tasks and how each is graded, e.g. a benchmark's
harness    the program that runs the agent, e.g. Terminus-2
runtime    where an agent runs: a local process, a Docker container, a remote sandbox
env        a taskset plus the agent that solves it: its harness and runtime
trace      Verifiers' record of one agent's run: each message and model call
group      the rollouts of one task, scored together
```

A Verifiers env is what a recipe configures (`vf.SingleAgentEnvConfig`). The two examples,
[`dapo_math/`](./dapo_math) and [`terminal_bench/`](./terminal_bench), each with its own README and
dependencies:

```text
env              taskset                          harness                  runtime
DAPO math        DAPO math, graded by             none: one model call     a local process
                 Math-Verify
Terminal-Bench   TMax or Terminal-Bench, graded   Terminus-2, many turns   a Docker container
                 by each task's tests                                      per rollout
```

Start with DAPO math: single-turn, no sandbox.

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

## Concurrency

One cap: the controller keeps at most K groups of G rollouts in flight, with
K = `(target_offpolicy_steps + 1) x num_prompts_per_train_step` and G = `num_samples_per_prompt`.
A group is in flight from when a worker starts it until it returns to the controller. The rest
follows from K x G:

| what | how many | set by |
|---|---|---|
| rollout workers, processes on the controller host | ceil(K x G / rollouts_per_worker) | `Rollouter.Config.rollouts_per_worker` (default 64) |
| rollouts per worker | a sizing target: whole groups are placed, so a worker can hold up to one group more (groups of 24 at 64: 72); a group larger than it raises | the same |
| sandboxes | one per running rollout: at most K x G, plus the validation tasks | the agent's runtime |
| interception servers per worker | one per `multiplex` rollouts in flight (Verifiers' default 32); never shrink | Verifiers: the env's `interception.multiplex` |
| renderers per worker | one per 256 model calls in flight (Verifiers' default `multiplex`); never shrink | fixed: the worker builds its train client without `multiplex` |
| threads per worker | `num_threads_per_worker`; None is Python's default, min(32, CPUs + 4) | `Rollouter.Config.num_threads_per_worker` (default 4) |
| controller host CPUs | one per worker, plus local sandboxes | the host |

- Placement: the Rollouter counts rollouts in flight per worker and sends each group to the
  worker with the lowest count. It is a list of numbers in the controller, so it works wherever
  the workers run. A group's count drops when the whole group returns.
- A whole group runs on one worker, so a group larger than `rollouts_per_worker` raises when the
  config is built. Raise `rollouts_per_worker` to at least the group size. Example: groups of 128
  at the default 64 raise; set `rollouts_per_worker=128` and 16 groups run on 16 workers, one
  group each.
- Each rollout starts its own sandbox and removes it when it ends. Validation sends all its tasks
  at once, beside training (`ValidationConfig.steps`). Usually far fewer run: a group stays in
  flight until its slowest rollout ends.
- Verifiers' pools grow with the load and never shrink. Loading a second renderer stops the
  worker's event loop for a moment. At the default 64 rollouts per worker and groups of at most
  64, a worker holds under 128 rollouts, so it keeps one renderer. A local runtime (Docker too)
  opens no tunnel; a remote provider without its own tunnel opens one per interception server.
  Set the interception pool in the env's `interception.multiplex`. The renderer pool stays at
  Verifiers' default: the worker builds its train client itself.
- Threads run renders and blocking runtime calls. A render waiting for the renderer lock holds a
  thread, so a small pool stalls other thread work, e.g. sandbox cleanup. More threads don't
  render faster: renders hold Python's GIL.
- CPUs: a local Docker sandbox shares the controller host's CPUs with the workers. `--cpus`
  limits a container but reserves nothing: count one CPU per worker plus each running sandbox's
  limit. A remote provider runs sandboxes on its own machines.

## Sandboxes

Each recipe picks a runtime for its agent, e.g. `vf.DockerConfig()` for local runs. Docker is a
placeholder: real training uses the sandbox provider of your choice, as a Verifiers runtime.

- A remote sandbox provider plugs in as a Verifiers custom runtime (`runtime.type = "custom"`;
  Verifiers' `docs/runtimes.md`). It implements `start`, `exec`, `write`, and `cleanup` or
  `teardown`.
- The agent's model calls come back to the worker's interception server. A provider that brings
  its own tunnel sets `is_local = True` and rewrites the URL in `host_url`. Then Verifiers opens no
  tunnel and needs no port per worker.
- Retry lost sandboxes with Verifiers' retries, e.g. `RetryRule(type="SandboxError")`. The worker
  releases the routing session of every attempt.
- Each run has one Verifiers run id (`VerifiersRolloutWorker.Config.run_id`, exported as
  `VF_RUN_ID`). It labels the run's sandboxes. Verifiers' run-wide limits (e.g.
  `creates_per_min`) count over every worker that shares a home directory: a lock file in
  `~/.cache/verifiers`.
- A worker that is killed can't clean up. A provider's leases reclaim its sandboxes. Docker
  containers stay. Once the run has stopped, remove them with the command below. Don't run it
  while the run is live: its containers carry the same label.

```bash
docker rm -f $(docker ps -aq --filter label=verifiers.run=<run_id>)
```

## Failed rollouts

Only a rollout that Verifiers finished cleanly trains, with Verifiers' grade as its reward. Every
other rollout gets reward None and doesn't train; the rest of its group does. Its status says what
went wrong:

```text
how the rollout ended                                 status                     trains
Verifiers dropped its trace                           error_infra                no
a deadline passed, and Verifiers recorded no error    error_timeout              no; validation keeps its grade
the prompt left no room for one output token          truncated_prompt_too_long  no
the harness failed after the policy's first token     error                      no
another error, e.g. a sandbox that never started, a   error_infra                no
  harness that failed to launch, a model call, or a
  deadline that came with an error
the policy never sampled a token                      error_infra                no
stopped at the turn limit                             truncated_max_turns        yes
stopped by another limit, e.g. a reply out of tokens  truncated_length           yes
completed                                             completed                  yes
```

`rollout_status` in [`rollouter.py`](./rollouter.py) picks the row.
`VerifiersRolloutWorker.score_group` sends only the "yes" rows to the rubric (the default rubric
is `RewardFromVerifiers`), so every Verifiers job follows this table. Validation also grades
`error_timeout`, so a timeout counts as a failure in the validation reward. The other rows are
left out of it.

The metrics:

- `rollout/status_rate/<status>`: the share of each status, e.g. `error_timeout` mostly for the
  agent's deadline (the log line's `stop` names the stage), `error_infra` for a sandbox, a harness
  launch or a generator that broke.
- `rollout/unscored_rate`: the share that doesn't train.
- `validation_reward` averages the graded rollouts; `validation_reward_all` averages all of them,
  an ungraded one as 0.

Each failed rollout logs one line. It is this WARNING, with its task, status, stop condition, last
error and phase times, or an ERROR with the traceback if its episode raised:

```text
Verifiers rollout failed: group=12 rollout=3 task=tmax/fix-git status=error_infra stop=sandbox_error
turns=7 dropped=None error=SandboxError: lease expired | boot=41s setup=12s agent=388s scoring=0s
```

`Rollout.logs["errors"]` keeps every error with its traceback. To write each failed rollout, with
these logs, to `rollout_samples.jsonl`, set this in the `Controller.Config`:

```python
rollout_recorder=RolloutSampleRecorder.Config(
    filter=KeepExtremeRewardsFilter.Config(keep_errors=True)
),
```

## Taskset ids

Verifiers loads a taskset, harness or env by id: the name of an installed package. It imports
the package's module ("-" becomes "_") and takes the one class of that kind in its `__all__`:

```text
id "torchtitan-verifiers-terminal-bench"
  import torchtitan_verifiers_terminal_bench
  __all__ = ["TerminalBenchTerminusHarness", "TerminalTaskset"]
  taskset -> TerminalTaskset, harness -> TerminalBenchTerminusHarness
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
