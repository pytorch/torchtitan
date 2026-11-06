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

A diagram of one group: the `VerifiersRollouter` docstring in [`rollouter.py`](./rollouter.py).

1. [`data.py`](./data.py) loads the taskset in the controller, which sends each group's task to a
   rollout worker.
2. [`env_server.py`](./env_server.py) runs the Verifiers env server as a local process on the
   controller host. The server runs the agents in the configured runtime, e.g. `SubprocessConfig`
   or `DockerConfig`. With a pool, its pool workers run them. The Terminal-Bench recipe runs 16.
   The server and each of its pool workers first run [`worker_setup.py`](./worker_setup.py). Every
   worker imports that module, so it must not import torch.
3. [`rollouter.py`](./rollouter.py) runs each group in a rollout worker process
   (`VerifiersRolloutWorker`), like TitanRL's own rollout groups. The worker sends the task to the
   env server (a plain dict, msgpack over ZMQ) and turns the trainable agent's trace of each
   rollout into TitanRL `RolloutTurn`s. The controller receives finished groups only.
4. [`generation_server.py`](./generation_server.py) serves Verifiers' model calls (token ids over
   HTTP, from the env server). For each call it gets a generator from `router.reserve`, then calls
   `generate` on it. Each rollout worker runs its own, on localhost. The server also records what
   Verifiers' traces leave out: the policy versions and the generation metrics.

The training dataset is the single source of the environment taskset config.
`VerifiersRollouter.Config` copies it into the environment-server config, so an
experiment must not configure the same taskset a second time.

For CPU-heavy envs, raise the env server's pool (`verifiers_env_server.serve.pool`, e.g.
`vf.StaticPoolConfig(num_workers=16)`). At high rollout rates, also raise the
rollouter's `worker_pool_size`: each rollout worker decodes one trace at a time.

## Add an experiment

Create one package per task, with its own dependency list:

```text
verifiers/
  data.py
  env_server.py
  generation_server.py
  rollouter.py
  worker_setup.py
  my_task/
    __init__.py
    recipe module
    data.py
    requirements.txt
    rollouter.py
```

1. In `my_task/data.py`, define the Verifiers `Taskset` and `TasksetConfig`.
2. In `my_task/rollouter.py`, put that config in
   `VerifiersTaskDataset.Config.verifiers_taskset`. Configure the agent,
   harness, runtime, and worker pool, but leave
   `verifiers_env_server.environment.taskset` unset; `VerifiersRollouter`
   derives it from the training dataset.
3. In `recipe module`, select the rollouter from `Controller.Config`.
4. Place the module under `torchtitan/rl/examples` or pass its fully qualified
   module path to `--module`.

Run the recipe with:

```bash
python -m torchtitan.rl.train \
  --module verifiers.my_task \
  --config <config-name>
```

See [DAPO Math](./dapo_math) for a complete single-turn example and
[Terminal-Bench](./terminal_bench) for a multi-turn agent example. This
integration pins Verifiers 0.4.0 and uses its `verifiers.v1` API.
