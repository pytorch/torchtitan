# Verifiers integration

This package contains the optional integration between TitanRL and
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers). TitanRL recipes
outside this directory do not import or require Verifiers.

The package is organized into reusable integration code and independent
experiments:

- [`components/`](./components) adapts Verifiers tasksets, environment servers,
  model clients, and rollout traces to TitanRL interfaces.
- [`dapo_math/`](./dapo_math) is a concrete single-turn math experiment.
- Future integrations should use another experiment package, for example
  `swe_bench/`, rather than adding task-specific code to `components/`.

## Integration flow

The shared components bridge Verifiers episode execution to TitanRL without
changing TitanRL's controller, generator, or training interfaces:

1. `components/dataset.py` loads one Verifiers taskset and converts its task
   data into a serializable TitanRL sample stream.
   Configure it with the taskset's concrete `TasksetConfig` subclass, such as
   `VerifiersMathTasksetConfig` or `SWEBenchVerifiedConfig`.
2. `components/env_server.py` starts the Verifiers environment service in a
   local process.
   `SingleAgentEnvConfig` combines the taskset with an `AgentConfig`. The agent
   selects a runtime such as `SubprocessConfig`, `DockerConfig`, or
   `PrimeConfig`, a harness config such as `NullHarnessConfig`,
   `RLMHarnessConfig`, or `CodexHarnessConfig`, and optional rollout limits.
   `ServeConfig` selects a `StaticPoolConfig` or `ElasticPoolConfig` and the ZMQ
   bind address.
3. `components/model_adapter.py` runs a `GenerationServer` that exposes TitanRL's generator callback through
   the HTTP model protocol expected by Verifiers and retains TorchTitan-only
   policy metadata.
4. `components/rollouter.py` sends samples to the environment service and
   converts the resulting Verifiers traces into scored TitanRL rollout turns.
   It constructs Verifiers' `TrainClientConfig` and per-request
   `SamplingConfig` from TitanRL's renderer, model, and generator settings;
   experiments do not construct these two configs directly.

The taskset config is used both by `VerifiersTaskDataset.Config` to load samples
and by `VerifiersEnvServer.Config` to reconstruct those samples. Their taskset
IDs and schemas must therefore agree.

## Add a Verifiers experiment

Create one subpackage per experiment. For example, a SWE-bench integration
would use this layout:

```text
verifiers/
  components/
  swe_bench/
    __init__.py
    config_registry.py
    requirements.txt
    rollouter.py
```

SWE-bench Verified is already published as the Verifiers taskset
`swebench-verified`, so this example does not need a local `taskset.py`. Add the
taskset package to `swe_bench/requirements.txt`:

```text
-r ../requirements.txt
swebench-verified @ git+https://github.com/PrimeIntellect-ai/prime-envs.git#subdirectory=environments/swe/swebench_verified
```

In `swe_bench/rollouter.py`, construct Verifiers' typed configuration classes.
Use the same taskset ID for the TitanRL dataset and the environment server:

```python
import verifiers.v1 as vf
from swebench_verified.taskset import SWEBenchVerifiedConfig
from verifiers.v1.harnesses.rlm import RLMHarnessConfig


_TASKSET_ID = "swebench-verified"

swebench_dataset = VerifiersTaskDataset.Config(
    taskset=SWEBenchVerifiedConfig(id=_TASKSET_ID),
    seed=42,
)

swebench_env_server = VerifiersEnvServer.Config(
    environment=vf.SingleAgentEnvConfig(
        taskset=SWEBenchVerifiedConfig(id=_TASKSET_ID),
        agent=vf.AgentConfig(
            runtime=vf.DockerConfig(),
            harness=RLMHarnessConfig(
                id="rlm",
                builtin_skills=["edit", "search"],
            ),
        ),
    ),
    serve=vf.ServeConfig(
        pool=vf.StaticPoolConfig(num_workers=1),
        address="tcp://127.0.0.1:0",
    ),
)
```

`VerifiersTaskDataset` loads the 500 SWE-bench Verified tasks and sends their
serialized task data through TitanRL. The environment server reconstructs each
task, starts the task's declared container image, runs the RLM coding harness in
that container, and invokes the packaged Harbor verifier after the harness
finishes. Docker must be available on the host; the `subprocess` runtime cannot
honor the per-task container image or provide suitable isolation for repository
tasks.

Use `DockerConfig` for local testing. For production, use `PrimeConfig` or
another managed remote sandbox. Do not use `SubprocessConfig`: it cannot honor
each task's container image and runs untrusted commands directly on the host.
Docker provides practical isolation, while a managed VM sandbox provides a
stronger security boundary.

Use these objects as the dataset and `env_server` fields of a
`VerifiersRollouter.Config` subclass, then reference that rollouter from the
experiment's `Controller.Config` in `config_registry.py`. The current integration
manages one Verifiers taskset schema per rollouter, so its dataset and
environment-server taskset IDs must agree. Define a local taskset beside the
rollouter only when an experiment needs to combine datasets or customize task
loading, setup, or scoring; export exactly one Verifiers `Taskset` subclass from
that module and use its dotted module path as the ID.

Finally, register `verifiers.swe_bench` in
`torchtitan/experiments/__init__.py`. Its recipe can then be invoked as:

```bash
python -m torchtitan.experiments.rl.train \
  --module verifiers.swe_bench \
  --config <swe-bench-config-name>
```

The existing DAPO Math package is invoked with:

```bash
python -m torchtitan.experiments.rl.train \
  --module verifiers.dapo_math \
  --config rl_dapo_qwen3_4b_verifiers_8k
```

See the [DAPO Math example](./dapo_math) for dependency installation and a
complete taskset implementation.

This integration pins Verifiers release 0.3.1 and uses its `verifiers.v1` API
namespace. The release number and API namespace are separate.
