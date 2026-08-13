# TerminalBench through Verifiers

TorchTitan does not provision a sandbox or run a coding harness itself in this
example. The controller launches a Verifiers environment server from
`controller.verifiers_env_server.config_path` and uses the same service boundary as
Prime-RL:

```text
TorchTitan Controller
  -> starts and owns Verifiers EnvServer
  -> passes its ZMQ address to Rollouter

Rollouter._run_single_rollout
  -- ZMQ run(task_data, model, sampling) --> Verifiers EnvServer

Verifiers environment worker
  -> rebuilds the TerminalBench task from dispatched TaskData
  -> provisions the configured runtime and sandbox
  -> launches the configured harness in that runtime
  -> runs tools and the hidden verifier
  -> returns Episode[Trace] over ZMQ

Verifiers TrainClient
  -- HTTP POST /inference/v1/generate --> TorchTitan model adapter
  -> TorchTitan GenerateFn -> generator actor
  <- exact completion token IDs and logprobs
```

The recipe configures the common `VerifiersRollouter` interface and
runs episodes through EnvServer. Search-R1 and the other native rollouters use
the base `Rollouter` implementation, which drives `TokenEnv` and `MessageEnv`.

## Environment server

Install the TorchTitan and EnvServer dependencies:

```bash
pip install -r torchtitan/experiments/rl/examples/terminal_bench/requirements.txt
```

The managed EnvServer is a child process using this same Python environment.

The example includes `verifiers_env.toml` with this service configuration:

```toml
[env.taskset]
id = "harbor"
dataset = "terminal-bench/terminal-bench-2"

[env.agent]
runtime = { type = "docker", allow = [], block = ["*"] }

[env.agent.harness]
id = "bash"

[serve.pool]
type = "static"
num_workers = 1
```

The Docker runtime configuration makes the agent execution phase framework-only:
the harness can reach the Verifiers model/interception endpoint, but the agent
cannot access the public internet. A task's network policy cannot widen this
base policy.

Verifiers 0.3 still opens public egress during its trusted task and harness setup
phase. Its Bash harness also installs `uv` and resolves PEP 723 dependencies in
that phase. Therefore, an all-lifecycle no-public-egress deployment additionally
requires an offline harness bootstrap and a Verifiers runtime that keeps setup
egress restricted. The production TerminalBench run should not rely on a host
proxy or temporarily relax this policy.

The stock `terminal-bench/terminal-bench-2` package is not offline-compatible:
all 89 tasks request internet access, and their verifier scripts install test
dependencies online. Verifiers and Prime-RL do not provide a separate
harness-ready image. Keep public egress blocked by default, and do not treat a
failed offline verifier as a valid zero reward. A real offline run requires
prebuilt task images, offline harness dependencies, and offline verifier tests.

The example starts one environment worker. Increase `serve.pool.num_workers`
when a larger environment-worker pool is useful.

`rl_grpo_qwen3_8b_terminal_bench` points at this file. The controller starts the
service automatically on a free local port and stops it during shutdown. The
current implementation starts the EnvServer on the controller host. A future
launcher may move the EnvServer to a separate host while retaining the same
ZMQ episode protocol.

The `harbor` entry above is Verifiers' TerminalBench taskset adapter. Under the
Verifiers 0.3 contract, TorchTitan's dataset adapter loads and selects Harbor
tasks, then sends each task's serialized data to the stateless EnvServer. Harbor
is not a TorchTitan sandbox API.

## TorchTitan connectivity

`verifiers_env_server.bind_address` selects the Verifiers ZMQ listen address; the default
`tcp://127.0.0.1:0` chooses a free local port. Set
`rollouter.model_adapter_base_url` to an HTTP address reachable from the
Verifiers worker processes.

The two protocols have different granularity:

- ZMQ carries one complete environment episode in each request and response.
- HTTP carries every individual harness model turn back to TorchTitan.

Verifiers owns the harness/interception/runtime lifecycle and returns its typed
message-graph trace. TorchTitan converts each trainable trace branch into the
existing `RolloutTurn` contract, preserving token IDs, loss-mask boundaries,
logprobs, and policy versions.

## Run

Download the Qwen3-8B assets expected by the recipe:

```bash
python scripts/download_hf_assets.py \
  --repo_id Qwen/Qwen3-8B \
  --local_dir torchtitan/experiments/rl/example_checkpoint/Qwen3-8B \
  --all
```

Start the TorchTitan controller, trainer, generator, and managed EnvServer with
one command:

```bash
python -m torchtitan.experiments.rl.train \
  --module terminal_bench \
  --config rl_grpo_qwen3_8b_terminal_bench \
  --dump-folder outputs/rl/qwen3_8b_terminal_bench
```

The default local topology uses four trainer GPUs and two generator GPUs.
