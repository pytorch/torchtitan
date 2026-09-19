# Environment variables

Most TorchTitan settings are config fields. This page lists environment
variables read or written by the root launcher, shared runtime, and user-facing
integrations. Test, CI, dependency, and integration-internal variables are
omitted.

## Launcher inputs

[`run_train.sh`](../run_train.sh) reads these variables.

| Name | Default | Effect |
| --- | --- | --- |
| `MODULE` | `llama3` | Passed to Python as `--module`. |
| `CONFIG` | `llama3_debugmodel` | Passed to Python as `--config`. |
| `NGPU` | `8` | Normal path: passed to `torchrun --nproc_per_node`. `fake_backend` and `local_tensor`: parsed as the integer world size. |
| `LOG_RANK` | `0` | Exported, passed to `torchrun --local-ranks-filter`, and read by the pipeline-loss visibility check. |
| `COMM_MODE` | Empty | When nonempty, bypasses `torchrun`, sets `LOCAL_RANK=0`, passes `--comm.mode`, and forces `--training.steps 1`. |
| `TORCHFT_LIGHTHOUSE` | `http://localhost:29510` | Normal path: passed in the `torchrun` environment. |

`MODEL` is not a `run_train.sh` input. The separate
[Flux inference launcher](../torchtitan/models/flux/run_infer.sh) reads it and
defaults it to `flux`.

`CUDA_VISIBLE_DEVICES` controls which GPUs CUDA exposes. `NGPU` separately sets
the worker count or simulated world size.

[Debugging](debugging.md#communication-mode-comm_mode-for-debugging) defines the
`COMM_MODE` values and limitations. The
[TorchFT README](../torchtitan/experiments/torchft/README.md) covers lighthouse
setup.

## Integration and experiment variables

[`WandBLogger`](../torchtitan/components/metrics.py) passes each W&B value
unchanged to `wandb.init`.

| Name | If absent | Reader or writer | Effect |
| --- | --- | --- | --- |
| `WANDB_TEAM` | `None` | W&B | Passed as `entity`. |
| `WANDB_PROJECT` | `torchtitan` | W&B | Passed as `project`. |
| `WANDB_RUN_NAME` | `None` | W&B | Passed as `name`. |
| `WANDB_RUN_ID` | `None` | W&B | Passed as `id`. |
| `WANDB_RUN_NOTES` | `None` | W&B | Passed as `notes`. |
| `WANDB_RUN_TAGS` | `None` | W&B | Passed as `tags`. |
| `WANDB_RUN_GROUP` | `None` | W&B | Passed as `group`. |
| `WANDB_RUN_JOB_TYPE` | `None` | W&B | Passed as `job_type`. |
| `WANDB_RESUME_FROM` | `None` | W&B | Passed as `resume_from`. |
| `WANDB_FORK_FROM` | `None` | W&B | Passed as `fork_from`. |
| `TITAN_STRUCT_LOGGER_HANDLERS` | Uses the default JSONL handler | [Structured logging](../torchtitan/observability/structured_logger/structured_logging.py) | A nonempty comma-separated list of fully qualified factory paths replaces the default handler. |
| `TORCHTITAN_SKIP_FINGERPRINT_CHECK` | A mismatch between two present fingerprints raises `ValueError` | [Graph Trainer](../torchtitan/experiments/graph_trainer/precompile.py) | Only `1` changes the error to a warning. |
| `PYTORCH_CUDA_ALLOC_CONF` | RL sets `expandable_segments:True` | [RL package import](../torchtitan/experiments/rl/__init__.py) | An existing value is preserved. If `torch` is already imported, RL warns that a newly set value may not affect the allocator. |

[Metrics](metrics.md) covers enabling W&B. The structured logger documents the
[handler factory format](../torchtitan/observability/structured_logger/README.md#custom-handlers).
RL has separate
[`WANDB_PROJECT` precedence](../torchtitan/experiments/rl/observability/metrics/README.md#backends)
and defaults it to `titan_rl`.

## Variables supplied by `torchrun`

The normal launcher does not require users to set these variables.

| Name | Main TorchTitan use |
| --- | --- |
| `LOCAL_RANK` | Required by the trainer as the local device index. |
| `RANK` | Used as the default structured-log rank. |
| `WORLD_SIZE` | Used to require one worker when creating a seed checkpoint. |

`COMM_MODE` bypasses `torchrun`. It sets `LOCAL_RANK=0`; it does not set `RANK`
or `WORLD_SIZE`. The
[`torchrun` reference](https://docs.pytorch.org/docs/stable/elastic/run.html#environment-variables)
defines the full launcher contract.

## Variables written by TorchTitan

The [root launcher](../run_train.sh),
[logging setup](../torchtitan/tools/logging.py), and
[distributed utilities](../torchtitan/distributed/utils.py) write these values.

| Name | When | Value |
| --- | --- | --- |
| `PYTORCH_ALLOC_CONF` | Normal `run_train.sh` path | `expandable_segments:True`; replaces an inherited value for the child process. |
| `KINETO_LOG_LEVEL` | Logger initialization | `5`; replaces an inherited value. |
| `CUBLAS_WORKSPACE_CONFIG` | `debug.deterministic=true` | `:4096:8`; replaces an inherited value. |
| `PYTHONHASHSEED` | A seed is applied | Effective seed modulo `2**32`; replaces an inherited value. |

Python reads `PYTHONHASHSEED` when the interpreter starts. TorchTitan writes it
after startup, so the write does not change hash randomization in the running
worker. See the
[`PYTHONHASHSEED` reference](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED).

When `init_distributed` initializes a normal process group, it replaces:

| Name | Value |
| --- | --- |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | `3` |
| `TORCH_FR_BUFFER_SIZE` | `comm.trace_buf_size` |

When `comm.trace_buf_size > 0`, it also replaces:

| Name | Value |
| --- | --- |
| `TORCH_NCCL_DUMP_ON_TIMEOUT` | `1` |
| `TORCH_FR_DUMP_TEMP_FILE` | `<dump_folder>/<comm.save_traces_folder>/<comm.save_traces_file_prefix>` |

When `comm.trace_buf_size <= 0`, existing values of
`TORCH_NCCL_DUMP_ON_TIMEOUT` and `TORCH_FR_DUMP_TEMP_FILE` are unchanged.

The first `set_batch_invariance(True)` call that enables the mode replaces:

```text
NCCL_LAUNCH_MODE=GROUP
NCCL_COLLNET_ENABLE=0
NCCL_NVLS_ENABLE=0
NCCL_P2P_NET_DISABLE=1
NCCL_MIN_NCHANNELS=1
NCCL_MAX_NCHANNELS=1
NCCL_PROTO=Simple
NCCL_ALGO=allreduce:tree
NCCL_NTHREADS=1
NCCL_SOCKET_NTHREADS=1
```

See [RL bitwise parity](../torchtitan/experiments/rl/docs/bitwise_parity.md) for
batch-invariant setup and constraints.

## Compiler settings

No TorchTitan code reads `TORCHINDUCTOR_*` environment variables. Core compiler
controls are config fields:

| Field | Default |
| --- | --- |
| `compile.enable` | `false` |
| `compile.components` | `["model", "loss"]` |
| `compile.backend` | `"inductor"` |

See [`CompileConfig`](../torchtitan/config/configs.py) and the separate
[Graph Trainer compiler options](../torchtitan/experiments/graph_trainer/README.md#compiler-optimizations).
PyTorch owns compiler environment variables; see its
[`torch.compile` troubleshooting guide](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html).

## Upstream variables

TorchTitan does not define the general CUDA, NCCL, or PyTorch environment
variable contracts. Use the upstream references:

- [PyTorch environment variables](https://docs.pytorch.org/docs/stable/torch_environment_variables.html)
- [CUDA environment variables](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)
- [ProcessGroupNCCL environment variables](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html)

`TORCH_NCCL_AVOID_RECORD_STREAMS` is a PyTorch variable. PyTorch avoids
record-stream synchronization by default; `0` selects the legacy behavior.
