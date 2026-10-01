## Enable Memory Profiling

Enable snapshots in a Python recipe:

```python
from torchtitan.models.llama3.config_registry import llama3_debugmodel


def llama3_debugmodel_memory_snapshot():
    config = llama3_debugmodel()
    config.profiler.enable_memory_snapshot = True
    config.profiler.save_memory_snapshot_folder = "memory_snapshot"
    config.profiler.memory_snapshot_freq = 3
    return config
```

## Profile CUDA with Cuspy

`Profiler.Config.cuspy` switches CUDA activity collection from Kineto to Cuspy, which can also sample GPU performance-monitor counters. It is not exposed on the CLI; set it in a config_registry function:

```python
from torchtitan.observability.profiler import CuspyProfilerConfig

profiler=Profiler.Config(
    enable_profiling=True,
    cuspy=CuspyProfilerConfig(
        enable_event_node_ids=True,
        performance_metrics=["sm__cycles_active.avg"],
    ),
)
```

Cuspy needs a torch build with `torch.profiler.CuspyConfig` and the `cupti-python` package; profiler startup raises a `ValueError` listing whatever is missing. Traces are exported synchronously when each profiling window ends.

`pm_sampling_interval_ms` and `pm_lookback_window_ms` set the PM sampling rate and buffer length. The buffer fills from profiler warmup onward, so for long steps raise `pm_lookback_window_ms` or lower `profiler_warmup`; otherwise the active steps lose their samples. The buffer is host memory and grows with lookback / interval (about 1.1 GiB at 100 s / 2 ms with 7 metrics).

Launch the recipe with:

```bash
MODULE=my_debug_configs CONFIG=llama3_debugmodel_memory_snapshot ./run_train.sh
```

* `profiler.enable_memory_snapshot`: enables memory profiling.
* `profiler.save_memory_snapshot_folder`: configures the folder in which memory snapshots are dumped (`profiling/memory_snapshot` under the dump folder by default).
* `profiler.memory_snapshot_freq`: controls how often regular memory snapshots are taken. When unset, it defaults to `profiler.profile_freq` for backward compatibility.
	+ In case of OOMs, the snapshots will be in `step_{step:012d}_exit` under that folder.
	+ Regular snapshots will be in `step_{step:012d}`.
	+ For example, set `profiler.memory_snapshot_freq = 3` to take a snapshot every three iterations independently of trace profiling.

You can find the saved pickle files in your output folder.
To visualize a snapshot file, you can drag and drop it to <https://pytorch.org/memory_viz>. To learn more details on memory profiling, please visit this [tutorial](https://pytorch.org/blog/understanding-gpu-memory-1/).

## Debugging Config Values

To inspect the complete resolved configuration returned by a recipe, pass
`--print-config` to the training launcher:

```bash
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh --print-config
```

To list the supported operational CLI options:

```bash
python -m torchtitan.train --help
```

Training behavior is configured in the recipe rather than through general
section flags. See [the configuration guide](../torchtitan/config/README.md).

## Fake Backend Debugging

TorchTitan has two fake-process-group modes because they answer different
debugging questions. They intentionally do not share a fallback path:

| Mode | Physical processes | Real communication | Use it to validate |
| --- | --- | --- | --- |
| `fake` | One | None | Configuration, logical mesh construction, rank-local model ownership, tensor shapes, and PyTorch-managed memory for one selected PP rank at SPMD coordinate zero. |
| `real_pp_fake_spmd` | Exactly one per PP rank | PP send/receive only | Pipeline scheduling, real PP buffers and transport, CUDA-graph capture, and rank-local memory while DP, TP, CP, and EP remain logically scaled. |

Use pure fake mode first when a full logical model would require more ranks than
are locally available. Escalate to real-PP/fake-SPMD when the question involves
pipeline transport or pipeline buffer lifetime. A real distributed run is still
required for SPMD communication, numerical, and performance evidence.

### Logical topology

`NGPU` is always the logical world size, not necessarily the number of launched
processes. For a pipeline degree `P`, each pipeline coordinate contains
`NGPU / P` flattened SPMD coordinates. These debugging modes always represent
SPMD coordinate zero, so the selected logical global rank is:

```text
logical_rank = pp_rank * (NGPU / P)
```

The logical world size must be divisible by `P`. The environment contract is:

| Variable | Pure fake | Real PP / fake SPMD | Meaning |
| --- | --- | --- | --- |
| `NGPU` | Required | Required | Complete logical world size used to construct the model mesh. |
| `FAKE_PP_RANK` | Required when `P > 1` | Invalid | Logical PP coordinate represented by the single process. |
| `RANK` | Unused | Set by `torchrun` | Physical rank and PP coordinate in hybrid mode. |
| `WORLD_SIZE` | Unused | Set by `torchrun` | Physical process count, which must equal `P`. |
| `LOCAL_RANK` | Set to `0` by `run_train.sh` | Set by `torchrun` | Physical device index for the process. |
| `MASTER_ADDR`, `MASTER_PORT` | Unused | Set by `torchrun` | Standard rendezvous settings for the real PP group. |
| `COMM_BACKEND` | `run_train.sh` convenience variable | Do not use | The shell launcher recognizes `fake`; hybrid mode is selected with `--comm-backend real_pp_fake_spmd`. |

### Fully fake example

Define a recipe whose pipeline and data-parallel shard degrees are both 2.
This command then constructs logical rank `1 * 2 = 2` of that four-rank job with
PP2 on one physical GPU. It validates that rank's stage, shards, prepared
weights, pipeline metadata, and memory ownership without creating NCCL process
groups or transferring peer data.

```bash
NGPU=4 \
FAKE_PP_RANK=1 \
COMM_BACKEND=fake \
MODULE=my_debug_configs \
CONFIG=llama3_debugmodel_pp2_fsdp2 \
./run_train.sh
```

Without PP, omit `FAKE_PP_RANK`; the represented rank is logical rank zero.
`run_train.sh` limits a pure-fake invocation to one training step by default so
this path remains a diagnostic rather than an accidental benchmark.

### Real PP / fake SPMD example

Using the same PP2/FSDP2 recipe, this command launches two physical processes
for PP2. Each process represents
SPMD coordinate zero of its PP rank in a four-rank logical job. `torchrun`
assigns physical ranks 0 and 1; those ranks are the PP coordinates. TorchTitan
creates one real NCCL PP group across them and fake groups for every other axis.

```bash
NGPU=4 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun \
  --nproc_per_node=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --role=rank \
  --tee=3 \
  -m torchtitan.train \
  --module my_debug_configs \
  --config llama3_debugmodel_pp2_fsdp2 \
  --comm-backend real_pp_fake_spmd
```

The physical world size must equal the PP degree. Do not set `FAKE_PP_RANK`:
the physical `RANK` already supplies that coordinate. All physical ranks use
SPMD coordinate zero and therefore form one PP line through the logical mesh.

### Memory debugging workflow

Keep the model, dtype, batch geometry, parallel degrees, activation
checkpointing, FSDP policy, and CUDA-graph settings identical to the intended
real job. Then:

1. Select the logical PP coordinate whose SPMD-zero ownership is under
   investigation.
2. Record allocator summaries or snapshots after initialization, after complete
   optimizer warmup, during steady-state forward/backward, and after optimizer
   completion.
3. Compare the same logical coordinate and observation point across candidate
   configurations.
4. Re-run with real PP/fake SPMD if PP transport or buffer lifetime matters.
5. Finish with a real distributed run when the claim depends on communication,
   numerics, or performance.

Fake execution represents PyTorch-managed parameters, optimizer state,
prepared quantized weights, activations, gradients, pipeline buffers, and
explicit model arenas such as DistMoE scratch and activation storage. It does
not faithfully represent NCCL communicator allocations, network registration,
collective scratch, SPMD collective latency, or communication overlap.

### Interpreting failures

- A divisibility or coordinate error is a launch-contract failure. Correct the
  logical topology instead of changing model shapes to bypass it.
- A hybrid world-size error means there is not exactly one physical process per
  PP rank.
- Pure fake success followed by hybrid failure isolates the problem to PP
  transport, PP buffer ownership, communicator initialization, or another path
  exercised only by real pipeline communication.
- Hybrid success followed by real-run failure points to a real SPMD collective,
  communication memory, or scale-dependent scheduling behavior.
- Success in either fake mode does not prove loss equivalence, distributed
  correctness, throughput, or communication overlap.

The [fake distributed backend skill](../.claude/skills/fake_distributed_backend/SKILL.md)
contains the operational checklist used for repeatable memory investigations.

## Distributed Breakpoints and LOG_RANK

`run_train.sh` defaults `LOG_RANK` to `0` and passes it to `torchrun` as `--local-ranks-filter`, so only rank 0's stdin/stdout are teed to the console. `torch.distributed.breakpoint(rank=N)` on a filtered rank therefore hangs and never prints a prompt.

To debug rank N, set `LOG_RANK` to N (or a comma-separated list that includes N) before launching. Do not change the default `LOG_RANK` in `run_train.sh`.

```bash
LOG_RANK=1 ./run_train.sh
# or, to keep rank 0 visible as well:
LOG_RANK=0,1 ./run_train.sh
```

## Troubleshooting jobs that timeout

If you encounter jobs that timeout, you'll need to debug them to identify the root cause. To help with this process, we've enabled Flight Recorder, a tool that continuously collects diagnostic information about your jobs.
When a job times out, Flight Recorder automatically generates dump files on every rank containing valuable debugging data. You can find these dump files in the `dump_folder` directory.
To learn how to analyze and diagnose issues using these logs, follow our step-by-step tutorial [link](https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html).



## Reproducibility between Runs

When debugging issues with multi-dimensional parallelism (combinations of FSDP, TP, PP, CP, EP), ensuring reproducible behavior is crucial for isolating and fixing problems. `torchtitan` provides several mechanisms to achieve deterministic training runs. For more information on ensuring reproducibility and managing randomness in PyTorch, you can refer to the official PyTorch documentation on randomness: [PyTorch Randomness Documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html).

### Seed Configuration
Set a consistent random seed in the selected recipe:

```python
config.debug.seed = 42
```

**Seed behavior with parallelism:**
- **Data Parallel (DP/FSDP), Tensor Parallel (TP), Context Parallel (CP):** All ranks use the same seed.
    - Note: For FSDP and TP, DTensor will do special RNG management to make sure a Replicate tensor get the same init across ranks, but a Shard tensor get "random"-like init across ranks.
- **Pipeline Parallel (PP):** Each PP stage gets a different seed to ensure different initialization across layers on different PP ranks.


### Deterministic Mode

Enable deterministic algorithms in the recipe to ensure bit-for-bit reproducibility across runs:

```python
config.debug.deterministic = True
```

**What it does:**
- Forces all CUDA operations to use deterministic algorithms
- Disables CuDNN benchmarking and enables deterministic mode
- Sets deterministic workspace configuration for CuBLAS operations
- **Note:** This will significantly reduce training performance but ensures exact reproducibility

`debug.deterministic_warn_only` is intended only for exploratory debugging and
must not be used for numerical validation.

### Activation Checkpointing Debugging ###

The following debug configs are available for AC.

`preserve_rng_state` - if deterministic output compared to non-checkpointed passes is required, set to true. Results in stashing and restoring the RNG state during each checkpoint, may be slower.

`determinism_check` - A string specifying the determinism function

`debug` - capture ac debug information. Will be slower.

See https://docs.pytorch.org/docs/stable/checkpoint.html for details.

### Seed-Checkpoint-based Reproducibility

For multiple experimental runs with different parallelism configs, we need to use a "seed" checkpoint to ensure model initializations are the same across runs. This is because in `torchtitan/train.py`, the model parameters are sharded first, and then have their weights initialized on each rank separately. As a result, it is not equivalent to initialize the model on one rank and then shard it. Using a seed checkpoint helps different runs load the same model weights from checkpoint -- DCP resharding will make sure the loaded weights are sharded correctly according to the parallelism configs.

#### Creating a Seed Checkpoint

Create a registry configuration with `create_seed_checkpoint=True`, a
non-`None` `checkpointer`, and every parallelism degree set to 1, then run it
on one device.

#### Loading Seed Checkpoints for Debugging

When using seed checkpoints for debugging or validation purposes, you can enable the `load_only` configuration to load checkpoints without saving any new ones during training. This is particularly useful when you only want to verify model correctness or compare different configurations without cluttering your disk:

Set `checkpointer=CheckpointManager.Config(load_only=True)` in the config
registry. The `load_only` setting prevents the training process from saving
any checkpoints, allowing you to:
- Run debugging sessions without generating unwanted checkpoint files
- Compare model behaviors using the same initial weights without checkpoint overhead

**Note**: Using a seed checkpoint will only make sure a model has same initial weights when configs change, but the training process may not be the same even after setting the seed and the `deterministic` mode, e.g. due to tensor shape change, data precision change, usage of randomness in model code, etc.

### Example: Reproducing loss curves with different parallelism configs

A common scenario is when you introduce a new parallelism strategy to the model, you need to ensure that the loss curve remains numerically equivalent to the previous parallelism config, thereby confirming the accuracy of your implementation. To achieve consistent behavior across multiple runs with varying parallelism configurations, it's crucial to make sure dataloader behaves consistently. We need to fix the DP degree (`dp_replicate * dpshard`) to ensure the dataloader operates consistently.

Here's a typical comparison setup (maintaining an overall DP degree of 4):
- Run 1: dp_shard = 4
- Run 2: dp_replicate = 2, dp_shard = 2, TP degree = 2
- Run 3: dp_replicate = 2, dp_shard = 2, CP degree = 2, PP degree = 2

To reproduce loss curves across above runs, you'll need to create a seed checkpoint, and then load the same seed checkpoint for all runs to ensure consistent model initialization on each rank. You might also need to set the `deterministic` mode to ensure consistent training behavior.

We also provided an example of verifying the numerical consistency across parallelism plans configs on Llama 3 in https://github.com/pytorch/torchtitan/blob/main/docs/converging.md.
