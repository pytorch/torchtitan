## Enable Memory Profiling

Launch training job with the following command (or alternatively set configs in your config_registry function)
```
MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh --profiling.enable_memory_snapshot --profiling.save_memory_snapshot_folder memory_snapshot
```
* `--profiling.enable_memory_snapshot`: to enable memory profiling
* `--profiling.save_memory_snapshot_folder`: configures the folder which memory snapshots are dumped into (`./outputs/memory_snapshot/` by default)
	+ In case of OOMs, the snapshots will be in `./outputs/memory_snapshot/iteration_x_exit`.
	+ Regular snapshots (taken every `profiling.profile_freq` iterations) will be in `memory_snapshot/iteration_x`.

You can find the saved pickle files in your output folder.
To visualize a snapshot file, you can drag and drop it to <https://pytorch.org/memory_viz>. To learn more details on memory profiling, please visit this [tutorial](https://pytorch.org/blog/understanding-gpu-memory-1/).

## Overriding Boolean Flags from Config via CLI

Boolean flags are treated as **actions**. To disable a flag from the command line, use the `--no` prefix.

For example, given the following in your config_registry function:

```python
def my_config() -> Trainer.Config:
    return Trainer.Config(
        profiling=ProfilingConfig(enable_memory_snapshot=True),
        # ...
    )
```
You can override it at runtime via CLI with:

```bash
--profiling.no_enable_memory_snapshot
--profiling.no-enable-memory-snapshot  # Equivalent
```

> Note: `--enable_memory_snapshot=False` will **not** work. Use `--no_enable_memory_snapshot` instead.

## Debugging Config Values

To inspect how configuration values are interpreted—including those from config_registry functions and CLI overrides—run the config manager directly:

```bash
python -m torchtitan.config.manager --module llama3 --config llama3_8b [your cli args...]
```

For example,

```bash
python -m torchtitan.config.manager --module llama3 --config llama3_8b --profiling.enable_memory_snapshot
```

To list all available CLI flags and usage:

```bash
python -m torchtitan.config.manager --module llama3 --config llama3_debugmodel --help
```

This will print a structured configuration to `stdout`, allowing you to verify that overrides are being applied correctly.

## Communication Mode (COMM_MODE) for Debugging

The `COMM_MODE` environment variable provides specialized debugging modes that allow you to test and validate your training setup without requiring full multi-GPU distributed execution. This is particularly useful for rapid iteration during development and debugging.

### Available Modes

#### 1. `fake_backend` - Configuration Validation Mode

This mode enables dry-run validation of your configuration, model setup, and rank-0 program logic without actual distributed communication:

```bash
NGPU=32 COMM_MODE="fake_backend" ./run_train.sh
```

**What it does:**
- Uses fake process groups that simulate distributed communication without actual data transfer
- Runs on a single GPU without `torchrun` or NCCL initialization
- Validates configuration parsing, model initialization, and overall training workflow
- Executes only one training step by default

**When to use it:**
- Quick validation of configuration files before launching expensive multi-GPU jobs
- Debugging training and parallelism logic that doesn't require actual communication. Note that No data-dependent logic should be validated with "fake_backend".

**Example use case:**
```bash
# Validate a 128-GPU configuration on a single GPU
NGPU=128 COMM_MODE="fake_backend" MODULE=llama3 CONFIG=llama3_70b ./run_train.sh
```

#### 2. `local_tensor` - Single-GPU Distributed Simulation

This mode simulates the full distributed training workflow on a single GPU by executing all communication and computation locally:

```bash
NGPU=32 COMM_MODE="local_tensor" ./run_train.sh
```

**What it does:**
- Simulates multi-GPU behavior on a single shared GPU
- Executes all collectives (all-reduce, all-gather, etc.) locally without network communication
- Maintains the same code paths as distributed training for accurate debugging
- Runs only one training step by default

**When to use it:**
- Debugging distributed training logic (FSDP, TP, PP, CP, EP) with data dependencies without multi-GPU setup. Note that local tensor doesn't support FSDP2 but should support SimpleFSDP.
- Verifying correctness of parallelism strategies locally
- Testing gradient synchronization and communication patterns
- Reproducing distributed training bugs in a simplified environment

**Example use case:**
```bash
# Debug 8-way TP + 2-way FSDP on a single GPU
NGPU=16 COMM_MODE="local_tensor" ./run_train.sh \
  --parallelism.tensor_parallel_degree 8 \
  --parallelism.data_parallel_shard_degree 2
```

### Limitations

- **Performance testing**: Neither mode provides accurate performance metrics; use actual distributed runs for benchmarking
- **Memory requirement**: Local tensor runs require more memory on a single GPU than the actual distributed runs

## Troubleshooting jobs that timeout

If you encounter jobs that timeout, you'll need to debug them to identify the root cause. To help with this process, we've enabled Flight Recorder, a tool that continuously collects diagnostic information about your jobs.
When a job times out, Flight Recorder automatically generates dump files on every rank containing valuable debugging data. You can find these dump files in the `dump_folder` directory.
To learn how to analyze and diagnose issues using these logs, follow our step-by-step tutorial [link](https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html).



## Reproducibility between Runs

When debugging issues with multi-dimensional parallelism (combinations of FSDP, TP, PP, CP, EP), ensuring reproducible behavior is crucial for isolating and fixing problems. `torchtitan` provides several mechanisms to achieve deterministic training runs. For more information on ensuring reproducibility and managing randomness in PyTorch, you can refer to the official PyTorch documentation on randomness: [PyTorch Randomness Documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html).

### Seed Configuration
Set consistent random seeds across all parallelism dimensions:

```bash
./run_train.sh --debug.seed 42
```

**Seed behavior with parallelism:**
- **Data Parallel (DP/FSDP), Tensor Parallel (TP), Context Parallel (CP):** All ranks use the same seed.
    - Note: For FSDP and TP, DTensor will do special RNG management to make sure a Replicate tensor get the same init across ranks, but a Shard tensor get "random"-like init across ranks.
- **Pipeline Parallel (PP):** Each PP stage gets a different seed to ensure different initialization across layers on different PP ranks.


### Deterministic Mode

Enable deterministic algorithms to ensure bit-for-bit reproducibility across runs:

```bash
./run_train.sh --debug.deterministic
```

**What it does:**
- Forces all CUDA operations to use deterministic algorithms
- Disables CuDNN benchmarking and enables deterministic mode
- Sets deterministic workspace configuration for CuBLAS operations
- **Note:** This will significantly reduce training performance but ensures exact reproducibility

Use `--debug.deterministic_warn_only` to only warn about (not stop running) kernel without deterministic implementation.

### Activation Checkipointing Debugging ###

The following debug configs are available for AC.

`preserve_rng_state` - if deterministic output compared to non-checkpointed passes is required, set to true. Results in stashing and restoring the RNG state during each checkpoint, may be slower.

`determinism_check` - A string specifying the determinism function

`debug` - capture ac debug information. Will be slower.

See https://docs.pytorch.org/docs/stable/checkpoint.html for details.

### Seed-Checkpoint-based Reproducibility

For multiple experimental runs with different parallelism configs, we need to use a "seed" checkpoint to ensure model initializations are the same across runs. This is because in `torchtitan/train.py`, the model parameters are sharded first, and then have their weights initialized on each rank separately. As a result, it is not equivalent to initialize the model on one rank and then shard it. Using a seed checkpoint helps different runs load the same model weights from checkpoint -- DCP resharding will make sure the loaded weights are sharded correctly according to the parallelism configs.

#### Creating a Seed Checkpoint

```bash
NGPU=1 MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh --checkpoint.enable --checkpoint.create_seed_checkpoint --parallelism.data_parallel_replicate_degree 1 --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 1 --parallelism.pipeline_parallel_degree 1 --parallelism.context_parallel_degree 1 --parallelism.expert_parallel_degree 1
```

#### Loading Seed Checkpoints for Debugging

When using seed checkpoints for debugging or validation purposes, you can enable the `load_only` configuration to load checkpoints without saving any new ones during training. This is particularly useful when you only want to verify model correctness or compare different configurations without cluttering your disk:

```bash
MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh --checkpoint.enable --checkpoint.load_only
```

The `--checkpoint.load_only` flag prevents the training process from saving any checkpoints, allowing you to:
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
