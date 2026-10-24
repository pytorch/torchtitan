# Enabling Fault Tolerance with TorchFT in TorchTitan

## Why Use TorchFT with TorchTitan?

TorchFT is designed to provide fault tolerance when training with replicated weights, such as in DDP or HSDP. By enabling TorchFT in TorchTitan, we can ensure that our training process can continue even if some machines fail. For more information on TorchFT, please refer to the [TorchFT repository](https://github.com/pytorch/torchft/).

**Note:** This is an ongoing development effort, and everything is subject to change.

## Prerequisites for Using TorchFT with TorchTitan

Before using TorchFT with TorchTitan, you need to install TorchFT by following the instructions in the [TorchFT README](https://github.com/pytorch/torchft/blob/main/README.md) to install TorchFT.

Alternatively, you can install TorchFT with `pip install torchft-nightly`.

## Configuring TorchTitan for Using TorchFT

When using TorchFT with TorchTitan, you need to launch multiple replica groups, each of which is a separate TorchTitan instance. Each replica group is responsible for maintaining a copy of the model weights. In case of a failure, the other replica groups can continue training without losing weight information.

For example, if you want to run HSDP on a single machine with eight GPUs, where weights are sharded within four GPUs with two replica groups (2, 4 device mesh), set `data_parallel_replicate_degree=2` and `data_parallel_shard_degree=4` in the selected recipe. However, to utilize TorchFT, you will need to launch two TorchTitan instances, each managing four GPUs and communicating with each other through TorchFT.

## Example Configuration

Let's consider an example where we want to run HSDP on a single machine with eight GPUs, where weights are sharded within four GPUs with two replica groups (2, 4 device mesh). Without using TorchFT, set the corresponding fields on `config.parallelism`. However, in the event of a trainer failure (emulating a real-world machine failure), the entire training process would need to stop and recover from the last checkpoint. This can lead to significant downtime and wasted resources.

With TorchFT, we can tolerate one replica group failure, ensuring that the training process continues uninterrupted. To achieve this, we can launch two TorchTitan instances, each managing four GPUs and communicating with each other through TorchFT. This setup allows for seamless fault tolerance and minimizes the impact of individual trainer failures.
### Launching TorchFT with TorchTitan (Example 1)

To launch TorchFT with TorchTitan, you need to execute the following three commands in different shell sessions:

1. Launch TorchFT lighthouse:

```bash
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

2. Launch the first TorchTitan instance:

```bash
NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 MODULE=torchtitan_recipes.torchft.llama3 CONFIG=llama3_torchft_replica_0 ./run_train.sh
```
3. Launch the second TorchTitan instance:

```bash
NGPU=4 CUDA_VISIBLE_DEVICES=4,5,6,7 MODULE=torchtitan_recipes.torchft.llama3 CONFIG=llama3_torchft_replica_1 ./run_train.sh
```

### Explanation

* We limit the visibility of GPUs for each TorchTitan instance using environment variables `NGPU` and `CUDA_VISIBLE_DEVICES`, as we are running on a single machine. In reality, each TorchTitan instance will not share machines, so these variables are not required.
* `fault_tolerance.enable` in each recipe enables TorchFT functionality.
* `fault_tolerance.group_size=2` tells TorchTitan that there are two replica groups.
* `fault_tolerance.replica_id` identifies each instance.
* Note that the alive replica group with the smallest replica ID will perform checkpointing saving.

In a real-world scenario, `torchft_lighthouse` would likely be on a different machine. The `TORCHFT_LIGHTHOUSE` environment variable is used to tell TorchFT how to communicate with `torchft_lighthouse`. The default value is `http://localhost:29510`.

### Using semi-synchronous training (Example 2)

TorchFT provides algorithms that do not require per-step synchronization and
the replica groups can synchronize weights every N steps.

**Note on Token Counts**: For DiLoCo, there is an important distinction in token-count terminology:

`training.num_tokens_per_train_step` controls the global token budget. The
number of forward/backward iterations accumulated before each optimizer step is
derived from that budget.

- **Tokens per step**: The total token count across all DiLoCo islands/replica groups
- **Inner tokens per step**: The token count within each individual DiLoCo island. This is determined by dividing tokens per step by the number of replica groups.

#### Replica Group 0
```bash
MODULE=torchtitan_recipes.torchft.llama3 CONFIG=llama3_torchft_replica_0 CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 ./run_train.sh
```

#### Replica Group 1
```bash
MODULE=torchtitan_recipes.torchft.llama3 CONFIG=llama3_torchft_replica_1 CUDA_VISIBLE_DEVICES=4,5,6,7 NGPU=4 ./run_train.sh
```

## Fault Tolerance Configuration Options

Configure fault tolerance in the selected Python recipe.

For all training modes:

- `--fault_tolerance.no-use_async_quorum`: Run the quorum synchronously, so the
  step waits for the quorum, including any state export or load it performs for
  healing, before the forward and backward passes. By default the quorum runs in
  the background of the step.

[Optional] Only for semi-synchronous training:

- `config.fault_tolerance.sync_steps`: The number of training steps before synchronization.
- `config.fault_tolerance.semi_sync_method`: Synchronization method (e.g., "local_sgd", "diloco")

For more semi-synchronous configuration options, see [config/job_config.py](config/job_config.py).

### Timeout configuration

For TorchFT to heal correctly, the timeouts below must form a strict hierarchy:
each layer must be longer than the one above it, so that lower-level timeouts
trigger recovery before higher-level timeouts kill the process.

| Parameter | Recommended value | Purpose |
| --- | --- | --- |
| `fault_tolerance.process_group_timeout_ms` | 60s | Shortest timeout — detects a stalled inter-replica collective (e.g. an allreduce hanging because a peer died). TorchFT catches it and marks the step as errored. Must be shorter than `comm.train_timeout_seconds`: TorchFT's FSDP all-reduce hook runs inline within FSDP ops, blocking the default process-group collectives. |
| `comm.train_timeout_seconds` | 120s | Default process-group timeout for intra-replica FSDP collectives during steady-state training. Must exceed `process_group_timeout_ms` so the TorchFT process group can time out and recover before the watchdog kills the process. |
| `comm.init_timeout_seconds` | 180s | Same as above, for initialization, with extra buffer for lazy init, data loading, and JIT compilation. |
| `TORCH_NCCL_NONBLOCKING_TIMEOUT` | 200s | NCCL's internal C++ timeout for non-blocking process-group ops. Must exceed the Python-level timeouts so PyTorch's timeout fires first, allowing graceful handling before NCCL aborts at the C++ level. |
| `TORCHFT_TIMEOUT_SEC` | 180s | Overall deadline for `ft.Manager` operations (quorum, should_commit, ...). Must be shorter than `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`. |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 900s | Must exceed all of the above so NCCL doesn't kill the process while TorchFT is still handling a timeout. Set via environment variable. |
| `TORCHFT_QUORUM_TIMEOUT_SEC` | 3600s | Longest timeout — how long a replica waits to join a quorum. Must cover node provisioning time. |

Note: `TORCH_NCCL_NONBLOCKING_TIMEOUT` only needs to beat the Python-level
timeouts above it, so its ordering relative to `TORCHFT_TIMEOUT_SEC` can be relaxed.

## Environment Variables

- `TORCHFT_LIGHTHOUSE`: URL of the lighthouse service
- `TORCHFT_MANAGER_PORT`: Port for the TorchFT manager
- `REPLICA_GROUP_ID`: Identifier for the replica group
- `RUST_LOGS`: Logging level for Rust components
- `RUST_BACKTRACE`: Enable backtrace for debugging
