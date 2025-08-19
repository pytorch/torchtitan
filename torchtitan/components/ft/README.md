# TorchFT - Fault Tolerance for TorchTitan

TorchFT provides fault tolerance capabilities for distributed training in TorchTitan, enabling resilient training across multiple replica groups with automatic failure recovery.

This component integrates with [TorchFT](https://github.com/pytorch/torchft), PyTorch's fault tolerance library. **Prerequisites from the TorchFT repository must be installed before using this functionality.**

## Fault Tolerance Configuration Options

The fault tolerance system can be configured using the following options:

- `--fault_tolerance.enable`: Enable fault tolerance mode
- `--fault_tolerance.group_size`: Number of replicas in the fault tolerance group
- `--fault_tolerance.replica_id`: Unique identifier for this replica within the group

For complete configuration options, see [job_config.py](../../config/job_config.py).

[Optional] Only for semi-synchronous training:

- `--fault_tolerance.sync_steps`: The number of training steps before synchronization.
- `--fault_tolerance.semi_sync_method`: Synchronization method (e.g., "local_sgd", "diloco")

For more semi-synchronouse configuration options, see [ft/config/job_config.py](config/job_config.py).

## Starting the Lighthouse Service

The lighthouse service coordinates fault tolerance across replica groups. Start it with:

```bash
# Requires 2 replica groups to join
RUST_LOGS=debug RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 2 --quorum_tick_ms 100 --join_timeout_ms 10000

# For single replica group (development)
RUST_LOGS=debug RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

## Examples: Running with Two Replica Groups

### Example 1: HSDP

Each replica group has 4 GPUs (assuming this host has 8 GPUs total, we use `CUDA_VISIBLE_DEVICES` to simulate two "hosts" for single host development experience). Each replica group has 4 GPUs which are sharded with FSDP, and they synchronize per-step for fault tolerant HSDP.

#### Replica Group 0
```bash
RUST_LOGS=debug TORCHFT_LIGHTHOUSE=http://<hostname>:29510 TORCHFT_MANAGER_PORT=29520 REPLICA_GROUP_ID=0 CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 ./run_train.sh --parallelism.data_parallel_shard_degree=4 --fault_tolerance.enable --fault_tolerance.group_size=2 --fault_tolerance.replica_id=0
```

#### Replica Group 1
```bash
TORCHFT_LIGHTHOUSE=http://<hostname>:29510 TORCHFT_MANAGER_PORT=29522 REPLICA_GROUP_ID=1 CUDA_VISIBLE_DEVICES=4,5,6,7 NGPU=4 ./run_train.sh --parallelism.data_parallel_shard_degree=4 --fault_tolerance.enable --fault_tolerance.group_size=2 --fault_tolerance.replica_id=1
```

### Example 2: With Semi-synchronous Training

TorchFT provides algorithms that do not require per-step synchronization and
the replica groups can sychronize weights every N steps.

**Note on Batch Sizes**: For DiLoCo, there's an important distinction in batch size terminology:

The `--training.global_batch_size` parameter refers to global batch size that will be split across all replica groups.

- **Global batch size**: The total batch size across all DiLoCo islands/replica groups
- **Inner global batch size**: The batch size within each individual DiLoCo island. This is determined by dividing global batch size by number of replica groups.

#### Replica Group 0
```bash
RUST_LOGS=debug TORCHFT_LIGHTHOUSE=http://<hostname>:29510 TORCHFT_MANAGER_PORT=29520 REPLICA_GROUP_ID=0 CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 ./run_train.sh --parallelism.data_parallel_shard_degree=4 --fault_tolerance.enable --fault_tolerance.group_size=2 --fault_tolerance.replica_id=0 --fault_tolerance.semi_sync_method="diloco"
```

#### Replica Group 1
```bash
TORCHFT_LIGHTHOUSE=http://<hostname>:29510 TORCHFT_MANAGER_PORT=29522 REPLICA_GROUP_ID=1 CUDA_VISIBLE_DEVICES=4,5,6,7 NGPU=4 ./run_train.sh --parallelism.data_parallel_shard_degree=4 --fault_tolerance.enable --fault_tolerance.group_size=2 --fault_tolerance.replica_id=1 --fault_tolerance.semi_sync_method="diloco"
```

## Environment Variables

- `TORCHFT_LIGHTHOUSE`: URL of the lighthouse service
- `TORCHFT_MANAGER_PORT`: Port for the TorchFT manager
- `REPLICA_GROUP_ID`: Identifier for the replica group
- `RUST_LOGS`: Logging level for Rust components
- `RUST_BACKTRACE`: Enable backtrace for debugging
