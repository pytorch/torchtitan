# Enabling Fault Tolerance with TorchFT in TorchTitan

## Why Use TorchFT with TorchTitan?

TorchFT is designed to provide fault tolerance when training with replicated weights, such as in DDP or HSDP. By enabling TorchFT in TorchTitan, we can ensure that our training process can continue even if some machines fail. For more information on TorchFT, please refer to the [TorchFT repository](https://github.com/pytorch/torchft/).

**Note:** This is an ongoing development effort, and everything is subject to change.

## Prerequisites for Using TorchFT with TorchTitan

Before using TorchFT with TorchTitan, you need to install TorchFT by following the instructions in the [TorchFT README](https://github.com/pytorch/torchft/blob/main/README.md) to install TorchFT.

## Configuring TorchTitan for Using TorchFT

When using TorchFT with TorchTitan, you need to launch multiple replica groups, each of which is a separate TorchTitan instance. Each replica group is responsible for maintaining a copy of the model weights. In case of a failure, the other replica groups can continue training without lossing weight information.

For example, if you want to run HSDP on a single machine with eight GPUs, where weights are sharded within four GPUs with two replica groups (2, 4 device mesh), you can do this with TorchTitan by specifying `--data_parallel_replica_degree=2` and `--data_parallel_shard_degree=4`. However, to utilize TorchFT, you will need to launch two TorchTitan instances, each managing four GPUs and communicating with each other through TorchFT.

## Example Configuration

Let's consider an example where we want to run HSDP on a single machine with eight GPUs, where weights are sharded within four GPUs with two replica groups (2, 4 device mesh). Without using TorchFT, you can launch such a training process by specifying `--parallelism.data_parallel_replica_degree=2 --parallelism.data_parallel_shard_degree=4`. However, in the event of a trainer failure (emulating a real-world machine failure), the entire training process would need to stop and recover from the last checkpoint. This can lead to significant downtime and wasted resources.

With TorchFT, we can tolerate one replica group failure, ensuring that the training process continues uninterrupted. To achieve this, we can launch two TorchTitan instances, each managing four GPUs and communicating with each other through TorchFT. This setup allows for seamless fault tolerance and minimizes the impact of individual trainer failures.

### Launching TorchFT with TorchTitan

To launch TorchFT with TorchTitan, you need to execute the following three commands in different shell sessions:

1. Launch TorchFT lighthouse:
```bash
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```
2. Launch the first TorchTitan instance:
```bash
NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh --fault_tolerance.enable --fault_tolerance.replica_id=0 --fault_tolerance.group_size=2 --parallelism.data_parallel_shard_degree=4
```
3. Launch the second TorchTitan instance:
```bash
NGPU=4 CUDA_VISIBLE_DEVICES=4,5,6,7 CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh --fault_tolerance.enable --fault_tolerance.replica_id=1 --fault_tolerance.group_size=2 --parallelism.data_parallel_shard_degree=4
```

### Explanation

* We limit the visibility of GPUs for each TorchTitan instance using environment variables `NGPU` and `CUDA_VISIBLE_DEVICES`, as we are running on a single machine. In reality, each TorchTitan instance will not share machines, so these variables are not required.
* `--fault_tolerance.enable` enables TorchFT functionality.
* `--fault_tolerance.group_size=2` tells TorchTitan that there are two replica groups.
* `--fault_tolerance.replica_id=1` tells TorchTitan that the replica ID of this instance is 1.
* Note that the alive replica group with the smallest replica ID will perform checkpointing saving.

In a real-world scenario, `torchft_lighthouse` would likely be on a different machine. The `TORCHFT_LIGHTHOUSE` environment variable is used to tell TorchFT how to communicate with `torchft_lighthouse`. The default value is `http://localhost:29510`.
