## TorchTitan & TorchComms Composability Testing

### Overview

This folder provides a framework for composability testing with TorchComms and distributed training in TorchTitan. It enables flexible experimentation with distributed communication primitives and various parallelism strategies in PyTorch.

> **TODO:** Additional documentation will be provided once TorchComms is publicly released.

### Quick Start

The following command uses Llama 3 as an example:

```bash
TEST_BACKEND=nccl TRAIN_FILE=torchtitan.experiments.torchcomms.train CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" ./run_train.sh
```

### Features

#### Distributed Training Utilities
- Custom communicator backend initialization via `torchcomms.new_comm`
- Compose torchcomms with DeviceMesh via the wrapper API `torchcomms.init_device_mesh`

#### Parallelism Support
Locally tested with:
- **FSDP** (`fully_shard`) - Fully Sharded Data Parallel
- **TP** - Tensor Parallelism
- **PP** - Pipeline Parallelism
- **EP** - Expert Parallelism
- **compile** - `torch.compile` integration

### Performance

**Setup**: Similar setting as [docs/converging.md](../../docs/converging.md) based on [torchtitan/models/llama3/train_configs/llama3_8b.toml](../torchtitan/models/llama3/train_configs/llama3_8b.toml), but `training.local_batch_size = 1`

| Run Name    | Parallelism        | Distributed Library | Remarks               |
| ----------- | ------------------ | ------------------- | --------------------- |
| (dist)DP8   | FSDP 8             | c10d.distributed    | Baseline              |
| DP8         | FSDP 8             | torchcomms          | 1D test set           |
| DP8_CP2_TP4 | FSDP 8, TP 4, CP 2 | torchcomms          | 3D test set           |
| DP8_CP8     | FSDP 8, CP 8       | torchcomms          | CP with larger degree |

**Results**:

![Loss Curves](./asserts/images/loss_curves.png)


### Known Issues

- **CP** (Context Parallelism) - Temporly not working
- **Async TP** - Temporly not working
- **Memory Overhead** - TorchComms requires higher peak memory usage. As a workaround, we need to reduce `local_batch_size` to avoid out of memory error.

## Roadmap

- [ ] Add N-D parallelism end-to-end performance and convergence tests
  - Test with additional models: DeepSeek-V3, Qwen3, Llama4, etc. on large scale
