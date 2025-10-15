## TorchTitan & TorchComms Composability Testing

### Overview

This folder provides a framework for composability testing with TorchComms and distributed training in TorchTitan. It enables flexible experimentation with distributed communication primitives and various parallelism strategies in PyTorch.

> **TODO:** Additional documentation will be provided once TorchComms is publicly released.

### Quick Start

The following command uses Llama 3 as an example, but should work with all models:

```bash
TEST_BACKEND=nccl TRAIN_FILE=torchtitan.experiments.torchcomms.train CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" ./run_train.sh
```

### Features

#### Distributed Training Utilities
- Custom communicator initialization via `torchcomms.new_comm`
- Device mesh setup using `torchcomms.init_device_mesh`

#### Parallelism Support
Fully integrated and tested with:
- **FSDP** (`fully_shard`) - Fully Sharded Data Parallel
- **TP** - Tensor Parallelism
- **PP** - Pipeline Parallelism
- **CP** - Context Parallelism

### Roadmap

- [ ] Integration and testing with `torch.compile`
