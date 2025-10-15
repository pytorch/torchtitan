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
- **CP** - Context Parallelism

### Roadmap

- [ ] Add N-D parallelism E2E perf and convergence tests
- [ ] Integrated and tested with Expert Parallelism
- [ ] Integration and testing with `torch.compile`
