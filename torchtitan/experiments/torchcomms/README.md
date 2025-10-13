# TorchTitan & TorchComms Composability Testing

This repository provides a framework for composability testing with **TorchComms** and distributed training in **TorchTitan**. The goal is to enable flexible experimentation with distributed communication primitives and parallelism strategies in PyTorch.
---
#### Example
```bash
TEST_BACKEND={backend} TRAIN_FILE=torchtitan.experiments.torchcomms.train ./run_train.sh --model.name torchcomms
```
---
## Available Features
- **Distributed Training Utilities**
  - Training with `torchcomms.new_comm`
  - Device mesh initialization with `torchcomms.init_device_mesh`
- **Composability Testing**
  - Integration and testing with `fully_shard` (FSDP)
---
## To Be Added
- Composability testing with additional parallelism strategies (e.g., tensor, pipeline, model parallelism)
- Integration and testing with torch.compile
---
