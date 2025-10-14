# TorchTitan & TorchComms Composability Testing

This folder provides a framework for composability testing with TorchComms and distributed training in TorchTitan. The goal is to enable flexible experimentation with distributed communication primitives and parallelism strategies in PyTorch.
---
#### Example

The command below uses Llama 3 as an example, but should work on all models.
```bash
TEST_BACKEND=nccl TRAIN_FILE=torchtitan.experiments.torchcomms.train CONFIG_FILE="./torchtitan/models/llama3/train_configs/debug_model.toml" ./run_train.sh
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
- Integration and testing with additional parallelism strategies (e.g., tensor, pipeline, model parallelism) other than fully_shard
- Integration and testing with torch.compile
---
