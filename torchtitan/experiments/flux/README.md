# FLUX model in torchtitan

## Overview

## Usage

Run the following command to train the model:
```
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=1 torchtitan/experiments/flux/train.py --job.config_file torchtitan/experiments/flux/train_configs/debug_model.toml

```

## TODO
- [ ] Supporting for multiple GPUs is comming soon (FSDP, etc)
- [ ] Implement test cases in CI for FLUX model
- [ ] More parallesim support (Tensor Parallelism, Context Parallelism, etc)
- [ ] Support for distributed checkpointing and loading
