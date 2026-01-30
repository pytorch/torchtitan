# Auto-Partition in torchtitan

## Overview

This folder provides an automatic partitioning method that considers the computation cost of embedding layers. 
This method involves calculating the floating-point operations (FLOPs) of the embedding layers and constructing an array that incorporates the FLOPs of both the transformer and embedding layers. Subsequently, a heuristic algorithm is employed to identify a balanced pipeline partition.

## Quick Start

Edit the run_train.sh
```bash
#!/bin/bash

NGPU=${NGPU:-"4"}
export LOG_RANK=${LOG_RANK:-0}
# Autopartition training configuration file (default: debug model config)
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/autopartition/train_configs/debug_model.toml"}
# Autopartition training script entry point
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.experiments.autopartition.train"}

# Launch distributed training with torchrun, set custom config module
torchrun --nproc_per_node ${NGPU} \
    -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} \
    --job.custom_config_module "torchtitan.experiments.autopartition.job_config" "$@"

```

```bash
# run
./run_train.sh
```

## Performance

Hardware configuration: 4x RTX 3090 24GB, pipeline parallelism dimension is 4.

### llama3 配置对比
|  hidden size|  layers |  autopipe TPS|  default TPS|  Speedup     |
|  ---------- |  ----   |  ----------  |  -----------|  ----------- |
|  dim=256    |  6      |  31,094      |  29,549     |  +5.2%       |
|  dim=256    |  12     |  21,803      |  21,923     |  -0.5%       |
|  dim=2048   |  12     |  3,348       |  2,616      |  +28.0%      |
|  dim=4096   |  12     |  981         |  761        |  +28.9%      |

### deepseekv3(without moe) 配置对比

|  hidden size|  layers |  autopipe TPS|  default TPS|  Speedup     |
|  ---------- |  ----   |  ----------  |  -----------|  ----------- |
|  dim=256    |  6      |  13,373      |  13,059     |  +2.4%       |
|  dim=256    |  12     |  7,714       |  6,859      |  +12.5%      |
|  dim=2048   |  12     |  4,331       |  3,810      |  +13.7%      |
|  dim=4096   |  12     |  2,888       |  2,561      |  +12.8%      |
|  dim=4096   |  16     |  2,207       |  2,008      |  +9.9%       |
|  dim=8192   |  16     |  4,331       |  3,935      |  +10.1%      |


### Known Issues

- **Not Support Moe** - Auto-Partition need flops for each layers, but current profiler from deepspeed not support computing flops for moe.
