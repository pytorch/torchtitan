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

Hardware configuration: 4x RTX 3090 24GB, pipeline parallelism dimension: 4.

### 1F1B Schedule

#### Llama3 Configuration Comparison
|  Hidden size|  Layers |  Autopipe TPS|  Default TPS|  Speedup     |
|  ---------- |  ----   |  ----------  |  -----------|  ----------- |
|  dim=256    |  6      |  31,094      |  29,549     |  +5.2%       |
|  dim=256    |  12     |  21,803      |  21,923     |  -0.5%       |
|  dim=2048   |  12     |  3,348       |  2,616      |  +28.0%      |
|  dim=4096   |  12     |  981         |  761        |  +28.9%      |

#### DeepseekV3(without moe) Configuration Comparison

|  Hidden size|  Layers |  Autopipe TPS|  Default TPS|  Speedup     |
|  ---------- |  ----   |  ----------  |  -----------|  ----------- |
|  dim=256    |  6      |  13,373      |  13,059     |  +2.4%       |
|  dim=256    |  12     |  7,714       |  6,859      |  +12.5%      |
|  dim=2048   |  12     |  4,331       |  3,810      |  +13.7%      |
|  dim=4096   |  12     |  2,888       |  2,561      |  +12.8%      |
|  dim=4096   |  16     |  2,207       |  2,008      |  +9.9%       |
|  dim=8192   |  16     |  4,331       |  3,935      |  +10.1%      |

### Multiple Schedules

#### Llama3 Configuration Comparison
|  Hidden size|  Layers |         Schedule       |Autopipe TPS|  Default TPS|  Speedup     |
|  ---------- |  ----   | ---------------------  | ---------- |  -----------|  ----------- |
|  dim=1024   |  16     |  1F1B                  | 7,532      |  6,471      |  +16.4%      |
|  dim=1024   |  16     |  Interleaved1F1B       | 7,808      |  6,556      |  +19.1%      |
|  dim=1024   |  16     |  InterleavedZeroBubble | 7,953      |  6,609      |  +20.3%      |
|  dim=1024   |  24     |  1F1B                  | 5,119      |  4,575      |  +11.9%      |
|  dim=1024   |  24     |  Interleaved1F1B       | 5,296      |  4,694      |  +12.8%      |
|  dim=1024   |  24     |  InterleavedZeroBubble | 5,382      |  4,692      |  +14.7%      |
|  dim=2048   |  24     |  1F1B                  | 1,770      |  1,570      |  +12.7%      |
|  dim=2048   |  24     |  Interleaved1F1B       | 1,835      |  1,626      |  +12.9%      |
|  dim=2048   |  24     |  InterleavedZeroBubble | 1,866      |  1,629      |  +14.5%      |

#### DeepseekV3 (without moe) Configuration Comparison

|  Hidden size|  Layers |         Schedule       |Autopipe TPS|  Default TPS|  Speedup     |
|  ---------- |  ----   | ---------------------  | ---------- |  -----------|  ----------- |
|  dim=1024   |  16     |  1F1B                  | 4,588      |  4,197      |  +9.3%       |
|  dim=1024   |  16     |  Interleaved1F1B       | 5,195      |  4,690      |  +10.8%      |
|  dim=1024   |  16     |  InterleavedZeroBubble | 5,267      |  4,592      |  +14.7%      |
|  dim=1024   |  24     |  1F1B                  | 3,071      |  2,912      |  +5.5%       |
|  dim=1024   |  24     |  Interleaved1F1B       | 3,478      |  3,295      |  +5.6%       |
|  dim=1024   |  24     |  InterleavedZeroBubble | 3,547      |  3,235      |  +9.6%       |
|  dim=2048   |  24     |  1F1B                  | 2,325      |  2,247      |  +3.5%       |
|  dim=2048   |  24     |  Interleaved1F1B       | 2,597      |  2,537      |  +2.4%       |
|  dim=2048   |  24     |  InterleavedZeroBubble | 2,773      |  2,525      |  +9.8%       |
