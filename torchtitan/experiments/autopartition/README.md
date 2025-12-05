# Auto-Partition in torchtitan

## Overview

This folder provides an automatic partitioning method that considers the computation cost of embedding layers. 
Thsi method involves calculating the floating-point operations (FLOPs) of the embedding layers and constructing an array that incorporates the FLOPs of both the transformer and embedding layers. Subsequently, a heuristic algorithm is employed to identify a balanced pipeline partition.

## Quick Start

### Compile

First, we need to compile `autopipe.cpp`.
```bash
pip install pybind11
cd ./torchtitan/experiments/autopartition/infra/cpp
mkdir build
cd build
cmake ..
make
mv *.so ../../
```

The following command uses Llama 3 as an example:

```bash
CONFIG_FILE="./torchtitan/experiments/autopartition/train_configs/debug_model.toml" ./run_train.sh
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
