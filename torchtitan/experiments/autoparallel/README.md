## Auto Parallel

### Overview

The Auto Parallel experiment integrates PyTorch's AutoParallel framework with TorchTitan to automatically optimize distributed training parallelism strategies given a device mesh. Instead of manually configuring parallelism layouts, AutoParallel uses cost-based analysis to determine optimal sharding placements for model parameters, activations, and gradients.

### Requirements

Requires installing [git@github.com:meta-pytorch/autoparallel.git](https://github.com/meta-pytorch/autoparallel)

### Single Node

**Llama3**

`CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.py ./run_train.sh --model.name autoparallel.llama3 --parallelism.tensor_parallel_degree 4`

**DeepSeekv3**

`NGPU=2 CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/debug_model.py ./run_train.sh --model.name autoparallel.deepseek_v3`

**DeepSeekv3 local_map**

This is a variant of titan's DSv3, which uses a local_map for the expert parallel region. This only supports 2D mesh right now. NOTE: the mesh provided are just to reuse torchtitan's trainer mesh setup code. Autoparallel is not bound to use dp2ep.

`NGPU=2 CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/debug_model.py ./run_train.sh --model.name autoparallel.local_map_deepseek_v3 --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2`

> **Note**: To use the autoparallel-specific config fields (e.g., `experimental.comms_bucket_reorder_strategy`), create a Python config file that merges the autoparallel config using `ConfigManager._merge_configs()`. See [docs/extension.md](../../../docs/extension.md) for details.
