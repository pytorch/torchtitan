## Auto Parallel

### Overview

The Auto Parallel experiment integrates PyTorch's AutoParallel framework with TorchTitan to automatically optimize distributed training parallelism strategies given a device mesh. Instead of manually configuring parallelism layouts, AutoParallel uses cost-based analysis to determine optimal sharding placements for model parameters, activations, and gradients.

### Requirements

Requires installing [git@github.com:meta-pytorch/autoparallel.git](https://github.com/meta-pytorch/autoparallel)

### Single Node

**Llama3**

`CONFIG_FILE=./torchtitan/models/llama3/train_configs/debug_model.toml ./run_train.sh --model.name auto_parallel.llama3 --parallelism.tensor_parallel_degree 4 --job.custom_config_module=torchtitan.experiments.auto_parallel.job_config`

**DeepSeekv3**

`CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/debug_model.toml ./run_train.sh --model.name auto_parallel.deepseek_v3 --job.custom_config_module=torchtitan.experiments.auto_parallel.job_config`
