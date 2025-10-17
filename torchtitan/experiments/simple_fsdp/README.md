## SimpleFSDP

[![integration tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_simple_fsdp.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_simple_fsdp.yaml?query=branch%3Amain)
[![arXiv](https://img.shields.io/badge/arXiv-2411.00284-b31b1b.svg)](https://arxiv.org/abs/2411.00284)

💡 **Note**: SimpleFSDP's composability with Mixed Precision Training and Tensor Parallel requires updates from latest PyTorch, which can be installed (e.g., for CUDA 12.6) via
```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
```

This folder includes an experimental frontend implementation for [SimpleFSDP: Simpler Fully Sharded Data Parallel with torch.compile](https://arxiv.org/abs/2411.00284). SimpleFSDP is a compiler-based Fully Sharded Data Parallel (FSDP) framework, which has a simple implementation for maintenance and composability, allows full computation-communication graph tracing, and brings performance enhancement via compiler backend optimizations.

### Run SimpleFSDP Training on Llama3 & DeepSeek_v3

#### Training Llama3 models

```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh --model.name simple_fsdp.llama3 --compile.enable
```

#### Training DeepSeek_v3 models

```bash
CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/debug_model.toml" ./run_train.sh --model.name simple_fsdp.deepseek_v3 --compile.enable
```

### Composability Support

Some of the features require the updates from PyTorch, with which we are working on providing composability support for the following features:

| Feature | Support |
| :--------: | :--------: |
|Meta Initialization| ✅ |
|Activation Checkpointing| ✅ |
|Mixed Precision Training| ✅ |
|Tensor Parallelism| ✅ |
|Context Parallelism| ✅ |
|Pipeline Parallelism| ✅ |
|Distributed Checkpointing| ✅ |
|Float8 Training| 🚧 |
|Expert Parallelism | ✅ |
|Expert Parallelism + Activation Checkpointing| 🚧 |
|Expert Parallelism + Pipeline Parallelism| 🚧 |


### Compiler Optimizations

SimpleFSDP relies on compiler backend to perform optimizations (i.e., bucketing & reordering) for good training performance. Currently, the following optimization passes are supported:

1. no optimization: default torch.compile backends (e.g., "inductor", "aot_eager", "eager")

2. auto optimization: perform auto-bucketing & reordering without user inputs. **Note: it is not guaranteed that users will get the most optimized training performance**
    - "aot_eager_autobucketing": perform autobucketing at aten fx-level, and perform code execution with aot_eager backend.


users can specify the pass (e.g., "aot_eager_autobucketing") via additional configs:

```bash
--job.custom_config_module=torchtitan.experiments.simple_fsdp.job_config  --compile.model_backend_override "aot_eager_autobucketing"
```

### Citation

If you find SimpleFSDP useful, please kindly consider citing the following paper:

```latex
@article{zhang2024simplefsdp,
  title={SimpleFSDP: Simpler Fully Sharded Data Parallel with torch. compile},
  author={Zhang, Ruisi and Liu, Tianyu and Feng, Will and Gu, Andrew and Purandare, Sanket and Liang, Wanchao and Massa, Francisco},
  journal={arXiv preprint arXiv:2411.00284},
  year={2024}
}
```
