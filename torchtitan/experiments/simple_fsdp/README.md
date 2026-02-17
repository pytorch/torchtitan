## SimpleFSDP

[![integration and numerics tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_simple_fsdp.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_simple_fsdp.yaml?query=branch%3Amain)
[![arXiv](https://img.shields.io/badge/arXiv-2411.00284-b31b1b.svg)](https://arxiv.org/abs/2411.00284)

💡 **Note 1**: SimpleFSDP's composability with Mixed Precision Training and Tensor Parallel requires updates from latest PyTorch, which can be installed (e.g., for CUDA 12.6) via
```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
```

💡 **Note 2**: Some of SimpleFSDP's functionalities (e.g., reshard_after_forward) is implemented with torch.compile. It is always recommended to open compile (`--compile.enable`) to see desired correct functionality.

This folder includes an experimental frontend implementation for [SimpleFSDP: Simpler Fully Sharded Data Parallel with torch.compile](https://arxiv.org/abs/2411.00284). SimpleFSDP is a compiler-based Fully Sharded Data Parallel (FSDP) framework, which has a simple implementation for maintenance and composability, allows full computation-communication graph tracing, and brings performance enhancement via compiler backend optimizations.

### Run SimpleFSDP Training on Llama3 & DeepSeek_v3

#### Training Llama3 models

```bash
MODEL=simple_fsdp.llama3 CONFIG=simple_fsdp_llama3_8b ./run_train.sh --compile.enable
```

#### Training DeepSeek_v3 models

```bash
MODEL=simple_fsdp.deepseek_v3 CONFIG=simple_fsdp_deepseek_v3_debugmodel ./run_train.sh --compile.enable --activation_checkpoint.mode "none"
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
    - "auto_bucketing": perform autobucketing at aten fx-level, and perform code execution with aot_eager backend. (We also support `inductor` backend).
      ```bash
      --compile.backend "aot_eager" --compile.graph_passes "auto_bucketing"
      ```

3. manual optimization: perform manual bucketing & reordering with user FQN inputs.
    - "transformer_block_bucketing": perform bucketing by transformer blocks at aten fx-level, and perform code execution with aot_eager backend. (We also support `inductor` backend).
      ```bash
      --compile.backend "aot_eager" --compile.graph_passes "transformer_block_bucketing"
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
