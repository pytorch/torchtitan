## Fullmodel

[![integration and numerics tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_fullmodel.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_fullmodel.yaml?query=branch%3Amain)
[![arXiv](https://img.shields.io/badge/arXiv-2411.00284-b31b1b.svg)](https://arxiv.org/abs/2411.00284)

This experiment demonstrates graph-based distributed training in torchtitan through toolkit-style usage of PyTorch's compiler technologies, including:
- [SimpleFSDP](https://arxiv.org/abs/2411.00284) as a compiler-friendly fully sharded data parallel implementation
- Dynamo and AOTAutograd as the joint graph capture frontend
- Provenance-tracking infrastructure as the user annotation backbone
- Graph optimization via FX graph passes

The goal is to give users more explicit control over the compiler stack in terms of performance, numerics, and debuggability during large-scale distributed training. Two compilation modes are currently supported:
- **AOT mode** (`--compile.mode aot`): Explicit joint graph export with a custom graph pass pipeline.
- **JIT mode** (`--compile.mode jit`): Standard `torch.compile()` with graph passes registered to custom backends.

### Prerequisites

Fullmodel requires the latest PyTorch nightly, which can be installed (e.g., for CUDA 12.8) via:
```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```
You can replace `cu128` with another version of cuda or an AMD GPU (e.g. `rocm6.3`).

### Quick Start

#### Training Llama3-8B

```bash
MODULE=fullmodel.llama3 CONFIG=fullmodel_llama3_8b ./run_train.sh
```
#### Training DeepSeek-v3-16B

```bash
MODULE=fullmodel.deepseek_v3 CONFIG=fullmodel_deepseek_v3_16b ./run_train.sh
```

### Customizing Parallelism

#### Training Llama3-8B with 2D parallelism (FSDP and TP)
```bash
NGPU=8 MODULE=fullmodel.llama3 CONFIG=fullmodel_llama3_8b ./run_train.sh --parallelism.data_parallel_shard_degree=4 --parallelism.tensor_parallel_degree=2
```
#### Training DeepSeek-v3-16B with 3D parallelism (FSDP, TP, and EP)

```bash
NGPU=8 MODULE=fullmodel.deepseek_v3 CONFIG=fullmodel_deepseek_v3_16b ./run_train.sh --parallelism.data_parallel_shard_degree=4 --parallelism.tensor_parallel_degree=2 --parallelism.expert_parallel_degree=2
```

### Customizing graph passes

```bash
# Auto bucketing for comm/compute overlap
MODULE=fullmodel.llama3 CONFIG=fullmodel_llama3_8b ./run_train.sh --compile.passes auto_bucketing

# transformer-block bucketing for comm/compute overlap
MODULE=fullmodel.llama3 CONFIG=fullmodel_llama3_8b ./run_train.sh --compile.passes transformer_block_bucketing

# CUDAGraph
MODULE=fullmodel.llama3 CONFIG=fullmodel_llama3_8b ./run_train.sh --compile.passes cudagraph

# Full Inductor compilation
MODULE=fullmodel.llama3 CONFIG=fullmodel_llama3_8b ./run_train.sh --compile.joint_passes inductor_decomposition --compile.passes full_inductor_compilation
```
