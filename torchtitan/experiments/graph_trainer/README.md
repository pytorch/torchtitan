## GraphTrainer

[![integration and numerics tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml?query=branch%3Amain)
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

GraphTrainer requires the latest PyTorch nightly, which can be installed (e.g., for CUDA 12.8) via:
```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```
You can replace `cu128` with another version of CUDA or an AMD GPU (e.g. `rocm6.3`).

### Quick Start

#### Training Llama3-8B

```bash
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh
```
#### Training DeepSeek-v3-16B

```bash
MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_16b ./run_train.sh
```

### Configuring Parallelism

#### Training Llama3-8B with 2D parallelism (FSDP and TP)
```bash
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --parallelism.data_parallel_shard_degree=4 --parallelism.tensor_parallel_degree=2
```
#### Training DeepSeek-v3-16B with 3D parallelism (FSDP, TP, and EP)

```bash
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_16b ./run_train.sh --parallelism.data_parallel_shard_degree=4 --parallelism.tensor_parallel_degree=2 --parallelism.expert_parallel_degree=2
```

### Compiler Optimizations

By default, the graph is captured with the AOT mode (you can switch to JIT mode via `--compile.mode jit`) and compiled with the `aot_eager` backend.

Graph passes can be applied to further optimize the graph by using the `--compile.joint_passes` and `--compile.passes` flags.

```bash
# Auto bucketing for comm/compute overlap
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.passes auto_bucketing

# Transformer-block bucketing for comm/compute overlap
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.passes transformer_block_bucketing

# CUDAGraph
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.passes cudagraph

# Full Inductor compilation (AOT)
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.joint_passes inductor_decomposition --compile.passes full_inductor_compilation

# Full Inductor compilation (JIT)
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.mode jit --compile.backend inductor
```

### Graph-based Pipeline Parallelism

Graph PP captures the joint forward/backward graph per pipeline stage, then applies
graph passes to partition FSDP collectives and enable zero-bubble scheduling. It
requires AOT mode (`--compile.mode aot`, the default).

Graph PP passes are auto-inferred from the parallelism config:
- `split_fsdp_collectives` is enabled when FSDP is active
- `split_dI_dW` is enabled for V-schedules (DualPipeV, ZBV)

> **Note:** Graph PP currently requires balanced MoE routing
> (`_debug_force_load_balance=True`) to avoid data-dependent ops
> (`_local_scalar_dense`) that Inductor cannot compile. The `16B_sdpa_balanced`
> and `debugmodel_balanced` configs have this enabled. This requirement is
> temporary and will be removed once Inductor supports non-even routing.

#### Training DeepSeek-v3-16B with graph PP (PP=2, FSDP=4, EP=4)
```bash
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_16b_sdpa_balanced ./run_train.sh \
  --parallelism.pipeline_parallel_degree 2 \
  --parallelism.data_parallel_shard_degree 4 \
  --parallelism.expert_parallel_degree 4
```

#### Training DeepSeek-v3-16B with graph PP and DualPipeV schedule
```bash
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_16b_sdpa_balanced ./run_train.sh \
  --parallelism.pipeline_parallel_degree 2 \
  --parallelism.data_parallel_shard_degree 4 \
  --parallelism.expert_parallel_degree 4 \
  --parallelism.pipeline_parallel_schedule DualPipeV
```

### Composability Support

Some of the features require the updates from PyTorch, with which we are working on providing composability support for the following features:

| Feature | Support |
| :--------: | :--------: |
|Meta Initialization| ✅ |
|Activation Checkpointing| ✅ |
|Activation Offloading| 🚧 |
|Mixed Precision Training| ✅ |
|Tensor Parallelism| ✅ |
|Context Parallelism| ✅ |
|Pipeline Parallelism| ✅ |
|Distributed Checkpointing| ✅ |
|CUDA Graphs| ✅ |
|Float8 Training| ✅ |
|Expert Parallelism| ✅ |
|Expert Parallelism + Activation Checkpointing| 🚧 |
|Expert Parallelism + Pipeline Parallelism| ✅ (AOT mode, balanced routing only) |
|Graph-based Pipeline Parallelism| ✅ (AOT mode, balanced routing only) |
|Micro-batch overlap| 🚧 |
|Pre-compile| 🚧 |

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
