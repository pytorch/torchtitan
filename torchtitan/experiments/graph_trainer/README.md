## GraphTrainer

[![integration and numerics tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml?query=branch%3Amain)
[![arXiv](https://img.shields.io/badge/arXiv-2411.00284-b31b1b.svg)](https://arxiv.org/abs/2411.00284)

This experiment demonstrates graph-based distributed training in torchtitan through toolkit-style usage of PyTorch's compiler technologies, including:
- [SimpleFSDP](https://arxiv.org/abs/2411.00284) as a compiler-friendly fully sharded data parallel implementation
- Dynamo and AOTAutograd as the joint graph capture frontend
- Provenance-tracking infrastructure as the user annotation backbone
- Graph optimization via FX graph passes

The goal is to give users more explicit control over the compiler stack in terms of performance, numerics, and debuggability during large-scale distributed training. Three compilation modes are currently supported:
- **AOT mode** (`--compile.mode aot`): Explicit joint graph export with a custom graph pass pipeline.
- **JIT mode** (`--compile.mode jit`): Standard `torch.compile()` with graph passes registered to custom backends.
- **AOT FX trace mode** (`--compile.mode aot_fx_trace`): Non-strict tracing of the full forward + loss + backward via `make_fx`, producing a single end-to-end graph without AOTAutograd partitioning.

### Prerequisites

GraphTrainer requires the latest PyTorch nightly, which can be installed (e.g., for CUDA 13.0) via:
```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --force-reinstall
```
You can replace `cu130` with another version of CUDA or an AMD GPU (e.g. `rocm6.3`).

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

By default, the graph is captured with the AOT mode (switch to JIT mode via `--compile.mode jit` or AOT FX trace mode via `--compile.mode aot_fx_trace`) and compiled with the `aot_eager` backend.

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

### Pre-compile (Compile-on-One-Rank)

Pre-compile lets you compile AOT graphs on a single GPU and save them to disk,
then load them on all ranks during training — skipping compilation entirely.
This uses compile-on-one-rank (CooR) to produce a rank-agnostic artifact.
Setting `--compile.precompile_artifact_dir` enables precompile in both steps.

**Artifact ephemerality:** Precompiled artifacts are tied to the exact PyTorch
version, CUDA version, model architecture, and parallelism configuration used
to create them. Changing any of these requires regenerating the artifacts.
Stale artifacts are detected automatically via config fingerprinting and
will raise an error at load time. Delete old artifacts and re-run
precompile when upgrading PyTorch or changing the model/parallelism setup.

```bash
# Step 1: precompile on a single process (needs only 1 GPU)
python -m torchtitan.experiments.graph_trainer.precompile_main \
    --module graph_trainer.llama3 \
    --config graph_trainer_llama3_debugmodel \
    --compile.passes full_inductor_compilation \
    --compile.joint_passes inductor_decomposition \
    --compile.precompile_artifact_dir /tmp/precompile_artifacts \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.tensor_parallel_degree 2

# Step 2: load and train with torchrun (uses all GPUs)
# Uses run_train_precompile.sh which passes --virtual-local-rank to torchrun.
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel \
    ./torchtitan/experiments/graph_trainer/run_train_precompile.sh \
    --compile.passes full_inductor_compilation \
    --compile.joint_passes inductor_decomposition \
    --compile.precompile_artifact_dir /tmp/precompile_artifacts \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.tensor_parallel_degree 2
```

**`--virtual-local-rank`:** This torchrun flag makes every worker process see
`LOCAL_RANK=0` and target `cuda:0`. torchrun isolates each worker's GPU via
`CUDA_VISIBLE_DEVICES`, so `cuda:0` maps to a different physical GPU per
worker. This is required for CooR because the precompiled artifact was
compiled on a single process targeting `cuda:0`, and CooR handles
rank-specific computation dynamically at runtime via
`_runtime_compute_coordinate_on_dim`.

Pre-compile works with any compiler pass that produces serializable output,
including `full_inductor_compilation` and `regional_inductor`. Use a shared
filesystem path for the artifact directory in multi-node setups.

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
|Expert Parallelism + Pipeline Parallelism| 🚧 |
|Graph-based Pipeline Parallelism| 🚧 |
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
