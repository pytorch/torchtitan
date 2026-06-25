## GraphTrainer

[![integration and numerics tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml?query=branch%3Amain) [![GraphTrainer H100 8 GPU Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer_h100.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer_h100.yaml?query=branch%3Amain)

This experiment demonstrates graph-based distributed training in torchtitan through toolkit-style usage of PyTorch's compiler technologies. The goal is to give users explicit control over the compiler stack in terms of performance, numerics, and debuggability during large-scale distributed training. See the [Manifesto](MANIFESTO.md) for the motivation and design philosophy behind GraphTrainer.

**Key features:**
- **Full train step graph capture** — `make_fx`-based `minimal_fx_tracer` traces forward + loss + backward (and optionally `optimizer.step`) into a single FX graph, without AOTAutograd partitioning, giving full visibility and control over the entire computation.
- **[SimpleFSDP](https://arxiv.org/abs/2411.00284)** — A compiler-based FSDP that represents sharding as parameterized collectives within the computation graph, making it fully tracer-friendly while achieving memory and throughput improvements over eager FSDP2.
- **Tensor-granularity memory policy** — Each activation can independently be saved, recomputed, or offloaded to the CPU, unlike module-level eager SAC. Different strategies mix freely within a single layer.
- **Graph pass pipeline** — Structured into default (numerics-preserving) and opt-in performance passes: bucketing for comm/compute overlap, async TP, regional/full Inductor compilation, CUDA graphs, CPU offload, and selective activation remat.
- **Pre-compile (Compile-on-One-Rank)** — Compile on a single GPU, serialize the artifact, and load on all ranks at training time — skipping compilation entirely. Config fingerprinting detects stale artifacts.
- **Composable parallelism** — FSDP + TP + EP in the graph, with async tensor parallel (micro-pipeline TP via symmetric memory) as an opt-in graph pass.
- **Debug tooling** — tlparse integration for browser-based graph inspection, and CUDA graph kernel annotations in profiler traces.

<details>
<summary>Legacy compilation modes (deprecated)</summary>

In addition to the default `aot_fx_trace` mode, two legacy modes exist but are deprecated and will be removed:
- **AOT mode** (`--compile.mode aot`): Explicit joint graph export with a custom graph pass pipeline.
- **JIT mode** (`--compile.mode jit`): Standard `torch.compile()` with graph passes registered to custom backends.
</details>

### Prerequisites

GraphTrainer requires the latest PyTorch nightly, which can be installed (e.g., for CUDA 13.0) via:
```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --force-reinstall
```
You can replace `cu130` with another version of CUDA.

### Quick Start

#### Training Llama3-8B

```bash
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh
```
#### Training DeepSeek-v3-16B

```bash
MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_16b ./run_train.sh
```

#### Training Qwen3-14B

```bash
MODULE=graph_trainer.qwen3 CONFIG=graph_trainer_qwen3_14b ./run_train.sh
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

### GraphPP Pipeline Parallelism

GraphPP is the `aot_fx_trace` pipeline-parallel path for GraphTrainer models.
It reuses TorchTitan's eager PP module splitting and PyTorch PP schedules, then
traces one representative microbatch per local stage with GraphTrainer's
`minimal_fx_tracer`. The resulting per-stage graph bundles are reused for later
microbatches; `GraphPPRunner` only executes the prebuilt callable for each PP
schedule action.

```bash
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel ./run_train.sh \
  --compile.mode aot_fx_trace \
  --parallelism.pipeline_parallel_degree 2 \
  --parallelism.pipeline_parallel_schedule Interleaved1F1B \
  --parallelism.data_parallel_shard_degree 4 \
  --parallelism.expert_parallel_degree 2
```

Supported runtime schedules include `Interleaved1F1B`, `ZBVZeroBubble`, and
`DualPipeV`. GraphPP builds stage-local forward/backward graphs, optional FSDP
`UNSHARD` / `REDUCE_GRAD` graphs, optional dI/dW graphs for split backward
schedules, and multiplexed graphs for `OVERLAP_F_B` actions. Regional and full
Inductor compilation reuse the existing GraphTrainer compilation passes on the
extracted GraphPP callables; GraphPP-specific handling stays in the GraphPP
stack before those passes are invoked. Multiplexed graphs keep the forward graph
as the destination module and ShapeEnv, insert backward placeholders/compute
before it, and transfer backward metadata into that ShapeEnv. This preserves
forward collective-size provenance for full Inductor without changing the shared
compiler passes.

GraphPP follows the same subclass boundary as the non-PP GraphTrainer tracer.
Extracted graphs run on flat plain tensor leaves. Values exposed to the PP
runtime are rewrapped from tracer metadata: stage forward outputs, input
gradients sent to previous stages, and parameter gradients before assignment to
live `param.grad`. Internal values remain flat because they never leave GraphPP
graph execution: saved-for-backward tensors, unsharded FSDP params, raw grad
leaves, reduce-grad inputs, and multiplexed intermediate outputs.

Current limitations: GraphPP does not load precompile artifacts yet, CUDA graph
capture should target the `GraphPPRunner` steady-state path in a future change,
and EP-overlap annotations will be composed with GraphPP in a later PR.

### Compiler Optimizations

The `aot_fx_trace` mode has a built-in pass pipeline controlled by dedicated flags.

```bash
# Full Inductor compilation (default is regional — compiles only tagged regions)
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.inductor_compilation full

# Numerics-changing optimizations (e.g. RMSNorm Inductor fusion)
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.numerics_changing_optim

# CPU activation offloading
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.memory_policy cpu_offload_all

# Disable CUDA graphs (for debugging)
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.disable_passes cudagraph_pass

# Disable specific passes by name
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.disable_passes custom_codegen_pass,cudagraph_pass

# Disable all graph passes (for debugging)
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.no-enable_passes
```

### Experimental AutoParallel Sharding

GraphTrainer can use AutoParallel to solve SPMD placement for supported models,
then trace and compile the placed model through the regular `aot_fx_trace`
flow. Enable it with `--compile.enable_autoparallel`; `--compile.mode
aot_fx_trace` is required.

Llama 3 debug model:

```bash
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel ./run_train.sh \
  --compile.mode aot_fx_trace \
  --compile.enable_autoparallel \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.tensor_parallel_degree 2
```

DeepSeek V3 debug model:

```bash
MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel_ep ./run_train.sh \
  --compile.mode aot_fx_trace \
  --compile.enable_autoparallel \
  --parallelism.data_parallel_shard_degree 4 \
  --parallelism.expert_parallel_degree 2
```

AutoParallel is only responsible for producing the placed model. After that,
GraphTrainer captures the full train step with `minimal_fx_tracer` and applies
the normal `aot_fx_trace` pipeline: the configured memory policy, selective
activation remat, CPU offload, bucketing and overlap passes, regional or full
Inductor compilation, CUDA graph compatibility checks, and any other enabled
GraphTrainer passes. This keeps AutoParallel placement composable with the same
compiler options used by manually parallelized GraphTrainer models.

AutoParallel can choose different sharding, collective schedules, and operator
lowerings from the manual parallelization path. This can change numerics, so
AutoParallel numerics tests check tight agreement with the eager baseline rather
than requiring bitwise-identical losses.

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

#### Llama3 (dense model)

```bash
# Step 1: precompile on a single process (needs only 1 GPU)
python -m torchtitan.experiments.graph_trainer.precompile_main \
    --module graph_trainer.llama3 \
    --config graph_trainer_llama3_debugmodel \
    --compile.precompile_artifact_dir /tmp/precompile_artifacts \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.tensor_parallel_degree 2

# Step 2: load and train with torchrun (uses all GPUs)
# Uses run_train_precompile.sh which passes --virtual-local-rank to torchrun.
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel \
    ./torchtitan/experiments/graph_trainer/run_train_precompile.sh \
    --compile.precompile_artifact_dir /tmp/precompile_artifacts \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.tensor_parallel_degree 2
```

#### DeepSeek-v3 (MoE model with expert parallelism)

```bash
# Step 1: precompile on a single process (needs only 1 GPU)
python -m torchtitan.experiments.graph_trainer.precompile_main \
    --module graph_trainer.deepseek_v3 \
    --config graph_trainer_deepseek_v3_debugmodel \
    --compile.precompile_artifact_dir /tmp/dsv3_precompile_artifacts \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.tensor_parallel_degree 2 \
    --parallelism.expert_parallel_degree 4

# Step 2: load and train with torchrun (uses all GPUs)
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel \
    ./torchtitan/experiments/graph_trainer/run_train_precompile.sh \
    --compile.precompile_artifact_dir /tmp/dsv3_precompile_artifacts \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.tensor_parallel_degree 2 \
    --parallelism.expert_parallel_degree 4
```

<details>
<summary><code>--virtual-local-rank</code> explained</summary>

This torchrun flag makes every worker process see `LOCAL_RANK=0` and target
`cuda:0`. torchrun isolates each worker's GPU via `CUDA_VISIBLE_DEVICES`, so
`cuda:0` maps to a different physical GPU per worker. This is required for
CooR because the precompiled artifact was compiled on a single process
targeting `cuda:0`, and CooR handles rank-specific computation dynamically
at runtime via `_runtime_compute_coordinate_on_dim`.
</details>

Pre-compile works with any compiler pass that produces serializable output,
including `full_inductor_compilation` and `regional_inductor`. Use a shared
filesystem path for the artifact directory in multi-node setups.

### Composability Support

Composability status for `aot_fx_trace` mode:

| Feature | Support |
| :--------: | :--------: |
|Meta Initialization| ✅ |
|Activation Checkpointing| ✅ |
|Activation Offloading| 🚧 |
|Mixed Precision Training| ✅ |
|Tensor Parallelism| ✅ |
|Context Parallelism| ✅ |
|Distributed Checkpointing| ✅ |
|CUDA Graphs| ✅ |
|Float8/MXFP8 Training| 🚧 |
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
