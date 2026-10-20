## GraphTrainer

[![integration and numerics tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml?query=branch%3Amain) [![GraphTrainer H100 8 GPU Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer_h100.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer_h100.yaml?query=branch%3Amain)

This experiment demonstrates graph-based distributed training in torchtitan through toolkit-style usage of PyTorch's compiler technologies. The goal is to give users explicit control over the compiler stack in terms of performance, numerics, and debuggability during large-scale distributed training. See the [Manifesto](MANIFESTO.md) for the motivation and design philosophy behind GraphTrainer.

> [!NOTE]
> GraphTrainer users and developers are welcome to join the [public PyTorch Slack channel](https://pytorch.slack.com/archives/C0C4ZRE5BL7) for questions, discussion, and updates.

**Key features:**
- **Full train step graph capture** — `make_fx`-based `minimal_fx_tracer` traces forward + loss + backward (and optionally `optimizer.step`) into a single FX graph, without AOTAutograd partitioning, giving full visibility and control over the entire computation.
- **[SimpleFSDP](https://arxiv.org/abs/2411.00284)** — A compiler-based FSDP that represents sharding as parameterized collectives within the computation graph, making it fully tracer-friendly while achieving memory and throughput improvements over eager FSDP2.
- **Tensor-granularity memory policy** — Each activation can independently be saved, recomputed, or offloaded to the CPU, unlike module-level eager SAC. Different strategies mix freely within a single layer.
- **Graph pass pipeline** — Structured into default (numerics-preserving) and opt-in performance passes: bucketing for comm/compute overlap, async TP, regional/full Inductor compilation, CUDA graphs, CPU offload, and selective activation remat.
- **Pre-compile (Compile-on-One-Rank)** — Compile on a single GPU, serialize the artifact, and load on all ranks at training time — skipping compilation entirely. Config fingerprinting detects stale artifacts.
- **Composable parallelism** — FSDP + TP + EP in the graph, with async tensor parallel (micro-pipeline TP via symmetric memory) as an opt-in graph pass.
- **Debug tooling** — tlparse integration for browser-based graph inspection, and CUDA graph kernel annotations in profiler traces.

### Prerequisites

GraphTrainer requires the latest PyTorch nightly, which can be installed (e.g., for CUDA 13.0) via:
```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --force-reinstall
```
You can replace `cu130` with another version of CUDA.

### Quick Start

#### Training Llama3-8B

```bash
MODULE=torchtitan_recipes.graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh
```
#### Training DeepSeek-V3 671B Dist-MoE

```bash
MODULE=torchtitan_recipes.graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_671b_dist_moe_bf16 ./run_train.sh
```

#### Training Qwen3-14B

```bash
MODULE=torchtitan_recipes.graph_trainer.qwen3 CONFIG=graph_trainer_qwen3_14b ./run_train.sh
```

### Configuring Parallelism

Put parallelism choices in a recipe. For example:

```python
from torchtitan_recipes.graph_trainer.llama3 import (
    graph_trainer_llama3_8b,
)

def llama3_8b_fsdp4_tp2():
    config = graph_trainer_llama3_8b()
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    return config
```

Launch it with `NGPU=8 MODULE=my_graph_configs
CONFIG=llama3_8b_fsdp4_tp2 ./run_train.sh`. Use the same pattern for EP and
other parallel axes.

### GraphPP Pipeline Parallelism

GraphPP is the pipeline-parallel path for GraphTrainer models.
It reuses TorchTitan's eager PP module splitting and PyTorch PP schedules, then
traces one representative microbatch per local stage with GraphTrainer's
`minimal_fx_tracer`. The resulting per-stage graph bundles are reused for later
microbatches; `GraphPipelineRuntime` only executes the prebuilt callable for each PP
schedule action.

Design references:
- GraphPP RFC: https://github.com/pytorch/torchtitan/issues/3780
- CUDA-graph-compatible GraphPP RFC: https://github.com/pytorch/torchtitan/issues/3820

```python
config.training.disable_cuda_graphs = True
config.parallelism.pipeline_parallel_degree = 2
config.parallelism.num_pp_microbatches = 8
config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
config.parallelism.data_parallel_shard_degree = 4
config.parallelism.expert_parallel_degree = 2
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

Current limitations: GraphPP does not load precompile artifacts yet, and
EP-overlap annotations will be composed with GraphPP in a later PR.

### Compiler Optimizations

GraphTrainer has a built-in pass pipeline controlled by its compile config.

```python
config.compile.inductor_compilation = "full"
config.compile.numerics_changing_optim = True
config.compile.memory_policy = "full"
config.compile.full_recompute_save_ops = (
    "layers.*.feed_forward.router.gate::aten.mm.dtype | "
    "layers.*.attention.wkv_a::aten.mm.default"
)
config.compile.disable_passes = ["custom_codegen_pass"]
# To disable every optional pass:
config.compile.enable_passes = False
```

The graph-level CUDA graph and kernel annotation passes remain available in
`cuda_graph.py`, but are not part of the built-in pass pipeline. Custom callers
can apply them explicitly with `construct_cuda_graph_passes`. By default, CUDA
graph capture is owned by the client around the complete forward/backward
execution. Users can disable CUDA graphs with
`config.training.disable_cuda_graphs = True`.

### Expert Parallel Overlap

Eager EP overlap supports non-blocking HybridEP in non-pipeline GraphTrainer.
Configure the model's `HybridEPTokenDispatcher` with a positive
`non_blocking_capacity_factor` and enable the two-chunk scheduler:

```python
config.compile.ep_overlap.enabled = True
config.compile.ep_overlap.module_fqn = "layers.*.feed_forward"
config.parallelism.data_parallel_shard_degree = 4
config.parallelism.expert_parallel_degree = 2
```

Selected MoE forwards are split into two equal chunks of the token axis before
tracing. Activations have shape `(T, D)`; padding masks remain replicated across
TP even when activations are sequence-sharded. Masks are split within each TP
shard and flattened back to replicated chunks, matching the local activations.
Router arguments and the full-call auxiliary-loss denominator are shared.
Dense feed-forward modules are not chunked.

The workflow is eager chunking -> tracing -> unshard normalization -> memory
policy -> collective bucketing -> EP scheduling -> regional
attention compilation -> whole forward/backward CUDA capture and replay.
Shared parametrized weights are read once outside chunk ownership, so both
chunks accumulate through one parameter-gradient reduction edge.

Each chunk owns an immutable native routing handle through dispatch, combine
and both backward exchanges. Expert payload outputs are separate allocations;
only native transport and preprocessing scratch are shared. Exchanges use one
communication stream with the same launch order on every EP rank. Native
device-side generation barriers protect shared storage; they are not CPU/GPU
size-discovery synchronizations. Every chunk has a group-uniform input row count.

Each exchange has explicit launch, phase, chunk and completion-wait annotations.
Waits retain the exact tensor input storage as graph arguments. Launch adapters
accept compiler layouts and normalize native contiguous inputs internally;
waits preserve storage rather than making independent layout copies. The
scheduler issues the second chunk's launch
before the first chunk's wait, then prefers ready launches, independent work
and finally waits. It preserves data and mutation dependencies;
it does not infer scheduling roles from node names or tensor shapes.

Chunked and unchunked training are not promised bitwise equal: chunking changes
expert-gradient accumulation order, and partitioned capacities can retain
different tokens. Scheduling-only validation compares the same two chunks,
routing and capacities with overlap disabled versus enabled. Numerical
differences must not substitute for checking completion and storage ownership.

The focused integration recipes use full forward/backward CUDA graphs for
the two-chunk control and overlap. They select native
varlen attention, which uses the existing FA4 provider on Blackwell, without
compiling attention through Inductor. They check actual
capture and compare ten training steps of full-precision loss and gradient
norm, including repeated replay; chunked-versus-unchunked equality is not the
comparison contract.

Varlen backend selection is explicit: use
`VarlenInnerAttention.Config(sdpa_backend=SDPBackend.FLASH_ATTENTION)` to select
the registered Flash implementation. Registering FA4 alone does not prevent
PyTorch from choosing cuDNN. The optional per-op preference preserves PyTorch's
automatic priority by default and restores backend flags after each call.

Current limitations:

- `expert_parallel_degree > 1` and `pipeline_parallel_degree == 1` are required.
- DeepEP and GraphPP overlap are outside this feature's scope.
- Full-graph Inductor is outside this feature's scope; expert ops
  execute natively. CUDA graphs capture the whole forward/backward callable.
- The token count must be even.
- Additional overlap or sharding modes require their own loss and trace checks.

### Experimental AutoParallel Sharding

GraphTrainer can use AutoParallel to solve SPMD placement for supported models,
then trace and compile the placed model through the regular GraphRuntime flow.
Enable it with `config.compile.enable_autoparallel = True` in the recipe.

Llama 3 debug model:

The tested Llama 3 layout also sets data-parallel sharding and tensor
parallelism to 2 in that recipe.

DeepSeek V3 debug model:

The tested DeepSeek V3 layout uses data-parallel sharding 4 and expert
parallelism 2.

AutoParallel is only responsible for producing the placed model. After that,
GraphTrainer captures the full train step with `minimal_fx_tracer` and applies
the normal graph pass pipeline: the configured memory policy, selective
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
Setting `compile.precompile_artifact_dir` in the selected recipe enables
precompile in both steps.

**Artifact ephemerality:** Precompiled artifacts are tied to the exact PyTorch
version, CUDA version, model architecture, and parallelism configuration used
to create them. Changing any of these requires regenerating the artifacts.
Stale artifacts are detected automatically via config fingerprinting and
will raise an error at load time. Delete old artifacts and re-run
precompile when upgrading PyTorch or changing the model/parallelism setup.

The integration runner contains complete Llama 3 and DeepSeek V3 precompile
recipes and runs both the single-process compile and distributed training
phases:

```bash
python -m torchtitan.experiments.graph_trainer.tests.run_precompile_tests \
    /tmp/precompile_test --ngpu 8
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

GraphRuntime composability status:

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
|MXFP8 Training| 🚧 |
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
