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

Current limitation: GraphPP does not load precompile artifacts yet.

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
policy and paging -> collective bucketing -> EP scheduling -> regional
attention compilation -> whole forward/backward CUDA capture and replay.
Shared parametrized weights are read once outside chunk ownership, so both
chunks accumulate through one parameter-gradient reduction edge.

Each chunk owns an immutable native routing handle through dispatch, combine
and both backward exchanges. Expert payload outputs are separate allocations;
only native transport and preprocessing scratch are shared. Exchanges use one
communication stream with the same launch order on every EP rank. Native
device-side generation barriers protect shared storage; they are not CPU/GPU
size-discovery synchronizations. Every chunk has a group-uniform input row count.

Native EP exchanges use a high-priority side stream. Set the same
`TORCHTITAN_HYBRIDEP_DISPATCH_SMS` and `TORCHTITAN_HYBRIDEP_COMBINE_SMS` values
on every rank before initialization to override the native SM targets
(16 each). Compute keeps its normal launch geometry; these targets do not
reserve compute SMs or change token capacity. CUDA priority does not preempt
already-running CTAs, so communication must be issued before independent
compute.

Each exchange has explicit launch, phase, chunk and completion-wait annotations.
Waits retain the exact tensor input storage as graph arguments. Launch adapters
accept compiler layouts and normalize native contiguous inputs internally;
waits preserve storage rather than making independent layout copies. The
scheduler issues the second chunk's launch
before the first chunk's wait, then prefers ready launches, independent work
and finally waits. It preserves data, mutation and paging queue dependencies;
it does not infer scheduling roles from node names or tensor shapes.
Among ready computations, launch ancestors go first so input gradients expose
their exchange before independent weight gradients. The latter can then cover
the final backward combine rather than being drained before it launches.
With eager paging, parameter-gradient annotations identify wgrad-only branches
outside the exchanges' input dependencies. The routed chunk0 branch waits to
become eligible until its terminal backward combine is issued. Independent
shared-expert and chunk1 work remain available earlier; reload and EP waits
stay late. Unchunked PP retains complementary F/B work rather than reserving
every backward wgrad for its own combine.

The whole-call output row budget is
`B = num_max_tokens_per_rank * ep_size * min(num_local_experts, top_k) * factor`.
Chunk capacities are `floor(B / 2)` and `B - floor(B / 2)`, each rounded to native
alignment. Factor 4 therefore gives each chunk factor 2 on the same full-call
basis, not factor 1 on that basis. Their combined capacity equals the unchunked
budget apart from alignment; the public factor is never changed. Transport
storage is shared and initialized separately from this output budget. Native
capacity drops set a device flag without synchronizing size discovery.

Chunked and unchunked training are not promised bitwise equal: chunking changes
expert-gradient accumulation order, and partitioned capacities can retain
different tokens. Scheduling-only validation compares the same two chunks,
routing and capacities with overlap disabled versus enabled. Numerical
differences must not substitute for checking completion and storage ownership.

Native capacity validation observes the real eager MoE wrapper at factor 4
and an odd, aligned budget; it does not hand-bind the expected capacity.

Set `config.compile.memory_policy = "sac_and_paged_stash"` to page saved expert
activations. Counts and stash commits belong to each layer/chunk, not just the
layer. Annotated stash, reload, prefetch and wait operations share the existing
paging stream and allocator. A true stash overflow invalidates backward and
uses the existing device-side optimizer-skip protocol; it is not a native
capacity drop. Initialization measures live pages once, retaining compact live
prefixes rather than every capacity-padded activation until backward. Reload
restores the capacity shape with the same live-prefix contract as real paging;
padding is not meaningful. Full-count inputs and unpaged reruns keep the original
tensor. This sizing step changes neither model computation nor recomputation.
Steady-state exchanges and paging do not read device counts on the host. The
CUDA page-buffer factor is headroom over that measured live-page peak,
independent of HybridEP's
static output capacity factor. Routing can change after calibration; choose
headroom for that variability or the deferred overflow contract stops the run.

Page copies and reloads use one high-priority stream, which serializes freelist
updates. Set `TORCHTITAN_PAGED_STASH_MAX_CTAS` before import to bound copy/pop
CTAs (2048 by default), without changing live-row counts or buffer sizing.
Copy/pop CTAs stride over flat multi-token tiles and copy payloads without
arithmetic. Page allocation, records, FIFO recycling and overflow semantics
are unchanged; lowering the grid cap does not shorten the copied live prefix.
Source annotations preserve each source's issue order and storage lifetimes;
independent multiplexed F/B sources may interleave on that one stream without
an additional whole-backward-before-forward paging chain. Per-slot records
remain FIFO and backward generations stay stage-owned.

For paging without activation recomputation, select
`config.compile.memory_policy = "none_and_paged_stash"` and
`config.activation_remat = None`. This saves forward activations and uses the
same annotated page operations and runner; the existing `sac_and_paged_stash`
mode retains its selective-recompute policy.

The focused integration recipes use full forward/backward CUDA graphs for
the two-chunk control, overlap, and overlap with paging. They select native
varlen attention, which uses the existing FA4 provider on Blackwell, without
compiling attention through Inductor. They check actual
capture and compare ten training steps of full-precision loss and gradient
norm, including repeated replay; chunked-versus-unchunked equality is not the
comparison contract. Paging covers both last-layer residency and
`config.compile.paged_stash.skip_immediate_backward = False`, which requires
a terminal stash join before backward reload. Keepalive activations belong to
the join, not to a functional reload that can return one of them during
calibration or residency.

Varlen backend selection is explicit: use
`VarlenInnerAttention.Config(sdpa_backend=SDPBackend.FLASH_ATTENTION)` to select
the registered Flash implementation. Registering FA4 alone does not prevent
PyTorch from choosing cuDNN. The optional per-op preference preserves PyTorch's
automatic priority by default and restores backend flags after each call.

Current limitations:

- `expert_parallel_degree > 1` and `pipeline_parallel_degree == 1` are required.
- DeepEP and GraphPP overlap are outside this feature's scope.
- Full-graph Inductor is outside this feature's scope; expert and paging ops
  execute natively. CUDA graphs capture the whole forward/backward callable.
- The token count must be even.
- Additional overlap or sharding modes require their own loss and trace checks.

### Pipeline Communication Overlap

Pipeline overlap uses `Interleaved1F1BCombined` or `DualPipeV` and graph
multiplexing, not EP token chunking. `Interleaved1F1BCombined` adds one warmup
forward, then pairs an independent
forward and backward as an explicit `OVERLAP_F_B` action. Both actions' receives
and parameter unshards precede the pair; normal send, gradient accumulation and
reduction ownership remain in the existing pipeline schedule.

`DualPipeV` uses the upstream V-shaped mapping with exactly two stages per
rank. Its combined actions use full backward; separate input-gradient and
weight-gradient actions fill the bubbles. Their saved activations remain live
until weight backward completes. The focused HybridEP recipes use native
varlen attention for both schedules, including actual dI/dW graph splitting.

```python
config.parallelism.pipeline_parallel_schedule = "Interleaved1F1BCombined"
config.compile.enable_graph_pp_comm_compute_overlap = True
config.compile.ep_overlap.enabled = False
config.compile.inductor_compilation = "regional"
config.training.disable_cuda_graphs = False
```

Use non-blocking `HybridEPTokenDispatcher` capacity. Enabling the GraphPP pass
selects explicit asynchronous native launches/waits without splitting tokens
or dividing each microbatch's capacity. Configure `sac_and_paged_stash` to
include page operations. Source graphs retain explicit forward/backward
ownership when multiplexed. The pass validates native request/input lifetimes,
preserves tensor/mutation and paging queue dependencies, launches ready
communication early and postpones waits behind independent compute.
Each EP or paging launch gives ready work in the opposite source graph priority,
including its ready completion waits, instead of draining one source first.
Full CUDA capture covers the entire pipeline forward/backward step, including every
microbatch's distinct native routing state.

For long multistream captures, PyTorch's native allocator can reclaim storage
after graph dependencies prove its asynchronous uses have joined:

```bash
PYTORCH_ALLOC_CONF=graph_capture_record_stream_reuse:True
```

This upstream option is experimental. Without it, cross-stream frees can wait
until capture ends, growing the graph pool with the microbatch count. Keep the
same allocator setting in matched comparisons; it does not replace request
ownership, stream joins, or the explicit overlap annotations.

The pass can be disabled with `compile.disable_passes =
["graph_pp_comm_compute_overlap_pass"]` for a same-schedule comparison. Without
combined actions, the requested overlap emits a warning. Full-graph Inductor
and DeepEP are outside this feature's scope.

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
