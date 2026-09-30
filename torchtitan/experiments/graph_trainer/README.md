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

### DeepSeek NVFP4 compilation gate

`graph_trainer_deepseek_v3_debugmodel_nvfp4` converts the eager NVFP4 recipe
without changing its dense/shared NVFP4, routed W13/W2 NVFP4, MXFP8 attention,
router/LM-head exclusions, dispatcher padding, or runtime buffers. It compiles
forward and backward together with full Inductor compilation, full activation
recomputation, and `numerics_changing_optim=False`. Both GraphTrainer replay
(`disable_passes=["cuda_graph_pass"]`) and inherited trainer capture
(`training.disable_cuda_graphs=True`) are disabled by default.

A four-GPU GB200 gate used sequence length 1024, eight sequences per rank,
FSDP 4 + EP 2, seed 42, deterministic algorithms, forced load balance,
`c4_test`, and the recipe's AdamW/scheduler defaults. One-GPU, FSDP 4, and
FSDP 4 + EP 2 smoke stages preceded four sequential 50-step arms.
All 50 TensorBoard losses and gradient norms were finite in every arm.

| Arm | Mean tokens/s/GPU, steps 10-50 | Peak active GiB, rank 0 | Step 50 loss |
| --- | ---: | ---: | ---: |
| Eager BF16 | 30,662 | 0.540 | 2.92338 |
| Eager NVFP4 + MXFP8 | 15,638 | 0.709 | 2.92619 |
| Compiled BF16 | 127,047 | 2.115 | 2.91534 |
| Compiled NVFP4 + MXFP8 | 65,280 | 2.084 | 2.91212 |

Over steps 10-50, the eager quantization loss gap was 0.01229228 and the
compiled gap was 0.00954233. Their difference, -0.00274995, passed the
<= 0.02 acceptance threshold. Eager and compiled loss/gradient norms differed
bitwise at all 50 steps for both precisions. This debugmodel gate does not
establish production-scale convergence or bitwise equivalence.

Each rank produced one `AOTCompiledArtifact` containing forward and backward;
the precompile NVFP4 graph included 2,358 backward operations. Only output
`getitem` unpacking remained outside that compiled region. Compilation includes
external calls: scaled GEMMs (`aten._scaled_mm_v2`, `_scaled_grouped_mm_v2`),
TorchAO quantization kernels, NCCL collectives, deterministic scatter, and
Inductor extern GEMMs. These kernels are not rewritten into fused Triton code.
The generated wrapper retains host routing/split-size work; full compilation
does not eliminate host operations. Saved compiler JSON, precompile FX graphs,
and generated source provide evidence beyond config flags.

The first FSDP compilation needed 61-83 seconds per rank in the graph-pass
pipeline. In the sequential gate, BF16 took 51-64 seconds and NVFP4 took
5.27-5.45 seconds with warmed compiler caches. NVFP4 total trainer wall time
was 108-110 seconds per rank, including tracing/startup and training; these
pass timings are not cold end-to-end compilation measurements. Memory above
is peak active tensor memory, not reserved memory or driver/compiler memory.

Two GraphTrainer correctness fixes are required: expose the fresh-buffer
`scaled_mm.out` writer as a functional result before recomputation (otherwise
remat reads a new uninitialized buffer), and stop FSDP unshard deduplication
before tuple-producing quantization (otherwise a tensor replaces the operand
tuple). Focused dense, structured W13, W2, and MXFP8 tests compare outputs,
input/weight gradients, and an FP32 SGD update at the specified tolerances.
Those controlled-RNG tests use Inductor `fallback_random=True` to match eager
stochastic-rounding streams; the training gate uses the normal compiler RNG.

The bounded 50-step replay-enabled run changed only `disable_passes` to `[]`
(and its output directory). Every rank skipped capture and reported zero
replays. Full replay is unsupported with this routing backend: token counts
are copied to CPU and converted to host-dependent split sizes, which the CUDA
graph compatibility gate rejects. Its measured throughput is therefore not a
replay speedup. The requested-replay run averaged 64,309 tokens/s/GPU,
with mean loss difference +0.00155179 over steps 10-50. Loss differed bitwise
at 47/50 steps and gradient norms at 48/50; maximum absolute gradient-norm
difference was 73.3948 (including startup). No equivalence is claimed for this
unsupported capture path. No dispatcher changes were made for capture.

Focused CPU recipe/pass checks passed (107 tests and 9 subtests), as did the
GraphPP regression suite (46 tests and 15 subtests) and all four controlled
projection compilation tests. The expanded GPU suite initially passed 42
checks, with one existing expected failure and three failures. Specifying
Triton in the TorchAO NVFP4 reference resolves two failures while preserving
bitwise tolerances: AUTO selected the installed CuteDSL backend. The remaining
raw-eager MXFP8 SimpleFSDP quantization-count test observes two quantizations
instead of one and also fails on the untouched merge in this same environment.
That pre-existing cache/count limitation is recorded; the compiled FSDP gate
passes, and this change does not introduce a runtime caching workaround.

Validated environment: Python 3.12.3 aarch64, official CUDA 13.0 nightly
`torch==2.15.0.dev20260928+cu130` (PyTorch commit
`68e0ae4967e1a7e39d914179aea0026036cc1a23`), wheel SHA256
`ecfeb2acc3e87f717bcfa9cd7eb845fa2e86cbfa5b36eaf47b4728ef82a6d9c6`;
TorchAO source `3972ed015091f659418dedf12edb980a8ca56b53` rebuilt as
`0.19.0+git3972ed01`. CUDA driver 595.84.01, build toolkit 13.4. Exact dependency
pins and reproducible four-arm launch/analysis scripts are retained locally in
`/home/me/nvfp4/graph-validation/{environment.txt,run_arm.py,run_gate.py,analyze.py}`;
`gate-commands.json` records argv and rank logs are retained for every launch.
Activate that isolated environment before reproducing the debugmodel gate:

```bash
source /home/me/nvfp4/graph-validation/venv/bin/activate
torchrun --standalone --nproc-per-node=4 \
  --log-dir /home/me/nvfp4/graph-validation/reproduce-ranks --redirects 3 --tee 3 \
  /home/me/nvfp4/graph-validation/run_arm.py compiled-nvfp4 \
  /home/me/nvfp4/graph-validation/reproduce 50 4 2 off
```

For the exact sequential four-arm comparison, run
`python /home/me/nvfp4/graph-validation/run_gate.py` and then
`python /home/me/nvfp4/graph-validation/analyze.py` from this checkout, using
new output directories for repeated runs. For clean one-GPU/FSDP-only smoke
runs, set `--nproc-per-node` and the final shard/EP arguments to 1 and `1 1`,
or 4 and `4 1`, respectively.

`graph_trainer_deepseek_v3_16b_nvfp4` and
`graph_trainer_deepseek_v3_671b_nvfp4_mixed` are configuration candidates for
larger-cluster testing, with sequence length 4096 by default and the same
compilation/replay defaults. The 16B recipe retains its BF16 dense FFN and
optional `bf16_tail_fraction`; 671B retains the eager mixed recipe's coverage.
CPU tests check conversion and routing preservation. Neither larger model nor
TP/CP/PP, precompilation, or overlap is validated by this milestone. Choose the
cluster topology explicitly before launching; no cluster jobs were submitted.

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

GraphPP is the pipeline-parallel path for GraphTrainer models.
It reuses TorchTitan's eager PP module splitting and PyTorch PP schedules, then
traces one representative microbatch per local stage with GraphTrainer's
`minimal_fx_tracer`. The resulting per-stage graph bundles are reused for later
microbatches; `GraphPipelineRuntime` only executes the prebuilt callable for each PP
schedule action.

Design references:
- GraphPP RFC: https://github.com/pytorch/torchtitan/issues/3780
- CUDA-graph-compatible GraphPP RFC: https://github.com/pytorch/torchtitan/issues/3820

```bash
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel ./run_train.sh \
  --training.disable_cuda_graphs \
  --parallelism.pipeline_parallel_degree 2 \
  --parallelism.num_pp_microbatches 8 \
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
capture should target the `GraphPipelineRuntime` steady-state path in a future change,
and EP-overlap annotations will be composed with GraphPP in a later PR.

### Compiler Optimizations

GraphTrainer has a built-in pass pipeline controlled by dedicated flags.

```bash
# Full Inductor compilation (default is regional — compiles only tagged regions)
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.inductor_compilation full

# Numerics-changing optimizations (e.g. RMSNorm Inductor fusion)
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.numerics_changing_optim

# Full recompute while saving selected module operations
MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_671b ./run_train.sh \
  --compile.memory_policy full \
  --compile.full_recompute_save_ops \
  'layers.*.moe.router.gate::aten.mm.dtype | layers.*.attention.wkv_a::aten.mm.default'

# CPU activation offloading
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.memory_policy cpu_offload_all

# Disable CUDA graphs (for debugging)
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.disable_passes cuda_graph_pass

# Disable specific passes by name
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.disable_passes custom_codegen_pass,cuda_graph_pass

# Disable all graph passes (for debugging)
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.no-enable_passes
```

### Expert Parallel Overlap

EP overlap is an experimental graph-trainer optimization for MoE models with
real expert-parallel collectives. Enable it only with
`expert_parallel_degree > 1`:

```bash
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel \
    ./run_train.sh \
    --compile.ep_overlap.enabled \
    --compile.ep_overlap.chunk_dim batch \
    --compile.ep_overlap.module_fqn layers.* \
    --parallelism.data_parallel_shard_degree 4 \
    --parallelism.expert_parallel_degree 2
```

Supported chunking selections are:

- `--compile.ep_overlap.chunk_dim batch --compile.ep_overlap.module_fqn layers.*`
  for transformer-block chunking.
- `--compile.ep_overlap.chunk_dim batch --compile.ep_overlap.module_fqn layers.*.moe`
  for MoE-only batch chunking.
- `--compile.ep_overlap.chunk_dim seq --compile.ep_overlap.module_fqn layers.*.moe`
  for MoE-only sequence chunking.

Selected module forwards are wrapped with eager chunking before tracing; the
chunk annotations recorded during tracing are the contract consumed by the
EP-overlap scheduler.

Current limitations:

- EP overlap is only meaningful when `expert_parallel_degree > 1`.
- Sequence chunking is restricted to `layers.*.moe`; chunking attention over
  sequence is not supported.
- EP overlap is validated against the tested DP/EP configurations in the
  graph-trainer numerics and integration suites. Composability with additional
  overlap or sharding modes should be validated with loss comparison before use.

### Experimental AutoParallel Sharding

GraphTrainer can use AutoParallel to solve SPMD placement for supported models,
then trace and compile the placed model through the regular GraphRuntime flow.
Enable it with `--compile.enable_autoparallel`.

Llama 3 debug model:

```bash
MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel ./run_train.sh \
  --compile.enable_autoparallel \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.tensor_parallel_degree 2
```

DeepSeek V3 debug model:

```bash
MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel ./run_train.sh \
  --compile.enable_autoparallel \
  --parallelism.data_parallel_shard_degree 4 \
  --parallelism.expert_parallel_degree 2
```

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
