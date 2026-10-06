# GraphTrainer vs Eager: DeepSeek V3 671B on 256 GB300 GPUs

This release compares two TorchTitan training engines on the same workload:

- **GraphTrainer (GT)**: traces the full training step into an FX graph,
  applies graph passes (FSDP bucketing, fusion, in-graph gradient reduction)
  and runs it as a CUDA graph.
- **Eager**: the standard TorchTitan trainer, also with full training and
  validation CUDA graphs.

The model is DeepSeek V3 671B with one MTP layer, MXFP8 compute and DistMoE
expert parallelism. Each engine is measured at two parallelism topologies
(SPMD and PP2), each with two routing modes:

- **Natural**: learned top-k routing.
- **Balanced**: an exact round-robin override that removes routing imbalance.

That is 8 runs of 60 optimizer updates each. Every run completed all 60
updates with finite loss, gradient norm and validation loss.

## Contents

| Path | Contents |
|---|---|
| [README.md](README.md) | This summary |
| [graph_trainer_perf_runs_instructions.md](graph_trainer_perf_runs_instructions.md) | Public setup, recipes, launch and measurement guide |
| [graph_trainer_stack.md](graph_trainer_stack.md) | The TorchTitan commit stack used by the recipes |
| [logs/](logs/) | GT training logs, one per pipeline rank |
| [plots/](plots/) | GT vs Eager loss, gradient norm and throughput plots |
| [Full profiler traces](#full-profiler-traces) | Rank-0 PyTorch profiler traces for every run, hosted on Google Drive |

## Results

### Throughput and memory

TPS is tokens/s/GPU, averaged over updates 3..60 excluding the profiler window
35..40 (52 updates). Each sample is one document padded to 4,096 tokens, and TPS
counts all 4,096 token slots, including padding. Global TPS is TPS/GPU x 256.
GT vs Eager is `GT TPS / Eager TPS - 1`; positive means GT is faster. Peak
memory is the largest step-60 reserved-memory high-water across all 256 ranks,
covering both the training and the validation records.

| Configuration | GT TPS/GPU | Eager TPS/GPU | GT vs Eager | GT TFLOPs/GPU | GT peak GiB | Eager peak GiB | GT memory saving GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| PP2 / natural | 4,826.63 | 4,823.94 | +0.06% | 1,408.67 | 232.62 | 247.89 | 15.27 |
| PP2 / balanced | 5,505.42 | 5,498.28 | +0.13% | 1,606.78 | 223.76 | 238.90 | 15.14 |
| SPMD / natural | 5,180.15 | 5,091.13 | +1.75% | 1,511.84 | 245.51 | 258.76 | 13.25 |
| SPMD / balanced | 5,659.18 | 5,594.02 | +1.16% | 1,651.65 | 236.65 | 249.90 | 13.25 |

Key observations:

- **Throughput:** GT and Eager are at parity for PP2. GT is 1.2-1.8% faster
  for SPMD.
- **Memory:** GT uses 13.3-15.3 GiB less peak reserved memory in every
  configuration.
- **Routing:** natural routing costs about 8-12% of throughput relative to
  balanced routing in both engines.

TFLOPs/GPU is TorchTitan's analytical model-FLOP estimate. MFU is not reported
because there is no agreed MXFP8 peak for this mix of kernels. Each
configuration was run once; differences are observed values, not confidence
intervals.

The step-60 memory high-water is not necessarily the lifetime peak: for natural
SPMD Eager, the startup logs already report 261.85 GiB, above its 258.76 GiB
step-60 value. Allocation retries and OOM counters were zero on the metrics
rank.

<details>
<summary>Per-run TPS distribution and selected-rank memory</summary>

| Configuration | Engine | Mean TPS/GPU | Median | Min..max | Global TPS | Mean step s | Active / reserved GiB (loss rank) |
|---|---|---:|---:|---:|---:|---:|---:|
| PP2 / natural | GT | 4,826.63 | 4,903.24 | 4,490.72..5,007.09 | 1,235,616 | 50.983 | 99.64 / 212.72 |
| PP2 / natural | Eager | 4,823.94 | 4,905.29 | 4,481.43..5,010.80 | 1,234,929 | 51.014 | 118.92 / 247.75 |
| PP2 / balanced | GT | 5,505.42 | 5,515.02 | 5,099.12..5,555.67 | 1,409,388 | 44.645 | 90.78 / 203.86 |
| PP2 / balanced | Eager | 5,498.28 | 5,501.24 | 5,394.55..5,529.81 | 1,407,560 | 44.698 | 110.06 / 238.76 |
| SPMD / natural | GT | 5,180.15 | 5,250.01 | 4,828.79..5,355.09 | 1,326,118 | 47.493 | 98.26 / 245.51 |
| SPMD / natural | Eager | 5,091.13 | 5,153.32 | 4,777.14..5,260.70 | 1,303,330 | 48.311 | 98.26 / 258.76 |
| SPMD / balanced | GT | 5,659.18 | 5,659.47 | 5,632.41..5,680.51 | 1,448,750 | 43.427 | 89.40 / 236.65 |
| SPMD / balanced | Eager | 5,594.02 | 5,594.00 | 5,579.56..5,623.68 | 1,432,070 | 43.933 | 89.40 / 249.89 |

Active / reserved are the TensorBoard `memory/max_active(GiB)` and
`memory/max_reserved(GiB)` values at update 60 on the rank that logs metrics
(the last pipeline stage for PP). They are single-rank values and differ from
the all-rank peak in the main table.

</details>

### Training plus validation cycle time

Validation runs after every update and is not included in TPS. The cycle time
below is the wall-clock interval between consecutive training-loss log lines,
so it includes one validation pass and one training step. It averages updates
4..60 excluding 35..40 (51 intervals).

| Configuration | GT cycle s | Eager cycle s | GT vs Eager cycle time |
|---|---:|---:|---:|
| PP2 / natural | 52.567 | 52.533 | +0.06% |
| PP2 / balanced | 46.149 | 46.115 | +0.07% |
| SPMD / natural | 53.349 | 49.782 | +7.17% |
| SPMD / balanced | 49.012 | 45.312 | +8.17% |

GT's SPMD training step is faster, but its full cycle is slower because GT
validation is slower (see [Profiler findings](#profiler-findings)). Two
additional 10-update balanced SPMD runs showed the same direction: training TPS
of 5,676.36 (GT) vs 5,565.27 (Eager) over updates 3..10, a 2.0% GT advantage,
and cycle times of 49.06 s vs 45.53 s over updates 4..10. Eight samples are too
few to establish significance on their own.

### Numerics

These runs were configured to measure performance, not numerical equivalence.
They are 60-step benchmarks with `deterministic=False` and without the
numerics-alignment changes listed under
[Bitwise matching](#bitwise-matching-with-eager), so GT and Eager are expected
to be close but not bitwise identical.

| Configuration | Step-60 train loss GT / Eager | Step-60 validation loss GT / Eager | Max abs grad-norm difference (all steps) |
|---|---:|---:|---:|
| PP2 / natural | 3.870094 / 3.885731 | 3.385651 / 3.396602 | 4.556576 |
| PP2 / balanced | 5.579585 / 5.592139 | 4.992837 / 4.992821 | 10.067969 |
| SPMD / natural | 3.853219 / 3.850643 | 3.378454 / 3.371248 | 1.144872 |
| SPMD / balanced | 5.624526 / 5.617516 | 5.027276 / 5.021752 | 17.551821 |

All pairs first differ in gradient norm at update 1, in validation loss at
update 2 and in reported training loss at update 3. The learning rate is zero
at update 1, so identical early losses do not imply identical gradients.
Repeated natural-routing GT runs on the same engine also showed scalar
differences after host placement changed, so part of the GT vs Eager gap is
ordinary run-to-run variability with `deterministic=False`.

Balanced routing has higher loss than natural routing because the round-robin
override ignores the learned router.

#### Bitwise matching with Eager

GraphTrainer is designed to reproduce Eager's numerics bitwise; that was not a
goal of these performance runs. A bitwise GT vs Eager comparison needs two
things:

1. **Deterministic execution.** Set `debug.deterministic=True` with a fixed
   seed and leave `debug.deterministic_warn_only=False`. The recipes expose this
   as `GT_RUN_MODE=verify`, and the TensorBoard reader in
   [Measurements](graph_trainer_perf_runs_instructions.md#4-measurements)
   checks exact equality of the stored loss and gradient norm at every update.
2. **Numerics-alignment fixes** that make GT perform the same floating-point
   operations in the same order as Eager. Some have landed; the rest are in
   review:

| Change | Status | Effect |
|---|---|---|
| [#5121](https://github.com/pytorch/torchtitan/pull/5121) Coalesce repeated FSDP gradient reductions | In review | A parameter used on several backward paths has its local contributions summed and reduced once, as Eager does, instead of reducing each contribution separately |
| [#5118](https://github.com/pytorch/torchtitan/pull/5118) Defer FSDP persistent-grad cast after accumulation | In review | Gradients accumulate in BF16 and are cast to FP32 once after accumulation, as Eager does, instead of casting every microbatch gradient |
| Eager deferred FSDP reduction order | In this branch's [commit stack](graph_trainer_stack.md) | Eager reduces all accumulated contributions once per optimizer step instead of issuing an early partial reduction |
| Eager MXFP8 FSDP shard order | In this branch's [commit stack](graph_trainer_stack.md) | Fixes a double reordering of `Shard(1)` weights before MXFP8 quantization |
| [#5207](https://github.com/pytorch/torchtitan/pull/5207) Match eager bucketing in PP unshard | Landed | PP all-gather buckets follow Eager's FSDP grouping |
| [#5117](https://github.com/pytorch/torchtitan/pull/5117) Apply expert FSDP to all decoder blocks | Landed | Expert FSDP sharding matches Eager, including the MTP block |
| [#3770](https://github.com/pytorch/torchtitan/pull/3770) DSv3 bucketing order for multi-node bitwise numerics | Landed | FSDP bucketing order matches Eager across nodes |

For a small-scale check, `scripts/graph_trainer_runs/run_paired_numerics.py`
runs a deterministic 4-GPU paired GT/Eager gate for SPMD and PP with the
DistMoE MTP model.

Plots of training loss, validation loss, gradient norm and throughput for every
update:

| Configuration | Plot |
|---|---|
| PP2 / natural | [pp2-natural.png](plots/pp2-natural.png) |
| PP2 / balanced | [pp2-force_balanced.png](plots/pp2-force_balanced.png) |
| SPMD / natural | [spmd-natural.png](plots/spmd-natural.png) |
| SPMD / balanced | [spmd-force_balanced.png](plots/spmd-force_balanced.png) |

The shaded region in each throughput panel is the excluded profiler window. The
dip at update 50 is a scheduled Python GC.

## Profiler findings

These findings come from PyTorch profiler traces of updates 38..39, analyzing
training and validation CUDA graph launches separately.

**SPMD training: GT overlaps gradient reduce-scatter with backward.** Eager
disables gradient synchronization during gradient accumulation and reduces all
gradients after the last microbatch. GT places the reductions inside the final
forward/backward graph, so ready gradient buckets reduce while backward work
continues. Both engines launch the same 123 reduce-scatter kernels per step.
"Exposed" is reduce-scatter time not overlapped by any other GPU kernel; that
other work includes DistMoE communication and waiting as well as compute.
Means of two training replays on global rank 0:

| Routing / engine | Training graph span s | Reduce-scatter time ms | Overlapped with other GPU work ms | Exposed ms |
|---|---:|---:|---:|---:|
| balanced / GT | 43.2984 | 433.0 | 396.0 | 37.0 |
| balanced / Eager | 43.7990 | 348.2 | 26.0 | 322.2 |
| natural / GT | 46.2770 | 491.3 | 420.6 | 70.7 |
| natural / Eager | 46.4862 | 453.0 | 25.6 | 427.3 |

For balanced routing, GT's 285 ms less exposed reduce-scatter accounts for most
of its 501 ms shorter graph span (the full-window training-time saving is
506 ms). GT's reduce-scatter kernels actually run longer than Eager's, so faster
NCCL kernels are not the explanation; the gain comes from overlap. For natural
routing, GT has 357 ms less exposed reduce-scatter but 138 ms more other GPU
work; its graph span is 209 ms shorter, while the full-window training-time
saving is 818 ms, so the two captured steps explain only part of the gap.

**SPMD validation: GT issues many more parameter all-gathers.** GT validation
launches 3,472 all-gather kernels versus Eager's 123, and its validation graph
spans about 5.4-5.7 s versus 1.2-1.3 s for Eager. GT validation reads
SimpleFSDP parameters outside the bucketed training path, so its parameter
unshards are not batched. The roughly 4 s longer validation graph is
consistent with GT's slower SPMD cycle time, and batching these unshards is the
natural next optimization.

**Kernel count:** Eager SPMD training launches about 16,000 more kernels than
GT, including unfused RoPE bounds checks and router count reductions.

**PP2:** both engines launch identical numbers of all-gather, reduce-scatter
and PP send/recv kernels per training replay.

**Allocator state.** Separate 3-update runs captured allocator snapshots on
global rank 0 after the update-3 validation. The CUDA graph private pool is
about 10 GiB smaller for GT on PP2:

| Configuration / engine | Reserved GiB | Allocated GiB | CUDA graph private pool GiB |
|---|---:|---:|---:|
| PP2 / balanced / GT | 223.76 | 105.58 | 127.87 |
| PP2 / balanced / Eager | 235.66 | 106.50 | 138.01 |
| PP2 / natural / GT | 232.62 | 114.44 | 127.87 |
| PP2 / natural / Eager | 244.50 | 115.36 | 138.01 |

## Workload

| Area | Configuration |
|---|---|
| Hardware | 256 NVIDIA GB300 GPUs: 64 hosts x 4 GPUs. Each block of 64 consecutive ranks shares an NVLink domain (required by DistMoE EP64). |
| Model | DeepSeek V3 671B: 61 decoder layers plus one MTP layer (MTP loss weight 0.1) |
| Batch | Sequence length 4,096; microbatch size 1; global batch 15,360 sequences (62,914,560 tokens per update); one document per sample, padded to 4,096 |
| Data | C4 (English), tokenized with the DeepSeek V3 tokenizer, in a fixed sample order shared by both engines |
| Initial weights | Pretrained BF16 DeepSeek V3 checkpoint (Hugging Face format, including the MTP layer); no checkpoints are saved during the run |
| Dense compute | TorchAO MXFP8 for attention projections, dense and shared-expert FFNs, SwiGLU and LM head |
| Attention | Causal BF16 SDPA (cuDNN / FlashAttention); fused MLA and RoPE; no tensor or context parallelism |
| Routed experts | DistMoE: 256 experts, top-8, staged asynchronous dispatch, MXFP8 grouped GEMMs (E4M3, fast-math kernels), BF16 backward activation gather |
| Router | Sigmoid; 8 groups with 4 selected; route scale 2.5; sequence auxiliary loss 0.01; bias update rate 0.001 |
| Precision | FP32 master parameters and optimizer updates; BF16 gathered parameters and reduced gradients |
| Optimizer | Fused AdamW with BF16 moments; lr 2.3238e-5; betas 0.9 / 0.95; eps 1e-8; weight decay 0 / 0.033 (selective); gradient clip 1.0 |
| LR schedule | 4 warmup steps from zero, cosine decay over 11,996 steps, minimum factor 0.21516481624924694 |
| Memory policy | No activation checkpointing, no CPU offload, no `torch.compile` |
| FSDP | `reshard_after_forward="never"`; automatic unshard lookahead; no symmetric-memory collectives; Eager defers gradient reduction to the end of accumulation |
| Pipeline | 16 logical stages, Interleaved1F1B schedule; static receive buffers; per-edge directed P2P (`TORCH_DISTRIBUTED_PIPELINE_PER_EDGE_P2P=1`); pending-send budget 16 |
| CUDA graphs | Full training and validation step graphs, shared memory pool; update 1 warms up, update 2 captures, update 3 onward replays. The optimizer runs outside the graph. |
| GT passes | Fusion enabled; numerics-changing optimizations disabled |
| Validation | After every update: 1,024 sequences globally, main LM loss |
| Determinism / GC | Seed 14,536; `deterministic=False`; Python GC every 50 steps |
| Profiler | Frequency 39, warmup 3, active 2, repeat 1: updates 35..37 warm up, 38..39 are recorded. TensorBoard logs every step. |
| Allocator / NCCL | `expandable_segments:True,graph_capture_record_stream_reuse:True`; public NCCL with DMA-BUF enabled, CTRAN disabled, 1,200 s timeout |

| Topology | PP / virtual stages per rank | DP / EP / expert FSDP | Train microbatches per DP rank | Validation microbatches | DistMoE activation slots per PP rank |
|---|---|---|---:|---:|---|
| SPMD | 1 / 1 | 256 / 64 / 4 | 60 | 4 | one shared arena |
| PP2 | 2 / 8 | 128 / 64 / 2 | 120 | 8 | 17 / 15 |

DistMoE preallocates expert scratch buffers sized by a capacity factor (CF).
Balanced runs use CF1. Natural runs use CF8 (8.86 GiB/GPU extra), because
learned routing exceeded the CF1 and CF2 buffers in these runs. CF8 was
sufficient here; it is not a guaranteed bound for arbitrary routing. All runs
use CF1 for saved activations and do not use VMM-backed buffers.

## Software

The public, pinned software stack is described in
[graph_trainer_perf_runs_instructions.md](graph_trainer_perf_runs_instructions.md#1-setup-and-tokenizer)
and [graph_trainer_stack.md](graph_trainer_stack.md): TorchTitan commit
`3515ae4430890bbb77007b2a072bfef29bfbd642`, PyTorch nightly
`2.16.0.dev20261013+cu132`, TorchAO nightly `0.19.0.dev20261014+cu132`,
FlashAttention 4, CUTLASS DSL and
[DistMoE](https://github.com/meta-pytorch/dist_moe).

The measured runs used pre-release builds of the same components: PyTorch
based on `dc0efe32b9ee884dfe0b56e24b3f49a954cfc543` (CUDA 13.0, SM103a),
DistMoE `d7ef47272afe0048cc9242487d0e5a44dd209343`, TorchAO
0.19.0+git6ded49303, CUTLASS 4.6.2 and FlashAttention 4.0.0b25. The
natural-routing runs also include a PP fix that propagates representative
values for dynamic activation metadata during CUDA graph capture. SPMD runs in
both engines set `TORCH_CUDAGRAPH_GC=1`.

## Reproducing

Follow [graph_trainer_perf_runs_instructions.md](graph_trainer_perf_runs_instructions.md).
It covers environment setup, data download, an NCCL / NVLink placement
preflight, the launch command and a TensorBoard reader that computes TPS,
memory and GT/Eager ratios. Each of the 8 runs maps to one factory in
`scripts/graph_trainer_runs/perf_configs.py`:

| Configuration | GT config | Eager config |
|---|---|---|
| SPMD / natural | `spmd_256gpu_graph_trainer_natural` | `spmd_256gpu_eager_natural` |
| SPMD / balanced | `spmd_256gpu_graph_trainer_force_balanced` | `spmd_256gpu_eager_force_balanced` |
| PP2 / natural | `pp2_256gpu_graph_trainer_natural` | `pp2_256gpu_eager_natural` |
| PP2 / balanced | `pp2_256gpu_graph_trainer_force_balanced` | `pp2_256gpu_eager_force_balanced` |

```bash
torchrun --nnodes=64 --nproc-per-node=4 \
  --node-rank="$NODE_RANK" \
  --rdzv-backend=static --rdzv-endpoint="$MASTER_ADDR:29500" \
  -m torchtitan.train \
  --module scripts.graph_trainer_runs.perf_configs \
  --config pp2_256gpu_graph_trainer_natural \
  --output-dir /shared/outputs/pp2-gt-natural
```

Set the tokenizer path, allocator settings and other environment variables
listed in the instructions before launching.

What to expect when comparing against these results:

- **Throughput and memory** should be directly comparable on the same hardware
  and NVLink placement.
- **Absolute loss values will differ.** The public recipes stream public C4,
  initialize weights from seed 42 and use the public package pins. The measured
  runs used a frozen, pre-tokenized C4 copy, a pretrained checkpoint, seed
  14,536 and the pre-release builds listed above.
- **Use the public measurement window.** The instructions recommend updates
  11..60 excluding 35..40 and three repetitions per configuration. The numbers
  above use updates 3..60 excluding 35..40 from a single run each.

## Released artifacts

| Artifact | Coverage | Location |
|---|---|---|
| Training logs | GT, all 4 configurations: SPMD global rank 0; PP2 global ranks 0/128 (one per pipeline rank) | [logs/](logs/) |
| Loss / grad-norm / TPS plots | GT vs Eager, all 4 configurations | [plots/](plots/) |
| Full profiler traces | GT and Eager, all 4 configurations, global rank 0 | [Google Drive](https://drive.google.com/drive/folders/1jD3buLCpwvsFXoffNAHDIctADcLroRKC?usp=sharing) |

The logs are the complete torchtitan stdout for each rank. Only the last
pipeline stage computes the loss; earlier stages print a placeholder negative
loss. Gradient norm, memory, TPS and validation lines are reported on every
rank.

### Full profiler traces

Complete PyTorch profiler traces of global rank 0 for every run, hosted on
[Google Drive](https://drive.google.com/drive/folders/1jD3buLCpwvsFXoffNAHDIctADcLroRKC?usp=sharing)
(about 105-115 MB compressed, about 2 GB uncompressed each). Each trace covers
the profiled updates 38..39: two complete training steps, the validation passes
and the optimizer steps, with CPU ops, CUDA runtime calls and GPU kernels. CUDA
graph streams are regrouped into semantic lanes (compute, NCCL all-gather,
reduce-scatter, and so on); event timestamps and durations are unchanged. The
profiler findings above were measured on the same captures. Download a file and open
it with **Open trace file** in [Perfetto](https://ui.perfetto.dev/); a trace this
size needs several GB of browser memory. Training and validation are separate
CUDA graph launches: select the training `cudaGraphLaunch` and follow its
correlated GPU kernels to inspect one training step. File names in the folder:

| Configuration | GT trace | Eager trace |
|---|---|---|
| SPMD / natural | `spmd_graph_trainer_natural_rank0.json.gz` | `spmd_eager_natural_rank0.json.gz` |
| SPMD / balanced | `spmd_graph_trainer_balanced_rank0.json.gz` | `spmd_eager_balanced_rank0.json.gz` |
| PP2 / natural | `pp2_graph_trainer_natural_rank0.json.gz` | `pp2_eager_natural_rank0.json.gz` |
| PP2 / balanced | `pp2_graph_trainer_balanced_rank0.json.gz` | `pp2_eager_balanced_rank0.json.gz` |

Not included: allocator snapshots, and tlparse graph dumps, which were not
captured for these runs.
