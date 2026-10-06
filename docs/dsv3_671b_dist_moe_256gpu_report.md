# DeepSeek V3 671B DistMoE 256-GPU Performance Report

Date: 2026-10-14. Hardware: 256 NVIDIA GB300 GPUs, 64 four-GPU hosts in QNO.

This report compares GraphTrainer (GT) with the Eager training engine for PP2,
PP4, and SPMD, using both learned natural routing and exact round-robin forced
balancing. **Both engines use full training and validation CUDA graphs.** The
historical comparison follows [Eager Prev Report](https://docs.google.com/document/d/1W1IIuFle-0HMg64piGN6RUiHzQc0r-eYbktsMfjZVz8/edit?tab=t.0),
last updated 2026-10-05. Its PP2/PP4 reference jobs use the Eager PP engine.

## Executive summary

All 12 main jobs completed 60 updates. Every current native attempt reports
COMPLETE, with all 64 workers COMPLETE and exit code 0. Every run has 60 finite
records for training mean loss, training max loss, gradient norm, and validation
loss. This establishes execution success; numerical equivalence remains open.

Training TPS is nearly equal for PP2 and balanced PP4. In the primary window,
balanced PP4 GT is 0.02% faster than Eager; natural PP4 GT is 1.77% slower.
GT is faster in SPMD training, but Eager completes shorter cycles when
every-update validation is included. GT uses less peak reserved memory in
all six matched pairs.

PP4 GT values throughout this report use the completed parameter-preparation
fix jobs listed below. Their workload fields are unchanged, their native
runtimes match the Eager references, and all 64 workers completed with exit
code 0 at native version 0 / attempt 0. Each comparison uses the selected
current jobs, including all derived TPS ratios, memory savings, cycles,
numerical deltas, plots and PP4 profile links.

**TPS below is tokens/s/GPU**, averaged over the 53-update reference window
defined below, using the same window for GT, Eager and Eager Prev Report.
Positive Eager delta means Eager has higher training TPS. Memory is the largest
step-60 reserved-memory high-water reported across all 256 physical ranks, rounded
to two decimal GiB. Eager Prev Report memory covers representative PP coordinates.
The detailed training statistics retain the primary 52-update comparison.

| Configuration | GT TPS/GPU | Eager TPS/GPU | Eager Prev Report TPS/GPU | TPS natural/balanced: GT / Eager | Eager delta | GT peak GiB | Eager peak GiB | Eager Prev Report peak GiB | GT saving GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PP2 / natural | 4,825.92 | 4,819.33 | -- | 0.88 / 0.88 | -0.14% | 232.62 | 247.89 | -- | 15.27 |
| PP2 / balanced | 5,502.46 | 5,495.09 | 5,324.21 | -- | -0.13% | 223.76 | 238.90 | 243.86 | 15.14 |
| PP4 / natural | 4,676.60 | 4,752.14 | -- | 0.87 / 0.89 | +1.62% | 232.48 | 243.12 | -- | 10.64 |
| PP4 / balanced | 5,361.71 | 5,360.78 | 5,178.22 | -- | -0.02% | 223.63 | 234.33 | 238.08 | 10.70 |
| SPMD / natural | 5,178.84 | 5,076.22 | -- | 0.92 / 0.91 | -1.98% | 245.51 | 258.76 | -- | 13.25 |
| SPMD / balanced | 5,656.79 | 5,591.63 | -- | -- | -1.15% | 236.65 | 249.90 | -- | 13.25 |

The 60-step jobs are profiling benchmarks, not convergence studies. No pair
has bitwise-identical scalar series. Balanced PP4 Eager has a finite gradient-norm
spike of **927.093994 at update 18**, versus GT **1.818364** at that update.
Performance parity and numerical correctness must be assessed separately.

## Measurement contract

1. The selected TensorBoard series is the loss-stage `throughput(tps)` record.
   Warmup is update 1, capture is update 2, and replay starts at update 3.
2. Primary matched comparison: updates **3..60 excluding 35..40**, 52 samples.
   The actual profiler schedule warms at 35..37, records at 38..39, and exports
   at 39; update 40 is also excluded for export effects.
3. Historical comparison: updates **3..60 excluding 39..43**, 53 samples, exactly
   the window specified in Eager Prev Report's document. That window retains profiler
   warmup/active updates 35..38 in these replicas, so it is shown separately.
4. Sensitivity check: updates **3..60 excluding 35..43**, 49 samples. GC at update
   50 and other remaining outliers are retained. Each arm has one full-length
   run; the percentages are observed differences, not confidence intervals.
5. TPS and `time_metrics/end_to_end(s)` describe training only. Validation runs
   after every update and is excluded from those timers. Mean global TPS is
   mean TPS/GPU multiplied by 256. Mean TPS is not recomputed from mean step time.
6. TFLOPs/GPU is TorchTitan's analytical model-FLOP estimate. No MXFP8 MFU is
   reported because an appropriate peak denominator is not defined.
7. Main memory values cover all 256 training and all 256 validation step-60
   records, not only the selected loss rank. TensorBoard allocation retries
   and OOM counters are zero on the selected rank; this is not an all-rank
   lifetime counter audit. For natural SPMD Eager, startup logs already report
   261.85 GiB, above its 258.76 GiB step-60 figure.
8. Quantitative trace analysis uses original traces and CUDA-graph launch
   correlations, separating training from validation. Compacted traces are
   visualization artifacts. Kernel-duration sums and category unions
   overlap and are not additive estimates of recoverable step time.

### Matched throughput and active/reserved memory

The following values come from each completed benchmark's loss-stage TensorBoard
training series at update 60. Active is `active_bytes.all.peak`; reserved is
`reserved_bytes.all.peak`. Both refer to the logged training interval on the
selected physical rank. They are distinct from the all-rank reserved high-water
table above and the separate update-3 allocator snapshots. All values are GiB.

TPS/GPU in this table uses Eager Prev Report's 53-update window, updates 3..60 excluding
39..43, matching the executive summary. Detailed training statistics below
also provide the primary 52-update window.
The Eager Prev Report columns refer to its original balanced Eager jobs. The active/reserved
split is extracted from their loss-stage TensorBoard records; the document's
peak reserved HBM is reported separately below. There is no natural or SPMD
reference in the document.

For each natural row, TPS natural/balanced is the ratio of that engine's natural
TPS to its balanced TPS at the same topology, using the same measurement window.
The two values are ordered GT / Eager.

| Configuration | GT TPS/GPU | Eager TPS/GPU | Eager Prev Report TPS/GPU | TPS natural/balanced: GT / Eager | GT active / reserved GiB | Eager active / reserved GiB | Eager Prev Report active / reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| PP2 / natural | 4,825.92 | 4,819.33 | -- | 0.88 / 0.88 | 99.64 / 212.72 | 118.92 / 247.75 | -- |
| PP2 / balanced | 5,502.46 | 5,495.09 | 5,324.21 | -- | 90.78 / 203.86 | 110.06 / 238.76 | 91.35 / 236.66 |
| PP4 / natural | 4,676.60 | 4,752.14 | -- | 0.87 / 0.89 | 89.90 / 175.27 | 124.96 / 232.43 | -- |
| PP4 / balanced | 5,361.71 | 5,360.78 | 5,178.22 | -- | 81.04 / 166.40 | 116.10 / 223.65 | 81.17 / 194.93 |
| SPMD / natural | 5,178.84 | 5,076.22 | -- | 0.92 / 0.91 | 98.26 / 245.51 | 98.26 / 258.76 | -- |
| SPMD / balanced | 5,656.79 | 5,591.63 | -- | -- | 89.40 / 236.65 | 89.40 / 249.89 | -- |

## Training statistics

The same 52-update window is used for every row. Every Eager value in this section comes from our matched Eager job listed below; Eager Prev Report's historical jobs are reported separately.

| Configuration | Engine | Mean TPS/GPU | Median | Min..max | Global TPS | Mean train s | TFLOPs/GPU |
|---|---|---:|---:|---:|---:|---:|---:|
| PP2 / natural | GT | 4,826.63 | 4,903.24 | 4,490.72..5,007.09 | 1,235,616 | 50.983 | 1,408.67 |
| PP2 / natural | Eager | 4,823.94 | 4,905.29 | 4,481.43..5,010.80 | 1,234,929 | 51.014 | 1,407.88 |
| PP2 / balanced | GT | 5,505.42 | 5,515.02 | 5,099.12..5,555.67 | 1,409,388 | 44.645 | 1,606.78 |
| PP2 / balanced | Eager | 5,498.28 | 5,501.24 | 5,394.55..5,529.81 | 1,407,560 | 44.698 | 1,604.69 |
| PP4 / natural | GT | 4,669.63 | 4,733.10 | 4,272.60..4,926.33 | 1,195,424 | 52.695 | 1,362.85 |
| PP4 / natural | Eager | 4,753.55 | 4,829.09 | 4,313.53..4,971.21 | 1,216,909 | 51.790 | 1,387.34 |
| PP4 / balanced | GT | 5,365.26 | 5,372.52 | 5,154.23..5,400.91 | 1,373,506 | 45.808 | 1,565.87 |
| PP4 / balanced | Eager | 5,364.33 | 5,366.62 | 5,256.58..5,409.70 | 1,373,269 | 45.814 | 1,565.60 |
| SPMD / natural | GT | 5,180.15 | 5,250.01 | 4,828.79..5,355.09 | 1,326,118 | 47.493 | 1,511.84 |
| SPMD / natural | Eager | 5,091.13 | 5,153.32 | 4,777.14..5,260.70 | 1,303,330 | 48.311 | 1,485.86 |
| SPMD / balanced | GT | 5,659.18 | 5,659.47 | 5,632.41..5,680.51 | 1,448,750 | 43.427 | 1,651.65 |
| SPMD / balanced | Eager | 5,594.02 | 5,594.00 | 5,579.56..5,623.68 | 1,432,070 | 43.933 | 1,632.64 |

## Comparison with Eager Prev Report's 256-GPU runs

The comparable historical arms are **balanced PP2 and balanced PP4**. The
document has no 256-GPU natural-routing or SPMD reference. The older October 6
report provides the presentation format; its GBS 4,096 workload is not used as
a throughput baseline for this GBS 15,360 experiment.

The following table uses the identical 53-update historical window for all arms.
The runtime and source revisions differ, and our runs use QNO while the reference
runs used LCO. Consequently, these deltas do not isolate the effect of either engine.

Peak active columns below are loss-stage TensorBoard training high-waters at update 60. Peak reserved columns cover representative PP coordinates for Eager Prev Report, and all 256 ranks for GT/Eager. All memory values are GiB; the document itself does not report an all-rank active peak.

| Topology | Eager Prev Report TPS/GPU | GT TPS/GPU (vs Eager Prev Report) | Eager TPS/GPU (vs Eager Prev Report) | Eager Prev Report peak active GiB | GT peak active GiB | Eager peak active GiB | Eager Prev Report peak reserved GiB | GT peak reserved GiB | Eager peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [PP2 / balanced](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-muse-spark-1-2-OSS-0fae39-p2p256-v1-256-sanketpurandare) | 5,324.21 | 5,502.46 (+3.35%) | 5,495.09 (+3.21%) | 91.35 | 90.78 | 110.06 | 243.86 | 223.76 | 238.90 |
| [PP4 / balanced](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-muse-spark-1-2-OSS-6b803c-p4p256-v1-256-sanketpurandare) | 5,178.22 | 5,361.71 (+3.54%) | 5,360.78 (+3.53%) | 81.17 | 81.04 | 116.10 | 238.08 | 223.63 | 234.33 |

Eager Prev Report reports peak reserved HBM over representative PP coordinates at step 60;
the current comparison covers every physical rank. The PP2 reference has mean
training time 46.1608 s and analytical throughput 1,553.89 TFLOPs/GPU; PP4 has
47.4625 s and 1,511.28 TFLOPs/GPU. Selected loss-rank reserved memory is smaller
than the reference peak and is not substituted for it.

## Training plus validation cycles

These are 51 consecutive training-loss log wall-time intervals: current updates
4..60 excluding 35..40. Each interval includes the preceding validation,
bookkeeping, and the next training update. The first interval starts at update
4 to avoid including update-2 validation capture. This is an observed wall-cycle
proxy, not an isolated validation GPU timer. Positive delta means a higher
Eager cycle rate, computed as GT mean cycle time / Eager mean cycle time - 1.

| Configuration | GT cycle s | Eager cycle s | Eager cycle-rate delta |
|---|---:|---:|---:|
| PP2 / natural | 52.567 | 52.533 | +0.06% |
| PP2 / balanced | 46.149 | 46.115 | +0.07% |
| PP4 / natural | 54.617 | 53.651 | +1.80% |
| PP4 / balanced | 47.565 | 47.501 | +0.14% |
| SPMD / natural | 53.349 | 49.782 | +7.16% |
| SPMD / balanced | 49.012 | 45.312 | +8.17% |

Two additional balanced SPMD 10-update repeats completed successfully. Over
replay updates 3..10, GT/Eager training TPS is 5,676.36/5,565.27 (-1.96% Eager),
and mean cycles over updates 4..10 are 49.063/45.534 s (+7.75% Eager cycle rate).
The direction repeats, but eight TPS samples and seven intervals do not replace
the full-length comparison or establish statistical significance.

## Configuration and reproduction

### Common contract

| Area | Configuration |
|---|---|
| Hardware/scheduler | 64 whole four-GPU GB300 hosts; subtype 200025; QNO; MuseSpark_1_2_Safety_DCT; HIGH, sub-priority 1 |
| Model | DeepSeek V3 671B; 61 decoder layers plus one routed MTP layer; MTP loss weight 0.1 |
| Batch/data | MBS1; sequence 4,096; GBS15,360; 62,914,560 tokens/update; frozen indexed MLPerf C4 |
| Data source | AirStore 590827 train, frozen 590819 validation; exact global sequence/microbatch order; one document per sample |
| Checkpoint | Official warmed BF16 model-only HF assets; 163 shards, 46,183 keys including MTP layer61; no checkpoint save |
| Assets path | `/mnt/mffuse/mlperf_dsv3_convergence/deepseekv3_checkpoint_bf16_hf_assets` |
| Dense compute | TorchAO MXFP8 attention projections, dense/shared-expert FFNs, fused SwiGLU and LM head |
| Attention | Dense causal BF16 SDPA; cuDNN/Flash selection; fused MLA/RoPE; no context or tensor parallelism |
| Routed experts | Staged asynchronous MXFP8 DistMoE; 256 experts; top8; grouped E4M3 fast-math kernels; BF16 backward activation gather |
| Router | Sigmoid; 8 groups / 4 selected groups; scale2.5; sequence auxiliary objective0.01 retained; bias update0.001 retained |
| Routing arms | Natural learned top-k; balanced exact round-robin override |
| Parameter precision | FP32 persistent/master parameters and updates; BF16 gathered parameters and reduced gradients |
| Optimizer | Fused AdamW; BF16 moments; lr2.3238e-5; betas0.9/0.95; eps1e-8; selective decay0/0.033; global clip1 |
| LR schedule | Warmup4 from zero; cosine horizon 11,996; min factor0.21516481624924694; horizon unchanged for short repeats |
| Activation policy | No outer activation checkpointing, no CPU offload, no torch.compile |
| FSDP | Reshard after forward never; no symmetric-memory collectives; automatic unshard lookahead; Eager deferred gradient reduction/waits |
| PP runtime | 16 logical stages; Interleaved1F1B; static receive buffers; directed P2P; pending-send budget16 |
| CUDA graphs | Full train and validation graphs; shared pool; warmup1, capture2, replay3; optimizer graph disabled |
| GT optimization | Fusion enabled; numerics-changing optimizations disabled |
| Validation | Every update; global1,024 sequences; main LM loss is the validation metric |
| Debug/GC | Seed14,536; deterministicFalse; deterministic_warn_onlyFalse; GC frequency50 |
| Profiler | Frequency39; warmup3; active2; repeat1; TensorBoard every step; benchmark allocator snapshots disabled |
| Allocator/communication | Expandable segments and graph capture record-stream reuse; DMA-BUF enabled; public NCCL; CTRAN disabled; timeout1,200s |

### Geometry

| Topology | PP / VPP | DP / EP / expert-FSDP | Train microbatches | Validation microbatches | DistMoE slots by PP coordinate |
|---|---|---|---:|---:|---|
| PP2 | 2 / 8 | 128 / 64 / 2 | 120 | 8 | 17 / 15 |
| PP4 | 4 / 4 | 64 / 64 / 1 | 240 | 16 | 19 / 17 / 15 / 13 |
| SPMD | 1 / 1 | 256 / 64 / 4 | 60 | 4 | One non-PP arena |

Balanced arms use CF1 device scratch. Natural-routing arms use **CF8 scratch**,
CF1 saved-activation capacity, and no VMM. They are declared derivatives of the
report workload: CF1 and CF2 encountered real natural-routing capacity traps.
CF8 adds 8.859375 GiB/GPU of device scratch and completed all six natural arms.
It is a measured sufficient reserve for this run, not a worst-case routing bound.
No forced balance is introduced in those natural arms.

### Frozen software

The measured source starts from frozen benchmark revision
`fef70b9b7ea44237c9d71c74217bbe32bb7e85c7`, including report adapters and reviewed
runtime fixes. Both PP4 GT jobs additionally backport parameter-preparation fix
`acb604989d9c446a16d907c096a9463205009cda`. The derivation records six GraphTrainer
files, matching source SHA256s in both workspaces, unchanged native packages,
and zero changed workload fields. This moves FSDP1 parameter preparation into
UNSHARD for PP. The report-bearing branch head is separate from these
executable benchmark sources.

| Arm | Immutable runtime fbpkg | Workspace fbpkg | PyTorch Python source |
|---|---|---|---|
| Balanced PP2/SPMD and PP4 Eager | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v2:b836fb1a52304c769916013e3f9c5181` | `torchtitan_workspace:44daefd757084e26b8b98cc136298f88` | `b30cdcd7603d54a05f311f97889037429a18708e` |
| Natural CF8 PP2/SPMD and PP4 Eager | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` | `torchtitan_workspace:6703336a972d40cdb183f9bd58f347f8` | `149ae98d98db6921d66e7ba527b395294a127b6e` |
| PP4 GT balanced, parameter-preparation fix | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v2:b836fb1a52304c769916013e3f9c5181` | `torchtitan_workspace:3b98b4582d3e480da7f02be6b5b9944c` | `b30cdcd7603d54a05f311f97889037429a18708e` |
| PP4 GT natural CF8, parameter-preparation fix | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` | `torchtitan_workspace:8692e7447cc74779a8f0eebaa24dafcc` | `149ae98d98db6921d66e7ba527b395294a127b6e` |

Both use native PyTorch base
`dc0efe32b9ee884dfe0b56e24b3f49a954cfc543` (CUDA13.0, SM103a),
DistMoE `d7ef47272afe0048cc9242487d0e5a44dd209343`, TorchAO0.19.0+git6ded49303,
CuTe/CUTLASS4.6.2 and FlashAttention4.0.0b25. The natural runtime includes the
generic PP representative-value propagation fix for dynamic activation metadata
during CUDA graph capture. The SPMD arms also use
`TORCH_CUDAGRAPH_GC=1` in both engines.

The runtime in Eager Prev Report is
`torchtitan_conda_sanket_mlperf_convergence_20260930_v3:2b311c1d3225410388d39a693c81ca7d`,
with CUDA13.1, cuDNN9.23.1, NCCL2.30.7, CuTe4.6.0.dev0 and
TorchAO0.19.0+git597c0aee4. The reference TorchTitan sources are
`0fae39dc0be9ac6695accfa44839353ccb29e428` (PP2) and
`6b803cdaaba41c5e86d5f88efd859bce892b69bb` (PP4).

### Reproduce an arm

Public setup and recipes for the 12-job topology/routing matrix are in
[GraphTrainer performance run instructions](graph_trainer_perf_runs_instructions.md).
Those recipes use public C4 and seed-initialized weights; the measured jobs
below use the frozen data/checkpoint and runtime described in this report.

Use the archived native job definition, full resolved configuration, immutable
package identities, and source/provenance bundle. Preserve the data/checkpoint
mounts, 64 hosts x 4 GPUs, all environment controls, and trainer selection.
Choose a new job/output/rendezvous name so existing benchmark artifacts remain
available. The worker entry point is:

```bash
python -m torchtitan.train \
  --module scripts.graph_trainer_runs.report_exact_configs \
  --config pp2_256gpu_graph_trainer_force_balanced \
  --output-dir /mnt/mffuse/outputs/NEW_JOB_NAME
```

For a natural-routing arm use module `report_natural_cf8_configs`; select the
corresponding `pp2_256gpu_*_natural`, `pp4_256gpu_*_natural`, or
`spmd_256gpu_*_natural` factory. Eager uses the matching `_eager_` factory.
The modules are in the frozen benchmark packages; the report-bearing checkout
alone does not contain that executable bundle. The native definition supplies
the distributed torchrun command, package mounts, CUDA compatibility libraries,
AirStore dependencies and strict rank-0 worker preflight.

Clone the chosen `main-results/<JOB>/mast-job-definition.json` from the evidence
bundle as `new-job-definition.json`. Update the job/output/rendezvous identities
throughout the definition, including the corresponding environment and metadata
fields. Keep the workload, geometry, priority, region, dataset and checkpoint
controls intact. If a package expired, republish its verified archived payload
and consistently replace that package identifier in the definition.

```bash
mslsc schedule --dry-run new-job-definition.json
mslsc schedule --freeze new-job-definition.json
mslsc unfreeze NEW_JOB_NAME:0
mslsc get-status --json NEW_JOB_NAME
```

Use the actual native version returned for the new name in the unfreeze command.
The dry run checks the definition; rank-0 worker preflight must also pass before
training begins.

Run all 60 updates with the scheduler horizon 11,996, every-update validation,
and profiler schedule intact. Recompute the three stated windows from raw
TensorBoard event files; collect every rank's step-60 memory line. Ephemeral
packages may expire; archive/source identities do not extend package lifetime.

## Completed jobs and native attempt selection

| Configuration | GraphTrainer job | Version / attempt | Eager job | Version / attempt |
|---|---|---|---|---|
| PP2 / natural | [GT](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report256-pp2-gt-natural-20261013-v2-ivankobzarev) | 7 / 0 | [Eager](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report256-pp2-eager-natural-20261013-v2-ivankobzarev) | 7 / 0 |
| PP2 / balanced | [GT](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report256-pp2-gt-force-balanced-20261013-v2-ivankobzarev) | 2 / 0 | [Eager](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report256-pp2-eager-force-balanced-20261013-v2-ivankobzarev) | 2 / 1 |
| PP4 / natural | [GT](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report256-pp4-gt-localprep-natural-20261014-ivankobzarev) | 0 / 0 | [Eager](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report256-pp4-eager-natural-20261013-v2-ivankobzarev) | 7 / 0 |
| PP4 / balanced | [GT](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report256-pp4-gt-localprep-balanced-20261014-ivankobzarev) | 0 / 0 | [Eager](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report256-pp4-eager-force-balanced-20261013-v2-ivankobzarev) | 2 / 0 |
| SPMD / natural | [GT](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report256-spmd-gt-natural-20261013-v2-ivankobzarev) | 7 / 1 | [Eager](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report256-spmd-eager-natural-20261013-v2-ivankobzarev) | 7 / 0 |
| SPMD / balanced | [GT](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report256-spmd-gt-force-balanced-20261013-v2-ivankobzarev) | 2 / 0 | [Eager](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report256-spmd-eager-balanced-20261013-v2-ivankobzarev) | 2 / 0 |

Both PP4 GT fix jobs completed native version 0 / attempt 0 with zero restarts.
Other job names containing `v2` do not identify the native execution version.
The remaining natural jobs use native version 7. SPMD natural GT uses
successful attempt 1 after dedicated-capacity preemption of attempt 0;
PP2 balanced Eager also uses attempt 1 after frozen staging. These selected
attempts retain their frozen controls and are not numerical-failure restarts.

## Numerical evidence

The following are step-60 training mean loss and validation main-LM loss.
Maximum gradient-norm differences are over all 60 updates, not only step 60.

| Configuration | Train loss GT / Eager | Validation loss GT / Eager | Max absolute grad-norm difference |
|---|---:|---:|---:|
| PP2 / natural | 3.870094 / 3.885731 | 3.385651 / 3.396602 | 4.556576 |
| PP2 / balanced | 5.579585 / 5.592139 | 4.992837 / 4.992821 | 10.067969 |
| PP4 / natural | 3.836451 / 3.834770 | 3.378187 / 3.352827 | 1.327934 |
| PP4 / balanced | 5.561720 / 5.568828 | 4.974211 / 4.981694 | 925.275630 |
| SPMD / natural | 3.853219 / 3.850643 | 3.378454 / 3.371248 | 1.144872 |
| SPMD / balanced | 5.624526 / 5.617516 | 5.027276 / 5.021752 | 17.551821 |

All six pairs first differ in gradient norm at update 1, validation at update 2,
and reported training losses at update 3. The initial LR multiplier is zero;
equal initial losses do not prove equal gradients or optimizer state.
Same-engine natural GT repeats also show scalar differences after placement
changes. This is evidence of observed run variability, not a root-cause assignment
to atomics, stream ordering, NCCL, or a specific kernel.

The PP4 spike, numerical drift, and convergence remain required follow-ups.
Clipping and finite metrics do not close them. Detailed raw scalar series and
pair plots are retained in the evidence archive.

### Paired scalar series

The plots retain all loss and gradient-norm samples. Their TPS panel starts at update3 and shades the primary excluded profiler window 35..40; the update 50 GC outlier remains visible. Gradient norm uses a logarithmic axis. Curves are observed scalar series, not numerical-equivalence proof.

| Configuration | PNG | SVG |
|---|---|---|
| PP2 / natural | [PNG](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pair-plots/pp2-natural.png) | [SVG](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pair-plots/pp2-natural.svg) |
| PP2 / balanced | [PNG](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pair-plots/pp2-force_balanced.png) | [SVG](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pair-plots/pp2-force_balanced.svg) |
| PP4 / natural | [PNG](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pp4-localprep-refresh-20261014/pair-plots/pp4-natural.png) | [SVG](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pp4-localprep-refresh-20261014/pair-plots/pp4-natural.svg) |
| PP4 / balanced | [PNG](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pp4-localprep-refresh-20261014/pair-plots/pp4-force_balanced.png) | [SVG](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pp4-localprep-refresh-20261014/pair-plots/pp4-force_balanced.svg) |
| SPMD / natural | [PNG](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pair-plots/spmd-natural.png) | [SVG](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pair-plots/spmd-natural.svg) |
| SPMD / balanced | [PNG](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pair-plots/spmd-force_balanced.png) | [SVG](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pair-plots/spmd-force_balanced.svg) |

## Profiler findings

The training and validation graph launches were analyzed separately on selected
physical coordinates. SPMD training has 123 named NCCL all-gather and 123
reduce-scatter kernel events per measured replay in both engines. GT SPMD validation has 3,472
all-gather kernels versus Eager's 123. In the selected original captures,
GT validation spans approximately 5.36-5.67s; Eager spans 1.16-1.28s across the
balanced/natural runs and selected ranks/replays.

GT validation directly reads SimpleFSDP parameters outside training's bucketing
path, which supports investigating parameter-unshard batching. The counts do
not establish equal bytes, exact operand/order equivalence, or the entire cause
of the cycle gap. A same-runtime native A/B must validate an optimization.

Eager SPMD training has 16,166 additional kernels in the selected comparison,
including unfused RoPE bounds checks and router count reductions. The router's
masked counts are required; removing unconsumed auxiliary counts requires a
proper producer API and SPMD/autograd ownership contract. Kernel sums are not
critical-path savings, and no speculative API patch is included here.

PP2 captures have equal selected collective counts in the two engines. Current
PP4 original profiles cover all four physical PP coordinates for fixed GT and
the Eager jobs used in the comparisons. Inspect each training launch separately
from validation and compare the same stage ownership. Equivalent counts
alone do not prove equivalent communication volume, arrival times, or critical
path. Optimization acceptance requires full-step TPS, native trace evidence,
numerical checks and no peak-HBM regression from the same matched environment.

### SPMD training reduction overlap

The primary 52-update window shows GT training TPS gains of 1.75% for natural routing and 1.16% for balanced routing. The table below uses means of two training replays from each original global-rank0 trace. Exposed reduction time is the reduce-scatter activity outside any non-NCCL GPU kernel activity; that other activity includes DistMoE communication and waiting as well as arithmetic.

| Routing / engine | Training graph span s | RS events | RS activity ms | RS overlapping other activity ms | Exposed RS ms |
|---|---:|---:|---:|---:|---:|
| balanced / GT | 43.2984 | 123 | 433.0 | 396.0 | 37.0 |
| balanced / Eager | 43.7990 | 123 | 348.2 | 26.0 | 322.2 |
| natural / GT | 46.2770 | 123 | 491.3 | 420.6 | 70.7 |
| natural / Eager | 46.4862 | 123 | 453.0 | 25.6 | 427.3 |

In the executed SPMD path, Eager disables gradient synchronization throughout accumulation and calls `finalize_backward()` after the loop. GT includes reductions in the final joint forward/backward graph, allowing ready gradient groups to reduce while remaining backward work continues. Balanced GT has 285ms less exposed reduce-scatter within a 501ms shorter graph span, compared with a 506ms mean training-window saving. This is a descriptive timeline partition; it does not establish the causal gain of a proposed patch. GT has longer reduce-scatter activity in these captures, with equal event counts, so faster standalone NCCL kernels are not the measured explanation.

Natural GT has 357ms less exposed reduce-scatter but 138ms more non-NCCL activity. Its captured graph-span saving is 209ms, compared with 818ms over the complete training window. Two rank-local captures do not explain that remaining difference. PP2 and balanced PP4 remain near parity in the full-window results; selected natural PP4 replays favor GT while the full natural PP4 run favors Eager. Routing evolution, other ranks and work outside the forward/backward graph require separate evidence.

[Offline comparison evidence](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/profile-analysis-20261015/gt-vs-eager-profile-analysis-20261015.tar.zst) contains the six paired comparisons, 12 rechecked original rank0 SHA256 identities, launch correlations, timeline summaries, collective exposure and kernel deltas. Full original profiles are archived separately below. The 90-day evidence bundle was downloaded and verified against SHA256 `855937a1feae3d701789c9f8fd9d3246123fe8ecc7216882fba30831e132a5e8`.

### Current PP4 replay measurements

These spans come from the complete original traces at the four selected PP coordinates, not the portable excerpts. For each rank, the two training replays are the CUDA graph launch correlations with at least half the maximum correlated kernel count; the two smaller launches are validation. A span runs from the first correlated GPU kernel start to the last end, including overlap and gaps. Ranges below cover those two profiled replays and do not replace the unprofiled training TPS or wall-cycle statistics.

| Routing / PP rank | Training span s: GT / Eager | Validation span s: GT / Eager | Training AG / RS / send-recv events: GT / Eager |
|---|---:|---:|---|
| natural / pp-rank0 | 48.858..48.913 / 49.177..49.207 | 1.203..1.254 / 1.181..1.345 | 17/17/3360 / 17/17/3360 |
| natural / pp-rank1 | 48.862..48.926 / 49.204..49.215 | 1.211..1.220 / 1.224..1.237 | 16/16/3840 / 16/16/3840 |
| natural / pp-rank2 | 48.825..48.850 / 48.975..49.024 | 1.188..1.241 / 1.221..1.244 | 16/16/3840 / 16/16/3840 |
| natural / pp-rank3 | 48.708..48.838 / 48.961..49.011 | 1.261..1.354 / 1.248..1.318 | 16/16/3360 / 16/16/3360 |
| balanced / pp-rank0 | 44.661..44.671 / 44.885..44.938 | 0.996..1.082 / 1.037..1.152 | 17/17/3360 / 17/17/3360 |
| balanced / pp-rank1 | 44.631..44.677 / 44.862..44.864 | 1.067..1.118 / 1.032..1.160 | 16/16/3840 / 16/16/3840 |
| balanced / pp-rank2 | 44.596..44.663 / 44.782..44.831 | 1.085..1.091 / 1.056..1.125 | 16/16/3840 / 16/16/3840 |
| balanced / pp-rank3 | 44.470..44.475 / 44.724..44.771 | 1.095..1.111 / 1.089..1.242 | 16/16/3360 / 16/16/3360 |

AG / RS counts are named NCCL all-gather / reduce-scatter kernel events per training replay; send-recv is the named NCCL PP send/recv count. They do not establish equivalent bytes or isolate the remaining natural-routing gap. Match stage ownership and CUDA graph correlations when inspecting the original profiles below.

## Archived profiles and fbcode validation

Public SPMD, PP2 and PP4 GPU timeline excerpts are checked into the branch and linked
from [Profiles](graph_trainer_perf_runs_instructions.md#5-profiles). SPMD uses
global rank 0; PP2 includes ranks 0/128; PP4 includes ranks 0/64/128/192,
covering pp-rank0/1/2/3. PP4 GT excerpts use the completed parameter-preparation
fix jobs. The section documents each one-second training window and metadata
omitted from these portable visualization files.

The current PP4 comparisons have **16 selected original traces** archived for 90 days, four physical PP coordinates per engine/routing. Total compressed bytes: 1,757,084,988. Every selected trace records `mesh_pp=[0,64,128,192]` and its expected global rank. [Open current PP4 archive](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pp4-localprep-refresh-20261014). The other current jobs retain their original attempt-qualified archives.

Verified primary/backup views for all 8 current SPMD/PP2 jobs are below; current PP4 views are in the following table.

Shared links use fbcode's `arvr/scripts/perfetto/share_trace.py` and are verified by downloading the published object and comparing SHA256. Viewer retention is 28 days from publishing; the manifest records exact upload times. For SPMD/PP2, rank0 on host0 is the primary trace analyzed and compacted; rank128 on host32 is the original backup. For PP4, the four representative ranks requested for this report are analyzed and published as original traces and compacted one-second excerpts. Other PP4 ranks are not processed.

| Configuration / engine | Rank0 original | Rank0 compacted | Rank128 raw backup |
|---|---|---|---|
| PP2 / natural / GT | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_4cb6cab0-397c-475e-915d-68c5a0a9ad8d_pp2-natural-gt-v7-a0_rank0_trace.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_76c3f448-c2eb-49c0-bba0-3013734a04ca_pp2-natural-gt-v7-a0_rank0_trace_compacted.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_8774ca79-1fa7-49c0-8c1d-79bee56c9ab2_pp2-natural-gt-v7-a0_rank128_trace.json.gz) |
| PP2 / natural / Eager | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_1a2e8441-e1c3-4355-bf0c-658dd8dc0930_pp2-natural-eager-v7-a0_rank0_trace.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_c37f2944-168c-425c-ad6e-3df1b602e206_pp2-natural-eager-v7-a0_rank0_trace_compacted.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_79672f72-2728-4fdd-8878-23391b850fa5_pp2-natural-eager-v7-a0_rank128_trace.json.gz) |
| PP2 / balanced / GT | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_8b9716da-b54c-4731-bff1-78eedff1dcfd_pp2-balanced-gt-v2-a0_rank0_trace.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e7043218-d057-4ad8-b54f-e4f98cce4940_pp2-balanced-gt-v2-a0_rank0_trace_compacted.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_ed39f30a-6e02-4d58-a8b0-6d840f059156_pp2-balanced-gt-v2-a0_rank128_trace.json.gz) |
| PP2 / balanced / Eager | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_8eb40b8c-9ca4-4f70-ba32-e4add6f8a940_pp2-balanced-eager-v2-a1_rank0_trace.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_52e5187a-e968-465c-b368-d8f6830411ce_pp2-balanced-eager-v2-a1_rank0_trace_compacted.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_5df52404-a0c6-4acd-bab4-f68115fb68a3_pp2-balanced-eager-v2-a1_rank128_trace.json.gz) |
| SPMD / natural / GT | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_1c45d04d-8631-4dee-95d6-497207bfec46_spmd-natural-gt-v7-a1_rank0_trace.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6d3d8dc5-813a-4907-aa0a-46e429d91635_spmd-natural-gt-v7-a1_rank0_trace_compacted.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_57c1df60-ad62-4d01-a121-09a7b39b54b2_spmd-natural-gt-v7-a1_rank128_trace.json.gz) |
| SPMD / natural / Eager | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_90ca1233-2c70-454a-bf83-22161883b268_spmd-natural-eager-v7-a0_rank0_trace.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_015bf4ea-55a2-4b19-92fa-f9bb3dc6a6d1_spmd-natural-eager-v7-a0_rank0_trace_compacted.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_1b3c8153-6650-4f57-b33d-6f6883aaab11_spmd-natural-eager-v7-a0_rank128_trace.json.gz) |
| SPMD / balanced / GT | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_248e108e-d908-4706-be9b-918905496c18_spmd-balanced-gt-v2-a0_rank0_trace.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_1463e573-f054-4eb9-a2e5-ad1d8932fd0a_spmd-balanced-gt-v2-a0_rank0_trace_compacted.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_7565e660-974f-495f-941a-502269b35100_spmd-balanced-gt-v2-a0_rank128_trace.json.gz) |
| SPMD / balanced / Eager | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_2a4fca7e-b46a-432b-88a8-527f9a11791c_spmd-balanced-eager-v2-a0_rank0_trace.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_bad678ab-b116-4b13-8839-c565c304dd6b_spmd-balanced-eager-v2-a0_rank0_trace_compacted.json.gz) | [Open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_d2ed0772-bdaa-4da9-a6c5-3d7946e75091_spmd-balanced-eager-v2-a0_rank128_trace.json.gz) |

### Current PP4 profiles by physical pipeline rank

The recorded PP group is `[0,64,128,192]`: global ranks 0/64/128/192 are pp-rank0/1/2/3 on hosts 0/16/32/48. Each cell links the full original trace for quantitative inspection and a portable compacted excerpt for scheduling inspection. Original shared objects have 28-day viewer retention and SHA256 download verification. Excerpts preserve kernel names, timestamps and durations; they are not full-step TPS measurements.

| Configuration / engine | pp-rank0 (global 0) | pp-rank1 (global 64) | pp-rank2 (global 128) | pp-rank3 (global 192) |
|---|---|---|---|---|
| PP4 / natural / GT | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_8ed084ff-1957-4bb3-93f8-6fe38bbd480e_pp4-natural-gt-localprep-v0-a0_rank0_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_graph_trainer_natural_rank0.json.gz) | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_7d334f1f-f1ba-4333-a0d7-b45daaefba82_pp4-natural-gt-localprep-v0-a0_rank64_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_graph_trainer_natural_rank64.json.gz) | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_32acaa37-9bcf-4571-8430-df6070fd27f4_pp4-natural-gt-localprep-v0-a0_rank128_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_graph_trainer_natural_rank128.json.gz) | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_959a9956-b7d8-4d71-8700-48e3e2658b81_pp4-natural-gt-localprep-v0-a0_rank192_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_graph_trainer_natural_rank192.json.gz) |
| PP4 / natural / Eager | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6d271fa8-0dc5-474e-b812-be0e0ec5b3a2_pp4-natural-eager-v7-a0_rank0_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_eager_natural_rank0.json.gz) | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e198c0a8-b62d-46e2-b6dd-d15724c80cb1_pp4-natural-eager-v7-a0_rank64_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_eager_natural_rank64.json.gz) | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_7758fc64-f2b7-43d0-bba5-d85c1b6d86da_pp4-natural-eager-v7-a0_rank128_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_eager_natural_rank128.json.gz) | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b119e580-38ae-463f-b71f-24b2fa4ab89f_pp4-natural-eager-v7-a0_rank192_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_eager_natural_rank192.json.gz) |
| PP4 / balanced / GT | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e95b20b4-a9e5-4ed3-b4b8-708c33d85000_pp4-balanced-gt-localprep-v0-a0_rank0_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_graph_trainer_balanced_rank0.json.gz) | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_d969b886-d275-47e2-8a51-6cdcc4dab9e0_pp4-balanced-gt-localprep-v0-a0_rank64_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_graph_trainer_balanced_rank64.json.gz) | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e3c51313-684b-4523-9ffa-725f26e069eb_pp4-balanced-gt-localprep-v0-a0_rank128_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_graph_trainer_balanced_rank128.json.gz) | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_249239ef-0b4c-4571-b436-a194bc212aad_pp4-balanced-gt-localprep-v0-a0_rank192_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_graph_trainer_balanced_rank192.json.gz) |
| PP4 / balanced / Eager | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_bd1fbc7b-1ba2-4a61-b606-43a008f8f508_pp4-balanced-eager-v2-a0_rank0_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_eager_balanced_rank0.json.gz) | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_f9cc99b6-4215-47de-90d2-e2c629476493_pp4-balanced-eager-v2-a0_rank64_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_eager_balanced_rank64.json.gz) | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_d3dc57fa-4c8f-4256-ad19-73c8ea8810d5_pp4-balanced-eager-v2-a0_rank128_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_eager_balanced_rank128.json.gz) | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_7021ca9c-1e54-407a-98ed-3368cc3b2076_pp4-balanced-eager-v2-a0_rank192_trace.json.gz) / [Excerpt](profiles/graph_trainer_256gpu/pp4_eager_balanced_rank192.json.gz) |

### Allocator diagnostic profiles

5 of the 12 separate three-update allocator diagnostics are complete and shared. Their frozen workload/source/runtime match the corresponding current benchmark; only training.steps60->3, enable_memory_snapshotFalse->True and memory_snapshot_freq40->3 change. These are allocator diagnostics, not TPS benchmarks. Post-fix PP4 GT allocator histories were not captured; its benchmark step-60 counters are reported above. Earlier PP4 GT diagnostic values are excluded from the current comparisons. Rank0 is the primary snapshot; rank128 is retained as a raw backup on another host. Both are archived for 90 days and shared with SHA256 round-trip verification.

Snapshots are taken after update 3 validation. They expose current allocation/block state and graph private pools. History begins when the profiler starts and has a one-million-event cap; they do not establish a complete lifetime history or replace benchmark step60 high-water measurements.

Natural diagnostics use native version 1 with a repaired launcher: two missing variables used in preflight JSON logging were restored after a version-0 failure before training. Model/runtime pins and the original performance benchmark results are unchanged. The repaired immutable workspace is included in the payload archives.

| Diagnostic configuration / engine | Native version / attempt | Rank0 MemoryViz | Rank128 raw backup |
|---|---|---|---|
| PP2 / balanced / GT | 0 / 0 | [Open](https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/ivankobzarev_4a178bb8-7f46-4fae-919d-95d2bfba80c6_torchtitan-report256-pp2-gt-mem3-balanced-20261014-ivankobzarev_v0_a0_000000_step_3.pickle) | [Backup](https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/ivankobzarev_36cffb30-90cd-4b08-bdbe-e1105042347f_torchtitan-report256-pp2-gt-mem3-balanced-20261014-ivankobzarev_v0_a0_000128_step_3.pickle) |
| PP2 / natural / GT | 1 / 1 | [Open](https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/ivankobzarev_a2df1ca7-f606-4345-84f3-9b63760b2c7d_torchtitan-report256-pp2-gt-mem3-natural-20261014-ivankobzarev_v1_a1_000000_step_3.pickle) | [Backup](https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/ivankobzarev_72a2422f-312e-4847-b8e0-f2c6820e0111_torchtitan-report256-pp2-gt-mem3-natural-20261014-ivankobzarev_v1_a1_000128_step_3.pickle) |
| PP2 / natural / Eager | 1 / 0 | [Open](https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/ivankobzarev_532b84c4-918f-410a-8d10-89788f3ea39f_torchtitan-report256-pp2-eager-mem3-natural-20261014-ivankobzarev_v1_a0_000000_step_3.pickle) | [Backup](https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/ivankobzarev_1fb2bd6c-df5e-4997-b487-8c3ca96c9437_torchtitan-report256-pp2-eager-mem3-natural-20261014-ivankobzarev_v1_a0_000128_step_3.pickle) |
| PP2 / balanced / Eager | 0 / 0 | [Open](https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/ivankobzarev_a383858b-2380-4f86-8a46-3b8e382497ad_torchtitan-report256-pp2-eager-mem3-balanced-20261014-ivankobzarev_v0_a0_000000_step_3.pickle) | [Backup](https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/ivankobzarev_6490c863-e6cb-4b56-a12b-e3d5e3a81b40_torchtitan-report256-pp2-eager-mem3-balanced-20261014-ivankobzarev_v0_a0_000128_step_3.pickle) |
| PP4 / natural / Eager | 1 / 0 | [Open](https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/ivankobzarev_2bdd24f5-0173-4224-aacc-234121813f9b_torchtitan-report256-pp4-eager-mem3-natural-20261014-ivankobzarev_v1_a0_000000_step_3.pickle) | [Backup](https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/ivankobzarev_638dfbb0-6ea2-4086-a3e0-5db5306ee8f3_torchtitan-report256-pp4-eager-mem3-natural-20261014-ivankobzarev_v1_a0_000128_step_3.pickle) |

5 of the 5 available primary histories reached the one-million-event cap. The earliest recorded events are truncated; current segment/block state remains available. A complete lifetime allocation history or startup allocation peak cannot be inferred from these snapshots.

| Diagnostic configuration / engine | Primary rank | Reserved GiB | Allocated GiB | Graph private-pool reserved GiB |
|---|---:|---:|---:|---:|
| PP2 / balanced / GT | 0 | 223.76 | 105.58 | 127.87 |
| PP2 / natural / GT | 0 | 232.62 | 114.44 | 127.87 |
| PP2 / natural / Eager | 0 | 244.50 | 115.36 | 138.01 |
| PP2 / balanced / Eager | 0 | 235.66 | 106.50 | 138.01 |
| PP4 / natural / Eager | 0 | 243.06 | 120.09 | 131.12 |

This table is the rank0 allocator state after diagnostic update 3 validation. It is separate from the benchmark all-rank step60 high-water table, and does not establish the lifetime maximum allocated bytes.

### Current PP4 evidence bundle

[Artifact manifest](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pp4-localprep-refresh-20261014/artifact-manifest.json) identifies the four current PP4 jobs, selected original profiles, source derivation and checksummed workspace payloads. [Download the current PP4 evidence bundle](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pp4-localprep-refresh-20261014/pp4-localprep-current-evidence-20261014.tar.gz) (1,033,855 bytes; SHA256 `9df79b690d596e77fdd812769045771007fcabd7faeaf74ee1dd58308e149c03`). Its 56 files include raw TensorBoard events, native definitions/status, unchanged resolved workload configurations, all-rank step-60 high-water records, current pair plots, numerical/cycle/TPS analyses and reproducible analysis scripts. The six-file source fix and source SHA256 proof are included. Original traces and executable workspace trees are archived separately for 90 days; shared original profile links have 28-day viewer retention. This bundle covers current PP4 GT/Eager data; it does not include allocator histories or checkpoint weights.

### Archived executable runtime payloads

The following 90-day archives retain fetched Conda runtimes, native libraries, launchers and CUDA compatibility assets. Verify SHA256 before extracting and republishing as fresh immutable FBPkg identifiers. Dataset/checkpoint mounts and source preflight remain required. Local import caches may be present; source/native identities remain those of the frozen benchmark packages.

| Payload | Archive | SHA256 |
|---|---|---|
| balanced-runtime | [Download](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/runtime-payloads/balanced-runtime-20261014.tar.zst) | `05c78f72efa045525b941cd66a32d42634d94789a7d2265db1b2d28d8caecefe` |
| natural-runtime | [Download](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/runtime-payloads/natural-runtime-20261014.tar.zst) | `3f0ee49ff47ba3f4255236658f568fb687e01b95e6c1841f897a798930046395` |
| balanced-256-workspace | [Download](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/runtime-payloads/balanced-256-workspace-20261014.tar.zst) | `16411a17b448b4e2589a23a32fad9295994d98304e1af98cd18a847bcfeab514` |
| natural-256-workspace | [Download](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/runtime-payloads/natural-256-workspace-20261014.tar.zst) | `e8dfbaf230b447428175a4ee118ae90a0fb1abb73e6de477e0bbc1707ee58262` |
| balanced-memory-workspace | [Download](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/runtime-payloads/balanced-memory-workspace-20261014.tar.zst) | `e1d46a2ba03f1f2cc366467ac63e9d49c7b30eeaaddd9172da9f1d5f24ec54fb` |
| natural-memory-workspace | [Download](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/runtime-payloads/natural-memory-workspace-20261014.tar.zst) | `bde6c10b6a1a12da5a61ac00d00034dc4c47c78293e47e134afb4885c1791c11` |
| balanced-512-workspace | [Download](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/runtime-payloads/balanced-512-workspace-20261014.tar.zst) | `bc15d8dbb1f12b661ef5e8b8074c9380a5ac5db7c31574d471ab044855304eae` |
| pp4-gt-balanced-localprep-workspace | [Download](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pp4-localprep-refresh-20261014/runtime-payloads/pp4-gt-balanced-localprep-workspace-20261014.tar.zst) | `6dee66339a9a3fbe3e4e815c776013bc5d87df4ee27ff447303965c292beb5b6` |
| pp4-gt-natural-localprep-workspace | [Download](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/pp4-localprep-refresh-20261014/runtime-payloads/pp4-gt-natural-localprep-workspace-20261014.tar.zst) | `e29c605bc0de56c236deea41f7bd0eed5f6e288171cdf75abb1ef881bfd30c85` |

## 512-GPU balanced sweep

Native status checked at 2026-10-15T16:56:26.183903+00:00. The six-point PP2/PP4/PP8 x GT/Eager matrix copies [Eager Prev Report](https://docs.google.com/document/d/1W1IIuFle-0HMg64piGN6RUiHzQc0r-eYbktsMfjZVz8/edit?tab=t.0)'s 512-GPU workload: 128 four-GPU GB300 hosts in LCO, EP64, MBS1, sequence 4,096, GBS15,360 and 60 updates. 4 unfinished jobs are frozen in MAST and require unfreezing before admission. Capacity preemptions receive MAST free requeues; they are not evidence of a training failure. Completion requires the latest native attempt and all 128 workers COMPLETE with exit code 0. Completed results are retained below; pending entries have no measured counterpart yet.

The 16-hour monitoring window ran from 2026-10-14T19:40:45+00:00 to 2026-10-15T11:40:45+00:00. At window end, 1/6 runs had completed. 14 PP4 GT attempt histories confirm MAST capacity preemptions with free requeues. Its attempt 19 reached update 11 before preemption; at window end, attempt 20 was running. No new runtime stack change was made during the window. The 955 recorded native status snapshots have a median cadence of 60.01 seconds; the longest interval is 96.28 seconds, including local monitor restarts. Finished PP2 GT results are preserved; interrupted PP4 measurements are excluded.

[Download the monitoring-window evidence](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/512-sweep/autonomy16h-20261014/autonomy16h-20261014-completed-window.tar.zst) for native snapshots, preemption histories, operational scripts and the numerical audit. The 90-day archive was downloaded and verified against SHA256 `e14906babd68d2be20bd61a8be5926999740e8d862a48590276f1a65783d0e8c`.

| Topology | DP / expert FSDP | Microbatches | GT job / state | Eager job / state |
|---|---|---:|---|---|
| PP2 / VPP8 | 256 / 4 | 60 | [GT](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report512-pp2-gt-balanced-20261014-ivankobzarev) / COMPLETE (v0, a4) | [EAGER](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report512-pp2-eager-balanced-20261014-ivankobzarev) / FROZEN, PENDING (v1, a1) |
| PP4 / VPP4 | 128 / 2 | 120 | [GT](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report512-pp4-gt-balanced-20261014-ivankobzarev) / COMPLETE (v1, a22) | [EAGER](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report512-pp4-eager-balanced-20261014-ivankobzarev) / FROZEN, PENDING (v1, a0) |
| PP8 / VPP2 | 64 / 1 | 240 | [GT](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report512-pp8-gt-balanced-20261014-ivankobzarev) / FROZEN, PENDING (v1, a2) | [EAGER](https://www.internalfb.com/msl/studio/runs/mast/torchtitan-report512-pp8-eager-balanced-20261014-ivankobzarev) / FROZEN, PENDING (v1, a0) |

### Completed training and memory measurements

TPS is tokens/s/GPU over the same 53-update reference window as the main comparison: updates 3..60 excluding 39..43. Eager Prev Report values are its rounded document values. Active memory is the full-precision TensorBoard high-water on the loss/metrics rank; reserved memory is the largest step-60 console high-water across all 512 ranks, rounded to two decimals. Reference reserved memory covers representative PP coordinates. Allocator snapshots and maximum active memory across every rank are not available for these 512-GPU benchmark jobs.

| Topology | GT TPS/GPU | Eager TPS/GPU | Eager Prev Report TPS/GPU | GT vs Eager | GT vs Eager Prev Report |
|---|---:|---:|---:|---:|---:|
| PP2 | 5,442.48 | -- | 5,262.68 | -- | +3.42% |
| PP4 | 5,340.41 | -- | 5,087.99 | -- | +4.96% |
| PP8 | -- | -- | 4,780.79 | -- | -- |

| Topology / engine | Active on loss rank GiB | Reserved on loss rank GiB | Reserved across all 512 ranks GiB | Eager Prev Report reserved GiB |
|---|---:|---:|---:|---:|
| PP2 / GT | 75.61 | 193.69 | 214.57 | 224.65 |
| PP2 / Eager | -- | -- | -- | 224.65 |
| PP4 / GT | 66.77 | 156.87 | 215.05 | 221.14 |
| PP4 / Eager | -- | -- | -- | 221.14 |
| PP8 / GT | -- | -- | -- | 233.96 |
| PP8 / Eager | -- | -- | -- | 233.96 |

### Primary training window and complete-cycle proxy

The primary training window contains 52 updates: 3..60 excluding 35..40. The cycle proxy averages 51 training-loss wall-time intervals with current updates 4..60 excluding 35..40. Each interval includes the preceding validation and the next training step. It is not an isolated CUDA-graph span; TPS of the cycle proxy uses 122,880 training tokens/GPU/update.

| Topology | GT primary TPS/GPU | Eager primary TPS/GPU | GT cycle seconds | Eager cycle seconds | GT cycle TPS/GPU | Eager cycle TPS/GPU |
|---|---:|---:|---:|---:|---:|---:|
| PP2 | 5,445.73 | -- | 23.2853 | -- | 5,277.15 | -- |
| PP4 | 5,344.34 | -- | 23.6586 | -- | 5,193.87 | -- |
| PP8 | -- | -- | -- | -- | -- | -- |

2/6 runs have verified 60 finite training mean/max-loss, gradient-norm and validation-loss records. These benchmarks use seed14536 and deterministic=False; finite scalars do not establish bitwise tensor equality or convergence. The 256-GPU audit likewise found no full-scale bitwise proof. Historical four-GPU gradient-hash matches use earlier reduced recipes, default updates1/2 and pre-optimizer gradients; they do not cover current full EP64 natural routing, full-step replay or updated parameter/optimizer/router state across all 256 ranks.

### Selected original profiles and archived evidence

Only rank0 and the pp-rank1 peer on another host are processed for each job: PP2 0/256, PP4 0/128, PP8 0/64. Rank0 originals support measurement; compaction preserves measured slices and changes display lanes only. The second original is a backup PP-coordinate view. Originals and evidence have 90-day archive retention; fbcode viewer links have 28-day retention. Shared links establish access, not reviewer validation.

| Topology / engine | Rank0 (pp-rank0) | Peer (pp-rank1) |
|---|---|---|
| PP2 / GT | [original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_9bc1044d-033a-496b-8dbf-cf87092a4252_rank0_trace.json.gz) / [compacted visualization](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_33bc3ac5-2d09-4b0c-8da6-37f1307118e7_rank0_trace_compacted.json.gz) / [archived original](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/512-sweep/torchtitan-report512-pp2-gt-balanced-20261014-ivankobzarev/v0-a4/profiling/traces/iteration_39/rank0_trace.json.gz) | rank256: [original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_3e566f50-7a92-47e8-b30c-46880c12e16d_rank256_trace.json.gz) / [archived original](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/512-sweep/torchtitan-report512-pp2-gt-balanced-20261014-ivankobzarev/v0-a4/profiling/traces/iteration_39/rank256_trace.json.gz) |
| PP4 / GT | [original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_c7d4f1d6-012d-441d-8cfe-6fcd91b39113_rank0_trace.json.gz) / [compacted visualization](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_423664e4-db2c-4630-ab31-11feb87eca7c_rank0_trace_compacted.json.gz) / [archived original](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/512-sweep/torchtitan-report512-pp4-gt-balanced-20261014-ivankobzarev/v1-a22/profiling/traces/iteration_39/rank0_trace.json.gz) | rank128: [original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_362615d4-7995-4e4a-b87f-70f5ecf77b1b_rank128_trace.json.gz) / [archived original](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/512-sweep/torchtitan-report512-pp4-gt-balanced-20261014-ivankobzarev/v1-a22/profiling/traces/iteration_39/rank128_trace.json.gz) |

Evidence bundles contain raw TensorBoard events, native definitions/status, all-rank step-60 high-water records, selected profile metadata/analysis, frozen configuration proof and analysis scripts. Original traces and executable workspace/runtime payloads are separate; checkpoint weights and allocator histories are external.

| Attempt | Evidence bundle | Bundle SHA256 | TensorBoard event SHA256 |
|---|---|---|---|
| PP2 / GT / v0-a4 | [Evidence](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/512-sweep/torchtitan-report512-pp2-gt-balanced-20261014-ivankobzarev/v0-a4/torchtitan-report512-pp2-gt-balanced-20261014-ivankobzarev-v0-a4-evidence.tar.zst) | `7aa952a1df95cb1f90a4cfa9880b5f2ce672650de968437f50e34c9d1d20eab0` | `9da0229c0ef3bd80fffc326454a7cc6113e5a6ab7c507b6c06a3c1e457688343` |
| PP4 / GT / v1-a22 | [Evidence](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/512-sweep/torchtitan-report512-pp4-gt-balanced-20261014-ivankobzarev/v1-a22/torchtitan-report512-pp4-gt-balanced-20261014-ivankobzarev-v1-a22-evidence.tar.zst) | `fdc93117b76f1879cca9801cef6a6c4ba76683f0ed7b73443b5f50222d476788` | `dd958d63c340496212e8b02c3478f09bfae88ae99306826d6c70870d4bdd30f9` |

Finalized schedules and static activation liveness match the reference slot counts:
PP2 `17/15`, PP4 `19/17/15/13`, and PP8 `16/21/19/17/15/13/11/9`.
PP8 pp-rank0 uses `microbatch` slots of depth5; its other seven coordinates use
`stage_microbatch` slots of depth4. CPU preflight covers all physical PP
coordinates and both engines; GPU allocation, capture/replay and numerics
require native execution.

All six jobs use balanced runtime `b836fb1a52304c769916013e3f9c5181`.
Completed PP2 GT retains workspace `a087ed92d5184495be2afe124f2b63d5`; the
other five current version1 definitions use `1c9d03ab00e14c36aa14f0e5ff247c14`,
including the PP4 parameter-preparation fix. Preserve this source difference
when comparing PP2 GT with its newer Eager counterpart. The warmed checkpoint,
frozen MLPerf data, routing and other workload controls are unchanged.
Historical source/runtime still differs from Eager Prev Report. The 256-GPU
controls ran in QNO; these 512-GPU jobs use LCO to match the document.
Cross-region ratios do not isolate pure DP/PP scaling. PP2/PP4/PP8 at fixed
512 GPUs form a fixed-hardware comparison, not a strong-scaling experiment.

The exact fixed 512-GPU workspace is [archived for 90 days](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/gt256-report-archive-20261014-ivankobzarev/512-sweep/runtime-payloads/fixed-512-workspace-1c9d03a-20261014.tar.zst); SHA256 `544707530e40982af483da498678089d5e34c4c50a1a488748167050869e8ada`. Its 1,059 publication files were checked against the fetched immutable package before archival. Republish that tree and the archived balanced runtime as fresh packages if ephemeral identifiers expire.

### Validation checklist

- Download the manifest and verify the job/version/attempt before comparing
  traces. Preserve raw trace SHA256 and the TensorBoard event-file checksum.
- In Perfetto, select the training `cudaGraphLaunch` correlation; validation is
  a separate launch. Inspect capture boundaries on each selected coordinate.
  PP4 uses global ranks 0/64/128/192, covering pp-rank0/1/2/3.
- Compaction changes display lanes only. Its self-validation checks measured
  slices and lane overlap. Current PP4 traces are analyzed at four selected
  coordinates, with one-second excerpts compacted for each. SPMD/PP2
  quantitative analysis uses rank0 with an original rank128 backup; published
  PP2 excerpts also cover pp-rank1.
- In MemoryViz, separate active allocations from reserved segments, CUDA graph
  private pools and fragmentation. Compare the same phase and local ownership.
  Diagnostic snapshot 3 is not the benchmark step 60 peak or a lifetime peak.
- The evidence bundle includes native definitions/statuses, full resolved
  configs, raw TensorBoard events, all-rank step-60 memory high-water records,
  selected-rank TensorBoard memory counters, pair analyses,
  provenance and plotting/analysis scripts. A shared link establishes access;
  it does not establish that an fbcode reviewer has validated correctness.

## Follow-up priorities

1. Locate the first gradient divergence and the PP4 Eager update 18 spike before
   claiming numerical parity or convergence.
2. Batch/unify GT SPMD validation unshards and measure complete train/validation
   cycles in a matched native A/B, with trace and allocator ownership evidence.
3. Explain the remaining natural-routing PP4 training gap using the current
   four-coordinate original traces. Balanced PP4 training is at parity in
   these completed runs. Preserve numerical and memory gates for further fixes.
4. Use required router-count contracts to remove measured redundant work, then
   verify complete native-step performance rather than summed kernel durations.
5. Replicate the fixed-GBS 512-GPU PP2/PP4/PP8 points from Eager Prev Report's document.
   Hold PP fixed for DP-axis scaling and DP fixed for PP-axis scaling; the three
   512-GPU points alone are a fixed-hardware comparison, not strong scaling.

The existing runtime fixes received independent critic review in the benchmark
campaign. This report does not convert the experimental report adapter or an
unexplained performance effect into a landable production change.
