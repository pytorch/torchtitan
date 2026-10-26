# DSv3 671B + MTP1: forced balanced MAST report, 64 / 32 / 16 GPUs

**Date:** 2026-10-26. **Live status snapshot:** 17:05:39-17:05:42 UTC (10:05 PDT).
**Hardware:** Nebius GB300 NVL72, four GPUs per host, approximately 276.62 GiB CUDA memory per GPU.

This report covers the recent forced balanced campaign: **23 distinct MAST job names, 10 completed benchmarks and 13 failed or replaced jobs**. Configurations are admitted by their submitted `forced_balanced_routing=True` and `_RoundRobinDeepSeekV3Router.Config`, rather than by job names alone. The October 24-25 learned-routing sweep is outside this report. The structure follows the [256-GPU reference report on `gt-ptc-runs-candidate-1310`](https://github.com/pytorch/torchtitan/blob/gt-ptc-runs-candidate-1310/docs/graphtrainer-ptc26-runs-release/README.md).

## Recommendations

- **Highest throughput on 64 GPUs: 64-A, B6/GA8.** SAC, GPU Adam moments, no activation CPU offload, CF2.5 and native save capacity factor 0.025. **28.33% MFU, 4,853.86 tokens/s/GPU, 310,646.84 tokens/s globally.** Peak training active/reserved memory is **254.64 / 267.11 GiB**.
- **64-GPU option with more allocator headroom: 64-B, B8/GA8.** SAC plus 100 GiB activation offload, GPU Adam moments, CF1 and eight loss chunks. **27.98% MFU, 4,793.44 tokens/s/GPU**, with **238.44 / 244.13 GiB** peak active/reserved. It gives up 1.24% throughput relative to 64-A and reduces peak reserve by 22.98 GiB. Batch, CF, native save policy and source also differ.
- **Highest throughput on 32 GPUs: 32-A, B10/GA8.** SAC plus 120 GiB activation offload and CPU BF16 Adam moments, CF1, eight loss chunks. **26.47% MFU, 4,534.30 tokens/s/GPU, 145,097.74 tokens/s globally.** Peak active/reserved is **252.70 / 263.15 GiB**. The scheduler-created attempt 1 completed; all 32 timing and scalar records were downloaded and checked for this report.
- **32-GPU option with more allocator headroom: 32-B, B8/GA8.** It reaches **25.76% MFU and 4,412.88 tokens/s/GPU** at **232.75 / 245.66 GiB** peak active/reserved. This is 2.68% less throughput than B10 while saving 17.49 GiB of peak reserve. B4/GA8 (32-C) lowers reserve further to 227.41 GiB at 4,029.98 tokens/s/GPU.
- **16 GPUs: no completed checkpoint-loaded benchmark.** B1 and B2 both failed while loading and stacking HF expert weights, before training. Their local memory forecasts do not qualify MFU or distributed fit.

The active figures above are rank-0 TensorBoard high-waters over training updates 1-27. All-rank allocated/reserved peaks are given below and agree at the displayed precision for these runs. These are training peaks, not lifetime peaks covering checkpoint loading.

## How to read the configurations

| Field | Meaning |
| --- | --- |
| BS | Local microbatch size: sequences per data rank in one forward/backward microbatch |
| GA | Number of microbatches accumulated before one optimizer update |
| GBS | Global sequences per update: GPUs x BS x GA here, because PP=TP=CP=1 |
| Sequence length | 4,096 tokens for every row; tokens/update = GBS x 4,096 |
| SAC | GraphTrainer selective activation checkpointing/rematerialization: retain selected outputs and recompute other intermediates in backward |
| Activation CPU offload | GraphTrainer offloads selected activations to pinned host memory; the GiB value is a ceiling per rank, not measured payload. AO100 means a 100 GiB activation offload ceiling |
| CPU Adam moments | Both BF16 Adam state tensors reside on CPU; updates use bounded GPU staging, 512 MiB per optimizer instance/device |
| Gradient CPU offload | Only the 16-GPU recipes use this: compact reduced gradient storage plus a 16 GiB CPU payload budget |
| CF | Native DistMoE scratch capacity factor; 1.0 is the balanced scratch sizing, 2.5 allocates additional capacity |
| Native save factor | Native MoE saved-state arena capacity, independent of outer SAC. 0 means minimum arena; 0.025 is a capacity interpolation, not 2.5% of layers/activations saved |
| WGrad fusion | Producer writes accumulate directly into gradient buffers; paired with terminal gradient reduction in the tested fused recipes |

All rows use **EP16, PP1, TP1, CP1, data replicate degree 1, always reshard after forward, MXFP8/native Mega DistMoE, and SAC**. Dense weights shard across all GPUs: FSDP64/32/16. Expert data sharding is 4/2/1 respectively. FSDP and EP overlap in their rank ownership and do not multiply the GPU count.

## Completed configurations and performance

All rows below completed 27 training updates, have 20 unprofiled measurements, and passed recomputation from every rank's raw timing records. `Peak active` and `Peak reserved` in this table are **rank-0 TensorBoard maxima over updates 1-27**, including warmup/capture and the two profile updates; units are GiB. See the memory table for all-rank coverage.

| ID | GPUs | BS | GA | GBS | Activation CPU offload | CPU Adam moments | SAC | MFU | Tokens/s/GPU | Peak active | Peak reserved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64-A | 64 | 6 | 8 | 3,072 | No | No | Yes | 28.33% | 4,853.86 | 254.64 | 267.11 |
| 64-B | 64 | 8 | 8 | 4,096 | Yes, 100 GiB | No | Yes | 27.98% | 4,793.44 | 238.44 | 244.13 |
| 64-C | 64 | 8 | 8 | 4,096 | Yes, 100 GiB | Yes | Yes | 26.41% | 4,523.74 | 248.34 | 256.44 |
| 64-D | 64 | 8 | 8 | 4,096 | Yes, 100 GiB | Yes | Yes | 26.53% | 4,544.29 | 248.34 | 256.62 |
| 32-A | 32 | 10 | 8 | 2,560 | Yes, 120 GiB | Yes | Yes | 26.47% | 4,534.30 | 252.70 | 263.15 |
| 32-B | 32 | 8 | 8 | 2,048 | Yes, 100 GiB | Yes | Yes | 25.76% | 4,412.88 | 232.75 | 245.66 |
| 32-C | 32 | 4 | 8 | 1,024 | Yes, 100 GiB | Yes | Yes | 23.52% | 4,029.98 | 192.77 | 227.41 |
| 32-D | 32 | 8 | 1 | 256 | Yes, 64 GiB | Yes | Yes | 23.67% | 4,054.35 | 236.36 | 261.69 |
| 32-E | 32 | 4 | 1 | 128 | Yes, 64 GiB | Yes | Yes | 19.49% | 3,338.59 | 267.07 | 269.94 |
| 32-F | 32 | 1 | 1 | 32 | Yes, 64 GiB | No | Yes | 16.67% | 2,856.00 | 252.07 | 257.88 |

### Runtime settings that distinguish the rows

Every completed row has forward/backward CUDA graphs enabled and no selected Inductor compute regions. Optimizer CUDA graph capture is disabled. `Every` means every microbatch; `Last` means extracted terminal FSDP gradient reductions occur on the last microbatch. Native operators can retain internal reductions.

| ID | Tokens/update | Mean update s | Global tokens/s | CF | Loss chunks | Native save factor | Gradient reduction | WGrad fusion | Pinned rounding cap MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64-A | 12,582,912 | 40.505520 | 310,646.84 | 2.5 | 4 | 0.025 | Every | No | Default |
| 64-B | 16,777,216 | 54.688051 | 306,780.28 | 1.0 | 8 | 0.0 | Every | No | Default |
| 64-C | 16,777,216 | 57.948518 | 289,519.33 | 1.0 | 8 | 0.0 | Last | No | Default |
| 64-D | 16,777,216 | 57.686466 | 290,834.53 | 1.0 | 8 | 0.0 | Last | Yes | Default |
| 32-A | 10,485,760 | 72.266873 | 145,097.74 | 1.0 | 8 | 0.0 | Every | No | 128 |
| 32-B | 8,388,608 | 59.404246 | 141,212.26 | 1.0 | 8 | 0.0 | Every | No | Default |
| 32-C | 4,194,304 | 32.524204 | 128,959.47 | 1.0 | 8 | 0.0 | Every | No | Default |
| 32-D | 1,048,576 | 8.082175 | 129,739.34 | 2.5 | 4 | 0.0 | Every | No | Default |
| 32-E | 524,288 | 4.907460 | 106,834.90 | 2.5 | 4 | 0.0 | Every | No | Default |
| 32-F | 131,072 | 1.434174 | 91,391.99 | 2.5 | 4 | 0.0 | Every | No | Default |

### Description of every completed configuration

- **64-A:** B6, GA8, GBS3,072. The balanced Save025 recipe uses default SAC without activation offload. FP32 parameters/gradients and BF16 Adam moments stay on GPU. CF2.5, loss chunks 4 and the 0.025 native saved-state arena are retained from its parent recipe. Gathers and reductions happen each microbatch. It is the fastest completed 64-GPU point, with about 9.51 GiB between maximum allocator reserve and CUDA total memory; that difference is not a whole-device free-memory guarantee.
- **64-B:** B8, GA8, GBS4,096. SAC combines with a 100 GiB activation offload ceiling, prefetch 1/defer 1. Adam moments remain on GPU. CF1, chunks 8, minimum native saved state, every-microbatch gather/reduce, no producer fusion. This carries one third more tokens/update than 64-A and has a longer update, while tokens/s/GPU remains close.
- **64-C:** Same B8/GA8 workload, with BF16 Adam moments offloaded to CPU. Parameters and FP32 gradients remain on GPU. Activation offload is 100 GiB; CF1/chunks 8/minimum native save. Gathers remain every microbatch, but terminal gradient reductions move to the last microbatch; explicit gradient accumulation is unfused. The corrected dense activation layout offload source enables this completed point.
- **64-D:** The matched producer-fusion sibling of 64-C. Same source, batch, activation policy, CPU moments, reduction cadence and native buffers; producer gradient accumulation fusion is enabled. Its small observed gain is discussed below.
- **32-A:** B10, GA8, GBS2,560. SAC, 120 GiB activation offload and CPU BF16 Adam moments; CF1/chunks 8/minimum native save; every-microbatch gather/reduce; fusion disabled. Pinned allocator rounding is capped at 128 MiB. This is the largest completed 32-GPU microbatch in the sweep. Attempt 0 was preempted for scheduler capacity; attempt 1 completed with the same version/package.
- **32-B:** B8, GA8, GBS2,048. SAC, 100 GiB activation offload and CPU BF16 moments; CF1/chunks 8/minimum native save; every gather/reduction, fusion disabled. It retains 97.32% of B10 throughput with substantially less allocator reserve.
- **32-C:** B4, GA8, GBS1,024. Same feature family as 32-B, with B4. It has the lowest all-27 training allocation/reservation among completed 32-GPU points. This is the lower-memory choice when the smaller GBS meets the training requirement.
- **32-D:** B8, GA1, GBS256. SAC plus 64 GiB activation offload and CPU BF16 moments; CF2.5/chunks 4/minimum native save; every gather/reduce. The single-microbatch ownership fix avoids retaining a cloned full gradient buffer. It updates the optimizer after each microbatch, so it amortizes CPU moment transfers less than GA8.
- **32-E:** Original B4, GA1, GBS128 baseline. Same SAC+64 GiB activation offload and CPU BF16 moments, CF2.5/chunks 4/minimum native save, but the earlier core/adapter. It fits with a high capture peak: 267.07 GiB allocated and 269.94 GiB reserved. It is slower and reserves more memory than the later B4/GA8 point; multiple settings/source revisions changed.
- **32-F:** B1, GA1, GBS32. SAC plus 64 GiB activation offload, with BF16 Adam moments kept on GPU. CF2.5/chunks 4/minimum native save and the single-microbatch ownership fix. It proves GPU moment residency fits on 32 GPUs at B1; it has the lowest throughput in the completed set. Scheduler attempt 1 is the successful run used here.

## Peak memory: measured training versus warmup/capture

Each number is a **maximum on one GPU**, not a sum over the job. Rank-0 active/reserved are the actual TensorBoard `memory/max_active(GiB)` / `memory/max_reserved(GiB)` fields: `active_bytes.all.peak` / `reserved_bytes.all.peak`. All-rank allocated/reserved come from exact per-rank allocator byte sidecars; allocated is a separate counter and is not relabeled as an all-rank active counter. Values are rounded to two decimals.

Updates 6-25 are the clean measurement interval. Updates 1-27 additionally include graph warmup/capture and profiling. **HF checkpoint loading precedes these records**, so neither scope establishes the lifetime peak of loading plus training. A separate live allocation at snapshot 27 is also not a peak.

| ID | Rank0 active / reserved, 6-25 | Rank0 active / reserved, 1-27 | All-rank allocated / reserved, 6-25 | All-rank allocated / reserved, 1-27 |
| --- | --- | --- | --- | --- |
| 64-A | 167.15 / 267.11 | 254.64 / 267.11 | 167.15 / 267.11 | 254.64 / 267.11 |
| 64-B | 156.05 / 244.13 | 238.44 / 244.13 | 156.05 / 244.13 | 238.44 / 244.13 |
| 64-C | 116.75 / 254.46 | 248.34 / 256.44 | 116.75 / 254.46 | 248.34 / 256.44 |
| 64-D | 116.75 / 252.70 | 248.34 / 256.62 | 116.75 / 252.70 | 248.34 / 256.62 |
| 32-A | 205.41 / 258.64 | 252.70 / 263.15 | 205.41 / 258.64 | 252.70 / 263.15 |
| 32-B | 196.28 / 233.95 | 232.75 / 245.66 | 196.28 / 233.95 | 232.75 / 245.66 |
| 32-C | 178.02 / 203.12 | 192.77 / 227.41 | 178.02 / 203.12 | 192.77 / 227.41 |
| 32-D | 212.29 / 243.38 | 236.36 / 261.69 | 212.29 / 243.38 | 236.36 / 261.69 |
| 32-E | 186.03 / 268.71 | 267.07 / 269.94 | 186.03 / 268.71 | 267.07 / 269.94 |
| 32-F | 245.30 / 257.88 | 252.07 / 257.88 | 245.30 / 257.88 | 252.07 / 257.88 |

The capture effect matters most for the CPU-moment pair: 64-C/D use about 116.75 GiB allocated during clean training but reach 248.34 GiB during the full 27 protocol. Optimizer offload reduces steady residency; it does not make the capture peak disappear. CUDA graph pools and cached/reserved segments also explain why reserve can remain much larger than active storage.

## What the comparisons establish

### Larger batch and GA on 32 GPUs

32-A B10/GA8 is **2.75% faster** than 32-B B8/GA8, but uses **17.49 GiB more all-27 reserve**. B10 also changes activation budget to 120 GiB and pinned rounding to 128 MiB, and uses a newer adapter. The next B12/GA8 point failed, so B10 is the largest successfully benchmarked microbatch in this submitted family.

B8/GA8 is **8.84% faster in tokens/s/GPU** than B8/GA1. GA8 increases useful work per optimizer update and amortizes CPU moment movement. The newer run also changes CF2.5->1, chunks 4->8, activation budget 64->100 GiB and the dense offload source, so this percentage does not isolate GA alone. The later B4/GA8 run reaches 4,029.98 tokens/s/GPU, versus 3,338.59 for the original B4/GA1 baseline, with much less capture reserve.

### Producer accumulation fusion on 64 GPUs

The completed 64-C/D pair uses the same core `4af09aedf` and adapter `23e9afe68`, workspace, BS8/GA8, SAC+AO100, CPU moments, CF1/chunks 8 and terminal reductions. Fusion changes mean update time from **57.948518 to 57.686466 s**, a **0.4543% throughput gain** (26.4054%->26.5253% MFU).

Clean-measurement reserve falls by 1.76 GiB, but the all-27 reserve increases by 0.18 GiB; both have the same allocated peaks. These are single runs on different host assignments. The observed small gain does not establish repeated-run significance. MXFP8 producer accumulation can change rounding, and full-scale numerical equivalence is not established by this timing pair.

The GPU-moment 64-B control is faster than either CPU-moment sibling, but also has a different reduction cadence and source. The difference cannot be assigned entirely to optimizer transfers.

### GPU count comparison

Best 32 B10 retains **93.42% of best 64 B6 per-GPU throughput** and **46.71% of its global throughput** with half the GPUs. This is a comparison of best observed configurations, not fixed-workload strong scaling: GBS2,560 vs3,072, BS10 vs6, different offload/native policies and source revisions. Raw seconds/update should not be ranked without accounting for tokens/update.

## Failed current configurations

All four jobs below are currently DEAD with failed native attempts. MFU/TPS and verified training peaks are unavailable; exception-time memory is reported separately below. Every row has forced balanced routing, EP16, SAC and CPU BF16 Adam moments.

| ID | GPUs | BS | GA | GBS | Activation offload | CPU moments | SAC | Gradient CPU offload | CUDA graphs | CF / chunks | MFU / TPS | Failure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32-X | 32 | 12 | 8 | 3,072 | Yes, 150 GiB | Yes | Yes | No | Yes | 1.0 / 16 | Unavailable | First training warmup: CUDA OOM, 2.25 GiB requested with 2.21 GiB free. |
| 32-Y | 32 | 8 | 8 | 2,048 | Yes, 100 GiB | Yes | Yes | No | No | 1.0 / 8 | Unavailable | First training update: CUDA OOM, 3.02 GiB requested with about 2.63 GiB free. |
| 16-A | 16 | 1 | 1 | 16 | Yes, 64 GiB | Yes | Yes | Yes, 16 GiB; compact | Yes | 1.0 / 8 | Unavailable | HF checkpoint expert stacking: CUDA OOM, 896 MiB requested with 890.81 MiB free; no training updates. |
| 16-B | 16 | 2 | 1 | 32 | Yes, 64 GiB | Yes | Yes | Yes, 16 GiB; compact | Yes | 1.0 / 8 | Unavailable | HF checkpoint expert stacking: CUDA OOM, 896 MiB requested with 694.81 MiB free; no training updates. |

### B12/GA8 and execution without CUDA graphs

**32-X:** GBS3,072, 12,582,912 tokens/update. Activation budget 150 GiB, CF1, chunks 16, minimum native saved state, pinned rounding 128 MiB, every gather/reduction and no fusion. It failed in the first CUDA graph warmup. A retained rank 3 exception reports 258.74 GiB PyTorch allocated, 2.22 GiB reserved but unallocated, and 274.40 GiB including non-PyTorch memory. Those values describe the failing instant, not audited all-rank high-waters.

**32-Y:** B8/GA8 transformed FX execution with CUDA graphs disabled; SAC+AO100, CPU moments, CF1/chunks 8, every gather/reduction and no fusion remain. It uses GraphTrainer's transformed graph, rather than an unmodified module eager engine. The first training update ran out of memory; rank 2 reports 263.16 GiB allocated, about 0.88 GiB reserved but unallocated and 273.97 GiB total process device usage. This attempt supplies no valid throughput comparison against 32-B.

### Why the 16-GPU runs failed

Both **16-A B1/GA1** and **16-B B2/GA1** use FSDP16/expert-FSDP1, SAC+64 GiB activation offload, CPU BF16 moments, a compact reduced-gradient pass and a 16 GiB gradient offload budget, CF1/chunks 8/minimum native save, every gather/reduce, fusion disabled, CUDA graphs and pinned rounding 128 MiB. GBS is 16/32, or 65,536/131,072 tokens/update.

The design keeps FP32 GPU parameters (158.94 GiB/rank), retains existing BF16 reduction outputs in compact storage and moves selected gradients to CPU. The source plan forecasts 233.50/239.82 GiB GPU usage and 753.75/804.56 GiB of pinned state/pools per four-GPU host. These forecasts describe the planned training storage and exclude checkpoint loading, graph-pool reservation, real transport and other consumers.

The real failure occurs **before training**, in `DeepSeekV3StateDictAdapter.from_hf` -> `_concatenate_expert_weights_dtensor` -> `torch.stack(sorted_experts, dim=0)`. Both jobs attempt an additional 896 MiB GPU allocation while forming expert weights from HF shards. The B1 retained rank 3 exception reports 274.07 GiB allocated and 890.81 MiB free; B2 reports 273.38 GiB allocated and 694.81 MiB free. This identifies checkpoint conversion/stacking as the observed admission failure. It does not show whether the planned training graph would fit after a different loading implementation.

The earlier local FakePG physical probe and small native numerical checks did not execute this same full checkpoint-loading path concurrently on all 16 ranks. **No 16-GPU MFU, tokens/s/GPU or completed training memory peak is assigned.** The supported launch scope was fresh model-only HF initialization; generic optimizer checkpoint/resume was already unqualified.

## Earlier failed and replaced balanced attempts

These nine job names complete the campaign inventory. Every row uses SAC and forced balanced routing. All end in DEAD; cancelled/preallocation failures are distinguished from GPU OOM. They have no completed20-update benchmark, so MFU/TPS and verified benchmark peaks are unavailable.

| ID | GPUs / BS / GA | GBS | Activation offload | CPU moments | SAC | CF / chunks / save | Reduction / fusion | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H01 | 64 / 6 / 8 | 3072 | No | No | Yes | 2.5 / 4 / 0.025 | Last / fused | Package rejected before GPU allocation: upstream timestamp violated the scheduler package age policy. |
| H02 | 64 / 6 / 8 | 3072 | No | No | Yes | 2.5 / 4 / 0.025 | Last / fused | CUDA OOM after package correction: 768 MiB requested with 154.81 MiB free. No completed benchmark. |
| H03 | 64 / 8 / 8 | 4096 | Yes, 100 GiB | No | Yes | 2.5 / 4 / 0.0 | Every / unfused | CUDA OOM in graph preparation: 3.95 GiB requested with 2.77 GiB free. Later B8 control changed CF/loss chunks and source. |
| H04 | 32 / 1 / 1 | 32 | Yes, 64 GiB | No | Yes | 2.5 / 4 / 0.0 | Every / unfused | Cancelled for replacement: GA1 runtime cloned a full 79.47 GiB gradient buffer beyond the graph estimate; corrected 32-F completed. |
| H05 | 32 / 8 / 1 | 256 | Yes, 64 GiB | Yes | Yes | 2.5 / 4 / 0.0 | Every / unfused | Cancelled for the same GA1 gradient ownership issue; corrected 32-D completed. |
| H06 | 64 / 8 / 8 | 4096 | Yes, 100 GiB | Yes | Yes | 1.0 / 8 / 0.0 | Last / unfused | Cancelled for replacement after the local plan exceeded device capacity (307.59 GiB forecast); that forecast is not a measured peak. |
| H07 | 64 / 8 / 8 | 4096 | Yes, 100 GiB | Yes | Yes | 1.0 / 8 / 0.0 | Last / fused | CUDA OOM on the earlier source: 3.02 GiB requested with 978.81 MiB free. |
| H08 | 64 / 8 / 8 | 4096 | Yes, 100 GiB | Yes | Yes | 1.0 / 8 / 0.0 | Last / unfused | CUDA OOM on the early accumulation source: 3.02 GiB requested with 1.01 GiB free; later dense offload layout fix enabled 64-C. |
| H09 | 64 / 8 / 8 | 4096 | Yes, 100 GiB | Yes | Yes | 1.0 / 8 / 0.0 | Last / fused | Cancelled while pending and replaced by the corrected dense offload source; 64-D is its completed successor. |

## Measurement contract and numerical scope

1. All completed benchmarks run27 updates: warmups 1-5, all 20 unprofiled measurements6-25, trace 26 and memory snapshot 27. No samples are trimmed. There is one world barrier/device synchronization after step 5 telemetry, outside timers; the records do not claim a fresh global timing barrier on every update.
2. For each measured update, select the largest elapsed time across every physical rank. Mean update time is the arithmetic mean of those20 maxima. It covers the training update, including optimizer work and offload. Checkpoint initialization and separate profile updates are excluded. These recipes have validation disabled.
3. `tokens/s/GPU = tokens_per_update / mean(max_rank_elapsed_seconds) / GPU_count`. Global throughput is that rate multiplied by GPU count. This differs from the historical256 report's arithmetic mean of per-update TensorBoard rates, so its values are not inserted as a directly matched baseline.
4. Useful model FLOPs/token are **291,853,541,376**, including MTP. `MFU% = tokens/s/GPU x FLOPs/token / 5e15 x100`. The denominator is **5 PFLOP/s dense FP8 per GB300**, derived from [NVIDIA Blackwell Ultra Datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf), whose10 PF sparse figure has dense performance at half that rate. Execution mixes MXFP8, BF16 and TF32; recomputation is excluded from useful FLOPs. This is a common analytical normalization, not measured hardware FLOP utilization.
5. The analysis rechecks rank identities, all 27 step records, the20 maximum-rank timings, token counts, MFU arithmetic and all-rank allocated/reserved maxima. Each completed run also has27 finite Python scalar records per rank for the supplied training metrics. Successful runs are benchmark evidence; they do not establish full-model convergence or bitwise equality between changed recipes.
6. These performance recipes use seed 14536 and `deterministic=False`. Each completed row is one observation, without independent repeat statistics. Small within-run variation does not establish statistical significance for small differences between jobs.

## Common model, data and optimizer setup

| Area | Configuration |
| --- | --- |
| Model | Official DeepSeek V3 671B plus MTP1; 61 decoder layers plus one MTP layer; 682,636,472,320 total parameters |
| Routed experts | 256 experts, top8; historical exact round-robin forced balanced selection; router auxiliary objective/bias settings retained |
| Compute | TorchAO MXFP8 dense projections/FFNs/head and native MXFP8 Mega DistMoE; BF16 SDPA/router; no selected Inductor compute regions |
| Parameters | FP32 persistent/master parameters; BF16 gathered parameters and reduction outputs; FP32 gradient storage in64/32 recipes |
| AdamW | BF16 moments, onGPU orCPU as specified; lr2.3238e-5, betas0.9/0.95, eps1e-8, norm/bias decay0 and other weight decay0.033; clip1 |
| LR schedule | Warmup4 from zero, cosine horizon11,996, minimum factor0.21516481624924694; benchmark length does not shorten the horizon |
| Checkpoint | BF16 HF model-only load from `/mnt/mffuse/mlperf_dsv3_convergence/deepseekv3_checkpoint_bf16_hf_assets`; checkpoint saving disabled/load-only |
| Checkpoint collectives | CPU metadata/planning collectives enabled; this is separate from training activation/optimizer offload |
| Data | Frozen indexed MLPerf C4, AirStore590827, sequence4,096, one document per sample; GBS follows each row |
| Outer activation policy | GraphTrainer SAC; `activation_remat=None` means no additional outer model checkpointing wrapper, rather than SAC disabled |
| FSDP | Always reshard after forward; every-microbatch unshard; gradient reduction cadence explicitly listed; symmetric-memory scope disabled |
| Native memory | Mega pipeline, VMM disabled; scratch and saved-state capacity explicitly listed |
| Offload scheduling | Activation prefetch1/defer1 when active; budgets interpreted as GiB by deployed `cpu_budget_gb *1024**3` implementation |
| Scheduler | `MastGenAICluster` via MSL scheduler (`mslsc`); Nebius subtype230009; tenant `msl_infra_pytorch_dct`; four GPUs/host |
| Native runtime | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266`, PyTorch revision149ae98d98db6921d66e7ba527b395294a127b6e |

## Job and evidence index

Every row below links its exact submitted configuration and current native status. Completed rows also link the raw summary used here. Version is 0 for all current jobs; attempt 1 is used for 32-A and 32-F. Statuses were refreshed with `mslsc`, which targets the MSL scheduler serving these jobs.

| ID | Exact job name | Version/attempt | Run | Submitted config | Measurements | Current status |
| --- | --- | --- | --- | --- | --- | --- |
| 64-A | `dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/force-balanced-20261026/submission-manifest.json) | [summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/force-balanced-20261026/archive/summary.json) | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev.status.json) |
| 64-B | `dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/early-accum-prepared/64-b8-activation-r1/submission-manifest.json) | [summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/early-accum-prepared/64-b8-activation-r1/archive/summary.json) | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev.status.json) |
| 64-C | `dsv3-nebius-64g-b8-balanced-cpu-unfused-r2-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-b8-balanced-cpu-unfused-r2-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/dense-offload-prepared/64-b8-unfused-r2/submission-manifest.json) | [summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/dense-offload-prepared/64-b8-unfused-r2/archive/summary.json) | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-64g-b8-balanced-cpu-unfused-r2-20261026-ivankobzarev.status.json) |
| 64-D | `dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/dense-offload-prepared/64-b8-fused-r3/submission-manifest.json) | [summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/dense-offload-prepared/64-b8-fused-r3/archive/summary.json) | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev.status.json) |
| 32-A | `dsv3-nebius-32g-ep16-b10-ga8-cpu-moments-r1-20261026-ivankobzarev` | v0/a1 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-32g-ep16-b10-ga8-cpu-moments-r1-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/expanded32-prepared-r3/32-b10-ga8-cpu-r1/attempt_1/submission-manifest.json) | [summary](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/b10/benchmark/summary.json) | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b10-ga8-cpu-moments-r1-20261026-ivankobzarev.status.json) |
| 32-B | `dsv3-nebius-32g-ep16-b8-ga8-cpu-moments-r1-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-32g-ep16-b8-ga8-cpu-moments-r1-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/32-accumulation-prepared/32-b8-ga8-cpu-r1/submission-manifest.json) | [summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/32-accumulation-prepared/32-b8-ga8-cpu-r1/archive/summary.json) | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b8-ga8-cpu-moments-r1-20261026-ivankobzarev.status.json) |
| 32-C | `dsv3-nebius-32g-ep16-b4-ga8-cpu-moments-r1-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-32g-ep16-b4-ga8-cpu-moments-r1-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/32-accumulation-prepared/32-b4-ga8-cpu-r1/submission-manifest.json) | [summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/32-accumulation-prepared/32-b4-ga8-cpu-r1/archive/summary.json) | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b4-ga8-cpu-moments-r1-20261026-ivankobzarev.status.json) |
| 32-D | `dsv3-nebius-32g-ep16-b8-ga1-cpu-moments-r1-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-32g-ep16-b8-ga1-cpu-moments-r1-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/single-microbatch-prepared/32-b8-cpu-r1/submission-manifest.json) | [summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/single-microbatch-prepared/32-b8-cpu-r1/archive/summary.json) | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b8-ga1-cpu-moments-r1-20261026-ivankobzarev.status.json) |
| 32-E | `dsv3-nebius-32g-ep16-b4-ga1-cpu-moments-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-32g-ep16-b4-ga1-cpu-moments-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/prepared/32-b4-cpu/submission-manifest.json) | [summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/prepared/32-b4-cpu/archive/summary.json) | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b4-ga1-cpu-moments-20261026-ivankobzarev.status.json) |
| 32-F | `dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev` | v0/a1 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/single-microbatch-prepared/32-b1-gpu-r1/attempt_1/submission-manifest.json) | [summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/single-microbatch-prepared/32-b1-gpu-r1/attempt_1/archive/summary.json) | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev.status.json) |
| 32-X | `dsv3-nebius-32g-ep16-b12-ga8-cpu-moments-r1-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-32g-ep16-b12-ga8-cpu-moments-r1-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/expanded32-prepared-r3/32-b12-ga8-cpu-r1/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b12-ga8-cpu-moments-r1-20261026-ivankobzarev.status.json) |
| 32-Y | `dsv3-nebius-32g-ep16-b8-ga8-fx-eager-r1-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-32g-ep16-b8-ga8-fx-eager-r1-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/expanded32-prepared-r3/32-b8-ga8-fx-eager-r1/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b8-ga8-fx-eager-r1-20261026-ivankobzarev.status.json) |
| 16-A | `dsv3-nebius-16g-ep16-b1-ga1-compact-cpu-r1-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-16g-ep16-b1-ga1-compact-cpu-r1-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/fit16-prepared/16-b1-compact-r1/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-16g-ep16-b1-ga1-compact-cpu-r1-20261026-ivankobzarev.status.json) |
| 16-B | `dsv3-nebius-16g-ep16-b2-ga1-compact-cpu-r1-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-16g-ep16-b2-ga1-compact-cpu-r1-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/fit16-prepared/16-b2-compact-r1/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-16g-ep16-b2-ga1-compact-cpu-r1-20261026-ivankobzarev.status.json) |
| H01 | `dsv3-nebius-ep16-b6-balanced-wgrad-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-ep16-b6-balanced-wgrad-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/force-balanced-wgrad-20261026/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-ep16-b6-balanced-wgrad-20261026-ivankobzarev.status.json) |
| H02 | `dsv3-nebius-ep16-b6-balanced-wgrad-r1-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-ep16-b6-balanced-wgrad-r1-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/force-balanced-wgrad-20261026/fusion-b6-provenance-r1/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-ep16-b6-balanced-wgrad-r1-20261026-ivankobzarev.status.json) |
| H03 | `dsv3-nebius-ep16-b8-balanced-offload-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-ep16-b8-balanced-offload-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/force-balanced-wgrad-20261026/b8-offload-prepared/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-ep16-b8-balanced-offload-20261026-ivankobzarev.status.json) |
| H04 | `dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/prepared/32-b1-gpu/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-20261026-ivankobzarev.status.json) |
| H05 | `dsv3-nebius-32g-ep16-b8-ga1-cpu-moments-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-32g-ep16-b8-ga1-cpu-moments-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/prepared/32-b8-cpu/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b8-ga1-cpu-moments-20261026-ivankobzarev.status.json) |
| H06 | `dsv3-nebius-64g-b8-balanced-cpu-unfused-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-b8-balanced-cpu-unfused-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/prepared/64-b8-unfused/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-64g-b8-balanced-cpu-unfused-20261026-ivankobzarev.status.json) |
| H07 | `dsv3-nebius-64g-b8-balanced-cpu-wgrad-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-b8-balanced-cpu-wgrad-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/prepared/64-b8-fused/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-64g-b8-balanced-cpu-wgrad-20261026-ivankobzarev.status.json) |
| H08 | `dsv3-nebius-64g-b8-balanced-cpu-unfused-r1-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-b8-balanced-cpu-unfused-r1-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/early-accum-prepared/64-b8-unfused-r1/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-64g-b8-balanced-cpu-unfused-r1-20261026-ivankobzarev.status.json) |
| H09 | `dsv3-nebius-64g-b8-balanced-cpu-wgrad-r2-20261026-ivankobzarev` | v0/a0 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r2-20261026-ivankobzarev) | [manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/early-accum-prepared/64-b8-fused-r2/submission-manifest.json) | Unavailable | [status](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r2-20261026-ivankobzarev.status.json) |

### Source and factory selectors

Source differences are preserved here to make reproduction and comparisons unambiguous. Use the full revision and workspace from each submitted manifest; the shortened revisions below are labels. The factory column is the actual `GT_REPORT_CONFIG_MODULE:CONFIG` selector.

| ID | Core | Adapter | Exact module:factory |
| --- | --- | --- | --- |
| 64-A | `4738aa897` | `cfd004055` | `force_balanced_save025_configs:dsv3_671b_ep16_fsdp64_lbs6_ga8_chunk4_cf25_mega_sac_save025_force_balanced` |
| 64-B | `a63953fef` | `23e9afe68` | `cpu_moment_offload_configs:dsv3_671b_ep16_b8_ga8_balanced_activation_offload_cf1_chunk8_gpu_moments` |
| 64-C | `4af09aedf` | `23e9afe68` | `cpu_moment_offload_configs:dsv3_671b_ep16_b8_ga8_balanced_cpu_moments_cf1_chunk8_last_reduce_unfused` |
| 64-D | `4af09aedf` | `23e9afe68` | `cpu_moment_offload_configs:dsv3_671b_ep16_b8_ga8_balanced_cpu_moments_cf1_chunk8_wgrad_fused` |
| 32-A | `2b47128a5` | `635358930` | `cpu_moment_offload_configs:dsv3_671b_32gpu_ep16_b10_ga8_balanced_cpu_moments_ao120_cf1_chunk8` |
| 32-B | `4af09aedf` | `5177fc3fb` | `cpu_moment_offload_configs:dsv3_671b_32gpu_ep16_b8_ga8_balanced_cpu_moments_cf1_chunk8` |
| 32-C | `4af09aedf` | `5177fc3fb` | `cpu_moment_offload_configs:dsv3_671b_32gpu_ep16_b4_ga8_balanced_cpu_moments_cf1_chunk8` |
| 32-D | `377f98fdd` | `23e9afe68` | `cpu_moment_offload_configs:dsv3_671b_32gpu_ep16_b8_ga1_balanced_cpu_moments` |
| 32-E | `ada0e8287` | `439908156` | `cpu_moment_offload_configs:dsv3_671b_32gpu_ep16_b4_ga1_balanced_cpu_moments` |
| 32-F | `377f98fdd` | `23e9afe68` | `cpu_moment_offload_configs:dsv3_671b_32gpu_ep16_b1_ga1_balanced_gpu_moments` |
| 32-X | `2b47128a5` | `635358930` | `cpu_moment_offload_configs:dsv3_671b_32gpu_ep16_b12_ga8_balanced_cpu_moments_ao150_cf1_chunk16` |
| 32-Y | `2b47128a5` | `635358930` | `cpu_moment_offload_configs:dsv3_671b_32gpu_ep16_b8_ga8_balanced_cpu_moments_cf1_chunk8_fx_eager` |
| 16-A | `b7ccd73be` | `635358930` | `fit16.recipes:compact` |
| 16-B | `b7ccd73be` | `635358930` | `fit16.recipes:compact` |

For16-A/B the factory is `fit16.recipes:compact`; the definition selects `FIT16_BATCH=1/2`, `FIT16_GRADIENT_OFFLOAD_GIB=16` and `FIT16_SCRATCH_FACTOR=1.0`. Both allocator aliases set `expandable_segments:True,graph_capture_record_stream_reuse:True,pinned_max_round_threshold_mb:128`. For64-A, the inherited100 GiB activation budget and prefetch/defer fields are inactive because activation offload is disabled.

### Retained profiles and artifacts

The prior collector verified/archive-retained all configured-rank trace 26 and memory27 artifacts for the nine completed rows preceding B10. B10's metrics/scalars/TensorBoard were independently downloaded from the original attempt 1 output for this report; this report does not certify full B10 profile archival. Existing archives have90-day retention. The following links point to retained original rank 0 traces; no new profile was published for this report.

| ID | Evidence | Compute profile |
| --- | --- | --- |
| 64-A | [archive](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev-verified64/dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev/v0-a0) | [rank0 original trace26](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev-verified64/dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev/v0-a0/profiling/traces/iteration_26/rank0_trace.json.gz) |
| 64-B | [archive](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev-verified64/dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev/v0-a0) | [rank0 original trace26](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev-verified64/dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev/v0-a0/profiling/traces/iteration_26/rank0_trace.json.gz) |
| 64-C | [archive](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-64g-b8-balanced-cpu-unfused-r2-20261026-ivankobzarev-verified64/dsv3-nebius-64g-b8-balanced-cpu-unfused-r2-20261026-ivankobzarev/v0-a0) | [rank0 original trace26](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-64g-b8-balanced-cpu-unfused-r2-20261026-ivankobzarev-verified64/dsv3-nebius-64g-b8-balanced-cpu-unfused-r2-20261026-ivankobzarev/v0-a0/profiling/traces/iteration_26/rank0_trace.json.gz) |
| 64-D | [archive](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev-verified64/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev/v0-a0) | [rank0 original trace26](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev-verified64/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev/v0-a0/profiling/traces/iteration_26/rank0_trace.json.gz) |
| 32-A | [original output](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-nebius-32g-ep16-b10-ga8-cpu-moments-r1-20261026-ivankobzarev/v_0/attempt_1) | Full profile archive not audited here |
| 32-B | [archive](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-32g-ep16-b8-ga8-cpu-moments-r1-20261026-ivankobzarev-verified32/dsv3-nebius-32g-ep16-b8-ga8-cpu-moments-r1-20261026-ivankobzarev/v0-a0) | [rank0 original trace26](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-32g-ep16-b8-ga8-cpu-moments-r1-20261026-ivankobzarev-verified32/dsv3-nebius-32g-ep16-b8-ga8-cpu-moments-r1-20261026-ivankobzarev/v0-a0/profiling/traces/iteration_26/rank0_trace.json.gz) |
| 32-C | [archive](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-32g-ep16-b4-ga8-cpu-moments-r1-20261026-ivankobzarev-verified32/dsv3-nebius-32g-ep16-b4-ga8-cpu-moments-r1-20261026-ivankobzarev/v0-a0) | [rank0 original trace26](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-32g-ep16-b4-ga8-cpu-moments-r1-20261026-ivankobzarev-verified32/dsv3-nebius-32g-ep16-b4-ga8-cpu-moments-r1-20261026-ivankobzarev/v0-a0/profiling/traces/iteration_26/rank0_trace.json.gz) |
| 32-D | [archive](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-32g-ep16-b8-ga1-cpu-moments-r1-20261026-ivankobzarev-verified32/dsv3-nebius-32g-ep16-b8-ga1-cpu-moments-r1-20261026-ivankobzarev/v0-a0) | [rank0 original trace26](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-32g-ep16-b8-ga1-cpu-moments-r1-20261026-ivankobzarev-verified32/dsv3-nebius-32g-ep16-b8-ga1-cpu-moments-r1-20261026-ivankobzarev/v0-a0/profiling/traces/iteration_26/rank0_trace.json.gz) |
| 32-E | [archive](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-32g-ep16-b4-ga1-cpu-moments-20261026-ivankobzarev-verified32/dsv3-nebius-32g-ep16-b4-ga1-cpu-moments-20261026-ivankobzarev/v0-a0) | [rank0 original trace26](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-32g-ep16-b4-ga1-cpu-moments-20261026-ivankobzarev-verified32/dsv3-nebius-32g-ep16-b4-ga1-cpu-moments-20261026-ivankobzarev/v0-a0/profiling/traces/iteration_26/rank0_trace.json.gz) |
| 32-F | [archive](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev-verified32/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev/v0-a1) | [rank0 original trace26](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev-verified32/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev/v0-a1/profiling/traces/iteration_26/rank0_trace.json.gz) |

### Local audit files

- [Full derived measurements, memory counters, validation and manifest hashes](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/analysis.json)
- [Recomputation script](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/analyze.py) and [B10 download script](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/collect_b10.py)
- [Current23-job status summary](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/current-status-summary.json)
- [MSL scheduler recent-job listing](/home/ivankobzarev/tmp/recent-mast-report-20261026-mslsc-jobs.json)
- [B10 original32-rank timing and scalar files](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/b10/benchmark/summary.json)
- [B12 failure log](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b12-ga8-cpu-moments-r1-20261026-ivankobzarev.failure.log)
- [FX eager failure log](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b8-ga8-fx-eager-r1-20261026-ivankobzarev.failure.log)
- [16-GPU B1 failure log](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-16g-ep16-b1-ga1-compact-cpu-r1-20261026-ivankobzarev.failure.log) and [B2 failure log](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-16g-ep16-b2-ga1-compact-cpu-r1-20261026-ivankobzarev.failure.log)

## Next configuration decisions

1. Select64-A for maximum measured throughput, or 64-B when the lower training reserve is more useful than its 1.24% throughput difference.
2. Select32-A for maximum measured throughput and the largest completed microbatch. Select32-B for more allocator headroom at a 2.68% throughput cost;32-C offers the lowest reserve in the completed32 family.
3. Resolve the observed HF expert-stacking loading peak before qualifying16 GPUs. The current B1/B2 attempts provide no training measurement.
4. A later B12 or FX eager retry needs a concrete reduction in the memory at its failing first update. Removing CUDA graphs alone did not make the submitted B8 transformed FX configuration fit.
5. Treat the0.4543% producer-fusion gain as a small single-pair observation until repeated measurements and appropriate numerical checks support choosing it over the simpler unfused configuration.
