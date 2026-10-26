# DeepSeek V3 671B + MTP1: best forced balanced configurations

**October 26, 2026 | 64, 32 and 16 GPUs | Nebius GB300 NVL72, 4 GPUs per host**

The fastest completed runs use **64 GPUs with no CPU offload (28.33% MFU)** and **32 GPUs with activation and Adam-state offload (26.47% MFU)**. The 16-GPU attempts failed during checkpoint loading, before training. Recent jobs were rechecked at 17:20 UTC.

## Summary

One winner is shown for each tested combination of GPU count and CPU offload mode. All use **forced balanced routing, EP16, SAC, 4,096-token sequences and forward/backward CUDA graphs**. Peak memory is **active / reserved in GiB**, measured on rank 0 across updates 1-27, including graph capture and profiling. Checkpoint loading is outside that interval.

| Best configuration | GPUs | BS | GA | GBS | CPU offload | SAC | WGrad fusion | MFU | Tokens/s/GPU | Peak active / reserved | Links |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [64-A: no CPU offload](#64-a) | 64 | 6 | 8 | 3,072 | None | Yes | No | 28.33% | 4,853.86 | 254.64 / 267.11 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev) / [Profile](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev-verified64/dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev/v0-a0/profiling/traces/iteration_26/rank0_trace.json.gz) |
| [64-B: activations on CPU](#64-b) | 64 | 8 | 8 | 4,096 | Activations 100 GiB | Yes | No | 27.98% | 4,793.44 | 238.44 / 244.13 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev) / [Profile](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev-verified64/dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev/v0-a0/profiling/traces/iteration_26/rank0_trace.json.gz) |
| [64-D: activations + Adam states on CPU](#64-d) | 64 | 8 | 8 | 4,096 | Activations 100 GiB + Adam states | Yes | Yes | 26.53% | 4,544.29 | 248.34 / 256.62 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev) / [Profile](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev-verified64/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev/v0-a0/profiling/traces/iteration_26/rank0_trace.json.gz) |
| [32-A: activations + Adam states on CPU](#32-a) | 32 | 10 | 8 | 2,560 | Activations 120 GiB + Adam states | Yes | No | 26.47% | 4,534.30 | 252.70 / 263.15 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-32g-ep16-b10-ga8-cpu-moments-r1-20261026-ivankobzarev) / [Output](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-nebius-32g-ep16-b10-ga8-cpu-moments-r1-20261026-ivankobzarev/v_0/attempt_1) |
| [32-F: activations on CPU](#32-f) | 32 | 1 | 1 | 32 | Activations 64 GiB | Yes | No | 16.67% | 2,856.00 | 252.07 / 257.88 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev) / [Profile](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev-verified32/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev/v0-a1/profiling/traces/iteration_26/rank0_trace.json.gz) |
| 16 GPUs: no completed run | 16 | 1 or 2 | 1 | 16 or 32 | Activations 64 GiB + Adam states + gradients 16 GiB | Yes | No | Unavailable | Unavailable | Unavailable | [B1 job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-16g-ep16-b1-ga1-compact-cpu-r1-20261026-ivankobzarev) / [B2 job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-16g-ep16-b2-ga1-compact-cpu-r1-20261026-ivankobzarev) |

**BS** is sequences per microbatch on each data rank; **GA** is microbatches per optimizer update; **GBS = GPUs x BS x GA**. Activation and gradient offload budgets are per GPU. Adam states are BF16; parameters remain on GPU. SAC saves selected activations and recomputes others during backward.

**Profile** links open the original rank-0 trace from update 26 in Manifold. For 32-A, the **Output** link opens its successful attempt 1 directory; its full profile archive has not been verified. Existing profile archives have 90-day retention.

## Which configuration to choose

Use **64-A** for maximum measured throughput: **310,646.84 tokens/s across 64 GPUs**. If you can move activations to CPU, **64-B** reduces peak reserved memory by **22.98 GiB** while giving up **1.24% throughput**. Its batch, native memory settings and source revision also differ.

Use **32-A** for maximum measured throughput with 32 GPUs: **145,097.74 tokens/s globally**. It retains **93.42% of the best 64-GPU per-GPU throughput**. The two runs have different batches and offload policies, so this comparison does not isolate GPU-count scaling.

Use **64-D** when Adam states must reside on CPU; it is the fastest completed 64-GPU run in that class and has **WGrad fusion enabled**. **32-F** is the completed 32-GPU choice with Adam states kept on GPU, but its B1/GA1 workload has much lower throughput than 32-A.

## What each selected configuration ran

### 64-A

**Fastest 64-GPU run, with no CPU offload.** BS6, GA8 and GBS3,072. SAC is enabled; FP32 parameters and gradients and BF16 Adam states stay on GPU. Native MoE scratch capacity factor is 2.5, loss is computed in 4 chunks, and native saved-state capacity factor is 0.025. Parameters are gathered and gradients are reduced every microbatch; WGrad fusion is disabled. Mean optimizer update: **40.51 s**.

### 64-B

**Fastest 64-GPU run with activation offload and GPU Adam states.** BS8, GA8 and GBS4,096. SAC is combined with a 100 GiB activation offload budget. Native scratch capacity factor is 1.0, loss uses 8 chunks, and native saved-state capacity is at its minimum. Gradient reduction runs every microbatch; WGrad fusion is disabled. Mean optimizer update: **54.69 s**. This is the 64-GPU option with the lowest peak reserve among the selected runs.

### 64-D

**Fastest 64-GPU run with both activation and Adam-state offload.** BS8, GA8 and GBS4,096. SAC, 100 GiB activation offload, CPU BF16 Adam states, scratch capacity factor 1.0, 8 loss chunks and minimum native saved-state capacity. Parameters are gathered every microbatch, while terminal FSDP gradient reduction runs after the last microbatch. **WGrad producers accumulate directly into gradient buffers.** Mean optimizer update: **57.69 s**.

### 32-A

**Fastest 32-GPU run with CPU Adam states.** BS10, GA8 and GBS2,560. SAC, a 120 GiB activation offload budget, CPU BF16 Adam states, scratch capacity factor 1.0, 8 loss chunks and minimum native saved-state capacity. Pinned allocator rounding is capped at 128 MiB. Gradient reduction runs every microbatch; WGrad fusion is disabled. Attempt 0 was preempted for scheduler capacity; **attempt 1 completed**. Mean optimizer update: **72.27 s**.

### 32-F

**Fastest completed 32-GPU run with GPU Adam states in this campaign.** BS1, GA1 and GBS32. SAC, 64 GiB activation offload, GPU BF16 Adam states, scratch capacity factor 2.5, 4 loss chunks and minimum native saved-state capacity. The corrected single-microbatch ownership path avoids retaining a cloned full gradient buffer. GA1 provides no inter-microbatch gradient accumulation to fuse. Successful attempt: 1; mean optimizer update: **1.43 s**.

Native scratch capacity factor controls MoE workspace size. Native saved-state capacity is separate from SAC: 0.025 sizes an internal arena and does not mean that 2.5% of layers are checkpointed. Activation offload uses prefetch 1 / defer 1. CPU Adam updates use a 512 MiB GPU staging budget per optimizer instance/device.

## WGrad fusion: what was scheduled and why most winners have it off

**Five forced balanced WGrad jobs were submitted. One completed.** The successful job is **64-D**, `dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev`; it is included in the summary with fusion marked **Yes**.

The matched unfused control, [64-C](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-b8-balanced-cpu-unfused-r2-20261026-ivankobzarev), used the same source revisions, BS8/GA8, activation budget, CPU Adam states and final-microbatch reduction. Fusion reduced mean update time from **57.95 to 57.69 s**, giving **0.45% higher throughput**: 4,523.74 to 4,544.29 tokens/s/GPU, or 26.41% to 26.53% MFU. This is one matched pair on different host assignments; repeat measurements would be needed to establish such a small gain reliably.

The other selected GA8 recipes explicitly reduce gradients **every microbatch**. In the deployed SPMD FSDP implementation, WGrad accumulation fusion requires `fsdp_grad_reduce_mode="last_microbatch"`. That mode retains unsharded gradient accumulators until the final reduction, increasing their lifetime. Enabling fusion therefore also requires a change to gradient reduction scheduling and memory ownership. The 32-GPU winners were submitted with every-microbatch reduction and `fuse_wgrad_accumulation="disabled"`; no 32-GPU WGrad variant appears in the recent job listing. The GA1 run has only one microbatch and no accumulation to fuse.

The other four WGrad submissions ended before a benchmark completed:

- [B6, initial submission](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-ep16-b6-balanced-wgrad-20261026-ivankobzarev): BS6/GA8, no CPU offload, last-microbatch reduction and fusion enabled. Rejected before GPU allocation because the package timestamp violated the scheduler package-age rule.
- [B6, corrected package](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-ep16-b6-balanced-wgrad-r1-20261026-ivankobzarev): same batch and offload class, last-microbatch reduction and fusion enabled. Ran out of GPU memory: 768 MiB requested with 154.81 MiB free. It produced no completed throughput benchmark.
- [B8, initial CPU-Adam fused run](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-b8-balanced-cpu-wgrad-20261026-ivankobzarev): BS8/GA8, activation offload and CPU Adam states. Ran out of GPU memory on the earlier source: 3.02 GiB requested with 978.81 MiB free.
- [B8, r2 CPU-Adam fused run](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r2-20261026-ivankobzarev): cancelled while pending and replaced by the corrected dense activation offload source. Its completed successor is **64-D / r3**.

These failed or replaced submissions do not provide a timing comparison. The faster unfused 64-A/64-B rows also use different offload and memory policies from 64-D.

## 16 GPUs: checkpoint loading runs out of memory

Both submitted recipes use SAC, 64 GiB activation offload, CPU BF16 Adam states and compact reduced-gradient storage with a 16 GiB CPU gradient budget. BS1/GA1 gives GBS16; BS2/GA1 gives GBS32. They use FSDP16, EP16, scratch capacity factor 1.0, 8 loss chunks, minimum native saved-state capacity and pinned rounding capped at 128 MiB. WGrad fusion is disabled.

Both fail **before the first training update**, while loading HF expert weights: `from_hf -> _concatenate_expert_weights_dtensor -> torch.stack(sorted_experts, dim=0)`. An additional **896 MiB** allocation is requested with only **890.81 MiB free for B1** and **694.81 MiB free for B2**. Changing BS did not avoid this checkpoint-conversion peak. No MFU, throughput or measured training peak can be assigned to either run.

The local training-memory forecasts excluded this loading path. The next step for 16 GPUs is to reduce the expert conversion/stacking peak, then measure whether the full training graph fits.

## How the numbers were measured

Each successful run completed **27 updates**: 5 warmup/capture updates, **20 measured updates (6-25)**, a compute profile at update 26 and a memory snapshot at update 27. For each measured update, the analysis takes the slowest elapsed time across all ranks, then averages those 20 values. Timers include optimizer work and offload; checkpoint loading and profile updates are excluded.

`tokens/s/GPU = GBS x 4,096 / mean_update_seconds / GPU_count`.

MFU uses **291,853,541,376 useful model FLOPs/token**, including MTP, divided by the GB300 **5 PFLOP/s dense FP8** peak from the [NVIDIA Blackwell Ultra datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf). Recomputation is excluded from useful model FLOPs. Execution mixes MXFP8, BF16 and TF32, so this is an analytical MFU normalization.

**Active memory** is the actual rank-0 TensorBoard `active_bytes.all.peak` counter; **reserved memory** is `reserved_bytes.all.peak`. All-rank allocated/reserved sidecars were checked independently and match the summary at the displayed precision. Allocated and active are separate allocator counters. Every peak is on one GPU, and the summary covers training updates 1-27. Capture can dominate: 64-D peaks at **116.75 GiB active during measured updates**, but reaches **248.34 GiB** over the full capture/profiling interval.

All-rank timing, token counts, MFU arithmetic, memory sidecars and finite scalar records were checked. Each result is one benchmark observation. These recipes use seed 14536 and `deterministic=False`; the measurements do not establish convergence or bitwise equivalence across configurations.

<details>
<summary>Common setup, exact recipes and evidence</summary>

### Common setup

- **Model:** official DeepSeek V3 671B plus one MTP layer; 61 decoder layers, 682,636,472,320 total parameters, 256 routed experts with top-8 selection. Submitted configs use `forced_balanced_routing=True` and `_RoundRobinDeepSeekV3Router.Config`.
- **Parallelism:** PP=TP=CP=1, data replication 1, EP16. Dense FSDP degree is 64/32/16; expert data sharding is 4/2/1. FSDP and EP overlap in rank ownership. Parameters always reshard after forward and are gathered every microbatch.
- **Compute:** TorchAO MXFP8 dense operations and native MXFP8 Mega DistMoE; BF16 SDPA/router. No selected Inductor compute regions, optimizer CUDA graphs, native VMM or symmetric-memory scope. SAC is supplied by GraphTrainer; `activation_remat=None` means there is no additional outer checkpointing wrapper.
- **Storage:** FP32 persistent parameters; BF16 gathered parameters and reduction outputs. The 64/32-GPU recipes use FP32 gradient storage. CPU metadata/checkpoint collectives are separate from training offload.
- **Optimizer:** AdamW with BF16 states, learning rate 2.3238e-5, betas 0.9/0.95, epsilon 1e-8, weight decay 0.033 except norm/bias decay 0, gradient clipping 1. LR warmup is 4 updates; cosine horizon 11,996; minimum factor 0.21516481624924694.
- **Data and checkpoint:** frozen indexed MLPerf C4, AirStore 590827, one document per 4,096-token sample. BF16 model-only HF initialization from `/mnt/mffuse/mlperf_dsv3_convergence/deepseekv3_checkpoint_bf16_hf_assets`; validation and checkpoint saving disabled.
- **Runtime:** `MastGenAICluster`, Nebius subtype 230009, tenant `msl_infra_pytorch_dct`, approximately 276.62 GiB CUDA memory per GPU. Conda runtime `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266`; PyTorch revision `149ae98d98db6921d66e7ba527b395294a127b6e`.

### Exact selected recipes

Source revisions below identify the submitted code. Each manifest contains the full revision, workspace/package and launch definition. All jobs use version 0; 32-A and 32-F use attempt 1, and the others use attempt 0.


**64-A**: `dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev`.

- Recipe: `force_balanced_save025_configs:dsv3_671b_ep16_fsdp64_lbs6_ga8_chunk4_cf25_mega_sac_save025_force_balanced`; core `4738aa897`; adapter `cfd004055`.
- Measured update: 40.505520 s; global throughput: 310,646.84 tokens/s. Peak active/reserved during updates 6-25: 167.15 / 267.11 GiB.
- [Submitted manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/force-balanced-20261026/submission-manifest.json) / [Raw summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/force-balanced-20261026/archive/summary.json) / [Status snapshot](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev.status.json) / [Artifacts](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev-verified64/dsv3-nebius-ep16-b6s025-balanced-20261026-ivankobzarev/v0-a0).


**64-B**: `dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev`.

- Recipe: `cpu_moment_offload_configs:dsv3_671b_ep16_b8_ga8_balanced_activation_offload_cf1_chunk8_gpu_moments`; core `a63953fef`; adapter `23e9afe68`.
- Measured update: 54.688051 s; global throughput: 306,780.28 tokens/s. Peak active/reserved during updates 6-25: 156.05 / 244.13 GiB.
- [Submitted manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/early-accum-prepared/64-b8-activation-r1/submission-manifest.json) / [Raw summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/early-accum-prepared/64-b8-activation-r1/archive/summary.json) / [Status snapshot](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev.status.json) / [Artifacts](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev-verified64/dsv3-nebius-64g-b8-balanced-activation-r1-20261026-ivankobzarev/v0-a0).


**64-D**: `dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev`.

- Recipe: `cpu_moment_offload_configs:dsv3_671b_ep16_b8_ga8_balanced_cpu_moments_cf1_chunk8_wgrad_fused`; core `4af09aedf`; adapter `23e9afe68`.
- Measured update: 57.686466 s; global throughput: 290,834.53 tokens/s. Peak active/reserved during updates 6-25: 116.75 / 252.70 GiB.
- [Submitted manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/dense-offload-prepared/64-b8-fused-r3/submission-manifest.json) / [Raw summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/dense-offload-prepared/64-b8-fused-r3/archive/summary.json) / [Status snapshot](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev.status.json) / [Artifacts](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev-verified64/dsv3-nebius-64g-b8-balanced-cpu-wgrad-r3-20261026-ivankobzarev/v0-a0).


**32-A**: `dsv3-nebius-32g-ep16-b10-ga8-cpu-moments-r1-20261026-ivankobzarev`.

- Recipe: `cpu_moment_offload_configs:dsv3_671b_32gpu_ep16_b10_ga8_balanced_cpu_moments_ao120_cf1_chunk8`; core `2b47128a5`; adapter `635358930`.
- Measured update: 72.266873 s; global throughput: 145,097.74 tokens/s. Peak active/reserved during updates 6-25: 205.41 / 258.64 GiB.
- [Submitted manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/expanded32-prepared-r3/32-b10-ga8-cpu-r1/attempt_1/submission-manifest.json) / [Raw summary](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/b10/benchmark/summary.json) / [Status snapshot](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b10-ga8-cpu-moments-r1-20261026-ivankobzarev.status.json) / [Artifacts](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-nebius-32g-ep16-b10-ga8-cpu-moments-r1-20261026-ivankobzarev/v_0/attempt_1).


**32-F**: `dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev`.

- Recipe: `cpu_moment_offload_configs:dsv3_671b_32gpu_ep16_b1_ga1_balanced_gpu_moments`; core `377f98fdd`; adapter `23e9afe68`.
- Measured update: 1.434174 s; global throughput: 91,391.99 tokens/s. Peak active/reserved during updates 6-25: 245.30 / 257.88 GiB.
- [Submitted manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/single-microbatch-prepared/32-b1-gpu-r1/attempt_1/submission-manifest.json) / [Raw summary](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/single-microbatch-prepared/32-b1-gpu-r1/attempt_1/archive/summary.json) / [Status snapshot](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev.status.json) / [Artifacts](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-671b-64gpu-archive-20261024-ivankobzarev/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev-verified32/dsv3-nebius-32g-ep16-b1-ga1-gpu-moments-r1-20261026-ivankobzarev/v0-a1).


**16-A**: `dsv3-nebius-16g-ep16-b1-ga1-compact-cpu-r1-20261026-ivankobzarev`, version 0 / attempt 0.

- Recipe: `fit16.recipes:compact`; `FIT16_BATCH=1`, `FIT16_GRADIENT_OFFLOAD_GIB=16`, `FIT16_SCRATCH_FACTOR=1.0`; core `b7ccd73be`; adapter `635358930`.
- [Submitted manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/fit16-prepared/16-b1-compact-r1/submission-manifest.json) / [Status snapshot](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-16g-ep16-b1-ga1-compact-cpu-r1-20261026-ivankobzarev.status.json).


**16-B**: `dsv3-nebius-16g-ep16-b2-ga1-compact-cpu-r1-20261026-ivankobzarev`, version 0 / attempt 0.

- Recipe: `fit16.recipes:compact`; `FIT16_BATCH=2`, `FIT16_GRADIENT_OFFLOAD_GIB=16`, `FIT16_SCRATCH_FACTOR=1.0`; core `b7ccd73be`; adapter `635358930`.
- [Submitted manifest](/home/ivankobzarev/local/e/tmp/dsv3-671b-64gpu-20261024/gpu-minimum-20261026/fit16-prepared/16-b2-compact-r1/submission-manifest.json) / [Status snapshot](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-16g-ep16-b2-ga1-compact-cpu-r1-20261026-ivankobzarev.status.json).


### Audit files and references

The selection was made from 23 forced balanced job names: 10 completed benchmarks and 13 failed or replaced jobs. The full inventory and derived measurements remain in the audit files:

- [All configurations, derived measurements and manifest hashes](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/analysis.json).
- [Latest recent-job listing](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/latest-jobs-for-readable-report.jsonl) and [23-job native status snapshot](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/current-status-summary.json).
- [Recomputation script](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/analyze.py) and [B10 original timing/scalar download script](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/collect_b10.py).
- [16-GPU B1 failure log](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-16g-ep16-b1-ga1-compact-cpu-r1-20261026-ivankobzarev.failure.log) / [B2 failure log](/home/ivankobzarev/tmp/dsv3-balanced-report-evidence-20261026/dsv3-nebius-16g-ep16-b2-ga1-compact-cpu-r1-20261026-ivankobzarev.failure.log).
- [256-GPU reference report on `gt-ptc-runs-candidate-1310`](https://github.com/pytorch/torchtitan/blob/gt-ptc-runs-candidate-1310/docs/graphtrainer-ptc26-runs-release/README.md). Its historical throughput aggregation differs from this report.

The audit-file links are local paths for this workspace; MAST and Manifold links above are web links.

</details>
