# DSv3 671B: GraphTrainer memory flexibility on 16 x GB300

**The 16-GPU result uses BF16 Adam moment buffers stored on CPU. Master weights remain FP32 on GPU. Eligible reduced gradients are retained in compact BF16 storage and restored to FP32 for gradient norm computation and Adam consumption.** These are separate precision and placement choices.

**BF16 moments are required for these hosts' RAM budget:** full FP32 moments would need approximately **1,271.51 GiB per four-GPU host**, exceeding the **845 GiB job assignment** and approximately **954.66 GiB physical RAM**. This requirement is specific to the current host arrangement and CPU-resident optimizer implementation.

Five real MAST configurations completed **17 optimizer updates on all 16 ranks**. The largest verified configuration is **Full recompute, local BS7 / GA1**, with **38 GiB/rank of CPU gradient-offload quota**: **3,129.90 tokens/s/GPU, 18.27% MFU, and 253.79 / 265.84 GiB allocated / reserved**. BS7 is the largest verified successful batch in this family; a real adjacent BS8 capacity boundary has not been established.

## 1. Main results

**Fixed for every row:** DeepSeek V3 671B plus one MTP module; 16 NVIDIA GB300 GPUs on four hosts with four workers each; FSDP16/EP16, PP1, sequence length 4096, GA1, **force balanced routing**, native MXFP8 Mega, FP32 GPU master weights, CPU BF16 Adam moments and compact gradients. Parameter gathering and gradient reduction use `every_microbatch`; **WGrad fusion is disabled**. Scratch factor 1, minimum activation save slots and eight loss chunks are shared. Seed is 14536, with deterministic algorithms disabled.

CPU activation and gradient columns are **selection quotas in binary GiB per rank**; **0 means disabled**. Adam moment offload is enabled in every row and is additional to those quotas. GPU peaks are the maximum recorded allocated/reserved values across all ranks and updates 1-17. Host RSS is the largest sampled sum of four workers on an actual host, rather than a transient peak. Every row passed all 16 ranks x 17 updates.

| Configuration / purpose | Local BS / GA | Global batch | Adam moments | CPU quota: activations / gradients GiB/rank | GPU allocated / reserved peak GiB | Host RSS GiB/host, sampled | Tokens/s/GPU | MFU (%) | Seconds/update | Job / profile / results |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full BS1: lowest CPU usage | 1 / 1 | 16 | BF16 on CPU | 0 / 0 | 250.96 / 256.60 | 662.01 | 1,012.70 | 5.91 | 4.0446 | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-16g-ep16-b1-fit16-full-b1-20261026-ivankobzarev) / [Profile](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-16g-ep16-b1-fit16-full-b1-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) / [Results](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-16g-ep16-b1-fit16-full-b1-20261026-ivankobzarev/v_0/attempt_0/benchmark/summary.json) |
| SAC + offload BS1: lowest GPU usage | 1 / 1 | 16 | BF16 on CPU | 64 / 16 | 235.32 / 241.21 | 776.38 | 973.12 | 5.68 | 4.2091 | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-16g-ep16-b1-fit16-ao-b1-20261026-ivankobzarev) / [Profile](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-16g-ep16-b1-fit16-ao-b1-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) / [Results](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-16g-ep16-b1-fit16-ao-b1-20261026-ivankobzarev/v_0/attempt_0/benchmark/summary.json) |
| Full BS4: gradient offload 16 | 4 / 1 | 64 | BF16 on CPU | 0 / 16 | 255.44 / 264.32 | 725.69 | 2,513.00 | 14.67 | 6.5197 | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-16g-ep16-b4-fit16-full-b4-g16-20261026-ivankobzarev) / [Profile](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-16g-ep16-b4-fit16-full-b4-g16-20261026-ivankobzarev/v_0/attempt_1/profiling/traces/iteration_16/rank0_trace.json.gz) / [Results](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-16g-ep16-b4-fit16-full-b4-g16-20261026-ivankobzarev/v_0/attempt_1/benchmark/summary.json) |
| Full BS6: gradient offload 32 | 6 / 1 | 96 | BF16 on CPU | 0 / 32 | 252.74 / 263.69 | 789.89 | 2,988.02 | 17.44 | 8.2248 | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-16g-ep16-b6-fit16-full-b6-g32-20261026-ivankobzarev) / [Profile](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-16g-ep16-b6-fit16-full-b6-g32-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) / [Results](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-16g-ep16-b6-fit16-full-b6-g32-20261026-ivankobzarev/v_0/attempt_0/benchmark/summary.json) |
| Full BS7: largest batch / highest throughput | 7 / 1 | 112 | BF16 on CPU | 0 / 38 | 253.79 / 265.84 | 810.20 | 3,129.90 | 18.27 | 9.1607 | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-16g-ep16-b7-fit16-full-b7-g38-20261026-ivankobzarev) / [Profile](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-16g-ep16-b7-fit16-full-b7-g38-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) / [Results](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-16g-ep16-b7-fit16-full-b7-g38-20261026-ivankobzarev/v_0/attempt_0/benchmark/summary.json) |

Profiles are original **rank 0, update 16** traces. Their exact remote paths were checked for existence, including **attempt 1 for BS4**; all other completed rows use version 0 / attempt 0. MFU is normalized to the same **5 PFLOP/s dense MXFP8 reference per GPU** as the [64-GPU report](https://github.com/pytorch/torchtitan/blob/gt-ptc-runs-candidate-1310/dsv3_64gpu_progression_report_20261027.md).

## 2. What "CPU BF16 Adam states" means

- **First moment (`exp_avg`, momentum): BF16, stored in pinned CPU memory.**
- **Second moment (`exp_avg_sq`, squared-gradient moving average): BF16, stored in pinned CPU memory.**
- **Master parameters: FP32, stored on GPU.** CPU moment offload does not offload the master parameters.
- **Gradient storage is configured independently.** Eligible terminal BF16 reduction outputs remain BF16 on GPU or pinned CPU. Gradients that require subsequent FP32 arithmetic keep FP32 storage. Gradient norms restore one parameter gradient to FP32 at a time, and Adam restores bounded FP32 gradient chunks before clipping and updating FP32 master weights.
- **Optimizer computation executes on GPU.** CPU offload describes where persistent moment buffers live. The native fused AdamW kernel updates chunks on GPU, then copies the updated moments back to CPU. This optimizer's `fused=True` setting is separate from the disabled **WGrad fusion** setting in the training graph.

The recipe explicitly sets `moment_dtype="bfloat16"`. BF16 moment storage changes rounding of Adam's recurrent state and is a numerical choice, even though master weights and gradient consumers are FP32. The [standard TorchTitan AdamW configuration](https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/optim/optimizer.py#L159) defaults to `moment_dtype="parameter"`; with FP32 training parameters, its moments are FP32. These runs therefore qualify the explicitly stated BF16-moment recipe. They do not establish equivalence to FP32-moment Adam training.

The compact-gradient pass removes an existing terminal BF16-to-FP32 expansion; it adds no new rounding operation at that storage boundary. This storage transformation and BF16 Adam moment precision should be evaluated separately.

## 3. Why BF16 moments are required on these hosts

The model contains approximately **682,636,472,320 parameters**, including MTP. Assuming approximately uniform ownership across 16 shards, Adam's two moment buffers require:

```text
BF16 moments per rank = parameters / 16 * 2 moments * 2 bytes = 158.94 GiB
FP32 moments per rank = parameters / 16 * 2 moments * 4 bytes = 317.88 GiB

Four workers per host:
BF16 moments =   635.75 GiB/host
FP32 moments = 1,271.51 GiB/host
```

The job receives **845 GiB of host memory per four-GPU host**. The recorded BS7 host has approximately **954.66 GiB physical RAM**. **FP32 moments alone exceed both the assignment and physical RAM.** BF16 moments are required for this hardware arrangement when both complete moment buffers remain resident in CPU RAM.

Replacing only the moment dtype while holding other BS7 allocations approximately constant adds **635.75 GiB/host**. Its measured **810.20 GiB** sampled host RSS would project to **1,445.95 GiB/host**, or about **1.41 TiB**, before additional headroom. Even Full BS1, with no activation or gradient offload, projects from **662.01** to **1,297.77 GiB/host**. These FP32 figures are storage calculations, **not measured FP32-state runs**. Larger RAM hosts could support the same 16-GPU geometry; a 2 TiB host would be a practical target for BS7.

GPU memory is a separate constraint. FP32 master weights require approximately **158.94 GiB/GPU**. A full resident FP32 gradient copy would add another **158.94 GiB**, giving **317.88 GiB/GPU** before activations or workspace, above the approximately **277.55 GiB NVML-reported capacity**. Compact gradient storage, recomputation and the selected gradient offload address this GPU constraint.

The offloaded optimizer caps combined GPU staging for its two moment buffers at **512 MiB per optimizer instance/device**. FP32 moments halve the number of elements per chunk while preserving that byte cap; they do not place the complete FP32 state on GPU. Moment transfer bytes would double. Similar GPU fit with larger host RAM is plausible but requires a real FP32-moment run to verify memory and performance.

**Interpretation:** the demonstrated 16-GPU fit combines BF16 CPU moments with GraphTrainer recomputation, compact gradients and, for the larger batches, gradient offload. BF16 CPU moments solve the host-memory requirement; the graph and gradient memory strategies solve the GPU-memory requirement.

## 4. What each configuration demonstrates

- **Full BS1: the smallest host-memory footprint among these successful runs.** Full recomputation regenerates eligible saved activations, with no CPU activation or gradient offload. Only Adam moments are offloaded to CPU. This is the simplest demonstrated configuration: **662.01 GiB sampled host RSS**, with **256.60 GiB GPU reservation**.
- **SAC + offload BS1: the smallest GPU-memory footprint.** Selective activation checkpointing (SAC) combines selective recomputation with CPU offload of chosen saved activations, under a **64 GiB/rank activation quota**, plus a **16 GiB/rank gradient quota**. Compared with Full BS1 at the same batch and source, reservation is **15.39 GiB lower**, measured throughput is **3.91% lower**, and sampled host RSS is **114.36 GiB higher**. Both activation policy and gradient placement change, so this comparison describes the combined memory option rather than isolating activation-offload overhead.
- **Full BS4: gradient offload supports a larger local batch.** It retains Full recomputation, disables CPU activation offload and uses a **16 GiB/rank gradient quota**. Global batch rises to **64**, with **2,513.00 tokens/s/GPU** and **264.32 GiB reservation**.
- **Full BS6: a larger gradient quota supports the next capacity point.** It uses **32 GiB/rank gradient offload**, no CPU activation offload, and global batch **96**. Throughput is **2,988.02 tokens/s/GPU**, with **263.69 GiB reservation** and **789.89 GiB sampled host RSS**.
- **Full BS7: the largest verified batch and highest measured throughput.** Increasing the gradient quota to **38 GiB/rank** supports global batch **112**. Relative to BS6 on the same source/package/runtime, throughput is **4.75% higher** and reservation **2.15 GiB higher**. Sampled host RSS is **810.20 GiB**, leaving **34.80 GiB below the assignment at the sampled points**; this is not a guaranteed transient-memory margin. Batch and offload quota both change.

The Full progression shows how choosing gradient placement supports local batches **1, 4, 6 and 7** while measured GPU reservation stays within **256.60-265.84 GiB**. It is a capability progression with different batches and quotas; there is no matched smaller-quota OOM control at each batch. WGrad fusion was held disabled throughout this family, so these results contain no 16-GPU WGrad-fusion ablation.

## 5. Numerical qualification and checkpoint startup

All real runs produced finite loss and gradient norms through 17 updates, with exact agreement of the recorded globally reduced scalars **within each job across all 16 ranks**. Seed 14536 with deterministic algorithms disabled does not establish exact numerical parity across configurations. The runs are short capacity/performance trials, rather than convergence studies.

[Six compact-gradient tests on the pinned runtime](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/fit16-analysis/pinned-compact-gradient-numerics/README.md) verify restored FP32 gradient norms and native Adam updates against the standard FP32-gradient-storage reference for ten updates, with the same moment precision. They cover GPU/CPU gradient retention and transfer/capture ordering. Their scope is the gradient storage transformation; they do not qualify full-model equivalence to FP32 Adam moments.

Every successful recipe uses the experimental **in-place HF checkpoint loader**. It loads the official checkpoint into existing native parameter views, avoiding approximately **154.88 GiB/rank of duplicate expert reconstruction** during startup. Earlier BS1/BS2 attempts failed before training during checkpoint reconstruction and are not training-capacity OOM controls. [Checkpoint loading qualification](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/fit16-analysis/README.md).

## 6. Measurement and reproduction

- **Protocol:** 17 optimizer updates; warmup 1-5; ten timed updates 6-15; compute trace at 16; memory snapshot at 17. Profiling and the measurement-alignment barrier are outside the throughput timers.
- **Throughput:** each update duration is the maximum across 16 ranks, then averaged over updates 6-15. Tokens/s/GPU = `local_BS * GA * 4096 / mean_update_seconds`.
- **MFU:** `100 * tokens_per_second_per_gpu * 291853541376 / 5e15`. This is model MFU against a dense MXFP8 reference; it excludes recomputation FLOPs and does not measure individual kernel utilization. The normalization and hardware reference match the 64-GPU report.
- **GPU memory:** maximum recorded PyTorch allocated/reserved peak across all 16 ranks and updates 1-17, including capture and profiling. Allocator reservation and NVML physical usage are separate counters. CUDA replay bookkeeping does not directly measure transient replay liveness.
- **CPU memory:** sampled worker RSS sums grouped by actual hostname at updates 0, 1, 2, 5 and 15. Samples are not globally synchronized or continuous transient-peak measurements. Selection quotas, pinned allocator allocation and RSS are distinct quantities.
- **Sources:** BS1 Full, BS1 SAC + offload and BS4 use `f06bdedd7a21de9d4cb8e6302363fe781a793318`, workspace `torchtitan_workspace:d73e6717e09a4255804b093033a5d972`. BS6/BS7 use `da189fccf9d5c4161296519f098a87dd6592aef9`, workspace `torchtitan_workspace:109f34f8141f4fc390ef789a235b3acb`, including collection at initial capture boundaries outside measured timers.
- **Runtime:** Torch `149ae98d98db6921d66e7ba527b395294a127b6e`. BS6/BS7 match source, package, runtime, policy and schedules while changing batch and gradient quota. BS4 uses completed attempt 1 after preemption of attempt 0.
- **Scope:** one completed hardware trial per configuration. The compact-gradient implementation currently requires **GA1 and PP1**. The 64/32-GPU report families use GA8 and different gradient storage; their throughput is not a controlled scaling comparison with these 16-GPU runs.

**Suggested paper statement:** "GraphTrainer trains DeepSeek V3 671B plus one MTP layer on 16 GB300 GPUs using FP32 master weights, CPU-offloaded BF16 Adam moments, full recomputation and compact gradient storage. CPU gradient offload of up to 38 GiB per rank enables the demonstrated BS7/GA1 configuration. BF16 moment precision is part of this result's experimental setup."

[All five configurations and original verification receipts](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/fit16-analysis/real16-results.json) / [fresh scheduler, benchmark and profile verification](/home/ivankobzarev/tmp/dsv3-16gpu-report-evidence-20261027/verification.json) / [matched BS6/BS7 configuration evidence](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/fit16-analysis/mast-monitor-full7-gc/matched-full6-full7-definition.json).
