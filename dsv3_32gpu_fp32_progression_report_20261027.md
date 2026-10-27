# DSv3 671B: GraphTrainer FP32 training state and memory progression on 32 x GB300

**Hardware: 32 x NVIDIA GB300 GPUs, eight hosts with four GPUs each.**

**All four hardware runs keep FP32 master weights and standard FP32 gradient storage on GPU, and both FP32 Adam moments on CPU.** Activation offload is the additional variable. The workload uses mixed MXFP8/BF16 computation; FP32 training state does not mean all operations compute in FP32.

Native MAST state rechecked at **2026-10-27 12:43 PDT**: three runs completed on all 32 ranks through all 17 optimizer updates; the BS10 control OOMed on all 32 ranks before completing an update. All use **v0/attempt0, with zero restarts**. The successful results were reconstructed from all 96 rank benchmark files, and all 96 update-16 profile files were verified.

Companion: [64 x GB300 progression report](https://github.com/pytorch/torchtitan/blob/dsv3-64gpu-startup-report-20261027/dsv3_64gpu_progression_report_20261027.md). Its BF16 Adam-moment results are a separate campaign.

## 1. Main progression table

**Fixed:** DSv3 671B + MTP1, FSDP32/EP32/PP1, TP1/CP1, sequence length 4096, GA8, full recomputation, forced balanced routing, native MXFP8 Mega, scratch factor 1, minimum activation slots, eight loss chunks, and WGrad fusion disabled. Parameter gathering and gradient reduction occur every microbatch. Seed 14536; deterministic algorithms disabled.

Global batch is `32 * local BS * GA8`: **2048 at BS8; 2560 at BS10**. GA is fixed; global batch changes with BS. Allocated/reserved peaks cover all 17 updates, including warmup and capture. Throughput and MFU use only ten unprofiled updates 6-15.

| ID | Policy | Local BS | Global batch | Gather / reduce during GA8 | Activation budget GiB/rank | Outcome | Allocated / reserved GiB | Tokens/s/GPU | MXFP8 MFU (%) | Seconds/update | MAST job | Profile 16, rank 0 |
| --- | --- | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| `full_recompute_bs8` | Full recompute | 8 | 2048 | Every microbatch / every microbatch | Disabled | PASS, all 32 x 17 | 254.82 / 258.39 | 4,582.42 | [26.75](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev/v_0/attempt_0/benchmark/summary.json) | 57.2065 | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev) | [Readable](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_98f6fecd-6967-48b5-96c5-3d2354834c2c_dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) / [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_3a5244fd-4ce5-49fa-932f-6ac6a59eef46_dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev__update16_rank0_original.json.gz) |
| `full_recompute_cpuoffload_bs8_ao24` | Full recompute + CPU activation offload | 8 | 2048 | Every microbatch / every microbatch | 24 | PASS, all 32 x 17 | 242.07 / 245.11 | 4,558.03 | [26.61](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev/v_0/attempt_0/benchmark/summary.json) | 57.5126 | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev) | [Readable](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b0aaf5d8-5379-4c41-aa5f-a4a8ee815bde_dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) / [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_ce54d94f-6a02-4061-9dce-7713476a0517_dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev__update16_rank0_original.json.gz) |
| `full_recompute_bs10` | Full recompute | 10 | 2560 | Every microbatch / every microbatch | Disabled | OOM, all 32 | ~263.08 / ~264.40 at failure; [~272.95 / ~280.82 projected](#bs10-oom-control-and-projections) | - | - | - | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-17u-20261027-ivankobzarev) | Not produced (OOM) |
| `full_recompute_cpuoffload_bs10_ao48` | Full recompute + CPU activation offload | 10 | 2560 | Every microbatch / every microbatch | 48 | PASS, all 32 x 17 | 253.26 / 257.96 | 4,673.54 | [27.28](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev/v_0/attempt_0/benchmark/summary.json) | 70.1138 | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev) | [Readable](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_1d18201a-9a03-4535-b59d-0a1e8cd28034_dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) / [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_2f415afc-18f9-43c6-ac8b-09a4180fe20a_dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev__update16_rank0_original.json.gz) |

The observed progression is **Full BS8 fits -> Full + activation offload BS8 uses less GPU memory -> Full BS10 OOM -> Full + activation offload BS10 fits**.

At BS8, the 24 GiB activation budget lowers allocated peak by **12.75 GiB** and reserved peak by **13.28 GiB**, with **0.53% lower throughput**. At BS10, a 48 GiB activation budget enables all 17 updates. Its reserved peak is **0.43 GiB below** Full BS8, while processing **25% more samples per update**, with **1.99% higher tokens/s/GPU** and 27.28% MFU. Update time increases from 57.21 to 70.11 seconds because the batch is larger.

These are observations from one hardware trial per configuration. The 0.53% throughput difference has no repeatability estimate.

### BS10 OOM control and projections

The control loaded the official checkpoint successfully and failed in update 1 while executing the **repeated gradient-accumulating forward/backward action**, after the initial non-accumulating microbatch. The rank-0 stack ends in `_fused_mla_k_rope_op` allocating its attention-key tensor. All 32 ranks report CUDA OOM, and no optimizer update or measured throughput was produced.

The OOM message reports **263.08 GiB currently allocated**, plus **1.32 GiB reserved but unallocated**, giving approximately **264.40 GiB reservation at failure**. Total process GPU use, including non-PyTorch memory, is **275.52 GiB** of **276.62 GiB** usable capacity. The failed allocation requests **1.88 GiB** with **1.09 GiB** free. These are values at the failed allocation, not recorded allocator high-water peaks or the memory needed for a complete update.

| BS10/GA8 policy | Activation budget GiB/rank | Fake graph + native arenas + optimizer staging GiB | Fake live CPU activations GiB/rank | Extrapolated reserved peak GiB | Hardware result |
| --- | ---: | ---: | ---: | ---: | --- |
| Full recompute | 0 | 272.95 | 0.00 | 280.82 | OOM, all 32 |
| Full recompute + activation offload | 24 | 259.90 | 13.05 | 267.05 | Local plan only; not submitted |
| Full recompute + activation offload | 48 | 247.95 | 25.00 | 255.10 | PASS; measured reserved peak 257.96 |

Fake projections include graph-owned GPU parameters/gradients, native DistMoE arenas and a conservative 1 GiB summed optimizer-staging cap. They exclude CUDA graph pool reservation, allocator padding/fragmentation, opaque CUDA/operator workspaces, real transport and checkpoint temporaries. Extrapolated reservation uses the corresponding BS8 measured reserved-minus-Fake margin: 7.87 GiB for Full and 7.15 GiB for Full + offload. These are planning estimates, not measured BS10 peaks. The actual 48 GiB result is 2.86 GiB above its reservation extrapolation.

## 2. CPU offload and GA strategy

| Setting | All four hardware runs |
| --- | --- |
| Model | DSv3 671B + one MTP layer, 61 decoder layers |
| Parallelism | FSDP32 / EP32 / PP1; TP1 / CP1 |
| Local BS / gradient accumulation | BS8 or BS10 / GA8 |
| Sequence length / global batch | 4096 / 2048 or 2560 |
| Master weights | FP32, GPU |
| Standard persistent gradient storage | FP32, GPU |
| Adam first and second moments | FP32, pinned CPU memory |
| CPU activation offload | Disabled, 24 GiB or 48 GiB selection budget per rank, as shown in each row |
| CPU gradient offload | Disabled |
| GA parameter strategy | All-gather every microbatch; gathered representations are released between microbatches |
| GA gradient reduction | Every microbatch |
| WGrad accumulation fusion | Disabled |
| Routing | Force balanced |
| Native runtime | MXFP8 Mega, scratch capacity factor 1, minimum activation slots |
| Loss chunks | 8 |
| Activation offload prefetch / defer | 1 / 1 |
| Optimizer staging | 512 MiB per optimizer instance; two instances, at most 1 GiB total |
| CUDA graphs | Forward/backward enabled; optimizer capture disabled |
| Seed / deterministic algorithms | 14536 / disabled; warn-only disabled |

**Optimizer offload:** both FP32 Adam moments occupy **158.94 GiB/rank**, or **635.75 GiB per four-GPU host**. `CPUOffloadedAdamW` stores moments in pinned CPU memory, streams bounded chunks through GPU staging buffers, runs native fused AdamW on GPU, and copies updated moments back to CPU. The optimizer computation remains on GPU. Both optimizers use `moment_dtype="parameter"`; persistent parameters are FP32.

FP32 master weights, standard gradients and both FP32 moments would require approximately **317.88 GiB/rank** if simultaneously on GPU, before activations and native arenas. That exceeds the 276.62 GiB usable capacity. Every hardware result in this report already uses CPU optimizer-state storage.

**Activation offload:** the 24 GiB BS8 plan keeps approximately **12.56 GiB** of activations live on CPU and has **33 offload/reload calls per microbatch**; the 48 GiB BS10 plan has **25.00 GiB** and **50 calls**. The selected tensors include saved BF16 layer states and native MoE saved activations. The configured budget is an activation-selection quota; it is not the bytes physically live at once, the total transfer volume, or a process RSS/pinned-memory cap.

### Observed GPU and host memory

| ID | Measured-window allocated / reserved GiB | Maximum pinned allocator allocation peak GiB/process | Maximum sampled RSS GiB/process | Minimum sampled host MemAvailable GiB | Maximum sampled NVML device use GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_recompute_bs8` | 196.24 / 256.10 | 159.93 | 168.92 | 174.41 | 267.05 |
| `full_recompute_cpuoffload_bs8_ao24` | 196.24 / 243.34 | 172.80 | 181.82 | 122.45 | 254.30 |
| `full_recompute_cpuoffload_bs10_ao48` | 205.37 / 255.51 | 185.18 | 194.19 | 73.34 | 268.25 |

Measured-window values cover updates 6-15; all-update capture peaks are in the main table. Host and NVML values use telemetry from **all 32 ranks and all eight hosts**. RSS, MemAvailable and NVML are point samples, so they do not establish the worst transient value. Pinned allocator peaks are cumulative counters at the sampled points. Host RAM is approximately **954.66 GiB per host**. Do not sum independent per-process maxima and label the result a simultaneous host measurement.

## 3. Additional runs for a complete progression

### Priority 1: two BS8 controls

Add **No AC BS8** and **SAC BS8**, retaining the exact FP32 optimizer placement, GA8, FSDP32/EP32 and every-microbatch gather/reduce settings. These complete the same-batch recomputation progression: **No AC OOM -> SAC OOM -> Full recompute fits -> Full + activation offload uses less memory**. The existing matched BS10 pair already demonstrates activation offload enabling the larger batch.

| Proposed ID | Policy | BS / GA | Global batch | Activation budget GiB/rank | Fake GPU GiB | Fake live CPU activation GiB/rank | Expected hardware outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `noac_bs8` | No AC | 8 / 8 | 2048 | 0 | 800.63 | 0.00 | OOM expected; not hardware tested |
| `sac_bs8` | SAC | 8 / 8 | 2048 | 0 | 312.44 | 0.00 | OOM expected; not hardware tested |

Both full-model FakePG/FakeTensor plans completed for both GA8 actions. Their combined forecasts are already above device capacity before unmodeled overhead. They allocate no real model tensors, load no checkpoint and measure no throughput. The OOM expectations require hardware runs before they become paper results.

### Priority 2: one additional budget point

A single **Full recompute BS8 + 16 GiB activation budget** run gives a same-batch three-point sweep with the existing budgets 0 and 24. It changes only activation policy/budget, keeping BS8/GA8 and global batch 2048 fixed.

| ID | Activation budget GiB/rank | Fake GPU GiB | Fake live CPU activation GiB/rank | Offload / reload calls per microbatch | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `full_recompute_bs8` | 0 | 250.51 | 0.00 | 0 / 0 | Hardware PASS, reused |
| `full_recompute_cpuoffload_bs8_ao16` | 16 | 242.01 | 8.50 | 22 / 22 | Local plan only; expected to fit |
| `full_recompute_cpuoffload_bs8_ao24` | 24 | 237.95 | 12.56 | 33 / 33 | Hardware PASS, reused |

### Optional: a matched SAC + offload example at BS6

For a direct SAC offload comparison, use **BS6/GA8, global batch 1536**, with SAC budgets **0 and 32 GiB**. This keeps the workload and all training-state precisions matched within the pair. Full BS8 and Full + offload BS10 remain the larger-batch choices.

| Proposed ID | Policy | BS / GA | Global batch | Activation budget GiB/rank | Fake GPU GiB | Fake live CPU activation GiB/rank | Purpose |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sac_bs6` | SAC | 6 / 8 | 1536 | 0 | 274.96 | 0.00 | Capacity control; hardware outcome pending |
| `sac_cpuoffload_bs6_ao32` | SAC + CPU activation offload | 6 / 8 | 1536 | 32 | 242.97 | 31.99 | Fit with selective recomputation and CPU memory; hardware outcome pending |

Both BS6 plans completed for both GA8 actions. SAC without activation offload projects **274.96 GiB GPU**, leaving only 1.66 GiB before real runtime overhead; OOM is expected. Adding a 32 GiB budget projects **242.97 GiB GPU** and **31.99 GiB live CPU activations/rank**, with 48 offload/reload calls per microbatch. Applying the measured 20.31 GiB device-over-Fake margin from BS10 Full + offload gives approximately **263.27 GiB device use**, or **13.35 GiB headroom**. The host estimate is **174.41 - 4 * 31.99 = 46.44 GiB** remaining MemAvailable. These are extrapolations; hardware must verify both GPU and host fit. This two-run pair would show SAC + CPU offload fitting a batch that SAC alone is expected to reject.

A BS8 SAC + 48 GiB plan was also evaluated: **264.44 GiB GPU** and **48.00 GiB live CPU activations/rank**. Adding the observed BS10 Full + offload device-over-Fake margin of 20.31 GiB gives approximately **284.75 GiB device use**. Using the Full BS8 minimum sampled host availability, 48 GiB more activations per rank projects **174.41 - 4 * 48 = -17.59 GiB** available per host, before any new padding. Increasing this budget would add more host pressure. These extrapolations motivate the smaller BS6 SAC pair; the BS8 SAC offload plan was not submitted.

**No additional MAST jobs were submitted while preparing this report.** The proposed rows are completed local plans and an experiment design. The minimum extension is two BS8 controls; the one-run budget addition and two-run SAC pair answer separate questions. Reuse the same 17-update protocol and source/runtime for each addition.

## 4. Numerical scope

Every successful run has **544 finite rank/update loss and gradient-norm readings**, with global scalars consistent across all ranks at stdout precision. The runs are short performance/capacity experiments with forced balanced routing; they do not establish long-run convergence under imbalanced routing.

| ID | Initial loss / grad norm | Final loss / grad norm |
| --- | ---: | ---: |
| `full_recompute_bs8` | 8.41613 / 47.046 | 6.52888 / 0.9162 |
| `full_recompute_cpuoffload_bs8_ao24` | 8.41613 / 46.95 | 6.55579 / 2.45 |
| `full_recompute_cpuoffload_bs10_ao48` | 8.40850 / 46.565 | 6.55999 / 1.4734 |

The matched BS8 Full and Full + offload runs have different scalar histories with deterministic algorithms disabled. This report establishes observed fit, memory and throughput; it does not claim bitwise numerical equivalence of activation offload. To support that claim, rerun the matched BS8 pair with **seed 42, deterministic=True and deterministic_warn_only=False**, then compare full-precision scalar sidecars and final parameter/gradient values. The BS10 run has a different global batch, so its scalar history is not an equivalence comparison to BS8.

## 5. Measurement protocol and MFU

| Updates | Role |
| --- | --- |
| 1-5 | Warmup and CUDA graph preparation/capture |
| 6-15 | Ten measured, unprofiled optimizer updates |
| 16 | Compute profiler trace on every rank |
| 17 | CUDA allocator memory snapshot |

The first measured interval follows device synchronization and a world barrier after step-5 telemetry, outside the timers. Each measured update includes forward/backward computation, GA8 communication, activation transfers and the CPU-moment Adam update. For each update, take the slowest rank's elapsed time; then average the ten updates. The all-rank reconstruction matches the published benchmark summaries.

Tokens/update are **8,388,608 at BS8** and **10,485,760 at BS10**. `tokens/s/GPU = tokens_per_update / (32 * mean_update_seconds)`.

MFU uses the archived model estimate **291,853,541,376 FLOPs/token** and a **5 PFLOP/s dense MXFP8 reference per GB300**: `100 * tokens_per_second_per_gpu * model_flops_per_token / 5e15`. Model FLOPs exclude recomputation. The workload mixes MXFP8, BF16 and FP32 operations, so this is model-FLOP utilization normalized to that reference. Each MFU cell links to its benchmark summary. [NVIDIA Blackwell Ultra datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf), page 5: FP8/FP6 10 PFLOP/s sparse, with dense at half sparse. Reference PDF SHA256: `ccdffc064731cc160016f4c7a77ecfa37983038bcaf60e39190ba4175b39c7c8`.

Graph preparation was not separately instrumented in these 32 GPU jobs. Warmup update time includes execution/capture and must not be labeled graph-preparation time or precompile savings. The 64 GPU startup/precompile measurements use their own timing boundaries.

## 6. Reproduction and artifact index

Compute source: `f66c0e6cd7aa3d5356d918e7b8d6a2a6f1fd1c65`. PyTorch source: `149ae98d98db6921d66e7ba527b395294a127b6e`; Torch `2.15.0a0+git149ae98`, CUDA 13.0, NCCL 2.30.7. The package preserves in-place official HF checkpoint loading, capture cleanup and the engine pinned pool from the qualified source.

Workspace: `torchtitan_workspace:f426b59d31f74b749092f6d9ace965ad`. Runtime: `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266`. All four hardware jobs use identical source/runtime packages. Both optimizer groups and all requested native job-definition fields were read back and verified. The constructed BS8 and BS10 matched pairs differ only at `/compile/memory_policy` (`full` -> `full_and_offload`) and `/compile/cpu_offload_budget_gb` (0 -> 24 or 48).

Initialization: official DSv3 BF16 HF checkpoint at `/mnt/mffuse/mlperf_dsv3_convergence/deepseekv3_checkpoint_bf16_hf_assets`; 163 shards and 46,183 keys verified by worker preflight. Workload uses the frozen C4 ingestion `airstore://590827`, seed 14536, sequence length 4096 and MTP1.

| ID | Benchmark / scalar artifacts | Original profile directory | Memory snapshots |
| --- | --- | --- | --- |
| `full_recompute_bs8` | [Summary](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev/v_0/attempt_0/benchmark/summary.json) / [All rank benchmarks](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev/v_0/attempt_0/benchmark) / [Scalar sidecars](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev/v_0/attempt_0/benchmark/scalars) | [All 32 profiles](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16) | [Update 17](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev/v_0/attempt_0/profiling/memory_snapshot/step_000000000017) |
| `full_recompute_cpuoffload_bs8_ao24` | [Summary](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev/v_0/attempt_0/benchmark/summary.json) / [All rank benchmarks](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev/v_0/attempt_0/benchmark) / [Scalar sidecars](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev/v_0/attempt_0/benchmark/scalars) | [All 32 profiles](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16) | [Update 17](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev/v_0/attempt_0/profiling/memory_snapshot/step_000000000017) |
| `full_recompute_bs10` | No completed update; [MAST error logs](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-17u-20261027-ivankobzarev) | Not produced | Not produced |
| `full_recompute_cpuoffload_bs10_ao48` | [Summary](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev/v_0/attempt_0/benchmark/summary.json) / [All rank benchmarks](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev/v_0/attempt_0/benchmark) / [Scalar sidecars](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev/v_0/attempt_0/benchmark/scalars) | [All 32 profiles](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16) | [Update 17](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev/v_0/attempt_0/profiling/memory_snapshot/step_000000000017) |

Main-table profile links open Perfetto directly. Readable copies compact CUDA replay streams into semantic lanes; original copies preserve measurement placement. Compaction validates that kernel names, timestamps, durations and arguments remain unchanged. All six shared profile copies were downloaded and SHA256-verified against their source files, with a 365-day TTL from publication on 2026-10-27. The original job profile paths are listed above.

Local evidence: `/home/ivankobzarev/tmp/dsv3-paper-32gpu-fp32-sweep-20261027/report-evidence/` contains native status/log receipts, all 96 rank benchmark files, profile listings, reconstructed measurements, scalar validation, and shared-profile checksums. Local Fake plans, resolved configs, launch manifests and readback receipts are in the same campaign directory. These paths are an evidence archive on the experiment host; the linked MAST/Manifold artifacts provide the remote records.
