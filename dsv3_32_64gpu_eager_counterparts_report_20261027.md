# DSv3 671B: eager counterparts of the 32 and 64 GPU progressions

**Hardware: 64 x NVIDIA GB300 or 32 x NVIDIA GB300 GPUs.**

Updated: 2026-10-28T12:09:45.436508+00:00. Autonomous experiment window ends 2026-10-28T12:56:10+00:00.

This report pairs the existing graph trainer presentation configurations with native TorchTitan eager execution. Results come from real MAST jobs. Prepared or queued runs have no measured throughput or memory result.

**Eager definition:** native `Trainer` / `TrainingEngine`, native FSDP, regular autograd and native chunked-loss backward, with CUDA graphs disabled and `local_compile_regions=[]`. Handwritten MXFP8, attention and router kernels retain their original implementations. A native CUDA graph control, when included, is labeled separately.

## Current results

- Requested eager counterparts: 8 passed, 12 CUDA OOM, 0 remaining. Supplemental runs are listed separately.
- Main 64-GPU Full recompute BS6 pair: graph trainer has 4.9% higher throughput and uses 31.58 GiB less reserved memory. Native SAC BS6 and Full recompute BS10 run out of memory; their graph references pass.
- All four requested 32-GPU native configurations run out of memory with FP32 GPU masters/gradients and CPU FP32 Adam moments. Three of their graph references pass; Full recompute BS10 without activation offload also fails in graph trainer.
- Native 64-GPU Full recompute BS10 is rescued by CPU activation offload: 208.44 GiB reserved, 4,270 tokens/s/GPU, 24.92% MFU, and 79.98 GiB actual CPU activation storage per rank.
- Both native 32-GPU BS6/GA8 baselines pass with GPU FP32 masters/gradients and CPU FP32 Adam moments. Their matching original-payload graph BS6 runs are included in the matched table.
- Matched 32-GPU Full recompute BS6 pair: graph trainer has 4.7% higher throughput and uses 35.74 GiB less reserved memory; both keep FP32 CPU Adam moments.
- Matched 32-GPU BS8/AO40 pair passes in both engines: graph trainer has 12.0% higher throughput and uses 23.02 GiB less reserved memory. The native retry resolves a host OOM by reducing profiling to rank 0 without shape recording; the measured training settings are unchanged.
- Strict native Full versus Full + offload and fused Every versus First gathering match all 17 recorded losses and gradient norms bitwise on all 64 ranks. Cross-engine scalar histories differ; see the numerical comparison section.
- Native WGrad fusion changes throughput by 0.32% in the single non-deterministic trials. This small difference does not establish a meaningful speed gain; fused and unfused scalar histories also differ.

## Shared configuration

| Setting | 64 GPUs | 32 GPUs |
| --- | --- | --- |
| Model | DSv3 671B + MTP1 | DSv3 671B + MTP1 |
| Parallelism | FSDP64 / EP64 / PP1 | FSDP32 / EP32 / PP1 |
| Sequence length / GA | 4096 / 8 | 4096 / 8 |
| Global batch | GPU count x local BS x 8 | GPU count x local BS x 8 |
| Master weights / persistent gradients | FP32 on GPU | FP32 on GPU |
| FSDP parameter / reduction dtype | BF16 / BF16 | BF16 / BF16 |
| Adam first and second moments | BF16 on GPU | FP32 on CPU |
| CPU optimizer staging cap | Not applicable | 512 MiB per optimizer instance; two instances |
| MAST RAM allocation per four-GPU task | 845 GiB | 845 GiB |
| Parameter gathering / gradient reduction | Every microbatch in main table | Every microbatch in main table |
| WGrad fusion / routing | Disabled / force balanced | Disabled / force balanced |
| Native expert standalone WGrad | BF16, then standard FP32 persistent gradient storage | BF16, then standard FP32 persistent gradient storage |
| Scratch / activation slots / loss chunks | Factor 1 / minimum / 8 | Factor 1 / minimum / 8 |
| Seed / deterministic algorithms | 14536 / disabled in main table | 14536 / disabled in main table |
| Updates | Warmup 1-5; measure 6-15; profile 16; memory snapshot 17 | Same |

## Policy mapping and comparison limits

The model, input stream, batch geometry, optimizer settings, checkpoint, kernels and measurement configuration are copied from the pinned graph reference and checked field by field. Backend changes are explicit: native FSDP/autograd replaces SimpleFSDP/transformed graphs, and native region checkpointing replaces graph trainer operator memory planning.

| Graph configuration | Native eager counterpart |
| --- | --- |
| No AC | No activation checkpoint wrapper |
| SAC | `torch_remat` saves all declared regions except `*routed_experts.w13.*`; code outside saved regions recomputes |
| Full recompute | `torch_remat` recomputes policy-controlled regions; model regions declared `recompute=False`, including Dist-MoE, remain retained |
| SAC or Full + CPU activation offload | Same native remat policy with budgeted pinned CPU copies through saved-tensor hooks |

These SAC/Full labels describe the native implementations; they do not imply identical saved-tensor sets or recompute communication counts. The adapter applies the policy to the MTP transformer block as well as the main decoder blocks.

Chunked LM-head communication also differs. The native default coalesces head gradients across loss chunks; the graph path reduces individual head contributions. The original GraphTrainer last-reduction profiles show 16 ReduceScatter kernel slices in each early microbatch and 80 in the last: the body reduction is deferred while head reductions remain earlier. The qualified native scheduling controls defer the head reduction to finalization as well. Use the within-engine Every/First pairs for gather-cadence comparisons; cross-engine schedule labels do not imply identical head communication.

CPU activation offload quotas are binary GiB per rank and cap live selected activation copies. Parameter and prepared-weight storage is excluded. Copies and restores use the current CUDA stream; graph trainer has its own prefetch/defer scheduling. The native adapter records actual selected bytes and skips additional copies if host available RAM falls below 8 GiB. This guard is needed because the measured 32 GPU FP32-moment baseline leaves less than 4 x 48 GiB available per host. A configured quota is not a guarantee that the full quota is used.

MFU uses the same model-FLOP estimate, excluding recompute, and 5 PFLOP/s dense MXFP8 reference per GB300. Each run has ten synchronized measured updates and one hardware trial; there are no uncertainty intervals. Local qualification is separate from the 671B memory and speed measurements.

## 64 GPU progression

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `noac_bs6` | 6 / 8 | Disabled | OOM | ~598.63 projected / - | - | - | OOM; 64 ranks logged | 267.50 / 268.17 at failure; +1.12 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-noac-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-noac-ga8-17u-20261027-ivankobzarev) | OOM before profiling |
| `sac_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 236.47 / 244.55 | 4,948.86 | 28.89 | OOM; 64 ranks logged | 267.62 / 268.69 at failure; +0.33 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-sac-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b368a8dd-9f0e-428a-bdf2-f80b28c840f0_dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 189.59 / 193.63 | 4,760.57 | 27.79 | PASS, all 64 x 17 | 219.08 / 225.21 | 4,536.09 | 26.48 | 1.049x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6548b64d-cb41-44a5-8160-34d958e4bb50_dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e49e9518-7a8d-4cc4-947a-c457d96ee662_rank0_trace.json.gz) |
| `sac_bs10` | 10 / 8 | Disabled | OOM | ~309.17 projected / - | - | - | OOM; 64 ranks logged | 263.01 / 263.65 at failure; +1.88 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b10-ga8-sac-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ga8-17u-20261027-r1-ivankobzarev) | OOM before profiling |
| `sac_cpuoffload_bs10_ao60` | 10 / 8 | 60 | PASS, all 64 x 17 (reference) | 252.94 / 258.08 | 4,778.03 | 27.89 | OOM; 64 ranks logged | 264.56 / 265.29 at failure; +1.88 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-paper-64g-ep64-b10-ga8-sac-ao60-17u-20261027-r1-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ao60-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-paper-64g-ep64-b10-ga8-sac-ao60-17u-20261027-r1-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `sac_cpuoffload_bs10_ao80` | 10 / 8 | 80 | PASS, all 64 x 17 (reference) | 234.58 / 239.19 | 4,708.92 | 27.49 | OOM; 64 ranks logged | 264.36 / 265.21 at failure; +2.50 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-paper-64g-ep64-b10-ga8-sac-ao80-17u-20261027-r1-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ao80-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-paper-64g-ep64-b10-ga8-sac-ao80-17u-20261027-r1-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `sac_cpuoffload_bs10` | 10 / 8 | 100 | PASS, all 64 x 17 (reference) | 224.45 / 228.70 | 4,672.37 | 27.27 | OOM; 64 ranks logged | 264.84 / 265.39 at failure; +0.55 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b10-ga8-sac-ao-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ao100-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6156cd14-86f0-4e88-bfd0-d8b2bd281d93_dsv3-nebius-64g-ep64-b10-ga8-sac-ao-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_bs10` | 10 / 8 | Disabled | PASS, all 64 x 17 (reference) | 235.96 / 240.56 | 4,826.62 | 28.17 | OOM; 64 ranks logged | 263.96 / 265.11 at failure; +1.88 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-full-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_d9c75545-1623-45fd-bc08-9cfbeca9cb55_dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |

## 32 GPU progression

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `full_recompute_bs8` | 8 / 8 | Disabled | PASS, all 32 x 17 (reference) | 254.82 / 258.39 | 4,582.42 | 26.75 | OOM; 32 ranks logged | 265.10 / 266.31 at failure; +1.00 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b8-full-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_98f6fecd-6967-48b5-96c5-3d2354834c2c_dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_cpuoffload_bs8_ao24` | 8 / 8 | 24 | PASS, all 32 x 17 (reference) | 242.07 / 245.11 | 4,558.03 | 26.61 | OOM; 32 ranks logged | 263.80 / 264.63 at failure; +3.02 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b8-full-ao24-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b0aaf5d8-5379-4c41-aa5f-a4a8ee815bde_dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_bs10` | 10 / 8 | Disabled | OOM | ~272.95 projected / - | - | - | OOM; 32 ranks logged | 264.54 / 265.44 at failure; +0.62 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b10-full-ga8-17u-20261027-r1-ivankobzarev) | OOM before profiling |
| `full_recompute_cpuoffload_bs10_ao48` | 10 / 8 | 48 | PASS, all 32 x 17 (reference) | 253.26 / 257.96 | 4,673.54 | 27.28 | OOM; 32 ranks logged | 262.34 / 263.03 at failure; +2.47 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b10-full-ao48-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_1d18201a-9a03-4535-b59d-0a1e8cd28034_dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |

## 64 GPU additional eager counterparts

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `native_eager_sac_bs6_det42` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 236.48 / 244.53 | 4,839.45 | 28.25 | OOM; 64 ranks logged | 267.49 / 268.71 at failure; +1.12 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-sac-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-sac-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-sac-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_det42` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 189.60 / 193.61 | 4,661.34 | 27.21 | PASS, all 64 x 17 | 219.10 / 225.21 | 4,421.89 | 25.81 | 1.054x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-full-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-full-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_f2de370f-8cde-41ad-a32f-684fc6e1034c_rank0_trace.json.gz) |
| `native_eager_full_recompute_cpuoffload_bs6_det42` | 6 / 8 | 80 | PASS, all 64 x 17 (reference) | 163.82 / 167.91 | 4,587.43 | 26.78 | PASS, all 64 x 17 | 172.95 / 177.81 | 4,077.67 | 23.80 | 1.125x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-fullao-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-fullao80-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-fullao-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_cb145ed0-d3ce-4c2e-b627-a13caf5827ba_rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_gather_every` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 213.35 / 221.66 | 4,726.74 | 27.59 | PASS, all 64 x 17 | 229.96 / 269.01 | 3,498.89 | 20.42 | 1.351x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-gather-every-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ge-f-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-gather-every-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_89da90f1-c852-48a4-b627-30f0084ad9cd_rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_gather_first` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 240.74 / 248.53 | 4,827.56 | 28.18 | PASS, all 64 x 17 | 254.39 / 269.03 | 3,624.44 | 21.16 | 1.332x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-gather-first-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-gf-f-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-gather-first-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_47160686-9f00-4516-9052-220d174f2f70_rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_reduce_last_unfused` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 213.33 / 223.84 | 4,822.62 | 28.15 | PASS, all 64 x 17 | 229.95 / 268.97 | 3,557.85 | 20.77 | 1.355x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-full-last-u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ge-u-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-full-last-u-20261026-ivankobzarev/v_0/attempt_1/profiling/traces/iteration_16/rank0_trace.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6c9cda47-fe23-4ac0-9749-60514193e813_rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_reduce_last_wgrad_fused` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 213.33 / 221.68 | 4,833.03 | 28.21 | PASS, all 64 x 17 | 229.95 / 268.99 | 3,569.11 | 20.83 | 1.354x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-full-last-f-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ge-f-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-full-last-f-20261026-ivankobzarev/v_0/attempt_1/profiling/traces/iteration_16/rank0_trace.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_bd35b9ea-4855-4818-84f7-c7e0796edea1_rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_first_gather_legacy_fused` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 240.80 / 250.17 | 4,798.39 | 28.01 | PASS, all 64 x 17 | 254.38 / 268.95 | 3,686.18 | 21.52 | 1.302x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-full-first-f-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-gf-f-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-full-first-f-20261026-ivankobzarev/v_0/attempt_1/profiling/traces/iteration_16/rank0_trace.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_1a01ea5f-16ac-4e7e-9078-33e3336c5817_rank0_trace.json.gz) |

## 64 GPU native CUDA graph controls

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `native_cuda_graphs_64g_sac_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 236.47 / 244.55 | 4,948.86 | 28.89 | OOM; 64 ranks logged | 267.62 / 268.71 at failure; +0.33 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-64g-b6-sac-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b368a8dd-9f0e-428a-bdf2-f80b28c840f0_dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `native_cuda_graphs_64g_full_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 189.59 / 193.63 | 4,760.57 | 27.79 | PASS, all 64 x 17 | 213.24 / 218.24 | 4,559.22 | 26.61 | 1.044x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-64g-b6-full-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6548b64d-cb41-44a5-8160-34d958e4bb50_dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) / [Native CG](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_ca84b859-52e6-4656-a7f6-dd0827a87917_rank0_trace.json.gz) |
| `native_cuda_graphs_64g_full_bs10` | 10 / 8 | Disabled | PASS, all 64 x 17 (reference) | 235.96 / 240.56 | 4,826.62 | 28.17 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-64g-b10-full-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_d9c75545-1623-45fd-bc08-9cfbeca9cb55_dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |

## 32 GPU native CUDA graph controls

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `native_cuda_graphs_32g_full_bs8` | 8 / 8 | Disabled | PASS, all 32 x 17 (reference) | 254.82 / 258.39 | 4,582.42 | 26.75 | OOM; 32 ranks logged | 266.10 / 267.23 at failure; +0.50 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-32g-b8-full-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_98f6fecd-6967-48b5-96c5-3d2354834c2c_dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |
| `native_cuda_graphs_32g_full_bs10` | 10 / 8 | Disabled | OOM | ~272.95 projected / - | - | - | OOM; 32 ranks logged | 263.29 / 264.27 at failure; +1.25 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-17u-20261027-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-32g-b10-full-ga8-17u-20261027-ivankobzarev) | OOM before profiling |
| `native_cuda_graphs_32g_full_bs6` | 6 / 8 | Disabled | Held for experiment priority | - | - | - | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-b6-full-ga8-17u-20261028-repeat-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-32g-b6-full-ga8-17u-20261028-bs6-ivankobzarev) | Not yet produced |

## Additional native scheduling settings

All rows below use Full recomputation, BS6/GA8 and 64 GPUs. The native head policy preserves the requested gathering and reduction across the entire update. The head communication difference from the graph references is described above.

| Native ID | Parameter gathering | Gradient reduction | WGrad fusion | Seed / strict deterministic |
| --- | --- | --- | --- | --- |
| `native_eager_full_recompute_bs6_gather_every` | Every microbatch | At finalization after GA8 | enabled | 42 / True |
| `native_eager_full_recompute_bs6_gather_first` | Once at the first microbatch; retain across GA8 | At finalization after GA8 | enabled | 42 / True |
| `native_eager_full_recompute_bs6_reduce_last_unfused` | Every microbatch | At finalization after GA8 | disabled | 14536 / False |
| `native_eager_full_recompute_bs6_reduce_last_wgrad_fused` | Every microbatch | At finalization after GA8 | enabled | 14536 / False |
| `native_eager_full_recompute_bs6_first_gather_legacy_fused` | Once at the first microbatch; retain across GA8 | At finalization after GA8 | enabled | 14536 / False |

## Matched BS6 32 GPU comparison

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `native_fit_full_recompute_bs6` | 6 / 8 | Disabled | PASS, all 32 x 17 (reference) | 231.15 / 233.09 | 4,541.68 | 26.51 | PASS, all 32 x 17 | 259.95 / 268.83 | 4,336.73 | 25.31 | 1.047x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-b6-full-ga8-17u-20261028-match1-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b6-full-ga8-17u-20261027-fit1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_54cdf4b4-4ae6-424b-a407-fa4245913f2b_rank0_trace.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b6ba5d08-ec4c-4089-979b-e2eae1702912_rank0_trace.json.gz) |
| `native_fit_full_recompute_cpuoffload_bs6_ao24` | 6 / 8 | 24 | PASS, all 32 x 17 (reference) | 218.68 / 221.06 | 4,515.63 | 26.36 | PASS, all 32 x 17 | 235.95 / 243.87 | 4,155.38 | 24.26 | 1.087x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-b6-full-ao24-ga8-17u-20261028-match1-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b6-full-ao24-ga8-17u-20261027-fit1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_603d9bab-28aa-4886-8d71-5580a362b25e_rank0_trace.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_fa3658b0-3638-4db2-baac-50c430103064_rank0_trace.json.gz) |

Both engines use the same BS6/GA8 batch geometry, FP32 GPU masters and gradients, FP32 CPU Adam moments, and the stated activation-offload quota. New graph runs clone the original 32-GPU definitions and immutable payload, changing only the batch from 8 to 6 and the job identity. Ratios are filled only after all-rank measurement verification.

## Matched BS8 32 GPU comparison with a 40 GiB activation quota

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `native_full_recompute_cpuoffload_bs8_ao40_profile0` | 8 / 8 | 40 | PASS, all 32 x 17 (reference) | 234.01 / 236.47 | 4,558.31 | 26.61 | PASS, all 32 x 17 | 251.32 / 259.49 | 4,071.72 | 23.77 | 1.120x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-b8-full-ao40-ga8-17u-20261028-match1-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b8-full-ao40-prof0-ga8-17u-20261028-r3-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_09e640ff-877d-4e12-b7a4-5f91fa59969e_rank0_trace.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_4540b13b-1016-4488-8daf-6a3cf089bc76_rank0_trace.json.gz) |

Both engines use BS8/GA8, GPU FP32 master weights and gradients, CPU FP32 Adam moments, and a 40 GiB/rank activation quota. The graph run clones the original BS8/AO24 definition and payload, changing only the activation quota and job identity. The native retry profiles rank 0 without shape recording to stay within task RAM; the graph run retains the original all-rank profiling. These differences occur after the measured updates 6-15 and are declared when comparing full-run memory peaks.

Within graph trainer, increasing the BS8 activation quota from 24 to 40 GiB lowers reserved peak memory by 8.63 GiB, from 245.11 to 236.47 GiB. Both runs achieve approximately 4,558 tokens/s/GPU in these single trials. Native eager fails with the 24 GiB quota and completes training with 40 GiB and the declared profiling adjustment.

### CUDA memory before profiling

Maximum recorded allocated/reserved CUDA peaks across every rank through update 15. This excludes the trace and memory snapshot updates, providing the same observation window even when profile coverage differs. Main tables retain the established full 17-update memory scope.

| Matched native configuration | Graph allocated / reserved GiB | Native allocated / reserved GiB |
| --- | ---: | ---: |
| `native_fit_full_recompute_bs6` | 231.15 / 233.09 | 259.95 / 268.83 |
| `native_fit_full_recompute_cpuoffload_bs6_ao24` | 218.68 / 221.06 | 235.95 / 243.87 |
| `native_full_recompute_cpuoffload_bs8_ao40_profile0` | 234.01 / 236.47 | 251.32 / 259.49 |

## Smaller-batch native 32 GPU baselines

These supplement the matched presentation tables with smaller-batch native training baselines. They retain FP32 master weights and gradients on GPU, FP32 Adam moments on CPU, and GA8. Their batch geometry differs from the presentation runs, so matched throughput ratios are omitted.

| ID | BS / GA | Global batch | CPU activation quota GiB/rank | Task RAM GiB/4 GPUs | Native outcome | Allocated / reserved peak GiB | Tokens/s/GPU | MFU % | MAST job | Profile, update 16 rank 0 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `native_boundary_full_recompute_bs7` | 7 / 8 | 1792 | Disabled | 845 | OOM; 32 ranks logged | 266.19 / 267.47 at failure; +1.73 requested | - | - | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b7-full-ga8-17u-20261028-bound1-ivankobzarev) | OOM before profiling |

## Native 64 GPU activation-offload rescue configurations

These add CPU activation offload to failed native configurations: SAC BS6 and Full recompute BS10. The initial quota is 80 GiB/rank; a Full recompute BS10 run with 24 GiB/rank tests a smaller budget. A separate SAC run adds failure telemetry. All retain the primary native optimizer precision, placement, GA8 and communication settings. They have no existing matched graph reference, so cross-engine throughput ratios are omitted.

| ID | BS / GA | Global batch | CPU activation quota GiB/rank | Task RAM GiB/4 GPUs | Native outcome | Allocated / reserved peak GiB | Tokens/s/GPU | MFU % | MAST job | Profile, update 16 rank 0 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `native_rescue_sac_cpuoffload_bs6_ao80` | 6 / 8 | 3072 | 80 | 845 | OOM; 64 ranks logged | 267.59 / 268.70 at failure; +0.33 requested | - | - | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-sac-ao80-ga8-17u-20261028-res1-ivankobzarev) | OOM before profiling |
| `native_rescue_full_recompute_cpuoffload_bs10_ao80` | 10 / 8 | 5120 | 80 | 845 | PASS, all 64 x 17 | 202.94 / 208.44 | 4,269.97 | 24.92 | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-full-ao80-ga8-17u-20261028-res1-ivankobzarev) | [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e7eadb26-d623-4c6d-a955-c05444f72815_rank0_trace.json.gz) |
| `native_diagnostic_sac_cpuoffload_bs6_ao80` | 6 / 8 | 3072 | 80 | 845 | OOM; 64 ranks logged | 267.59 / 268.70 at failure; +0.33 requested | - | - | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-sac-ao80-diag-ga8-17u-20261028-ivankobzarev) | OOM before profiling |
| `native_full_recompute_cpuoffload_bs10_ao24` | 10 / 8 | 5120 | 24 | 845 | PASS, all 64 x 17 | 257.83 / 265.10 | 4,536.08 | 26.48 | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-full-ao24-ga8-17u-20261028-swp1-ivankobzarev) | [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_104cef00-b208-48be-bc38-5a4c31229da3_rank0_trace.json.gz) |

Within native eager at BS10/GA8, lowering the activation quota from 80 to 24 GiB increases throughput by 6.2% and reserved GPU memory by 56.66 GiB. Both complete all 17 updates; actual peak live CPU copies are 24.00 and 79.98 GiB/rank respectively. The configuration with offload disabled failed after one warmup update. These single trials illustrate the memory/throughput tradeoff; their non-deterministic scalar comparison is reported below.

## Native 32 GPU activation-offload rescue configuration

This increases the CPU activation quota from 24 to 40 GiB/rank at the original BS8/GA8 geometry while retaining GPU FP32 master weights and gradients, CPU FP32 Adam moments, and the 512 MiB staging cap per optimizer instance. The first AO40 run completed update 16 before a host-memory OOMD kill during profiler shutdown. A retry reduces profiler host memory: rank 0 trace without shape recording at update 16, rank 0 memory snapshot at update 17. All ranks retain ten measured updates 6-15 and 17 total training updates. These are supplemental capacity experiments, with no matched graph throughput ratio.

| ID | BS / GA | Global batch | CPU activation quota GiB/rank | Task RAM GiB/4 GPUs | Native outcome | Allocated / reserved peak GiB | Tokens/s/GPU | MFU % | MAST job | Profile, update 16 rank 0 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `native_rescue_full_recompute_cpuoffload_bs8_ao40` | 8 / 8 | 2048 | 40 | 845 | Host OOMD kill after update 16; incomplete benchmark | - | - | - | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b8-full-ao40-ga8-17u-20261028-res1-ivankobzarev) | Host OOM at profiler shutdown |

## Native startup observations

These are the maximum reported rank-local durations for two separate stages. Checkpoint loading is outside the ten measured training updates. Queueing, process launch, and the first update are excluded. The stage maxima are not summed into a synchronized total preparation time.

Checkpoint restoration varies with cache and storage conditions across sequential jobs. These observations are not controlled startup trials and do not establish a precompile preparation-time benefit.

| ID | Constructor ranks observed | Constructor max seconds | Checkpoint ranks observed | Checkpoint load max seconds |
| --- | ---: | ---: | ---: | ---: |
| `64g_sac_bs6` | 64 | 51.38 | 64 | 2,815.79 |
| `64g_full_bs6` | 64 | 47.83 | 64 | 2,167.19 |
| `64g_full_bs10` | 64 | 46.91 | 64 | 560.37 |
| `64g_sac-ao_bs10_ao100` | 64 | 46.99 | 64 | 558.88 |
| `64g_noac_bs6` | 64 | 50.98 | 64 | 632.55 |
| `64g_sac_bs10` | 64 | 45.77 | 64 | 550.34 |
| `32g_full_bs8` | 32 | 45.88 | 32 | 569.72 |
| `32g_full-ao_bs8_ao24` | 32 | 44.80 | 32 | 550.36 |
| `32g_full_bs10` | 32 | 46.34 | 32 | 549.26 |
| `32g_full-ao_bs10_ao48` | 32 | 45.85 | 32 | 560.79 |
| `64g_sac-ao_bs10_ao60` | 64 | 48.19 | 64 | 558.02 |
| `64g_sac-ao_bs10_ao80` | 64 | 46.69 | 64 | 545.64 |
| `native_cuda_graphs_64g_sac_bs6` | 64 | 44.36 | 64 | 557.49 |
| `native_cuda_graphs_64g_full_bs6` | 64 | 46.03 | 64 | 546.66 |
| `native_cuda_graphs_32g_full_bs8` | 32 | 44.07 | 32 | 570.24 |
| `native_cuda_graphs_32g_full_bs10` | 32 | 47.67 | 32 | 556.69 |
| `native_eager_sac_bs6_det42` | 64 | 44.74 | 64 | 534.34 |
| `native_eager_full_recompute_bs6_det42` | 64 | 45.02 | 64 | 532.81 |
| `native_eager_full_recompute_cpuoffload_bs6_det42` | 64 | 46.54 | 64 | 542.62 |
| `native_eager_full_recompute_bs6_gather_every` | 64 | 44.03 | 64 | 547.37 |
| `native_eager_full_recompute_bs6_gather_first` | 64 | 43.62 | 64 | 525.82 |
| `native_eager_full_recompute_bs6_reduce_last_unfused` | 64 | 45.25 | 64 | 542.10 |
| `native_eager_full_recompute_bs6_reduce_last_wgrad_fused` | 64 | 43.51 | 64 | 528.88 |
| `native_eager_full_recompute_bs6_first_gather_legacy_fused` | 64 | 43.13 | 64 | 529.75 |
| `native_fit_full_recompute_bs6` | 32 | 43.92 | 32 | 571.10 |
| `native_fit_full_recompute_cpuoffload_bs6_ao24` | 32 | 44.24 | 32 | 542.62 |
| `native_rescue_sac_cpuoffload_bs6_ao80` | 64 | 45.27 | 64 | 532.60 |
| `native_rescue_full_recompute_cpuoffload_bs10_ao80` | 64 | 47.09 | 64 | 541.78 |
| `native_boundary_full_recompute_bs7` | 32 | 45.06 | 32 | 559.00 |
| `native_rescue_full_recompute_cpuoffload_bs8_ao40` | 32 | 46.73 | 32 | 578.01 |
| `native_diagnostic_sac_cpuoffload_bs6_ao80` | 64 | 44.17 | 64 | 544.15 |
| `native_full_recompute_cpuoffload_bs10_ao24` | 64 | 45.81 | 64 | 542.15 |
| `native_full_recompute_cpuoffload_bs8_ao40_profile0` | 32 | 45.34 | 32 | 578.13 |

## Native activation offload actually used

Counters cover all 17 updates. Peak live bytes and copy count are maximums over ranks; host-pressure skips are summed over ranks. A zero remaining-live count verifies that all activation payloads were released.

| ID | Quota GiB/rank | Peak live CPU GiB/rank | Maximum copies/rank | Host-pressure skips, all ranks | Maximum remaining live bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `native_eager_full_recompute_cpuoffload_bs6_det42` | 80 | 48.74 | 48824 | 0 | 0 |
| `native_fit_full_recompute_cpuoffload_bs6_ao24` | 24 | 24.00 | 24752 | 0 | 0 |
| `native_rescue_full_recompute_cpuoffload_bs10_ao80` | 80 | 79.98 | 56168 | 0 | 0 |
| `native_full_recompute_cpuoffload_bs10_ao24` | 24 | 24.00 | 19176 | 0 | 0 |
| `native_full_recompute_cpuoffload_bs8_ao40_profile0` | 40 | 40.00 | 37128 | 0 | 0 |

## Native SAC offload diagnosis

A reduced real FSDP2/EP2, MTP1, GA8 native SAC run completed all 17 updates with a 0.25 GiB CPU activation quota. Each rank made 3128 CPU copies, reached 0.249 GiB peak live activation storage, and recorded zero host-pressure skips. Prepared FSDP weight wrappers remained on GPU. This verifies that the SAC hook path makes CPU copies locally; it does not measure offload use in the failed 671B jobs.

The supplemental SAC BS6/AO80 diagnostic uses the same native configuration and pack/unpack decisions, with top-level forward counters and an exception logger added to the experiment adapter. It records actual quota use on OOM. The earlier failed runs have no completed-update offload sidecar, so their actual offload use remains unknown.

The diagnostic adapter matched its original native SAC/offload control on real FSDP2/EP2: all 17 scalar readings and all 88 first-update weight and pre-clipping gradient tensors were bitwise equal on both ranks. CPU copy counts, peak quota use and remaining-live counters also matched exactly.

Counters below include the failed update. Forward counts refer to top-level model calls within that update, with one call per microbatch at PP1. Failed-allocation GPU memory remains in the job result table; memory queried after exception unwinding is not used as that snapshot.

| ID | Ranks observed | CPU peak min / max GiB/rank | Copies min / max per rank | Host-pressure skips | Forward calls started min / max | Forward calls completed min / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `native_diagnostic_sac_cpuoffload_bs6_ao80` | 64 | 80.00 / 80.00 | 174 / 174 | 0 | 1 / 1 | 0 / 0 |

The hardware diagnostic filled the 80 GiB quota on every rank, making 174 CPU copies per rank with zero host-pressure skips. Every rank failed in its first microbatch forward before reaching backward or completing an update. The CUDA failure is in the MTP attention MXFP8 projection. This establishes a first-microbatch capacity failure with active CPU offload; it does not identify which remaining GPU storage would need a different policy.

## Why GraphTrainer helps at BS6

### 64 GPU Full recompute: capture alone explains a small part of the speed gap

All three runs use BS6/GA8 on 64 x GB300, FP32 GPU master weights and persistent gradients, BF16 GPU Adam moments, and parameter gathering and gradient reduction every microbatch. All 64 ranks completed 17 updates; throughput is reconstructed from the ten measured updates.

| Execution | Update seconds | Tokens/s/GPU | MFU % | Allocated / reserved peak GiB | Rank-0 profile |
| --- | ---: | ---: | ---: | ---: | --- |
| Native eager | 43.343 | 4,536.09 | 26.48 | 219.08 / 225.21 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e49e9518-7a8d-4cc4-947a-c457d96ee662_rank0_trace.json.gz) |
| Native CUDA graphs | 43.123 | 4,559.22 | 26.61 | 213.24 / 218.24 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_ca84b859-52e6-4656-a7f6-dd0827a87917_rank0_trace.json.gz) |
| GraphTrainer | 41.299 | 4,760.57 | 27.79 | 189.59 / 193.63 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_09c20ce1-2119-412e-ae88-60c97992c1c5_dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev__update16_rank0_original.json.gz) |

Native capture improves throughput by 0.51%, accounting for 0.220 seconds of the 2.044-second eager-to-GraphTrainer gap, or approximately 11%. GraphTrainer remains 4.42% faster than native CUDA graphs. The native profile records one `cudaGraphLaunch` for the whole GA8 forward/backward, confirming actual replay.

### The captured programs execute different amounts of GPU work

The table counts only GPU slices with a nonzero CUDA graph ID, excluding the outside-graph token-count reduction, optimizer and input preparation. Counts are robust structural observations; durations from the separately profiled update are not a decomposition of the ten-update throughput result.

| Captured GPU observation | Native CUDA graphs | GraphTrainer |
| --- | ---: | ---: |
| Total GPU slices | 207,422 | 169,121 |
| `memcpy128` kernel slices | 20,272 | 1,544 |
| `memcpy32_post` kernel slices | 11,625 | 1,897 |
| BF16 conversion/copy kernel slices | 12,016 | 4,992 |
| Dense GEMM slices | 15,248 | 14,760 |
| Attention kernel slices | 1,488 | 1,488 |
| Dist-MoE GEMM slices | 3,776 | 3,776 |
| All-gather kernel slices | 1,048 | 1,024 |
| Reduce-scatter kernel slices | 520 | 640 |

GraphTrainer executes 18.5% fewer captured GPU slices. The reductions are primarily tensor copies, dtype conversions, checks and 488 dense GEMM slices. Whole-graph tracing can eliminate unused computations and temporary materializations; native region replay and FSDP/autograd retain their runtime operations inside capture. The major Dist-MoE GEMMs have essentially identical aggregate kernel time, about 17.14 seconds in both profiles. These observations point to lower auxiliary GPU work, rather than faster expert GEMMs.

Communication is not uniformly reduced: GraphTrainer has more reduce-scatter slices because its chunked head reductions differ. The earlier eager trace's 42-second symmetric-memory barrier time is dominated by profiling disturbance; under native capture and GraphTrainer the corresponding unions are 0.72 and 0.75 seconds.

The captured GPU span is 41.60 seconds for native capture and 41.25 seconds for GraphTrainer in this single rank-0 profile. This smaller profile gap cannot assign the entire 1.824-second difference in measured, slowest-rank update times to individual kernels. Rank timing variation and single-trial variance remain limitations. The added 32-GPU native-capture run and original-payload GraphTrainer repeat test the BS6 comparison with FP32 CPU optimizer moments.

### The remaining memory advantage is in the graph workspace pool

Rank-0 update-17 allocator snapshots place almost the entire native-capture-to-GraphTrainer reserved-memory difference in the CUDA graph pool.

| Snapshot observation, GiB | Native CUDA graphs | GraphTrainer |
| --- | ---: | ---: |
| Reserved outside the graph pool | 107.13 | 107.12 |
| Reserved in the graph pool | 111.11 | 86.51 |
| Allocated in the graph pool after the update | 39.735 | 39.735 |
| Inactive blocks retained in the graph pool | 71.38 | 46.77 |

The 24.61 GiB difference is workspace reserved for replay. The near-identical outside-pool allocation and retained gradient storage support a buffer-lifetime/materialization explanation; optimizer and gradient precision have not been lowered. Captured temporary addresses remain reserved while the graph exists.

Native eager's reconstructed update-17 peak is 219.0846 GiB, agreeing with its measured 219.0848 GiB peak within allocator rounding. It contains 20.02 GiB from 61 block-sized residual outputs and 27.19 GiB attributed to the retained Dist-MoE region. GraphTrainer Full also forces layer-boundary outputs to be saved, so those 61 residual tensors cannot simply be counted as eliminated memory. History starts after capture and lacks the internal graph-pool allocation stacks; the exact buffer-by-buffer explanation of the 24.61 GiB workspace difference still needs capture-time allocation history or focused ablations.

### 32 GPU BS6 offload: fewer bytes transferred and more overlap

Both engines use a 24 GiB/rank activation quota and FP32 CPU Adam moments. Traffic below is the increase over each engine's matching run without activation offload, removing the common CPU optimizer traffic.

| Rank-0 update-16 observation | Native eager | GraphTrainer |
| --- | ---: | ---: |
| Additional activation D2H traffic, GiB/update | 191.99 | 99.75 |
| Additional activation H2D traffic, GiB/update | 191.99 | 99.75 |
| Additional D2H transfers/update | 1,456 | 304 |
| H2D/D2H interval overlap with compute, seconds | 0.025 | 0.629 |
| Throughput loss relative to no offload | 4.18% | 0.57% |

GraphTrainer's selected activation-transfer set is smaller under the same quota: approximately 12.47 GiB transferred each direction per microbatch, versus 24.00 GiB for native eager. This is measured traffic, not a GraphTrainer peak-live CPU counter. Its offload pass also prefetches reloads and defers waits; native saved-tensor hooks copy and restore on the current compute stream. The trace shows more transfer/compute overlap for GraphTrainer. Selection and scheduling both contribute to its lower offload cost; this is not an equal-byte transfer-speed comparison.

The baselines both write 158.94 GiB of FP32 optimizer state back to CPU per update. This confirms the optimizer offload traffic is comparable. The measured BS6 offload pair achieves 4,515.63 tokens/s/GPU in GraphTrainer versus 4,155.38 in eager, with reserved peaks of 221.06 versus 243.87 GiB. Their profile links and job links are in the matched BS6 table.

The strongest supported conclusions are a smaller replay workspace, fewer auxiliary GPU operations, and a more efficient activation-offload selection and schedule. The exact attribution of every saved GiB and every fraction of the speed gap remains open. Cross-engine numerical differences described elsewhere in this report also remain.



## 32 GPU CPU memory observations

These are outside-timer samples after updates 1, 2, 5 and 15 on every rank. Available RAM is per four-GPU host; RSS and pinned allocation are per worker. They are point observations, not continuous host-memory peaks. Pinned allocations include CPU optimizer states and cached activation buffers, so they are not activation-only bytes.

| ID | Samples | Minimum observed host available GiB | Maximum observed worker RSS GiB | Maximum observed pinned allocation peak GiB/worker |
| --- | ---: | ---: | ---: | ---: |
| `native_fit_full_recompute_bs6` | 128 | 185.50 | 166.95 | 159.90 |
| `native_fit_full_recompute_cpuoffload_bs6_ao24` | 128 | 81.83 | 192.38 | 185.34 |
| `graph_full_recompute_bs6_fp32_cpu_moments` | 128 | 172.18 | 168.86 | 159.93 |
| `graph_full_recompute_cpuoffload_bs6_ao24_fp32_cpu_moments` | 128 | 121.40 | 181.36 | 172.40 |
| `native_rescue_full_recompute_cpuoffload_bs8_ao40` | 128 | 52.51 | 209.03 | 206.97 |
| `native_full_recompute_cpuoffload_bs8_ao40_profile0` | 128 | 52.37 | 207.55 | 206.97 |
| `graph_full_recompute_cpuoffload_bs8_ao40_fp32_cpu_moments` | 128 | 88.50 | 189.70 | 180.74 |

### BS8/AO40 host-memory failure

The first native BS8/AO40 run completed 16 updates on all 32 ranks, including the ten measured updates, but OOMD killed a task during profiler shutdown (exit 137, signal 9). Actual CPU activation quota use reached 40.00 GiB/rank with zero host-pressure skips. The maximum pinned allocator peak was 206.97 GiB/worker before profiling; four workers share an 845 GiB task RAM allocation. At update 15 the hosts still had at least 53.49 GiB physically available.

The final benchmark/scalar files are written after training returns and were absent for this killed run. Its clean synchronized throughput cannot be reconstructed from those files, so no tokens/s or MFU result is reported. A never-started retry requested 896 GiB but MAST normalized it back to 845 GiB; readback caught the mismatch and that definition was cancelled. The profiler-only retry preserves training math and measurement steps, and explicitly changes profile coverage to rank 0 with shape recording disabled.

The profiler-only adapter completed 17 updates with real FSDP2/EP2, MTP1, GA8, CPU FP32 Adam moments, and strict seed 42. All 17 average losses, maximum losses and gradient norms match the original native Full/offload control bitwise on both ranks. The trace has no input-shape recording, and only rank 0 produces the update 16 trace and update 17 memory snapshot.

## Full recompute BS6 profile interpretation

The synchronized measured update averages are 41.30 seconds for graph trainer and 43.34 seconds for native eager. Update 16 is separately profiled and excluded from those averages.

| Original rank-0 update-16 observation | Graph trainer | Native eager |
| --- | ---: | ---: |
| GPU slice interval union, seconds | 43.19 | 85.68 |
| Symmetric-memory barrier slice union, seconds | 0.75 | 42.27 |
| NCCL all-gather GPU slices | 1024 | 1048 |
| NCCL reduce-scatter GPU slices | 640 | 520 |
| Trace events | 355,167 | 2,953,367 |

The native profiled update has substantial instrumentation disturbance and long symmetric-memory waits. Its trace duration must not replace the measured training time or establish a steady-state communication penalty. GPU slice counts are not logical collective counts, and individual category durations can overlap. Analysis uses the original trace files, not compacted display lanes.

## Communication observations from original profiles

All counts below are rank-0 GPU kernel slices from the separately profiled update 16. They verify the executed communication pattern, but are not logical collective counts. The GPU interval union and barrier time are affected by profiling overhead and must not replace synchronized measured update times.

| ID | Measured update seconds | Profile GPU union seconds | All-gather GPU slices | ReduceScatter GPU slices | Barrier union seconds | Profile |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `64g_full_bs6` | 43.34 | 85.68 | 1048 | 520 | 42.27 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e49e9518-7a8d-4cc4-947a-c457d96ee662_rank0_trace.json.gz) |
| `native_cuda_graphs_64g_full_bs6` | 43.12 | 44.81 | 1048 | 520 | 0.72 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_ca84b859-52e6-4656-a7f6-dd0827a87917_rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_det42` | 44.46 | 88.26 | 1048 | 520 | 43.02 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_f2de370f-8cde-41ad-a32f-684fc6e1034c_rank0_trace.json.gz) |
| `native_eager_full_recompute_cpuoffload_bs6_det42` | 48.22 | 89.21 | 1048 | 520 | 40.15 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_cb145ed0-d3ce-4c2e-b627-a13caf5827ba_rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_gather_every` | 56.19 | 89.11 | 592 | 64 | 43.79 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_89da90f1-c852-48a4-b627-30f0084ad9cd_rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_gather_first` | 54.25 | 80.83 | 64 | 64 | 37.71 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_47160686-9f00-4516-9052-220d174f2f70_rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_reduce_last_unfused` | 55.26 | 80.27 | 592 | 64 | 37.03 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6c9cda47-fe23-4ac0-9749-60514193e813_rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_reduce_last_wgrad_fused` | 55.09 | 84.63 | 592 | 64 | 41.70 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_bd35b9ea-4855-4818-84f7-c7e0796edea1_rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_first_gather_legacy_fused` | 53.34 | 83.45 | 64 | 64 | 40.73 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_1a01ea5f-16ac-4e7e-9078-33e3336c5817_rank0_trace.json.gz) |
| `native_fit_full_recompute_bs6` | 45.34 | 86.95 | 1048 | 520 | 42.11 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b6ba5d08-ec4c-4089-979b-e2eae1702912_rank0_trace.json.gz) |
| `native_fit_full_recompute_cpuoffload_bs6_ao24` | 47.31 | 90.55 | 1048 | 520 | 42.25 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_fa3658b0-3638-4db2-baac-50c430103064_rank0_trace.json.gz) |
| `native_rescue_full_recompute_cpuoffload_bs10_ao80` | 76.74 | 91.79 | 1048 | 520 | 17.41 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e7eadb26-d623-4c6d-a955-c05444f72815_rank0_trace.json.gz) |
| `graph_full_recompute_bs6_fp32_cpu_moments` | 43.29 | 45.94 | 1024 | 640 | 0.72 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_54cdf4b4-4ae6-424b-a407-fa4245913f2b_rank0_trace.json.gz) |
| `graph_full_recompute_cpuoffload_bs6_ao24_fp32_cpu_moments` | 43.54 | 46.86 | 1024 | 640 | 1.01 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_603d9bab-28aa-4886-8d71-5580a362b25e_rank0_trace.json.gz) |
| `native_full_recompute_cpuoffload_bs10_ao24` | 72.24 | 89.44 | 1048 | 520 | 18.27 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_104cef00-b208-48be-bc38-5a4c31229da3_rank0_trace.json.gz) |
| `native_full_recompute_cpuoffload_bs8_ao40_profile0` | 64.38 | 62.18 | 1048 | 520 | 1.62 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_4540b13b-1016-4488-8daf-6a3cf089bc76_rank0_trace.json.gz) |
| `graph_full_recompute_cpuoffload_bs8_ao40_fp32_cpu_moments` | 57.51 | 60.87 | 1024 | 640 | 0.85 | [Original](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_09e640ff-877d-4e12-b7a4-5f91fa59969e_rank0_trace.json.gz) |

## Native failure locations

- `64g_sac_bs6`: Native forward in MTP transformer attention projection, torch.nn.functional.scaled_mm; no optimizer update completed. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `64g_full_bs10`: Second optimizer update forward in native fused MLA key/RoPE preparation (_fused_mla_k_rope_op); one warmup update completed. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `64g_sac-ao_bs10_ao100`: Native forward in attention projection (torch.nn.functional.scaled_mm); no optimizer update completed with configured 100 GiB/rank CPU activation quota. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
  No after-update sidecar was emitted because the first update failed; actual copied bytes are unavailable for this attempt.
- `64g_noac_bs6`: Native forward in fused MLA key/RoPE preparation (_fused_mla_k_rope_op); no optimizer update completed. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `64g_sac_bs10`: Native forward in fused MLA key/RoPE preparation (_fused_mla_k_rope_op); no optimizer update completed. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `32g_full_bs8`: Native forward in ReportDenseAttention scaled-dot-product attention under Full remat; no optimizer update completed. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `32g_full-ao_bs8_ao24`: Native cuDNN scaled-dot-product attention backward (_cudnn_attention_backward) with configured 24 GiB/rank CPU activation quota; no optimizer update completed. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
  No after-update sidecar was emitted because the first update failed; actual copied bytes are unavailable for this attempt.
- `32g_full_bs10`: Forward MXFP8 quantization. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `32g_full-ao_bs10_ao48`: Forward native chunked cross-entropy. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `64g_sac-ao_bs10_ao60`: Forward fused MLA key/RoPE. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `64g_sac-ao_bs10_ao80`: Forward MTP MXFP8 attention projection (scaled_mm). GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `native_cuda_graphs_32g_full_bs8`: Native CUDA graph warmup on its capture stream: forward MTP attention MXFP8 quantization; graph capture had not started. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `native_cuda_graphs_32g_full_bs10`: Native CUDA graph warmup on its capture stream: forward cuDNN SDPA; graph capture had not started. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `native_eager_sac_bs6_det42`: Forward fused MLA query/RoPE output allocation (_fused_mla_q_rope_out_op). GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `native_rescue_sac_cpuoffload_bs6_ao80`: Forward MTP attention MXFP8 projection (scaled_mm). GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `native_boundary_full_recompute_bs7`: Forward native chunked cross-entropy, FP32 intermediate (BS7). GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
- `native_rescue_full_recompute_cpuoffload_bs8_ao40`: Profiler shutdown after update 16: host-memory OOMD kill (exit 137, signal 9).
- `native_diagnostic_sac_cpuoffload_bs6_ao80`: First microbatch forward, MTP attention MXFP8 scaled_mm; all 64 ranks started one model forward and completed none. GPU memory in OOM rows is measured at the failed allocation; it is not a successful training peak or an exact capacity forecast.
  Failure telemetry on all 64 ranks records 80.00 GiB peak live CPU activations, 174 copies per rank and zero host-pressure skips.

## Cross-engine scalar observations

The completed Full recompute BS6 performance pair uses seed 14536 with deterministic algorithms disabled. Its globally aggregated rank-0 scalar histories are not bitwise equivalent. The first loss agrees within 0.000001, while gradients and subsequent losses differ. These observations do not establish a root cause or convergence.

| Scalar | Graph update 1 / 17 | Native update 1 / 17 | Largest absolute difference over 17 updates |
| --- | ---: | ---: | ---: |
| `global_avg_loss` | 8.406084 / 6.556857 | 8.406085 / 6.558824 | 0.042672 at update 12 |
| `global_max_loss` | 8.568663 / 6.714091 | 8.568663 / 6.715082 | 0.039763 at update 12 |
| `grad_norm` | 45.760014 / 1.684176 | 45.345272 / 1.677615 | 9.343822 at update 13 |

Strict seed-42 Full recompute and Full + offload counterparts completed all 17 updates on all 64 ranks; their scalar comparison appears below. The local MTP annotation and native offload qualification still match bitwise against their native controls; that does not establish equivalence across native and graph engines.

## Reduced-model gradient localization

A real FSDP2/EP2 DSv3 + MTP1 comparison uses GA8, strict seed 42 and the production BF16 GPU Adam settings. Snapshots are taken before first-update gradient clipping and optimizer execution. All 88 initial parameter tensors match bitwise on both ranks. Of 88 first-update parameter gradients, 87 match bitwise; only `lm_head.weight` differs.

| Rank | LM-head gradient relative L2 difference | Maximum absolute gradient difference |
| --- | ---: | ---: |
| 0 | 0.0038652 | 0.00005198 |
| 1 | 0.0035316 | 0.00000156 |

This localizes the initial reduced-model discrepancy before Adam runs. Native head coalescing and graph head reduction order differ. This reduced-model observation does not establish the full-model cause or convergence.

A separate native control synchronizes the head after every chunk while preserving every-microbatch body reduction. It also completed 17 updates; 87 of 88 first-update gradients still match, and the head relative L2 differences are 0.0029796, 0.0024435. The maximum average-loss difference is 0.0006638. Per-chunk synchronization narrows the discrepancy but does not make the engines bitwise equivalent. The recorded presentation jobs retain their original native chunk loop.

## Strict hardware scalar comparisons

Only completed, all-rank-verified 17-update runs are included. These are recorded scalar comparisons; they do not establish convergence or equality of every tensor. Cross-engine reduction and accumulation orders can differ.

| Native counterpart ID | Graph / native average loss at update 1 | Maximum average-loss difference | Maximum gradient-norm difference |
| --- | ---: | ---: | ---: |
| `strict_64g_full_bs6` | 8.4076576 / 8.4076576 | 0.0257945 (update 5) | 12.8320246 (update 3) |
| `strict_64g_full-ao_bs6_ao80` | 8.4076576 / 8.4076576 | 0.0257945 (update 5) | 12.8320246 (update 3) |
| `schedule_gather_every_det42` | 8.4076576 / 8.4076576 | 0.0626588 (update 5) | 25.9283993 (update 4) |
| `schedule_gather_first_det42` | 8.4076576 / 8.4076576 | 0.0626588 (update 5) | 25.9283993 (update 4) |

| Native pair | Ranks x updates checked | All scalar values bitwise equal | Maximum average-loss / gradient-norm difference |
| --- | ---: | --- | ---: |
| `strict_64g_full_bs6` versus `strict_64g_full-ao_bs6_ao80` | 64 x 17 | True | 0.0000000 / 0.0000000 |
| `schedule_gather_every_det42` versus `schedule_gather_first_det42` | 64 x 17 | True | 0.0000000 / 0.0000000 |

## Native scalar comparisons with deterministic algorithms disabled

These pairs use the primary performance settings. They completed 17 finite updates, but their scalar histories differ and are not numerical-equivalence or convergence evidence. The strict seed-42 Every/First gathering pair above is the qualified scheduling comparison.

| Native pair | Ranks x updates checked | Maximum average-loss difference | Maximum gradient-norm difference |
| --- | ---: | ---: | ---: |
| `64g_full_bs6` versus `native_cg_64g_full_bs6` | 64 x 17 | 0.0510969 (update 14) | 9.6593959 (update 13) |
| `schedule_reduce_last_unfused` versus `schedule_reduce_last_fused` | 64 x 17 | 0.0914626 (update 8) | 6.2225356 (update 14) |
| `schedule_reduce_last_fused` versus `schedule_first_legacy_fused` | 64 x 17 | 0.0579176 (update 16) | 170.5535989 (update 15) |
| `fit_32g_full_bs6` versus `fit_32g_full-ao_bs6_ao24` | 32 x 17 | 0.0471563 (update 16) | 4.5601285 (update 15) |
| `sweep_64g_full_bs10_ao24` versus `rescue_64g_full_bs10_ao80` | 64 x 17 | 0.1042328 (update 16) | 9.6117245 (update 17) |

## Additional execution controls

The main twelve counterparts use uncaptured native execution with seed 14536, deterministic algorithms disabled, every-microbatch gathers and reductions, and WGrad fusion disabled. The following rows record overrides for subsequent controls; their configuration IDs in the tables explicitly identify the backend or strict seed.

| ID | Execution | Gather / reduce during GA8 | WGrad fusion | Seed / strict deterministic |
| --- | --- | --- | --- | --- |
| `native_cuda_graphs_64g_sac_bs6` | Native CUDA graphs, whole GA8 forward/backward | every_microbatch / every_microbatch | disabled | 14536 / False |
| `native_cuda_graphs_64g_full_bs6` | Native CUDA graphs, whole GA8 forward/backward | every_microbatch / every_microbatch | disabled | 14536 / False |
| `native_cuda_graphs_64g_full_bs10` | Native CUDA graphs, whole GA8 forward/backward | every_microbatch / every_microbatch | disabled | 14536 / False |
| `native_cuda_graphs_32g_full_bs8` | Native CUDA graphs, whole GA8 forward/backward | every_microbatch / every_microbatch | disabled | 14536 / False |
| `native_cuda_graphs_32g_full_bs10` | Native CUDA graphs, whole GA8 forward/backward | every_microbatch / every_microbatch | disabled | 14536 / False |
| `native_eager_sac_bs6_det42` | Native eager | every_microbatch / every_microbatch | disabled | 42 / True |
| `native_eager_full_recompute_bs6_det42` | Native eager | every_microbatch / every_microbatch | disabled | 42 / True |
| `native_eager_full_recompute_cpuoffload_bs6_det42` | Native eager | every_microbatch / every_microbatch | disabled | 42 / True |
| `native_eager_full_recompute_bs6_gather_every` | Native eager | every_microbatch / last_microbatch | enabled | 42 / True |
| `native_eager_full_recompute_bs6_gather_first` | Native eager | first_microbatch / last_microbatch | enabled | 42 / True |
| `native_eager_full_recompute_bs6_reduce_last_unfused` | Native eager | every_microbatch / last_microbatch | disabled | 14536 / False |
| `native_eager_full_recompute_bs6_reduce_last_wgrad_fused` | Native eager | every_microbatch / last_microbatch | enabled | 14536 / False |
| `native_eager_full_recompute_bs6_first_gather_legacy_fused` | Native eager | first_microbatch / last_microbatch | enabled | 14536 / False |
| `native_rescue_sac_cpuoffload_bs6_ao80` | Native eager | every_microbatch / every_microbatch | disabled | 14536 / False |
| `native_rescue_full_recompute_cpuoffload_bs10_ao80` | Native eager | every_microbatch / every_microbatch | disabled | 14536 / False |
| `native_rescue_full_recompute_cpuoffload_bs8_ao40` | Native eager | every_microbatch / every_microbatch | disabled | 14536 / False |
| `native_diagnostic_sac_cpuoffload_bs6_ao80` | Native eager | every_microbatch / every_microbatch | disabled | 14536 / False |
| `native_full_recompute_cpuoffload_bs10_ao24` | Native eager | every_microbatch / every_microbatch | disabled | 14536 / False |
| `native_full_recompute_cpuoffload_bs8_ao40_profile0` | Native eager | every_microbatch / every_microbatch | disabled | 14536 / False |
| `native_cuda_graphs_32g_full_bs6` | Native CUDA graphs, whole GA8 forward/backward | every_microbatch / every_microbatch | disabled | 14536 / False |

## Native MTP checkpoint qualification

The first SAC submission was cancelled while loading its checkpoint after a local MTP1 qualification exposed missing `recompute_needs_tensor` consumer annotations in the native MTP transformer block. It produced no measured update and is not a capacity OOM result. Queued remat jobs were held for correction.

The correction marks the MTP projection, attention and feed-forward outputs before unwrapped consumers read them during recomputation. It preserves forward math. The original native MTP execution without a checkpoint wrapper and the corrected Full checkpoint execution completed 17 updates with real FSDP2/EP2, GA8, FP32 CPU Adam moments, seed 42 and strict deterministic algorithms. All 17 global average losses, maximum losses and gradient norms matched bitwise on both ranks.

The source package declares the base compute commit plus the explicit MTP annotation patch and hashes both the original and modified file. Patch SHA256: `95c8130efe635f9c5748183c0f88a4e12a2463b93eb50c4170b29d5d0ca5c6cb`. The No AC control uses the original source because it does not execute native remat.

The No AC attempt 0 was preempted during checkpoint loading on borrowed capacity by our own queued Full recompute job. It completed no update and is not a CUDA OOM observation. Dispatch now keeps only one 64 GPU job or at most two 32 GPU jobs unfrozen to prevent this interruption between our experiments. MAST preemption attempts are recorded separately from training failures.

Native `EAGER_INITIALIZATION_END` times trainer/model construction. Checkpoint restoration happens later and has separate existing `Loading the checkpoint` / `Finished loading the checkpoint in ... seconds` log markers. The constructor sidecar scope text incorrectly includes checkpoint load; do not use that sidecar as total preparation time.

The native chunked-loss wrapper originally re-enabled gradient synchronization and resharded the head after each microbatch, overriding whole-update accumulation and retention policies. The scheduling counterparts use an experiment-only subclass that preserves the incoming FSDP synchronization and reshard settings while reusing the original chunk loop. Core TorchTitan source is unchanged by this head correction. Their original frozen payloads never ran; MAST required terminating those versions before accepting the corrected definitions as version 1. All five replacement definitions were read back and verified.

Head qualification completed 17 updates with real FSDP2/EP2, MTP1, GA8, strict seed 42 and FP32 CPU Adam moments. Corrected every-microbatch execution matches the original native control bitwise. Every-gather versus First-gather pairs with last-update reduction match all 17 average losses, maximum losses and gradient norms, in both fused and unfused execution. Explicit FSDP head instrumentation verifies one reducing backward and one finalization per update; First gather requires one unshard transition per update. Head-only per-op captures cover all eight microbatches of update 2 on both ranks: matched operator/phase/shape summaries and output norm hashes agree. These rounded per-op summaries supplement the full-precision scalar checks. Whole-model DebugMode capture was incompatible with storage-free remat placeholders, so the numeric capture is scoped to the corrected head loop.

Both fused gather arms recorded 6069 uses per rank of an existing MXFP8 running gradient. Native fusion enables model `inplace_wgrad_accum` flags; graph fusion is expressed in transformed backward graphs. Fused versus unfused arithmetic has different BF16 rounding and is not claimed to match bitwise. The native First gather configuration retains parameters across the update; the archived legacy graph First configuration has backward regathers, as explained in the original graph report. These local checks qualify execution semantics, not 671B capacity, convergence, or cross-engine equivalence.

## Evidence and reproducibility

- [64 GPU graph progression](https://github.com/pytorch/torchtitan/blob/dsv3-64gpu-startup-report-20261027/dsv3_64gpu_progression_report_20261027.md).
- [32 GPU FP32 graph progression](https://github.com/pytorch/torchtitan/blob/dsv3-32gpu-fp32-progression-report-20261027/dsv3_32gpu_fp32_progression_report_20261027.md).
- Local eager recipes, configuration comparisons, launch/readback receipts, scalar checks and collected profiles: `/home/ivankobzarev/tmp/dsv3-eager-progression-20261027`.
- CPU activation copy unit checks verify exact reconstruction, repeated restore, parameter exclusion, gradient equality and live-quota accounting.
- The local native qualification uses seed 42 and strict deterministic algorithms on real FSDP2/EP2. Its reduced model is a correctness/execution check, not a 671B capacity forecast.
- Main performance runs use deterministic algorithms disabled. Finite losses do not establish numerical equivalence between native and graph engines.

| Payload | Compute source | Workspace package | Runtime package |
| --- | --- | --- | --- |
| 64 GPU No AC control | `1457497a35d2fd7570b87566a8af0b3fd22b7538` | `torchtitan_workspace:d36df3dd6514412ebf3aa3a66efefacf` | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` |
| 64 GPU native main, strict and capture controls | `1457497a35d2fd7570b87566a8af0b3fd22b7538` + MTP annotation patch | `torchtitan_workspace:302fe6d01e964cdd9076676a027fbde3` | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` |
| 32 GPU native counterparts | `f66c0e6cd7aa3d5356d918e7b8d6a2a6f1fd1c65` + MTP annotation patch | `torchtitan_workspace:aa3efdac1f8d413da72d4f5ac8f94b37` | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` |
| 64 GPU scheduling payload before head qualification (held, superseded) | `2eb747d1b479616880b27e21c068578d4cb333e6` + MTP annotation patch | `torchtitan_workspace:34c5271cac874defb21c29c02c1d5a89` | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` |
| 64 GPU native scheduling with qualified head policy | `2eb747d1b479616880b27e21c068578d4cb333e6` + MTP annotation patch | `torchtitan_workspace:6b7c7969487b435b85105645768736d9` | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` |
| 64 GPU native SAC with failure telemetry | `1457497a35d2fd7570b87566a8af0b3fd22b7538` + MTP annotation patch | `torchtitan_workspace:185df65145ea45478b3322c3bcfd172b` | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` |
| 32 GPU native AO40 with rank 0 profiling, no shapes | `f66c0e6cd7aa3d5356d918e7b8d6a2a6f1fd1c65` + MTP annotation patch | `torchtitan_workspace:128996753322495ab5172acfb58000a7` | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` |
| Original 32 GPU graph payload reused for matched BS6 and BS8/AO40 controls | `f66c0e6cd7aa3d5356d918e7b8d6a2a6f1fd1c65` | `torchtitan_workspace:f426b59d31f74b749092f6d9ace965ad` | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` |
