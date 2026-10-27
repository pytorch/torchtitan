# DSv3 671B: eager counterparts of the 32 and 64 GPU progressions

**Hardware: 64 x NVIDIA GB300 or 32 x NVIDIA GB300 GPUs.**

Updated: 2026-10-27T22:56:46.944920+00:00. Autonomous experiment window ends 2026-10-28T12:56:10+00:00.

This report pairs the existing graph trainer presentation configurations with native TorchTitan eager execution. Results come from real MAST jobs. Prepared or queued runs have no measured throughput or memory result.

**Eager definition:** native `Trainer` / `TrainingEngine`, native FSDP, regular autograd and native chunked-loss backward, with CUDA graphs disabled and `local_compile_regions=[]`. Handwritten MXFP8, attention and router kernels retain their original implementations. A native CUDA graph control, when included, is labeled separately.

## Shared configuration

| Setting | 64 GPUs | 32 GPUs |
| --- | --- | --- |
| Model | DSv3 671B + MTP1 | DSv3 671B + MTP1 |
| Parallelism | FSDP64 / EP64 / PP1 | FSDP32 / EP32 / PP1 |
| Sequence length / GA | 4096 / 8 | 4096 / 8 |
| Global batch | GPU count x local BS x 8 | GPU count x local BS x 8 |
| Master weights / persistent gradients | FP32 on GPU | FP32 on GPU |
| Adam first and second moments | BF16 on GPU | FP32 on CPU |
| CPU optimizer staging cap | Not applicable | 512 MiB per optimizer instance; two instances |
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

CPU activation offload quotas are binary GiB per rank and cap live selected activation copies. Parameter and prepared-weight storage is excluded. Copies and restores use the current CUDA stream; graph trainer has its own prefetch/defer scheduling. The native adapter records actual selected bytes and skips additional copies if host available RAM falls below 8 GiB. This guard is needed because the measured 32 GPU FP32-moment baseline leaves less than 4 x 48 GiB available per host. A configured quota is not a guarantee that the full quota is used.

MFU uses the same model-FLOP estimate, excluding recompute, and 5 PFLOP/s dense MXFP8 reference per GB300. Each run has ten synchronized measured updates and one hardware trial; there are no uncertainty intervals. Local qualification is separate from the 671B memory and speed measurements.

## 64 GPU progression

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `sac_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 236.47 / 244.55 | 4,948.86 | 28.89 | RUNNING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-sac-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b368a8dd-9f0e-428a-bdf2-f80b28c840f0_dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 189.59 / 193.63 | 4,760.57 | 27.79 | PASS, all 64 x 17 | 219.08 / 225.21 | 4,536.09 | 26.48 | 1.049x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6548b64d-cb41-44a5-8160-34d958e4bb50_dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e49e9518-7a8d-4cc4-947a-c457d96ee662_rank0_trace.json.gz) |
| `full_recompute_bs10` | 10 / 8 | Disabled | PASS, all 64 x 17 (reference) | 235.96 / 240.56 | 4,826.62 | 28.17 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-full-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_d9c75545-1623-45fd-bc08-9cfbeca9cb55_dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `sac_cpuoffload_bs10` | 10 / 8 | 100 | PASS, all 64 x 17 (reference) | 224.45 / 228.70 | 4,672.37 | 27.27 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b10-ga8-sac-ao-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ao100-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6156cd14-86f0-4e88-bfd0-d8b2bd281d93_dsv3-nebius-64g-ep64-b10-ga8-sac-ao-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `noac_bs6` | 6 / 8 | Disabled | OOM | ~598.63 projected / - | - | - | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-noac-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-noac-ga8-17u-20261027-ivankobzarev) | Not yet produced |
| `sac_bs10` | 10 / 8 | Disabled | OOM | ~309.17 projected / - | - | - | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b10-ga8-sac-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ga8-17u-20261027-r1-ivankobzarev) | Not yet produced |
| `sac_cpuoffload_bs10_ao60` | 10 / 8 | 60 | PASS, all 64 x 17 (reference) | 252.94 / 258.08 | 4,778.03 | 27.89 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-paper-64g-ep64-b10-ga8-sac-ao60-17u-20261027-r1-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ao60-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-paper-64g-ep64-b10-ga8-sac-ao60-17u-20261027-r1-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `sac_cpuoffload_bs10_ao80` | 10 / 8 | 80 | PASS, all 64 x 17 (reference) | 234.58 / 239.19 | 4,708.92 | 27.49 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-paper-64g-ep64-b10-ga8-sac-ao80-17u-20261027-r1-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ao80-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-paper-64g-ep64-b10-ga8-sac-ao80-17u-20261027-r1-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_cuda_graphs_64g_sac_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 236.47 / 244.55 | 4,948.86 | 28.89 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-64g-b6-sac-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b368a8dd-9f0e-428a-bdf2-f80b28c840f0_dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `native_cuda_graphs_64g_full_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 189.59 / 193.63 | 4,760.57 | 27.79 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-64g-b6-full-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6548b64d-cb41-44a5-8160-34d958e4bb50_dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `native_cuda_graphs_64g_full_bs10` | 10 / 8 | Disabled | PASS, all 64 x 17 (reference) | 235.96 / 240.56 | 4,826.62 | 28.17 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-64g-b10-full-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_d9c75545-1623-45fd-bc08-9cfbeca9cb55_dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `native_eager_sac_bs6_det42` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 236.48 / 244.53 | 4,839.45 | 28.25 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-sac-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-sac-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-sac-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_det42` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 189.60 / 193.61 | 4,661.34 | 27.21 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-full-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-full-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_cpuoffload_bs6_det42` | 6 / 8 | 80 | PASS, all 64 x 17 (reference) | 163.82 / 167.91 | 4,587.43 | 26.78 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-fullao-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-fullao80-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-fullao-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_gather_every` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 213.35 / 221.66 | 4,726.74 | 27.59 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-gather-every-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ge-f-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-gather-every-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_gather_first` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 240.74 / 248.53 | 4,827.56 | 28.18 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-gather-first-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-gf-f-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-gather-first-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_reduce_last_unfused` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 213.33 / 223.84 | 4,822.62 | 28.15 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-full-last-u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ge-u-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-full-last-u-20261026-ivankobzarev/v_0/attempt_1/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_reduce_last_wgrad_fused` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 213.33 / 221.68 | 4,833.03 | 28.21 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-full-last-f-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ge-f-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-full-last-f-20261026-ivankobzarev/v_0/attempt_1/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_first_gather_legacy_fused` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 240.80 / 250.17 | 4,798.39 | 28.01 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-full-first-f-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-gf-f-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-full-first-f-20261026-ivankobzarev/v_0/attempt_1/profiling/traces/iteration_16/rank0_trace.json.gz) |

## 32 GPU progression

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `full_recompute_bs8` | 8 / 8 | Disabled | PASS, all 32 x 17 (reference) | 254.82 / 258.39 | 4,582.42 | 26.75 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b8-full-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_98f6fecd-6967-48b5-96c5-3d2354834c2c_dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_cpuoffload_bs8_ao24` | 8 / 8 | 24 | PASS, all 32 x 17 (reference) | 242.07 / 245.11 | 4,558.03 | 26.61 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b8-full-ao24-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b0aaf5d8-5379-4c41-aa5f-a4a8ee815bde_dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_bs10` | 10 / 8 | Disabled | OOM | ~272.95 projected / - | - | - | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b10-full-ga8-17u-20261027-r1-ivankobzarev) | Not yet produced |
| `full_recompute_cpuoffload_bs10_ao48` | 10 / 8 | 48 | PASS, all 32 x 17 (reference) | 253.26 / 257.96 | 4,673.54 | 27.28 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b10-full-ao48-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_1d18201a-9a03-4535-b59d-0a1e8cd28034_dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |
| `native_cuda_graphs_32g_full_bs8` | 8 / 8 | Disabled | PASS, all 32 x 17 (reference) | 254.82 / 258.39 | 4,582.42 | 26.75 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-32g-b8-full-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_98f6fecd-6967-48b5-96c5-3d2354834c2c_dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |
| `native_cuda_graphs_32g_full_bs10` | 10 / 8 | Disabled | OOM | - | - | - | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-17u-20261027-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-32g-b10-full-ga8-17u-20261027-ivankobzarev) | Not yet produced |

## Native startup observations

These are the maximum reported rank-local durations for two separate stages. Checkpoint loading is outside the ten measured training updates. Queueing, process launch, and the first update are excluded. The stage maxima are not summed into a synchronized total preparation time.

| ID | Constructor ranks observed | Constructor max seconds | Checkpoint ranks observed | Checkpoint load max seconds |
| --- | ---: | ---: | ---: | ---: |
| `64g_sac_bs6` | 64 | 51.38 | 0 | - |
| `64g_full_bs6` | 64 | 47.83 | 64 | 2,167.19 |

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

## Additional execution controls

The main twelve counterparts use uncaptured native execution with seed 14536, deterministic algorithms disabled, every-microbatch gathers and reductions, and WGrad fusion disabled. The following rows record overrides for subsequent controls; their configuration IDs in the progression table explicitly identify the backend or strict seed.

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

## Native MTP checkpoint qualification

The first SAC submission was cancelled while loading its checkpoint after a local MTP1 qualification exposed missing `recompute_needs_tensor` consumer annotations in the native MTP transformer block. It produced no measured update and is not a capacity OOM result. Queued remat jobs were held for correction.

The correction marks the MTP projection, attention and feed-forward outputs before unwrapped consumers read them during recomputation. It preserves forward math. The original native MTP execution without a checkpoint wrapper and the corrected Full checkpoint execution completed 17 updates with real FSDP2/EP2, GA8, FP32 CPU Adam moments, seed 42 and strict deterministic algorithms. All 17 global average losses, maximum losses and gradient norms matched bitwise on both ranks.

The source package declares the base compute commit plus the explicit MTP annotation patch and hashes both the original and modified file. Patch SHA256: `95c8130efe635f9c5748183c0f88a4e12a2463b93eb50c4170b29d5d0ca5c6cb`. The No AC control uses the original source because it does not execute native remat.

The No AC attempt 0 was preempted during checkpoint loading on borrowed capacity by our own queued Full recompute job. It completed no update and is not a CUDA OOM observation. Dispatch now keeps only one 64 GPU job or at most two 32 GPU jobs unfrozen to prevent this interruption between our experiments. MAST preemption attempts are recorded separately from training failures.

Native `EAGER_INITIALIZATION_END` times trainer/model construction. Checkpoint restoration happens later and has separate existing `Loading the checkpoint` / `Finished loading the checkpoint in ... seconds` log markers. The constructor sidecar scope text incorrectly includes checkpoint load; do not use that sidecar as total preparation time.

The native gather-retention qualification uses last-microbatch gradient reduction in both arms and matches every logged loss and gradient norm across all 17 updates. The fused pair also matches bitwise, with 5950 calls per rank using an existing MXFP8 running gradient. Native fusion enables the model `inplace_wgrad_accum` flags; graph fusion is expressed in transformed backward graphs. Fused versus unfused arithmetic has different BF16 rounding and is not claimed to match bitwise. The native First gather configuration retains values through the whole update; the archived legacy graph First configuration has backward regathers, as explained in the original graph report.

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
| 64 GPU native scheduling counterparts | `2eb747d1b479616880b27e21c068578d4cb333e6` + MTP annotation patch | `torchtitan_workspace:34c5271cac874defb21c29c02c1d5a89` | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` |
