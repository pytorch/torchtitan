# DSv3 671B: eager counterparts of the 32 and 64 GPU progressions

**Hardware: 64 x NVIDIA GB300 or 32 x NVIDIA GB300 GPUs.**

Updated: 2026-10-27T21:19:56.912841+00:00. Autonomous experiment window ends 2026-10-28T12:56:10+00:00.

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

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Eager outcome | Eager allocated / reserved peak GiB | Eager tokens/s/GPU | Eager MFU % | Graph / eager throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `sac_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 236.47 / 244.55 | 4,948.86 | 28.89 | RUNNING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-sac-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b368a8dd-9f0e-428a-bdf2-f80b28c840f0_dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 189.59 / 193.63 | 4,760.57 | 27.79 | PENDING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6548b64d-cb41-44a5-8160-34d958e4bb50_dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_bs10` | 10 / 8 | Disabled | PASS, all 64 x 17 (reference) | 235.96 / 240.56 | 4,826.62 | 28.17 | PENDING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-full-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_d9c75545-1623-45fd-bc08-9cfbeca9cb55_dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `sac_cpuoffload_bs10` | 10 / 8 | 100 | PASS, all 64 x 17 (reference) | 224.45 / 228.70 | 4,672.37 | 27.27 | PENDING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b10-ga8-sac-ao-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ao100-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6156cd14-86f0-4e88-bfd0-d8b2bd281d93_dsv3-nebius-64g-ep64-b10-ga8-sac-ao-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `noac_bs6` | 6 / 8 | Disabled | OOM | ~598.63 projected / - | - | - | PENDING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-noac-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-noac-ga8-17u-20261027-ivankobzarev) | Not yet produced |
| `sac_bs10` | 10 / 8 | Disabled | OOM | ~309.17 projected / - | - | - | PENDING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b10-ga8-sac-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ga8-17u-20261027-ivankobzarev) | Not yet produced |
| `sac_cpuoffload_bs10_ao60` | 10 / 8 | 60 | COMPLETE; summary available | 252.94 / 258.08 | 4,778.03 | 27.89 | PENDING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-paper-64g-ep64-b10-ga8-sac-ao60-17u-20261027-r1-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ao60-ga8-17u-20261027-ivankobzarev) | Not yet produced |
| `sac_cpuoffload_bs10_ao80` | 10 / 8 | 80 | COMPLETE; summary available | 234.58 / 239.19 | 4,708.92 | 27.49 | PENDING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-paper-64g-ep64-b10-ga8-sac-ao80-17u-20261027-r1-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ao80-ga8-17u-20261027-ivankobzarev) | Not yet produced |

## 32 GPU progression

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Eager outcome | Eager allocated / reserved peak GiB | Eager tokens/s/GPU | Eager MFU % | Graph / eager throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `full_recompute_bs8` | 8 / 8 | Disabled | PASS, all 32 x 17 (reference) | 254.82 / 258.39 | 4,582.42 | 26.75 | PENDING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b8-full-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_98f6fecd-6967-48b5-96c5-3d2354834c2c_dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_cpuoffload_bs8_ao24` | 8 / 8 | 24 | PASS, all 32 x 17 (reference) | 242.07 / 245.11 | 4,558.03 | 26.61 | PENDING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b8-full-ao24-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b0aaf5d8-5379-4c41-aa5f-a4a8ee815bde_dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_bs10` | 10 / 8 | Disabled | OOM | ~272.95 projected / - | - | - | PENDING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b10-full-ga8-17u-20261027-ivankobzarev) | Not yet produced |
| `full_recompute_cpuoffload_bs10_ao48` | 10 / 8 | 48 | PASS, all 32 x 17 (reference) | 253.26 / 257.96 | 4,673.54 | 27.28 | PENDING | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b10-full-ao48-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_1d18201a-9a03-4535-b59d-0a1e8cd28034_dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |

## Evidence and reproducibility

- [64 GPU graph progression](https://github.com/pytorch/torchtitan/blob/dsv3-64gpu-startup-report-20261027/dsv3_64gpu_progression_report_20261027.md).
- [32 GPU FP32 graph progression](https://github.com/pytorch/torchtitan/blob/dsv3-32gpu-fp32-progression-report-20261027/dsv3_32gpu_fp32_progression_report_20261027.md).
- Local eager recipes, configuration comparisons, launch/readback receipts, scalar checks and collected profiles: `/home/ivankobzarev/tmp/dsv3-eager-progression-20261027`.
- CPU activation copy unit checks verify exact reconstruction, repeated restore, parameter exclusion, gradient equality and live-quota accounting.
- The local native qualification uses seed 42 and strict deterministic algorithms on real FSDP2/EP2. Its reduced model is a correctness/execution check, not a 671B capacity forecast.
- Main performance runs use deterministic algorithms disabled. Finite losses do not establish numerical equivalence between native and graph engines.

| Payload | Compute source | Workspace package | Runtime package |
| --- | --- | --- | --- |
| Eager 64 GPU | `1457497a35d2fd7570b87566a8af0b3fd22b7538` | `torchtitan_workspace:d36df3dd6514412ebf3aa3a66efefacf` | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` |
| Eager 32 GPU | `f66c0e6cd7aa3d5356d918e7b8d6a2a6f1fd1c65` | `torchtitan_workspace:6a658caa1fcf4ef2a437223bf20f0dca` | `torchtitan_conda_ivankobzarev_gt_report256_20261013_v3:c57edcb8f7a74a58836a31c4a3d44266` |
