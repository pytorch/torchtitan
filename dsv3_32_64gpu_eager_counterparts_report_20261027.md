# DSv3 671B: eager counterparts of the 32 and 64 GPU progressions

**Hardware: 64 x NVIDIA GB300 or 32 x NVIDIA GB300 GPUs.**

Updated: 2026-10-28T03:13:17.496831+00:00. Autonomous experiment window ends 2026-10-28T12:56:10+00:00.

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
| FSDP parameter / reduction dtype | BF16 / BF16 | BF16 / BF16 |
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

Chunked LM-head communication also differs. The native default coalesces head gradients across loss chunks; the graph path reduces individual head contributions. The original GraphTrainer last-reduction profiles show 16 ReduceScatter kernel slices in each early microbatch and 80 in the last: the body reduction is deferred while head reductions remain earlier. The qualified native scheduling controls defer the head reduction to finalization as well. Use the within-engine Every/First pairs for gather-cadence comparisons; cross-engine schedule labels do not imply identical head communication.

CPU activation offload quotas are binary GiB per rank and cap live selected activation copies. Parameter and prepared-weight storage is excluded. Copies and restores use the current CUDA stream; graph trainer has its own prefetch/defer scheduling. The native adapter records actual selected bytes and skips additional copies if host available RAM falls below 8 GiB. This guard is needed because the measured 32 GPU FP32-moment baseline leaves less than 4 x 48 GiB available per host. A configured quota is not a guarantee that the full quota is used.

MFU uses the same model-FLOP estimate, excluding recompute, and 5 PFLOP/s dense MXFP8 reference per GB300. Each run has ten synchronized measured updates and one hardware trial; there are no uncertainty intervals. Local qualification is separate from the 671B memory and speed measurements.

## 64 GPU progression

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `noac_bs6` | 6 / 8 | Disabled | OOM | ~598.63 projected / - | - | - | OOM; 64 ranks logged | 267.50 / 268.17 at failure; +1.12 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-noac-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-noac-ga8-17u-20261027-ivankobzarev) | Not yet produced |
| `sac_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 236.47 / 244.55 | 4,948.86 | 28.89 | OOM; 64 ranks logged | 267.62 / 268.69 at failure; +0.33 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-sac-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b368a8dd-9f0e-428a-bdf2-f80b28c840f0_dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 189.59 / 193.63 | 4,760.57 | 27.79 | PASS, all 64 x 17 | 219.08 / 225.21 | 4,536.09 | 26.48 | 1.049x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6548b64d-cb41-44a5-8160-34d958e4bb50_dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e49e9518-7a8d-4cc4-947a-c457d96ee662_rank0_trace.json.gz) |
| `sac_bs10` | 10 / 8 | Disabled | OOM | ~309.17 projected / - | - | - | OOM; 64 ranks logged | 263.01 / 263.65 at failure; +1.88 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b10-ga8-sac-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ga8-17u-20261027-r1-ivankobzarev) | Not yet produced |
| `sac_cpuoffload_bs10_ao60` | 10 / 8 | 60 | PASS, all 64 x 17 (reference) | 252.94 / 258.08 | 4,778.03 | 27.89 | OOM; 64 ranks logged | 264.56 / 265.29 at failure; +1.88 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-paper-64g-ep64-b10-ga8-sac-ao60-17u-20261027-r1-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ao60-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-paper-64g-ep64-b10-ga8-sac-ao60-17u-20261027-r1-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `sac_cpuoffload_bs10_ao80` | 10 / 8 | 80 | PASS, all 64 x 17 (reference) | 234.58 / 239.19 | 4,708.92 | 27.49 | OOM; 64 ranks logged | 264.36 / 265.21 at failure; +2.50 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-paper-64g-ep64-b10-ga8-sac-ao80-17u-20261027-r1-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ao80-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-paper-64g-ep64-b10-ga8-sac-ao80-17u-20261027-r1-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `sac_cpuoffload_bs10` | 10 / 8 | 100 | PASS, all 64 x 17 (reference) | 224.45 / 228.70 | 4,672.37 | 27.27 | OOM; 64 ranks logged | 264.84 / 265.39 at failure; +0.55 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b10-ga8-sac-ao-17u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-sac-ao100-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6156cd14-86f0-4e88-bfd0-d8b2bd281d93_dsv3-nebius-64g-ep64-b10-ga8-sac-ao-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_bs10` | 10 / 8 | Disabled | PASS, all 64 x 17 (reference) | 235.96 / 240.56 | 4,826.62 | 28.17 | OOM; 64 ranks logged | 263.96 / 265.11 at failure; +1.88 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b10-full-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_d9c75545-1623-45fd-bc08-9cfbeca9cb55_dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |

## 32 GPU progression

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `full_recompute_bs8` | 8 / 8 | Disabled | PASS, all 32 x 17 (reference) | 254.82 / 258.39 | 4,582.42 | 26.75 | OOM; 32 ranks logged | 265.10 / 266.31 at failure; +1.00 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b8-full-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_98f6fecd-6967-48b5-96c5-3d2354834c2c_dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_cpuoffload_bs8_ao24` | 8 / 8 | 24 | PASS, all 32 x 17 (reference) | 242.07 / 245.11 | 4,558.03 | 26.61 | OOM; 32 ranks logged | 263.80 / 264.63 at failure; +3.02 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b8-full-ao24-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b0aaf5d8-5379-4c41-aa5f-a4a8ee815bde_dsv3-fp32-32g-ep32-b8-full-ao24-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |
| `full_recompute_bs10` | 10 / 8 | Disabled | OOM | ~272.95 projected / - | - | - | OOM; 32 ranks logged | 264.54 / 265.44 at failure; +0.62 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b10-full-ga8-17u-20261027-r1-ivankobzarev) | Not yet produced |
| `full_recompute_cpuoffload_bs10_ao48` | 10 / 8 | 48 | PASS, all 32 x 17 (reference) | 253.26 / 257.96 | 4,673.54 | 27.28 | OOM; 32 ranks logged | 262.34 / 263.03 at failure; +2.47 requested | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b10-full-ao48-ga8-17u-20261027-r1-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_1d18201a-9a03-4535-b59d-0a1e8cd28034_dsv3-fp32-32g-ep32-b10-full-ao48-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |

## 64 GPU additional eager counterparts

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `native_eager_sac_bs6_det42` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 236.48 / 244.53 | 4,839.45 | 28.25 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-sac-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-sac-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-sac-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_det42` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 189.60 / 193.61 | 4,661.34 | 27.21 | PASS, all 64 x 17 | 219.10 / 225.21 | 4,421.89 | 25.81 | 1.054x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-full-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-full-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_f2de370f-8cde-41ad-a32f-684fc6e1034c_rank0_trace.json.gz) |
| `native_eager_full_recompute_cpuoffload_bs6_det42` | 6 / 8 | 80 | PASS, all 64 x 17 (reference) | 163.82 / 167.91 | 4,587.43 | 26.78 | PASS, all 64 x 17 | 172.95 / 177.81 | 4,077.67 | 23.80 | 1.125x | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-fullao-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-fullao80-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-fullao-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) / [Eager](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_cb145ed0-d3ce-4c2e-b627-a13caf5827ba_rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_gather_every` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 213.35 / 221.66 | 4,726.74 | 27.59 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-gather-every-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ge-f-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-gather-every-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_gather_first` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 240.74 / 248.53 | 4,827.56 | 28.18 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-gather-first-det42-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-gf-f-d42-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-gather-first-det42-20261026-ivankobzarev/v_0/attempt_0/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_reduce_last_unfused` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 213.33 / 223.84 | 4,822.62 | 28.15 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-full-last-u-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ge-u-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-full-last-u-20261026-ivankobzarev/v_0/attempt_1/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_reduce_last_wgrad_fused` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 213.33 / 221.68 | 4,833.03 | 28.21 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-full-last-f-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-ge-f-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-full-last-f-20261026-ivankobzarev/v_0/attempt_1/profiling/traces/iteration_16/rank0_trace.json.gz) |
| `native_eager_full_recompute_bs6_first_gather_legacy_fused` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 240.80 / 250.17 | 4,798.39 | 28.01 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-full-first-f-20261026-ivankobzarev) / [Eager](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-64g-b6-full-gf-f-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/manifold/explorer/torchtrain_datasets/tree/outputs/dsv3-night-64g-ep64-b6-full-first-f-20261026-ivankobzarev/v_0/attempt_1/profiling/traces/iteration_16/rank0_trace.json.gz) |

## 64 GPU native CUDA graph controls

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `native_cuda_graphs_64g_sac_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 236.47 / 244.55 | 4,948.86 | 28.89 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-64g-b6-sac-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_b368a8dd-9f0e-428a-bdf2-f80b28c840f0_dsv3-nebius-64g-ep64-b6-ga8-sac-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `native_cuda_graphs_64g_full_bs6` | 6 / 8 | Disabled | PASS, all 64 x 17 (reference) | 189.59 / 193.63 | 4,760.57 | 27.79 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-64g-b6-full-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6548b64d-cb41-44a5-8160-34d958e4bb50_dsv3-nebius-64g-ep64-b6-ga8-full-17u-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |
| `native_cuda_graphs_64g_full_bs10` | 10 / 8 | Disabled | PASS, all 64 x 17 (reference) | 235.96 / 240.56 | 4,826.62 | 28.17 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-64g-b10-full-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_d9c75545-1623-45fd-bc08-9cfbeca9cb55_dsv3-night-64g-ep64-b10-full64-b10-20261026-ivankobzarev__update16_rank0_compacted.json.gz) |

## 32 GPU native CUDA graph controls

| Configuration ID | BS / GA | CPU activation quota GiB | Graph outcome | Graph allocated / reserved peak GiB | Graph tokens/s/GPU | Graph MFU % | Native outcome | Native allocated / reserved peak GiB | Native tokens/s/GPU | Native MFU % | Graph / native throughput | MAST jobs | Profiles, update 16 rank 0 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `native_cuda_graphs_32g_full_bs8` | 8 / 8 | Disabled | PASS, all 32 x 17 (reference) | 254.82 / 258.39 | 4,582.42 | 26.75 | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-32g-b8-full-ga8-17u-20261027-ivankobzarev) | [Graph](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_98f6fecd-6967-48b5-96c5-3d2354834c2c_dsv3-fp32-32g-ep32-b8-full-17u-20261027-ivankobzarev__update16_rank0_compacted.json.gz) |
| `native_cuda_graphs_32g_full_bs10` | 10 / 8 | Disabled | OOM | - | - | - | Held for experiment priority | - | - | - | - | [Graph](https://www.internalfb.com/msl/studio/runs/mast/dsv3-fp32-32g-ep32-b10-full-17u-20261027-ivankobzarev) / [Native CG](https://www.internalfb.com/msl/studio/runs/mast/dsv3-nativecg-32g-b10-full-ga8-17u-20261027-ivankobzarev) | Not yet produced |

## Smaller-batch native 32 GPU baselines

These supplement the matched presentation tables with smaller-batch native training baselines. They retain FP32 master weights and gradients on GPU, FP32 Adam moments on CPU, and GA8. Their batch geometry differs from the presentation runs, so matched throughput ratios are omitted.

| ID | BS / GA | Global batch | CPU activation quota GiB/rank | Native outcome | Allocated / reserved peak GiB | Tokens/s/GPU | MFU % | MAST job | Profile, update 16 rank 0 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `native_fit_full_recompute_bs6` | 6 / 8 | 1536 | Disabled | Held for experiment priority | - | - | - | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b6-full-ga8-17u-20261027-fit1-ivankobzarev) | Not yet produced |
| `native_fit_full_recompute_cpuoffload_bs6_ao24` | 6 / 8 | 1536 | 24 | Held for experiment priority | - | - | - | [Job](https://www.internalfb.com/msl/studio/runs/mast/dsv3-eager-32g-b6-full-ao24-ga8-17u-20261027-fit1-ivankobzarev) | Not yet produced |

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
| `native_eager_full_recompute_bs6_det42` | 64 | 45.02 | 64 | 532.81 |
| `native_eager_full_recompute_cpuoffload_bs6_det42` | 64 | 46.54 | 64 | 542.62 |

## Native activation offload actually used

Counters cover all 17 updates. Peak live bytes and copy count are maximums over ranks; host-pressure skips are summed over ranks. A zero remaining-live count verifies that all activation payloads were released.

| ID | Quota GiB/rank | Peak live CPU GiB/rank | Maximum copies/rank | Host-pressure skips, all ranks | Maximum remaining live bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `native_eager_full_recompute_cpuoffload_bs6_det42` | 80 | 48.74 | 48824 | 0 | 0 |

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

## Native failure locations

- `64g_sac_bs6`: Native forward in MTP transformer attention projection, torch.nn.functional.scaled_mm; no optimizer update completed. All reported memory values are at the failed allocation, not a successful training peak or an exact capacity forecast.
- `64g_full_bs10`: Second optimizer update forward in native fused MLA key/RoPE preparation (_fused_mla_k_rope_op); one warmup update completed. All reported memory values are at the failed allocation, not a successful training peak or an exact capacity forecast.
- `64g_sac-ao_bs10_ao100`: Native forward in attention projection (torch.nn.functional.scaled_mm); no optimizer update completed with configured 100 GiB/rank CPU activation quota. All reported memory values are at the failed allocation, not a successful training peak or an exact capacity forecast.
  No after-update sidecar was emitted because the first update failed; actual copied bytes are unavailable for this attempt.
- `64g_noac_bs6`: Native forward in fused MLA key/RoPE preparation (_fused_mla_k_rope_op); no optimizer update completed. All reported memory values are at the failed allocation, not a successful training peak or an exact capacity forecast.
- `64g_sac_bs10`: Native forward in fused MLA key/RoPE preparation (_fused_mla_k_rope_op); no optimizer update completed. All reported memory values are at the failed allocation, not a successful training peak or an exact capacity forecast.
- `32g_full_bs8`: Native forward in ReportDenseAttention scaled-dot-product attention under Full remat; no optimizer update completed. All reported memory values are at the failed allocation, not a successful training peak or an exact capacity forecast.
- `32g_full-ao_bs8_ao24`: Native cuDNN scaled-dot-product attention backward (_cudnn_attention_backward) with configured 24 GiB/rank CPU activation quota; no optimizer update completed. All reported memory values are at the failed allocation, not a successful training peak or an exact capacity forecast.
  No after-update sidecar was emitted because the first update failed; actual copied bytes are unavailable for this attempt.
- `32g_full_bs10`: Forward MXFP8 quantization. All reported memory values are at the failed allocation, not a successful training peak or an exact capacity forecast.
- `32g_full-ao_bs10_ao48`: Forward native chunked cross-entropy. All reported memory values are at the failed allocation, not a successful training peak or an exact capacity forecast.
- `64g_sac-ao_bs10_ao60`: Forward fused MLA key/RoPE. All reported memory values are at the failed allocation, not a successful training peak or an exact capacity forecast.
- `64g_sac-ao_bs10_ao80`: Forward MTP MXFP8 attention projection (scaled_mm). All reported memory values are at the failed allocation, not a successful training peak or an exact capacity forecast.

## Cross-engine scalar observations

The completed Full recompute BS6 performance pair uses seed 14536 with deterministic algorithms disabled. Its globally aggregated rank-0 scalar histories are not bitwise equivalent. The first loss agrees within 0.000001, while gradients and subsequent losses differ. These observations do not establish a root cause or convergence.

| Scalar | Graph update 1 / 17 | Native update 1 / 17 | Largest absolute difference over 17 updates |
| --- | ---: | ---: | ---: |
| `global_avg_loss` | 8.406084 / 6.556857 | 8.406085 / 6.558824 | 0.042672 at update 12 |
| `global_max_loss` | 8.568663 / 6.714091 | 8.568663 / 6.715082 | 0.039763 at update 12 |
| `grad_norm` | 45.760014 / 1.684176 | 45.345272 / 1.677615 | 9.343822 at update 13 |

Strict seed-42 Full recompute and Full + offload counterparts are prioritized after the main 32 GPU comparisons and budget sweep. The local MTP annotation and native offload qualification still match bitwise against their native controls; that does not establish equivalence across native and graph engines.

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

| Native pair | Ranks x updates checked | All scalar values bitwise equal | Maximum average-loss / gradient-norm difference |
| --- | ---: | --- | ---: |
| `strict_64g_full_bs6` versus `strict_64g_full-ao_bs6_ao80` | 64 x 17 | True | 0.0000000 / 0.0000000 |

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
