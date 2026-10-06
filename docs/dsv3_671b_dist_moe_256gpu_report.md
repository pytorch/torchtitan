# DeepSeek V3 671B DistMoE 256-GPU Performance Report

Date: 2026-10-06

## Executive summary

The retained results cover the Chien-Chin PP1/no-MTP configuration and both
Sanket PP2/VPP8 configurations: MTP disabled and MTP1 enabled. All results use
64 hosts with four NVIDIA GB300 GPUs per host.

| Configuration | Eager step average (tokens/s/GPU) | GraphTrainer step average (tokens/s/GPU) |
| --- | ---: | ---: |
| PP1, MTP disabled | 5,822.00 | 5,830.50 |
| PP2/VPP8, MTP disabled | 5,615.50 | 5,575.50 |
| PP2/VPP8, MTP1 | 5,548.44 | 5,517.33 |

The PP1 runs are a directly matched pair on the same cluster and software
revisions. The PP2/no-MTP runs used different software packages and are
independent successful results, not a controlled eager-versus-GraphTrainer
comparison. The PP2/MTP1 rows are matched runs on the same cluster and
software package.

The PP2/MTP1 GraphTrainer path requires coalescing the two local `lm_head`
gradient contributions before their shared FSDP reduction. This is specific to
the GraphPP shared-parameter path: PP1/MTP1 does not use this fix.

## Configuration

| Setting | Value |
| --- | --- |
| Model | DeepSeek V3 671B DistMoE |
| Precision | BF16 parameters, MXFP8 DistMoE experts |
| Sequence length | 4,096 |
| Tensor/context parallelism | TP1, CP1 |
| Tokens per optimizer step | 16,777,216 |
| FSDP lifetime | `fsdp_reshard_after_forward="never"` |
| Activation checkpointing | Disabled |
| Data | C4 test source, concat-then-split packing |
| CUDA graph | One outer graph for the complete training step |
| Hardware | 64 x 4 NVIDIA GB300, 256 GPUs total |

| Configuration | Pipeline | Microbatches | Data/expert parallelism |
| --- | --- | ---: | --- |
| Chien-Chin, MTP disabled | PP1 | 16 | FSDP256, EP64 |
| Sanket, MTP disabled | PP2/VPP8, Interleaved 1F1B | 32 | FSDP128, EP64 |
| Sanket, MTP1 | PP2/VPP8, Interleaved 1F1B | 32 | FSDP128, EP64 |

MTP1 adds one MTP layer on the final virtual stage, full-vocabulary logits, and
an MTP loss scale of 0.1. The no-MTP runs omit this extra model and loss work.

## Latest successful runs

### PP1, MTP disabled

The average excludes the profiler- and memory-snapshot-contaminated logging
interval.

| Implementation | Run | Step average (tokens/s/GPU) | Peak memory |
| --- | --- | ---: | ---: |
| Eager | [Chien-Chin r3](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-baseline-eager-chien-r3-256-ivankobzarev) | 5,822.00 | 265.08 GiB |
| GraphTrainer | [parent-bucketing r1](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-test-gt-chien-parent-bucket-r1-256-ivankobzarev) | 5,830.50 | 249.38 GiB |

The corresponding clean compute rates are 1,636.87 TFLOP/s/GPU for eager and
1,639.29 TFLOP/s/GPU for GraphTrainer. Both jobs completed 60 steps with all
64 tasks complete, no restarts, and finite loss and gradient norm. Eager moved
from loss `12.30927`, grad norm `17.1941` at step 1 to loss `2.95984`, grad
norm `14.5442` at step 60. GraphTrainer moved from loss `12.26398`, grad norm
`17.4585` to loss `2.84121`, grad norm `6.7638`.

### PP2/VPP8, MTP disabled

| Implementation | Run | Step average (tokens/s/GPU) | Peak memory |
| --- | --- | ---: | ---: |
| Eager | [r16](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-sanket-max8-expand-r16-256-ivankobzarev) | 5,615.50 | 263.25 GiB on rank 0 |
| GraphTrainer | [r5](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-sanket-gt-parent-r5-v5-256-ivankobzarev) | 5,575.50 | 252.10 GiB on rank 0 |

The GraphTrainer run had cluster stalls in later logging intervals, so the
table retains the stable uncontaminated average.

Eager r16 completed with finite loss and gradient norm: loss moved from
`12.25781` at step 1 to `3.22177` at step 41. GraphTrainer rank 0 is not the
loss stage and therefore reports the pipeline sentinel; the final stage
reported finite loss, and all reported gradient norms were finite.

### PP2/VPP8, MTP1

These are representative matched long runs. The clean means exclude startup
and the profiler/garbage-collection intervals at steps 50 and 100.

| Implementation | Representative run | Step average (tokens/s/GPU) |
| --- | --- | ---: |
| Eager | [r3](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-p0-eager-sanket-mtp1-v11-long-r3-256-ivankobzarev) | 5,548.44 |
| GraphTrainer | [r3](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-p0-gt-sanket-mtp1-v11-long-r3-256-ivankobzarev) | 5,517.33 |

Both jobs completed. The eager run averaged 5,549.00 tokens/s/GPU and 1,619.50
TFLOP/s/GPU over every logged step from step 20 onward, with 238.30 GiB peak
memory on rank 0. The GraphTrainer run averaged 5,516.73 tokens/s/GPU and
1,610.08 TFLOP/s/GPU over the same range, with 231.17 GiB peak memory on
rank 0.

The [canonical-order safety run](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-p0-gt-sanket-mtp1-canonical-rs-v15-long-r1-256-ivankobzarev) also
completed with finite training values. It validates that communication-only
gradient reductions follow the same deterministic order on every rank.

## Shared profiler traces

Raw traces are authoritative for measurement. Compacted traces preserve slice
names, timestamps, durations, arguments, and process IDs while packing the
CUDA-graph replay streams into semantic lanes. The merged views align both PP
ranks and add send/receive flow arrows. These links use the default 28-day
sharing lifetime.

### PP1, MTP disabled

PP1 has no second pipeline rank to merge, so the rank-0 compacted trace is the
recommended view.

| Profile | Raw rank 0 | Compacted rank 0 |
| --- | --- | --- |
| Eager Chien-Chin r3 | [open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_9f81cf85-45c0-4ef4-aaf0-a8cf13815154_baseline_chien_r3_rank0_trace.json.gz) | [open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_5c56f7b5-3007-4a15-9615-208c083cd75c_baseline_chien_r3_rank0_trace_compacted.json.gz) |
| GraphTrainer parent-bucketing r1 | [open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_03cf5dbd-e036-48bc-9a64-cc410808cdde_gt_chien_parent_bucket_r1_rank0_trace.json.gz) | [open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_368bf727-5f97-4b15-a6d5-153469dbca91_gt_chien_parent_bucket_r1_rank0_trace_compacted.json.gz) |

Both profiles contain one complete-step CUDA graph replay. The GraphTrainer
trace verifies that the FSDP layout is reconstructed once per optimizer step
and that routed-expert parameters use the parent-scope buckets.

### PP2/VPP8, MTP disabled

| Profile | Merged compacted | Rank 0 | Rank 128 |
| --- | --- | --- | --- |
| Eager r16 | [open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_8fd49cb9-347c-4ae4-90ea-21ab812dd7b6_pp_traces_merged_compacted.json.gz) | [raw](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_fdf19333-901e-436d-976c-346d9d531a9d_rank0_trace.json.gz) / [compacted](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_4e82653a-7fb0-4f51-85b4-fdcd4169e883_rank0_trace_compacted.json.gz) | [raw](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_5f31188d-31a5-4d96-beb4-572f5c5ee5de_rank128_trace.json.gz) / [compacted](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_25ea3a91-dadf-4b9c-8700-c1a068a82be9_rank128_trace_compacted.json.gz) |
| GraphTrainer r5 | [open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e7d55a90-f347-479c-a49f-4cb3e25d30e6_pp_traces_merged_compacted.json.gz) | [raw](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_841e9fe4-7ee6-451f-ae9e-daee6f9e60f2_rank0_trace.json.gz) / [compacted](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_473ca8cf-4691-418d-a9f3-426c2299c4fd_rank0_trace_compacted.json.gz) | [raw](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_5808c90e-b045-4873-afbb-835e04cb2f00_rank128_trace.json.gz) / [compacted](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_d6d99767-9bcd-4ce0-a42a-e1bdcf9faf5e_rank128_trace_compacted.json.gz) |

Each raw trace contains one complete-step CUDA graph replay. In GraphTrainer
r5, all 960 pipeline sends and receives in the capture window are matched.

### PP2/VPP8, MTP1

The throughput numbers above come from the v11 long runs. The following are
separate representative profile runs with the same PP2/VPP8 MTP1 workload.

| Profile | Merged compacted | Rank 0 | Rank 128 |
| --- | --- | --- | --- |
| Eager v8 | [open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e7ad696d-29ac-4d13-986c-44be09a0ba1e_pp_traces_merged_compacted.json.gz) | [raw](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_f1458a55-f90f-418e-8961-4bc0a471cd29_rank0_trace.json.gz) / [compacted](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_87383ce6-2533-4b23-92a1-32c65d9274e3_rank0_trace_compacted.json.gz) | [raw](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_c0fe8269-3776-4c3e-a905-84ee7f898c4b_rank128_trace.json.gz) / [compacted](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_79060e23-afce-48b2-a170-7231675599a3_rank128_trace_compacted.json.gz) |
| GraphTrainer canonical v16 | [open](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_0470211b-59f7-456f-9ec8-332480921e65_pp_traces_merged_compacted.json.gz) | [raw](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_8149a947-9877-421a-b4cb-593f263dc628_rank0_trace.json.gz) / [compacted](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_aeed6661-8ea1-4ea9-86b6-060b360886c7_rank0_trace_compacted.json.gz) | [raw](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_ac0b89ba-2c6e-4d63-b090-f43ef9bc97ac_rank128_trace.json.gz) / [compacted](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_fcb6be52-8451-48f4-ae58-047b08f9c8b5_rank128_trace_compacted.json.gz) |

The canonical GraphTrainer profile contains two complete-step replays on each
rank. Compaction maps 1,019 replay streams on rank 0 and 1,016 on rank 128 to
10 semantic lanes each. The merged window matches 2,054 pipeline send/receive
pairs; its 13 unmatched endpoints are at the truncated capture boundary. All
64 FSDP unshard pairs are matched.

## Landing status

### Already landed in `origin/main`

These changes are the baseline for the reported runs and require no further
landing work:

- [#5058](https://github.com/pytorch/torchtitan/pull/5058) extracts the FSDP
  split/pad/cat layout with the final reduction instead of repeating it in
  every microbatch.
- [#5059](https://github.com/pytorch/torchtitan/pull/5059) lets WGrad
  accumulation fuse through supported view chains.
- [#5067](https://github.com/pytorch/torchtitan/pull/5067) buckets DistMoE
  routed experts at the captured parent scope.

### Remaining changes to land

The remaining required changes in this reproduction stack are:

- `565fe74ef`: define the functional DistMoE WGrad output dtype.
- `dcd09fecb` and `200d0893d`: restore DistMoE external state after graph
  metadata inference.
- `1ffe43e58`: give the outer full-step CUDA graph exclusive ownership of
  capture.
- `9145b841c` and `2fa2fffde`: preserve in-place WGrad accumulation semantics
  in GraphTrainer and GraphPP.
- `0a6077e3b`: support the MTP embedding/loss shared-parameter topology.
- `3b1023300`: for PP2/MTP1, add the two local `lm_head` gradients first and
  perform their shared FSDP reduction once. This change is not used by PP1.
- `7388e096a`: use a canonical cross-rank order for communication-only
  reductions, avoiding rank-dependent collective order at scale.

This list intentionally excludes experimental launch support and optional
sub-percent refinements. The source branch and complete diff are available at
[the reproduction branch](https://github.com/pytorch/torchtitan/tree/dist-moe-256gpu-repro-final-20261005)
and [`main...reproduction branch`](https://github.com/pytorch/torchtitan/compare/main...dist-moe-256gpu-repro-final-20261005).
