# Matched 64-GPU parameter-gather scheduling

Both native jobs are COMPLETE. Each has 64 ranks x 17 finite scalar/update records; public global-average loss, global-maximum loss and gradient norm are exactly equal within each job and across all 17 matched updates.

| Parameter unshard | Attempt | Max-rank mean update (s) | Tokens/s/GPU | Training allocated (GiB) | Training reserved (GiB) | Rank0 profile16 AG / RS kernel slices |
|---|---:|---:|---:|---:|---:|---:|
| every_microbatch | 0 | 41.59480784 | 4726.743799 | 213.346777 | 221.656250 | 1024 / 192 |
| first_microbatch | 0 | 40.72619280 | 4827.556579 | 240.744277 | 248.529297 | 64 / 192 |

## Allocator counter breakdown

| Update | Every allocated / reserved (GiB) | First allocated / reserved (GiB) |
|---|---:|---:|
| 1 | 198.618000 / 200.728516 | 226.994932 / 229.162109 |
| 2 | 213.346777 / 221.652344 | 240.744277 / 248.525391 |
| 3 | 146.854381 / 221.656250 | 146.854381 / 248.529297 |
| 17 (same reservation as 3-16) | 146.854381 / 221.656250 | 146.854381 / 248.529297 |

Both recorded allocated maxima appear at update 2 (first capture); reserved memory is nearly final at 2, rises by 4 MiB at 3, and stays fixed through 17. This is the training counter scope, including capture. The combined First FakeTensor forecast 199.189 GiB did not establish the real capture/runtime ownership represented by the recorded 240.744 GiB allocated peak and 248.529 GiB reservation. The receipts do not identify the storage owner responsible for that difference, and no real memory-benefit claim follows from the forecast.

## Matched setup and observed difference

Source `2eb747d1b479616880b27e21c068578d4cb333e6`, immutable package `torchtitan_workspace:00ebfb8073c842a9848e8efd2e2ecbab`, pinned Torch `149ae98d98db6921d66e7ba527b395294a127b6e`. The fetched native-environment recipe comparison covers 11,022 leaves; only `/compile/spmd_gradient_accumulation/fsdp_param_unshard_mode` differs.

64 GB300 GPUs, FSDP64/EP64, expert FSDP1, BS6/GA8 (GBS 3072), sequence 4096, Full recomputation, balanced router, GPU FP32 parameters/gradients, GPU BF16 Adam moments, native Mega scratch 1/activation-slot-factor 0, loss in 8 chunks, last-microbatch gradient reduction, WGrad fusion enabled, seed 42 and strict deterministic algorithms. Both jobs use the same allocator and data adapter. Updates 1-5 warm up, 6-15 provide ten measured optimizer updates, 16 is profiled, and 17 snapshots memory.

First versus Every: observed update-time change -2.0883%, throughput change +2.1328%, and training reservation change +26.873047 GiB. These are two single trials, with no uncertainty interval or maximum-speed claim.

Both arms have exactly two positive lowered accumulation-graph records on each of 64 workers: MM=1, MXFP8=434, DistMoE=59. FIRST initializes accumulators; REPEAT and LAST apply the producer fusion. MXFP8 fusion can change rounding relative to unfused arithmetic; here fusion settings match between arms.

## Trace interpretation

The original rank0/update16 traces confirm Every has 128 NCCL all-gather kernel slices per microbatch (1024 total), whereas First has 64 in microbatch 0 and zero in microbatches 1-7. Both have 16 reduce-scatter slices in each early microbatch and 80 in the last (192 total). Saved fully fake lowered action graphs corroborate the placement, including required retained FSDP boundaries. First retains parameter values for later microbatches without the old source 145 backward gather recomputation.

Original traces are preserved read-only and hashed; compacted copies only change display lanes and are for Perfetto reading. Kernel-slice durations and interval intersections are rank-local profile observations, not the measured benchmark, communication bytes, independently decoded logical messages, whole-world critical path, or synchronized speed limit. Profile 16 is separate from measured 6-15.

## Measurement scope

Memory is the maximum recorded allocator peak across training updates 1-17 and all 64 ranks. CUDA graph replay does not replay allocator peak-counter updates, so the measured-update allocated counter is not a replay activation-memory peak. Maximum training reserved memory includes graph pools; initialization/current checkpoint counters are separately recorded. NVML, process RSS, host availability and pinned allocator values are point observations outside timers, with raw records and explicit scopes in each verification JSON.

Exact JSON logged scalar equality is whole-model scalar evidence for these 17 updates. It does not establish all tensor equality or convergence. The independent local repair qualification covers exact small-model gradients and capture/replay and is linked separately.

## Evidence

- [Exact matched results and scopes](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/mast-monitor/matched-results.json)
- [Exact CSV](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/mast-monitor/matched-results.csv)
- [Per-update max-rank CSV](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/mast-monitor/measured-updates-max-rank.csv)
- [All-rank update1/2/3-17 allocator breakdown](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/mast-monitor/capture-memory-breakdown.json)
- [All 17 scalar comparison](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/mast-monitor/matched-scalar-comparison.json)
- [Matched recipe binding](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/matched-config-verification.json)
- [Source repair qualification](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/QUALIFICATION.md)
- [Preserved trace summaries and compaction](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/summary.json)
- [Actual rank0 gather placement](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/placement-verification.json)

- [gather-every-det42](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-gather-every-det42-20261026-ivankobzarev) / [local 64-rank verification](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/mast-monitor/dsv3-night-64g-ep64-b6-gather-every-det42-20261026-ivankobzarev/verified-results-attempt0.json)
- [gather-first-det42](https://www.internalfb.com/msl/studio/runs/mast/dsv3-night-64g-ep64-b6-gather-first-det42-20261026-ivankobzarev) / [local 64-rank verification](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/mast-monitor/dsv3-night-64g-ep64-b6-gather-first-det42-20261026-ivankobzarev/verified-results-attempt0.json)

## Related 64 GPU reports

- [Memory progression and complete job/profile index](dsv3_64gpu_progression_report_20261027.md)
- [Scheduling profiles and Perfetto traces](dsv3_64gpu_profiles_report_20261027.md)

[Archived source report](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/mast-monitor/matched-results.md). Local artifact links refer to the preserved files on the experiment host. MAST and Manifold links provide web access to jobs and profiles.
