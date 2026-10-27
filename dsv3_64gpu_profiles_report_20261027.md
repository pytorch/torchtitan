# Source2eb matched gather scheduling traces

Two original rank0/update16 profiles are preserved read-only beside compacted Perfetto copies and inspected PNG previews. Both jobs are version0/attempt0, native COMPLETE, with all64x17 scalar/update verification. Source2eb and package00eb match; only parameter unshard cadence differs. Last reduction and WGrad fusion are enabled, seed42 is strict, GPU moments are BF16.

- [Original trace summaries and provenance](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/summary.json)
- [Actual per-microbatch NCCL slice placement](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/placement-verification.json)
- [Exact profile/benchmark comparison CSV](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/comparison.csv)
- [Profile microbatch count CSV](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/microbatch-collectives.csv)
- [Existing lowered Fake action-graph corroboration](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/lowered-action-graph-audit.json)
- [Compaction and visual inspection receipt](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/qualification.json)

Every has1024 NCCL all-gather kernel slices (128 per microbatch). First has64, all in microbatch0, with zero later all-gather slices. Both have192 reduce-scatter slices (16 in each early microbatch,80 in the final one). Counts are explicit NCCL kernel slices in original traces, corroborated by already-qualified lowered graphs; no captured CPU all-gather/reduce-scatter records or message-size/sequence metadata independently establish network bytes or bucket sizes. Nested eager all-reduce wrappers are not separate replay collectives.

Profile16 is separate from timed6-15. Device duration sums and interval intersections are one-rank profile observations. They are not benchmark estimates, whole-world synchronization/critical-path measurements, communication-byte claims or maximum-speed bounds. Compaction only changes display placement; it does not supply measurement data. The generic PP:0FULL_FORWARD_BACKWARD annotation does not imply pipeline parallelism.

[Every compacted trace](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/gather-every-det42/rank0_trace_compacted.json.gz) / [preserved original](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/gather-every-det42/rank0_trace.json.gz) / [preview](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/gather-every-det42/compacted-preview.png)

[First compacted trace](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/gather-first-det42/rank0_trace_compacted.json.gz) / [preserved original](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/gather-first-det42/rank0_trace.json.gz) / [preview](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/gather-first-det42/compacted-preview.png)

## Related 64 GPU reports

- [Memory progression and complete job/profile index](dsv3_64gpu_progression_report_20261027.md)
- [Matched parameter-gather scheduling](dsv3_64gpu_scheduling_report_20261027.md)

[Archived source report](/home/ivankobzarev/tmp/dsv3-paper-experiments-20261026/overnight-20261026/gather-retention/scheduling-traces/README.md). Local artifact links refer to the preserved files on the experiment host. MAST and Manifold links provide web access to jobs and profiles.
