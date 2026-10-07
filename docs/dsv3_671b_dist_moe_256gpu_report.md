# DeepSeek V3 671B DistMoE 256-GPU Report

This report retains only successful 256-GPU jobs. It compares eager and
GraphTrainer for Chien-Chin PP1 and Sanket PP2/VPP8, with representative no-MTP
and MTP1 results. The final-geometry Sanket MTP1 rerun is still initializing,
so its provisional jobs and values are not included.

## Performance results

Each value is an average over every available post-warmup measurement that does
not overlap profiler warmup, recording, trace export, memory-snapshot export,
or scheduled garbage collection. A ten-step log represents the entire preceding
ten-step interval; it is not a measurement of only the named step. The report
therefore gives one aggregate value per run and no individual interval values.

Projected MFU is average logged TFLOP/s/GPU divided by the 2.5 PFLOP/s GB300
BF16 peak. Step time is tokens per global optimizer step divided by aggregate
throughput. Every profile link was published with `share_trace.py`; compacted
traces are for visualization, while their corresponding raw traces remain the
measurement evidence.

| Configuration | Eager | GraphTrainer | GraphTrainer versus eager |
| --- | --- | --- | --- |
| Chien-Chin PP1, no MTP | [MAST](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-baseline-eager-chien-r3-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_5c56f7b5-3007-4a15-9615-208c083cd75c_baseline_chien_r3_rank0_trace_compacted.json.gz); **5,822.00 tokens/s/GPU**, 265.08 GiB, 65.47% projected MFU, 11.257 s/step | [MAST](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-test-gt-chien-parent-bucket-r1-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_368bf727-5f97-4b15-a6d5-153469dbca91_gt_chien_parent_bucket_r1_rank0_trace_compacted.json.gz); **5,830.50 tokens/s/GPU**, 249.38 GiB, 65.57% projected MFU, 11.240 s/step | +0.15% throughput, -0.15% step time, -15.70 GiB |
| Chien-Chin PP1, MTP1 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/signal2-profile-eager-cc-mtp1-r1-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_c8a4cd41-ec4c-4909-b8f9-b7700b75cd68_rank0_trace_compacted.json.gz); **5,207.25 tokens/s/GPU**, 246.15 GiB, 60.79% projected MFU, 12.586 s/step | [MAST](https://www.internalfb.com/msl/studio/runs/mast/signal2-profile-gt-cc-mtp1-r1-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_f66aae36-718e-4fbf-a5f0-3c93c34316b6_rank0_trace_compacted.json.gz); **5,328.75 tokens/s/GPU**, 229.49 GiB, 62.21% projected MFU, 12.299 s/step | +2.33% throughput, -2.28% step time, -16.66 GiB |
| Sanket PP2/VPP8, no MTP | [MAST](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-sanket-max8-expand-r16-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_8fd49cb9-347c-4ae4-90ea-21ab812dd7b6_pp_traces_merged_compacted.json.gz); **5,583.75 tokens/s/GPU**, 263.25 GiB, 62.80% projected MFU, 11.737 s/step | [MAST](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-sanket-gt-parent-r5-v5-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e7d55a90-f347-479c-a49f-4cb3e25d30e6_pp_traces_merged_compacted.json.gz); **5,545.61 tokens/s/GPU**, 252.10 GiB, 62.37% projected MFU, 11.818 s/step | -0.68% throughput, +0.69% step time, -11.15 GiB; independent software stacks, not a controlled pair |
| Sanket PP2/VPP8, MTP1 representative | [MAST](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-p0-eager-sanket-mtp1-v11-long-r3-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e7ad696d-29ac-4d13-986c-44be09a0ba1e_pp_traces_merged_compacted.json.gz); **5,548.44 tokens/s/GPU**, 238.30 GiB, 64.77% projected MFU, 11.812 s/step | [MAST](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-p0-gt-sanket-mtp1-v11-long-r3-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_0470211b-59f7-456f-9ec8-332480921e65_pp_traces_merged_compacted.json.gz); **5,517.33 tokens/s/GPU**, 231.17 GiB, 64.41% projected MFU, 11.878 s/step | -0.56% throughput, +0.56% step time, -7.13 GiB; profiles are separate same-workload captures |

The PP1 no-MTP and MTP1 rows use all four clean ten-step intervals, covering 40
clean training steps per run. The PP2 no-MTP row uses all 28 individually logged
clean steps after ten warmup steps and before the three-step profiler cycle. The
representative PP2 MTP1 row uses all nine clean ten-step intervals after warmup;
the two instrumentation/garbage-collection intervals are excluded.

### Performance analysis

GraphTrainer matches PP1 no-MTP throughput and improves PP1 MTP1 throughput by
2.33%, while reserving 15.70-16.66 GiB less memory. The PP1 MTP1 raw traces
show why: the GPU span decreases from 12,679.758 ms to 12,412.663 ms while
non-barrier compute remains effectively unchanged (10,848.406 versus
10,855.580 ms). NCCL/compute overlap rises from 38.92% to 62.63%, exposed NCCL
falls from 508.589 to 329.126 ms, barrier time falls from 1,085.486 to
1,014.775 ms, and GPU idle gaps fall from 227.850 to 205.950 ms. The gain comes
from scheduling and communication overlap, not faster model math. The remaining
PP1 opportunity is the 329 ms of exposed NCCL and 206 ms of GPU idle time.

The PP2 no-MTP delta is not attributable to GraphTrainer because those two jobs
used different runtime packages and both contain cluster stalls; the row is
representative performance, not a controlled comparison. The archived PP2
MTP1 pair is controlled and leaves a 0.56% GraphTrainer gap. Its 32-microbatch
geometry is not the final Sanket 120-microbatch objective. The next useful PP2
performance conclusion must come from the in-flight combined
performance-and-profile pair on the final source and geometry.

Potential performance work should remain generic: reduce exposed PP collectives
and pipeline bubbles, then verify that shared `lm_head` fan-in and deferred FSDP
reductions do not lengthen the last virtual stage. No additional optimization
should be accepted from throughput alone without the deterministic numerics
gate described below.

## Numerics validation

The 256-GPU jobs disable TensorBoard and print rounded values, so they can prove
only finite training health, not bitwise equality. Differences below compare
the same clean measurement set used for performance. They are diagnostics, not
full-precision parity evidence.

| Configuration | Eager health | GraphTrainer health | Eager versus GraphTrainer bitwise status and observed difference |
| --- | --- | --- | --- |
| Chien-Chin PP1, no MTP | Finite loss and grad norm through completion | Finite loss and grad norm through completion | **Not measured bitwise.** Rounded clean means: loss 3.150270 versus 2.992180 (-5.02%); grad norm 12.938550 versus 8.054675 (-37.75%). The performance jobs were not a deterministic parity gate. |
| Chien-Chin PP1, MTP1 | Finite loss and grad norm through completion | Finite loss and grad norm through completion | **Not measured bitwise at 256 GPUs.** Rounded clean means: loss 3.327117 versus 3.306025 (-0.63%); grad norm 10.267900 versus 9.034025 (-12.02%). Maximum pointwise relative differences are 1.26% and 29.93%. |
| Sanket PP2/VPP8, no MTP | Finite final-stage loss and all reported grad norms | Finite final-stage loss and all reported grad norms | **Not comparable bitwise.** The independent packages give rounded final-stage clean means of 4.453365 versus 3.530156 loss (-20.73%) and 20.541204 versus 12.146968 grad norm (-40.87%); this is not controlled numerics evidence. |
| Sanket PP2/VPP8, MTP1 representative | Finite final-stage loss and all reported grad norms | Finite final-stage loss and all reported grad norms | **Not measured bitwise at 256 GPUs.** Rounded final-stage clean means: loss 3.289473 versus 3.229851 (-1.81%); grad norm 6.256367 versus 6.232422 (-0.38%). |

Local deterministic checks provide stronger evidence for the current fixes.
With seed 42 and deterministic execution, PP1 DP4/EP2 GA2 and GA4 matched eager
and GraphTrainer loss, full-precision grad norm, and all 340 parameter-gradient
hashes on every rank. The singleton-FSDP reduce-dtype fix also matched PP1
DP4/EP4 GA4 and PP2/VPP8 DP2/EP2 for two optimizer steps, including 166 hashes
on the first pipeline half and 175 on the second. These exact checks were made
on the pre-rebase source; the final-source and packaged-runtime paired gates
remain required before claiming final bitwise parity.

The 256-GPU rounded gaps can come from data/order differences that a performance
recipe does not control, but they cannot be dismissed. The required next check
is the tracked deterministic paired harness with identical input and initial
state, full-precision TensorBoard loss and grad norm, and per-parameter gradient
hashes. If that fails, bisect the first divergent microbatch before changing
scheduling or arithmetic.

## Changes and reproducibility

The landable stack is based directly on `origin/main` at
`948d65c868c5fa8f0290bcf9e54b69f004721f54`. It contains generic fixes for
repeated FSDP gradient fan-in, MXFP8 shard-order preservation, auxiliary-module
FSDP policy, DistMoE WGrad dtype and external-state restoration, exclusive
outer CUDA-graph ownership, pipeline runtime state, MTP pipeline integration,
in-place WGrad accumulation, shared-parameter fan-in, deterministic reduction
ordering, parent-scope expert bucketing, asynchronous unshard lookahead,
pipeline loss ordering, and singleton-FSDP reduce-dtype preservation. Benchmark
recipes, launch packaging, profiling, and this document remain isolated in
`[not-for-land]` commits.

The final packaged PP1 MTP1 result is pinned to runtime source
[`8f8337e566c0b32822b9bd2fcd22efff859ad72d`](https://github.com/IvanKobzarev/torchtitan/commit/8f8337e566c0b32822b9bd2fcd22efff859ad72d),
runtime fbpkg
`torchtitan_conda_dist_moe_256gpu_sm103_final:2f07dc452cd7453b984d0e3352221320`,
launcher fbpkg
`torchtitan_muse_spark_launcher_dist_moe_mtp1_signal_20261007:fc43bf7a58b64c64bd077b284285f7dc`,
and checkpoint-agent fbpkg
`checkpoint_agent:0d3fa4116fd446ea931c0699c748dedb`. It uses PyTorch
`8f709e9258aff0d5f3e74b40b7631ed30fe1c6ca`, DistMoE
`18b4f4887ab9a97e35193d0921cff51a249202ee`, and torchao
`a701b6a6058720c21f95908b7ae4a24bf0cae1b6`.

Use the [reproduction branch](https://github.com/pytorch/torchtitan/tree/dist-moe-256gpu-repro-final-20261007-v6),
the [complete diff](https://github.com/pytorch/torchtitan/compare/main...dist-moe-256gpu-repro-final-20261007-v6),
and the [runbook](dsv3_671b_dist_moe_256gpu_runbook.md) for exact package
verification, local correctness gates, MAST submission, measurement exclusion,
trace retrieval, compaction, and sharing commands. Package and trace links have
a 28-day lifetime and must be republished after expiry.

## Comparing configurations and launch parameters

This section is intentionally last so every report ends with the exact launch
contract used to interpret its results.

Common 256-GPU launch:

- 64 workers with four NVIDIA GB300 GPUs each; LCO single-region placement;
  Normal/Regular priority; zero task retries; `OFFLINE_TRAINING`; tenant
  `gen_ai/msl/fair_research/fair_prod/Alignement/MuseSpark_1_2_Safety_DCT`.
- DeepSeek V3 671B; sequence length 4,096; TP1 and CP1; 256 routed experts with
  top-k 8; deterministic round-robin routing; BF16 parameters and reductions;
  MXFP8 routed experts.
- No activation checkpointing; `fsdp_reshard_after_forward="never"`; one outer
  full-training-step CUDA graph. Eager owns DistMoE in-place WGrad accumulation
  and deferred FSDP reduction; GraphTrainer owns WGrad accumulation, extracted
  reduction graphs, parent-scope buckets, and asynchronous overlap. GraphTrainer
  standalone CUDA-graph passes are disabled while the outer graph is enabled.
- Fused AdamW with BF16 moments, learning rate 2.2e-4, betas (0.9, 0.95), weight
  decay 0.1, and gradient clipping at 1.0. Metrics are logged every ten steps
  except the archived PP2 no-MTP run, which logs each step.

Chien-Chin PP1:

- PP1, DP256, EP64, dense FSDP256 and routed-expert FSDP4; 16 effective
  microbatches; 4,096 global sequences and 16,777,216 tokens per optimizer
  step; seed 42.
- Committed C4 test JSON, committed test tokenizer, no shuffle, and repeat
  enabled. The runs complete 60 optimizer steps.
- No-MTP omits the auxiliary decoder depth. MTP1 adds one full-vocabulary MTP
  depth with loss scale 0.1 and one extra target depth. Sequence-wise auxiliary
  loss coefficient 0.01 applies to all 58 main routed layers and the routed MTP
  layer. PP1 profiles one active iteration at iteration 41 and records its
  memory snapshot there.

Sanket PP2/VPP8:

- PP2 with VPP8 and Interleaved 1F1B; DP128, EP64, dense FSDP128 and
  routed-expert FSDP2; TP1 and CP1; the asymmetric 4/4/.../4/1 layer split;
  maximum 16 outstanding sends, eight active unsharded stages, and automatic
  unshard lookahead.
- The archived no-MTP and representative MTP1 rows use 32 microbatches, 4,096
  global sequences, 16,777,216 tokens per optimizer step, and seed 42. They use
  the `/mnt/mffuse/c4` streaming mirror and the packaged DeepSeek V3.1 tokenizer.
  Archived MTP1 adds one full-vocabulary MTP depth and loss scale 0.1; its
  performance run completes 120 optimizer steps.
- The final Sanket MTP1 comparison changes to 120 microbatches, 15,360 global
  sequences, 62,914,560 tokens per optimizer step, seed 14,536, and 60 steps.
  It applies sequence-wise auxiliary loss coefficient 0.01 to 59 routed depths,
  profiles with three warmup iterations followed by two active iterations, and
  records the memory snapshot during that cycle. Its eager and GraphTrainer
  jobs are still initializing, so this report does not mix their incomplete
  state with the successful archived representative row.

The Sanket reproduction matches topology, arithmetic, and seed where stated,
but it does not have the historical warmed checkpoint, AirStore validation
stream, or validation pass. Historical document values remain targets rather
than same-cluster baselines.
