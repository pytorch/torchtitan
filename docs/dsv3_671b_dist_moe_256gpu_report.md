# DeepSeek V3 671B DistMoE 256-GPU Report

This report contains successful 256-GPU jobs only. It compares eager and
GraphTrainer on the Chien-Chin PP1 and Sanket PP2/VPP8 configurations, including
the final MTP1 geometry, full-scale deterministic numerics gates, and profiles.
The FSDP symmetric-memory mode is stated explicitly because the legacy PP1 eager
runs used it while GraphTrainer did not.

## Performance results

For the final deterministic MTP1 runs, the aggregate uses every non-warmup step
from 21 through 60 (40 samples). No sample is removed for logging or garbage
collection. Archived runs use every available non-warmup, non-profiled sample;
the profiled interval itself is excluded. A ten-step metric represents the
preceding ten-step interval, not only the named step.

Projected MFU is logged TFLOP/s/GPU divided by the 2.5 PFLOP/s GB300 BF16 peak.
The profiler-disabled final jobs provide the performance and memory values. A
separate successful capture supplies a profile link when noted. Compacted traces
are visualization artifacts; the raw traces remain the measurement evidence.

| Configuration and effective FSDP symmetric-memory mode | Eager baseline | GraphTrainer test | GraphTrainer versus eager |
| --- | --- | --- | --- |
| Chien-Chin PP1, no MTP, legacy mixed mode: eager **with FSDP symmetric memory**; GraphTrainer **no FSDP symmetric memory** | [MAST](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-baseline-eager-chien-r3-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_5c56f7b5-3007-4a15-9615-208c083cd75c_baseline_chien_r3_rank0_trace_compacted.json.gz); **5,822.00 tokens/s/GPU**, 265.08 GiB, 65.47% projected MFU, 11.257 s/step | [MAST](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-test-gt-chien-parent-bucket-r1-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_368bf727-5f97-4b15-a6d5-153469dbca91_gt_chien_parent_bucket_r1_rank0_trace_compacted.json.gz); **5,830.50 tokens/s/GPU**, 249.38 GiB, 65.57% projected MFU, 11.240 s/step | **+0.15% throughput**, -0.15% step time, -15.70 GiB |
| Chien-Chin PP1, MTP1, legacy mixed mode: eager **with FSDP symmetric memory**; GraphTrainer **no FSDP symmetric memory** | [MAST](https://www.internalfb.com/msl/studio/runs/mast/signal2-profile-eager-cc-mtp1-r1-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_c8a4cd41-ec4c-4909-b8f9-b7700b75cd68_rank0_trace_compacted.json.gz); **5,207.25 tokens/s/GPU**, 246.15 GiB, 60.79% projected MFU, 12.586 s/step | [MAST](https://www.internalfb.com/msl/studio/runs/mast/signal2-profile-gt-cc-mtp1-r1-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_f66aae36-718e-4fbf-a5f0-3c93c34316b6_rank0_trace_compacted.json.gz); **5,328.75 tokens/s/GPU**, 229.49 GiB, 62.21% projected MFU, 12.299 s/step | **+2.33% throughput**, -2.28% step time, -16.66 GiB |
| Chien-Chin PP1, MTP1, final matched mode: eager **no FSDP symmetric memory**; GraphTrainer **no FSDP symmetric memory** | [performance](https://www.internalfb.com/msl/studio/runs/mast/final-pr5118-pp1-e-r2-256-ivankobzarev) / [profile capture](https://www.internalfb.com/msl/studio/runs/mast/final-nosymm-prof-pp1-e-r1-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_2c30af45-c018-4705-83e2-ed8a23d54ef7_rank0_trace_compacted.json.gz); **5,118.65 +/- 4.85 tokens/s/GPU**, 244.28 GiB, 59.76% projected MFU, 12.803 s/step | [performance](https://www.internalfb.com/msl/studio/runs/mast/final-pr5118-pp1-gt-r2-256-ivankobzarev) / [profile capture](https://www.internalfb.com/msl/studio/runs/mast/final-nosymm-prof-pp1-gt-r1-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e1525ccf-25d3-4df9-a33e-c8a18e283f95_rank0_trace_compacted.json.gz); **5,242.64 +/- 12.41 tokens/s/GPU**, 229.49 GiB, 61.20% projected MFU, 12.501 s/step | **+2.42% throughput**, -2.36% step time, -14.79 GiB |
| Sanket PP2/VPP8, no MTP: eager **no FSDP symmetric memory**; GraphTrainer **no FSDP symmetric memory** | [MAST](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-sanket-max8-expand-r16-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_8fd49cb9-347c-4ae4-90ea-21ab812dd7b6_pp_traces_merged_compacted.json.gz); **5,583.75 tokens/s/GPU**, 263.25 GiB, 62.80% projected MFU, 11.737 s/step | [MAST](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-sanket-gt-parent-r5-v5-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_e7d55a90-f347-479c-a49f-4cb3e25d30e6_pp_traces_merged_compacted.json.gz); **5,545.61 tokens/s/GPU**, 252.10 GiB, 62.37% projected MFU, 11.818 s/step | -0.68% throughput, +0.69% step time, -11.15 GiB; independent software stacks, not a controlled pair |
| Sanket PP2/VPP8, MTP1 final 120-microbatch geometry: eager **no FSDP symmetric memory**; GraphTrainer **no FSDP symmetric memory** | [performance](https://www.internalfb.com/msl/studio/runs/mast/final-pr5118-pp2-e-r2-256-ivankobzarev) / [profile capture](https://www.internalfb.com/msl/studio/runs/mast/profilefix-eager-sanket-mtp1-r1-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_6d152d10-82b5-4e9d-a1cb-8f3ad87f0c01_pp_traces_merged_compacted.json.gz); **5,498.69 +/- 140.91 tokens/s/GPU**, 238.38 GiB, 64.19% projected MFU, 44.725 s/step | [performance](https://www.internalfb.com/msl/studio/runs/mast/final-pr5118-pp2-gt-r2-256-ivankobzarev) / [profile capture](https://www.internalfb.com/msl/studio/runs/mast/profilefix-gt-sanket-mtp1-r1-256-ivankobzarev) / [profile](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_1dd89952-21e2-47d0-8126-7099757d2e81_pp_traces_merged_compacted.json.gz); **5,528.74 +/- 125.27 tokens/s/GPU**, 217.26 GiB, 64.54% projected MFU, 44.475 s/step | **+0.55% throughput**, -0.56% step time, -21.13 GiB |

### Performance analysis

The final matched PP1 MTP1 pair is the primary controlled result. GraphTrainer
is 2.42% faster and reserves 14.79 GiB less memory. The legacy mixed-mode pair
was 2.33% faster, so moving from a symmetric-memory eager counterpart to the
matched no-symmetric-memory comparison changes the relative GraphTrainer
advantage by only +0.09 percentage points. In absolute terms, the final eager
result is 1.70% below the legacy eager result, while final GraphTrainer is 1.62%
below legacy GraphTrainer. This is not a pure symmetric-memory A/B: the source
stack, runtime, deterministic metrics cadence, and capture mode also changed.
It therefore does not support attributing the 1.6-1.7% absolute movement to
symmetric memory alone.

The legacy PP1 MTP1 traces explain the GraphTrainer advantage. GPU span falls
from 12,679.758 ms to 12,412.663 ms while non-barrier compute is essentially
unchanged (10,848.406 versus 10,855.580 ms). NCCL/compute overlap increases
from 38.92% to 62.63%; exposed NCCL falls from 508.589 to 329.126 ms; barrier
time falls from 1,085.486 to 1,014.775 ms; and GPU idle gaps fall from 227.850
to 205.950 ms. The gain comes from communication scheduling and overlap, not
faster model math. In the final matched no-symmetric-memory captures, the raw
rank-0 GPU `ProfilerStep` span falls from 12,724.330 ms to 12,335.759 ms
(-388.571 ms, -3.05%). This single captured step is directionally consistent
with the +2.42% aggregate throughput result; it is not used to replace the
40-sample performance aggregate.

The final PP2/VPP8 MTP1 pair is also controlled: GraphTrainer is 0.55% faster
and reserves 21.13 GiB less memory. Its relatively large per-step standard
deviation affects both sides and comes from including every non-warmup,
non-profiled step as required, including cluster and logging stalls. The merged
PP traces use the final 120-microbatch geometry. Their compaction matched 3,600
send/receive flows on each side with no unmatched flows, making the pipeline
relationship readable without changing measurement evidence. The captures use
the matching workload geometry but an earlier runtime package; throughput and
numerics claims come only from the final `f7b71b57` performance jobs.

The archived PP2 no-MTP row remains context rather than a controlled result:
its eager and GraphTrainer jobs used different runtime packages. No causal
GraphTrainer conclusion should be drawn from its -0.68% delta.

## Numerics validation

The final PP1 and PP2 MTP1 jobs enable deterministic execution with their
matched recipe seeds (42 for PP1 and 14,536 for PP2) and record full-precision
TensorBoard metrics every step. The comparison covers all 60 optimizer steps,
including warmup, because numerical equality must hold for the full trajectory.

| Configuration | Eager | GraphTrainer | Bitwise result |
| --- | --- | --- | --- |
| Chien-Chin PP1 MTP1, both **no FSDP symmetric memory** | [MAST](https://www.internalfb.com/msl/studio/runs/mast/final-pr5118-pp1-e-r2-256-ivankobzarev); finite through step 60 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/final-pr5118-pp1-gt-r2-256-ivankobzarev); finite through step 60 | **Exact 60/60:** `global_avg_loss`, `global_max_loss`, `grad_norm`, and `n_tokens_seen` |
| Sanket PP2/VPP8 MTP1, both **no FSDP symmetric memory** | [MAST](https://www.internalfb.com/msl/studio/runs/mast/final-pr5118-pp2-e-r2-256-ivankobzarev); finite through step 60 | [MAST](https://www.internalfb.com/msl/studio/runs/mast/final-pr5118-pp2-gt-r2-256-ivankobzarev); finite through step 60 | **Exact 60/60:** `global_avg_loss`, `global_max_loss`, `grad_norm`, and `n_tokens_seen` |

The auxiliary observability metric
`microbatch_wise_load_balance_loss/mean` is not bitwise equal: eager records the
side-effect accumulator while GraphTrainer reports zero because graph capture
bypasses that reporting side effect. This does not enter the optimized loss;
the exact global losses and gradient norm prove equality of the training
trajectory. It remains an observability bug to fix separately.

## Changes and reproducibility

The final full-stack performance and numerics jobs use source
[`1e1d0bdd0a4150890f65658ecc205580cb385670`](https://github.com/IvanKobzarev/torchtitan/commit/1e1d0bdd0a4150890f65658ecc205580cb385670),
runtime fbpkg
`torchtitan_conda_dist_moe_256gpu_sm103_final:f7b71b57d0c149a2b9972b90a5b9e15b`,
launcher fbpkg
`torchtitan_muse_spark_launcher_dist_moe_mtp1_signal_20261007:93ed0028d9024ecaa9f55c16c9a2805e`,
PyTorch `f96a1d292d53556cc7017424fd8b070ace79d375`, DistMoE
`18b4f4887ab9a97e35193d0921cff51a249202ee`, and torchao
`a701b6a6058720c21f95908b7ae4a24bf0cae1b6`.

The landable stack is based on `origin/main` at
`948d65c868c5fa8f0290bcf9e54b69f004721f54`. Generic changes cover MTP FSDP,
coalesced FSDP gradient fan-in, DistMoE WGrad accumulation, PP gradient
accumulation and fusion, persistent-gradient casting, pipeline runtime state,
parent-scoped expert bucketing, asynchronous unshard lookahead, canonical
reduction ordering, loss reporting order, and rejection of unsupported
GraphTrainer FSDP symmetric memory. Recipes, packaging, profiling, and this
report are isolated in `[not-for-land]` commits.

Use the [reproduction branch](https://github.com/IvanKobzarev/torchtitan/tree/dist-moe-256gpu-repro-final-20261007-v7),
the [complete diff](https://github.com/IvanKobzarev/torchtitan/compare/main...dist-moe-256gpu-repro-final-20261007-v7),
and the [runbook](dsv3_671b_dist_moe_256gpu_runbook.md) for package verification,
local correctness gates, MAST submission, measurement selection, trace
retrieval, compaction, and sharing. Shared trace links have a 28-day lifetime.

## Comparing configurations and launch parameters

This section is intentionally last so the report ends with the exact launch
contract used to interpret the results.

Common 256-GPU launch:

- 64 workers with four NVIDIA GB300 GPUs each, pinned with machine subtype
  `T20_CIC_GB300_278GB_HBM3E_NVLSO_NSF_LP`; LCO single-region placement;
  Normal/Regular priority; zero task retries; `OFFLINE_TRAINING`.
- DeepSeek V3 671B; sequence length 4,096; TP1 and CP1; 256 routed experts with
  top-k 8; deterministic round-robin routing; BF16 parameters and reductions;
  MXFP8 routed experts.
- No activation checkpointing; `fsdp_reshard_after_forward="never"`; one outer
  full-training-step CUDA graph. Eager owns DistMoE in-place WGrad accumulation
  and deferred FSDP reduction; GraphTrainer owns WGrad accumulation, extracted
  reduction graphs, parent-scope buckets, and asynchronous overlap. GraphTrainer
  CUDA-graph passes are disabled while the outer graph is enabled.
- Fused AdamW with BF16 moments, learning rate 2.2e-4, betas (0.9, 0.95), weight
  decay 0.1, and gradient clipping at 1.0.

FSDP symmetric-memory modes:

- Legacy PP1 eager: **with FSDP symmetric memory**,
  `fsdp_symm_mem_scope="dense"`; dense FSDP uses symmetric memory.
- Legacy PP1 GraphTrainer: **no FSDP symmetric memory** in effect. Its recipe
  also declared `"dense"`, but the GraphTrainer SimpleFSDP path did not
  implement that mode and silently ignored it at the time.
- Final PP1 eager and GraphTrainer: matched **no FSDP symmetric memory**,
  `fsdp_symm_mem_scope=None`. The current GraphTrainer guard rejects a requested
  symmetric-memory scope instead of silently producing a mixed comparison.
- PP2 eager and GraphTrainer: **no FSDP symmetric memory**,
  `fsdp_symm_mem_scope=None`.

Chien-Chin PP1:

- PP1, DP256, EP64, dense FSDP256 and routed-expert FSDP4; 16 effective
  microbatches; 4,096 global sequences and 16,777,216 tokens per optimizer
  step; seed 42; 60 optimizer steps.
- Committed C4 test JSON, committed test tokenizer, no shuffle, and repeat
  enabled.
- No-MTP omits the auxiliary decoder depth. MTP1 adds one full-vocabulary MTP
  depth with loss scale 0.1 and one extra target depth. Sequence-wise auxiliary
  loss coefficient 0.01 applies to all 58 main routed layers and the routed MTP
  layer.
- Profile captures record one active iteration at iteration 41 and a memory
  snapshot. The final deterministic performance jobs disable profiling and log
  full-precision TensorBoard metrics every step.

Sanket PP2/VPP8:

- PP2 with VPP8 and Interleaved 1F1B; DP128, EP64, dense FSDP128 and
  routed-expert FSDP2; TP1 and CP1; asymmetric `4/4/.../4/1` layer split;
  maximum 16 outstanding sends, eight active unsharded stages, and automatic
  unshard lookahead.
- The no-MTP context row uses 32 microbatches and 4,096 global sequences. The
  final MTP1 pair uses 120 microbatches, 15,360 global sequences, 62,914,560
  tokens per optimizer step, seed 14,536, and 60 optimizer steps.
- MTP1 adds one full-vocabulary MTP depth with loss scale 0.1 and applies
  sequence-wise auxiliary loss coefficient 0.01 to 59 routed depths.
- Final performance/numerics jobs disable profiling and log full-precision
  TensorBoard metrics every step. Separate profile captures use the same final
  120-microbatch geometry and merge rank 0 with rank 128 for PP flow analysis.

The Sanket reproduction matches topology and arithmetic but does not have the
historical warmed checkpoint, AirStore validation stream, or validation pass.
Historical document values remain targets rather than same-cluster baselines.
