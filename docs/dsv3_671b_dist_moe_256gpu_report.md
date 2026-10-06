# DeepSeek V3 671B DistMoE MTP1 256-GPU Report

## Status

This report covers the final MTP1 comparison only. The required matrix is:

| Line | Topology | Implementation | Final-stack run |
| --- | --- | --- | --- |
| Baseline Chien-Chin | PP1, DP256, EP64 | eager | Pending final packaged MAST run |
| Test GraphTrainer Chien-Chin | PP1, DP256, EP64 | GraphTrainer | Pending final packaged MAST run |
| Baseline Sanket | PP2/VPP8, DP128, EP64 | eager | Pending final packaged MAST run |
| Test GraphTrainer Sanket | PP2/VPP8, DP128, EP64 | GraphTrainer | Pending final packaged MAST run |

No 256-GPU performance value or profiler link is currently attributed to the
final stack. Previous no-MTP jobs and pre-final-stack MTP1 jobs are intentionally
excluded: they do not validate the software and objective described here.
Failed or dead jobs are never included in this report.

The final report branch is
`dist-moe-256gpu-repro-final-20261006`. It is rebased directly onto
`origin/main` commit `948d65c868c5fa8f0290bcf9e54b69f004721f54` and contains
exactly 21 commits above that base: 20 individually scoped, landable commits
from `880029205` through `85de4b5b9`, followed by one top
`[not-for-land]` reproduction commit. The top commit isolates configs,
benchmarking, profiling, debugging, packaging, documentation, and the pinned
runtime compatibility revert described below.

Before packaging, that top commit is frozen as immutable runtime-source commit
S and retained at tag
`dist-moe-256gpu-repro-runtime-source-20261006-v5`. The runtime fbpkg and MAST
launcher identify and verify S. After the packaged gates and MAST runs finish,
only this report and its runbook are amended in the same single
`[not-for-land]` commit, producing report-tip commit R on the final report
branch. The immutable publication tag
`dist-moe-256gpu-repro-report-20261006-v5` will identify R without requiring a
commit to contain its own hash. S and R therefore have the same landable parent
and each has exactly one `[not-for-land]` commit above it. The immutable S hash,
the runtime and launcher fbpkg identifiers, and all final MAST and trace links
remain pending until their respective publication stages complete.

The `v5` tags are immutable retry refs. The abandoned `v1` publication plan is
not reused. The immutable `v2` source ref records a package attempt that failed
before publication because its relocated build environment could not import the
unpinned build backend. The immutable `v3` source ref records the next package
attempt, which passed that build then failed before publication because a host
CUDA 12.8 compiler was selected for a CUDA 13.0 PyTorch runtime. Neither attempt
produced a runtime package or MAST jobs. The immutable v4 source produced
runtime package
`torchtitan_conda_dist_moe_256gpu_sm103_final:2f07dc452cd7453b984d0e3352221320`,
but it was superseded before MAST submission because its recursive provenance
manifests included mutable `.pyc` files. V5 excludes only Python bytecode and
verifies every stable source and native payload after relocation. The runtime
and launcher fbpkgs expire 28 days after publication.
Rebuilds are possible only while every exact binary-input UUID in the runbook
remains fetchable. S and this report do not reconstruct the base runtime or
launcher template after those inputs expire. Without owner-preserved inputs or
separate authoritative immutable rebuild recipes, the report is historical only;
a mutable package alias is never an acceptable substitute.

## Pair-matched configuration

Both pairs use DeepSeek V3 671B, MXFP8 routed experts, one MTP depth,
full-vocabulary MTP logits, and MTP loss scale 0.1. The data loader produces one
extra target depth. All 58 main-model routed layers and the routed MTP layer use
complementary sequence-wise auxiliary loss with coefficient 0.01. TorchTitan
implements that objective with `MicrobatchWiseLoadBalanceLoss`.

The eager and GraphTrainer runs within each row pair use identical model, input,
batch, routing, and optimizer configurations. The Chien-Chin and Sanket pairs do
not use the same input source or tokenizer: Chien-Chin uses the committed test
tokenizer and `tests/assets/c4_test/data.json`, while Sanket uses the packaged
DeepSeek V3.1 tokenizer and the `/mnt/mffuse/c4` streaming mirror. Every report
recipe replaces learned top-k routing with deterministic round-robin routing.
These runs therefore compare execution performance and numerical parity; they
are not convergence reproductions of the historical training jobs.

The inherited base factory activates fused MLA and fused SwiGLU config
overrides. Config loading applies 124 overrides in every final recipe: 62 MLA
sites and 62 SwiGLU sites, including the MTP depth.

| Property | Chien-Chin | Sanket |
| --- | ---: | ---: |
| GPUs | 256 GB300 | 256 GB300 |
| Pipeline | PP1 | PP2, VPP8, Interleaved 1F1B |
| Data parallel | DP256 | DP128 |
| Expert parallel | EP64 | EP64 |
| Sequence length | 4,096 | 4,096 |
| Effective microbatches per step | 16 | 120 |
| Tokens per step | 16,777,216 | 62,914,560 |
| Seed | 42 | 14,536 |
| MTP depths | 1 | 1 |
| Routed auxiliary-loss depths | 59 | 59 |
| Full-step CUDA graph | enabled | enabled |

The eager and GraphTrainer members of each pair intentionally use different
accumulation implementations. Eager uses DistMoE in-place WGrad accumulation and
defers FSDP gradient reduction until the full local accumulation is complete.
GraphTrainer disables those eager mechanisms and owns the WGrad, accumulation,
and reduction graphs. TorchTitan owns the outer full-step CUDA graph
in both cases. Generic CUDA-graph ownership logic omits GraphTrainer's standalone
per-callable capture whenever this outer capture is enabled.

The Sanket recipes match the published PP2 geometry, arithmetic, and seed, but
do not have the historical internal warmed checkpoint, AirStore validation
stream, or validation pass. Historical results are performance targets rather
than exact reproductions or same-cluster baselines.

The exact performance recipe functions are:

- `deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance`
- `graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance`
- `deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance`
- `graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance`

Separate profile jobs use:

- `deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile`
- `graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile`
- `deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile`
- `graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile`

## Numerical validation

### PP1 internal local acceptance

The current PP1 accumulation design was checked on four GB300s with the 16B DistMoE
model, DP4/EP2 (EDP2), sequence length 512, MTP1, and deterministic seed 42.
Eager and GraphTrainer used the same inputs, parameters, and optimizer setup.
These values came from the earlier diagnostic harness and remain internal
evidence until the tracked paired gate now included in the reproduction commit
reruns them on the final source and packaged runtime.

| Case | Loss | Full-precision grad norm | Per-rank gradient SHA-256 |
| --- | --- | --- | --- |
| GA2, step 1 | exact: 13.093971252441406 | exact: 8.923309326171875 | 340/340 exact on ranks 0-3 |
| GA2, step 2 | exact: 8.990411758422852 | exact: 15.093811988830566 | 340/340 exact on ranks 0-3 |
| GA4, step 1 | exact: 13.103599548339844 | exact: 8.289326667785645 | 340/340 exact on ranks 0-3 |
| GA4, step 2 | exact: 9.43763542175293 | exact: 14.259662628173828 | 340/340 exact on ranks 0-3 |

The audit includes token embeddings, final norm, LM head, dense weights, and
routed-expert weights. Ordered native MXFP8 fan-in raises the local MXFP8 fusion
count from 168 to 170 by covering the two repeated LM-head contributions.

An earlier composed-stack check also compared outer-CUDA-graph disabled and
enabled execution for ten PP1 steps. Loss and full-precision grad norm matched
for all ten steps, gradient hashes matched for all 340 parameters at sampled
steps, and capture produced one graph followed by replay. This check must be
repeated once more on the final rewritten commit stack before publishing the
runtime artifact.

### Singleton FSDP reduce-dtype fix and exact pre-rebase proof

The EDP1 correctness gap is resolved by landable commit `85de4b5b9`
(`[graph_trainer] Preserve singleton FSDP reduce dtype`). A singleton FSDP
reduction mesh emits no collective, so GraphPP previously had no operation from
which to identify the once-per-step reduce-gradient boundary. It consequently
left the terminal BF16-to-FP32 persistent-gradient cast in every repeated
microbatch compute action. GraphTrainer then accumulated those microbatch
gradients in FP32, while eager FSDP accumulated in BF16 and cast once, causing
the routed-expert drift tracked by pytorch/torchtitan#5043.

The fix propagates the configured FSDP reduce dtype, resolves and explicitly
marks the persistent-gradient cast before graph extraction, and extracts it as
a zero-collective reduction epilogue. Strict FQN, mesh-axis, dtype, tensor, and
collective-ancestry provenance validation makes malformed new traces fail
instead of silently restoring the incorrect accumulation order.

The fix was validated on the pre-rebase source before it was carried into the
current final stack. These are local acceptance results, not substitutes for the
still-pending final-source and packaged-runtime gates:

| Case | Step | Eager loss | GraphTrainer loss | Exact local parameter and gradient records |
| --- | ---: | ---: | ---: | --- |
| PP1, DP4/EP4, EDP1, GA4 | 1 | 13.099918365478516 | 13.099918365478516 | 340/340 on each of ranks 0-3 |
| PP2/VPP8, DP2/EP2, EDP1 | 1 | 13.09231185913086 | 13.09231185913086 | 166/166 on ranks 0-1; 175/175 on ranks 2-3 |
| PP2/VPP8, DP2/EP2, EDP1 | 2 | 8.543712615966797 | 8.543712615966797 | same captured manifest coverage |

The eager and GraphTrainer JSON manifests are byte-for-byte equal for every
listed rank and include both parameter and pre-optimizer gradient SHA-256 data.
The PP2 test exercises the formerly failing no-collective path directly. The
tracked paired harness must still repeat PP1 and PP2, with and without the
outer full-step CUDA graph where specified by the runbook, on the final rebased
source and again from the published runtime fbpkg.

Those five source and five packaged paired gates are the only final
full-precision loss/grad-norm and exact gradient evidence. The 256-GPU MAST
recipes intentionally disable TensorBoard, and console metrics are rounded;
their loss and grad norm establish only finite training health.

## Performance results

The final table will contain exactly four lines: Baseline Chien-Chin, Test
GraphTrainer Chien-Chin, Baseline Sanket, and Test GraphTrainer Sanket. Every
comparison within a pair will use the same cluster allocation, runtime package,
recipe objective, and profiler-disabled step window. Tokens/s/GPU,
aggregate tokens/s, TFLOP/s/GPU, MFU, sample standard deviation, peak memory,
and eager delta will be calculated from successful final jobs only.

The predeclared headline window is the intervals ending at steps 20, 30, 40,
and 60. Step 10 is startup; step 50 includes configured host garbage collection.
The reported variability uses sample standard deviation (denominator `n - 1`).
Profile-job throughput is never used in the headline.

Historical values from external documents are targets, not same-cluster
baselines or exact objective reproductions, and will be labeled separately if
retained for context.

## Profiler traces

Only `share_trace.py` links from successful, separate profile jobs will be
listed. PP1 uses the representative rank-0 trace. PP2 uses rank 0 and rank 128
plus a compacted, aligned two-rank timeline. The PP2 profile schedule is
frequency 43, warmup 3, active 2; the PP1 profile captures iteration 41. Profile
jobs are not used for headline throughput.

## Changes required to reproduce

The 20-commit landable stack is based directly on `origin/main` commit
`948d65c868c5fa8f0290bcf9e54b69f004721f54` and is ordered with the general
FSDP fan-in correction first. Its changes cover:

1. coalescing repeated functional and in-place FSDP gradient fan-in before one
   reduction;
2. preserving nonzero-dimension MXFP8 FSDP shard order;
3. applying expert FSDP policy to auxiliary decoder blocks;
4. defining functional DistMoE WGrad output dtype;
5. coordinating outer and GraphTrainer CUDA-graph ownership;
6. restoring external runtime state after pipeline metadata inference;
7. providing a generic model-owned pipeline runtime and DeepSeek MTP pipeline
   integration as separate commits;
8. fusing DistMoE, PP, and ordered native WGrad accumulation;
9. preserving scoped FSDP buckets, MTP parent buckets, asynchronous unshard
   lookahead, and deterministic communication-only reduction order;
10. finalizing deferred eager FSDP reductions once per optimizer step and
    preserving pipeline loss-reporting order; and
11. preserving eager reduce-dtype accumulation semantics for singleton FSDP
    meshes with an explicit zero-collective reduction boundary (`85de4b5b9`).

Benchmark recipes, profiler settings, packaging helpers, this report, and the
runbook are isolated in one `[not-for-land]` commit. That commit also contains a
runtime-compatibility revert of TorchTitan's
`MixedPrecisionPolicy.param_dtype_override_fn` forwarding because the pinned
PyTorch runtime predates that API. The revert is package-only and must not land;
remove it when reproducing with a PyTorch revision that supports the API. The
report branch, immutable runtime-source tag, base, landable fix hashes, and
commands are recorded now. The publisher archives `HEAD` and records that same
commit in the runtime provenance, so embedding an artifact identifier into S
would change both its source hash and package contents. Runtime-source commit S
is therefore frozen before packaging and retained immutably; package identifiers,
successful job links, measurements, and trace links are added only to the later
docs-only report tip R. Those publication-time values remain explicit
placeholders until the corresponding gates and jobs complete.
