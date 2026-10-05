# DeepSeek V3 671B DistMoE 256-GPU GraphTrainer Report

Date: 2026-10-05

## Executive summary

This report retains only the best current, directly comparable Chien-Chin
runs. Both used the same 64-host x four-GPU NVIDIA GB300 topology from the
`MuseSpark_1_2_Safety_DCT` tenant, the same PyTorch, TorchAO, and DistMoE
revisions, the same model and data configuration, and one outer CUDA graph
for the complete training step.

| Line | Clean tokens/s/GPU | TFLOP/s/GPU | Derived MFU | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| **Baseline Chien-Chin r3** | 5,822.00 +/- 24.79 | 1,636.87 | 65.47% | 265.08 GiB |
| **Test GraphTrainer Chien-Chin parent-bucketing r1** | 5,830.50 +/- 8.58 | 1,639.29 | 65.57% | 249.38 GiB |

The latest GraphTrainer result is 0.15% above the same-cluster eager baseline
and uses 15.70 GiB less peak memory. It is 2.82% faster than the superseded
Cat-fix-only GraphTrainer result. The new profile verifies both optimizations:
the selected FSDP gradient-layout `CatArrayBatchedCopy` kernel still executes
244 times per optimizer step, while the parent expert bucket removes one
all-gather and one reduce-scatter per each of the 58 MoE layers. GraphTrainer
therefore issues 121 all-gathers and 121 reduce-scatters instead of 179 of
each.

Both retained jobs completed 60 steps on 64 hosts with four GPUs per host,
all 64 tasks complete, zero task restarts, finite loss and gradient norm, one
full-step `cudaGraphLaunch`, profiler traces, and memory snapshots.

No Sanket performance result is published here. The previous GraphTrainer
number omitted MTP1 and was therefore not compute parity. A Sanket result
should be added only after both the eager baseline and GraphTrainer test run
the same MTP1 PP2/VPP8 configuration successfully on this cluster.

## Latest successful runs

| Configuration | MAST run |
| --- | --- |
| Test GraphTrainer Chien-Chin parent-bucketing r1 | [run](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-test-gt-chien-parent-bucket-r1-256-ivankobzarev) |
| Baseline Chien-Chin r3 | [run](https://www.internalfb.com/msl/studio/runs/mast/MS-Ivan-baseline-eager-chien-r3-256-ivankobzarev) |

Supporting links:

- [Chien-Chin source definition, P2483940224](https://www.internalfb.com/phabricator/paste/view/P2483940224?view=markdown)
- [MuseSpark capacity tenant](https://msl-capacity.internalmeta.com/resource-groups?tenantPath=gen_ai%2Fmsl%2Ffair_research%2Ffair_prod%2FAlignement%2FMuseSpark_1_2_Safety_DCT&customColumnsMachineTypeAllocation=MACHINE_TYPE_T20_GRAND_TETON_HBM3_ROCE_GENAI_PCI&resourcetypeTopCapacitySpenders=T20_CIC_GB300_278GB_HBM3E_NVLSO_NSF_LP)
- [Reproduction source branch](https://github.com/pytorch/torchtitan/tree/dist-moe-256gpu-repro-final-20261005)
- [`main...reproduction branch` diff](https://github.com/pytorch/torchtitan/compare/main...dist-moe-256gpu-repro-final-20261005)

## Configuration

Common configuration:

| Setting | Value |
| --- | --- |
| Model | DeepSeek V3 671B |
| Precision | BF16 parameters, MXFP8 DistMoE experts |
| Sequence length | 4,096 |
| MTP | Disabled, matching the Chien-Chin definition |
| Parallelism | PP1, DP/FSDP256, EP64, expert-FSDP4, TP1, CP1 |
| Gradient accumulation | 16 microbatches |
| Tokens per optimizer step | 16,777,216 |
| FSDP lifetime | `fsdp_reshard_after_forward="never"` |
| Dense FSDP transport | Symmetric memory |
| Activation checkpointing | Disabled |
| Data | C4 test source, concat-then-split packing |
| Logging | Every ten steps |
| Profile | Iteration 41, one active step, memory snapshot enabled |
| CUDA graph | One outer full-step graph owned by TrainingEngine |
| Hardware | 64 x 4 NVIDIA GB300, 256 GPUs total |
| Priority | Normal (`REGULAR`) |

The two implementations differ only where their runtimes express gradient
accumulation:

| Setting | Baseline Chien-Chin r3 | Test GraphTrainer Chien-Chin parent-bucketing r1 |
| --- | --- | --- |
| Training implementation | Eager TorchTitan | GraphTrainer |
| DistMoE WGrad accumulation | Eager in-place accumulating backward | GraphTrainer producer fusion |
| FSDP reduction | Eager deferred reduction | Last-microbatch GraphTrainer action |
| Parameter unshard | Eager FSDP lifetime | First-microbatch GraphTrainer action |
| Graph compilation | No local compile regions | Regional, numerics-changing optimization disabled |
| GraphTrainer CUDA-graph pass | Not applicable | Removed automatically because the outer graph is enabled |

CUDA-graph ownership is exclusive: `outer_cudagraphs_enabled=True` and
`graphtrainer_cudagraphs_enabled=False`. The runtime asserts that both cannot
be enabled simultaneously.

## Performance results

MXFP8 training did not emit a native MFU value. MFU is derived as logged
TFLOP/s divided by the 2.5 PFLOP/s BF16 peak used for GB300 comparisons.
Step 50 is excluded because iteration 41 profiling and memory-snapshot export
contaminate that ten-step interval.

The `+/-` values in the summary are sample standard deviations across the
four clean logged intervals shown below, not variation across independent
jobs.

| Line | Step 20 | Step 30 | Step 40 | Step 60 | Clean mean | Aggregate tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline Chien-Chin r3** | 5,837 | 5,835 | 5,831 | 5,785 | 5,822.00 | 1,490,432 |
| **Test GraphTrainer Chien-Chin parent-bucketing r1** | 5,840 | 5,835 | 5,826 | 5,821 | 5,830.50 | 1,492,608 |

The direct same-cluster delta is:

```text
(5,830.50 / 5,822.00 - 1) * 100 = +0.15%
```

The corresponding clean logged compute rates are 1,636.87 TFLOP/s/GPU for
Baseline Chien-Chin and 1,639.29 TFLOP/s/GPU for Test GraphTrainer.

## Correctness and completion evidence

| Line | Step 1 | Final logged step | Completion |
| --- | --- | --- | --- |
| Baseline Chien-Chin r3 | loss `12.30927`, grad norm `17.1941` | step 60: loss `2.95984`, grad norm `14.5442` | 64/64 tasks complete, zero restarts |
| Test GraphTrainer Chien-Chin parent-bucketing r1 | loss `12.26398`, grad norm `17.4585` | step 60: loss `2.84121`, grad norm `6.7638` | 64/64 tasks complete, zero restarts |

These runs establish finite 60-step training and execution stability. They do
not claim bitwise numerical identity between eager and GraphTrainer; that
requires a dedicated deterministic loss comparison rather than rounded MAST
stdout.

## Latest profiler links

The links were produced with `arvr/scripts/perfetto/share_trace.py` on
2026-10-05. They use its default 28-day TTL and should expire around
2026-11-02. Raw traces are authoritative for measurement. Compacted traces
retain slice names, timestamps, durations, arguments, and process IDs while
mapping hundreds of CUDA replay streams into readable semantic lanes.

### Test GraphTrainer Chien-Chin parent-bucketing r1

- [Compacted rank 0 (recommended)](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_368bf727-5f97-4b15-a6d5-153469dbca91_gt_chien_parent_bucket_r1_rank0_trace_compacted.json.gz)
- [Raw rank 0](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_03cf5dbd-e036-48bc-9a64-cc410808cdde_gt_chien_parent_bucket_r1_rank0_trace.json.gz)
- Raw SHA-256: `3f4e81189d8fac56398224ff3422fb0b0084241b86ad332e77672d646d6653a0`
- Compacted SHA-256: `689417a9ac64ed4ba445f430eb12c37d6fcd08054dc38758cb05791efce6c6f4`
- Compaction: 234,783 measured GPU slices, 126 replay streams -> 9 lanes.
- CUDA graph launches: 1.

### Baseline Chien-Chin r3

- [Compacted rank 0 (recommended)](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_5c56f7b5-3007-4a15-9615-208c083cd75c_baseline_chien_r3_rank0_trace_compacted.json.gz)
- [Raw rank 0](https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fivankobzarev_9f81cf85-45c0-4ef4-aaf0-a8cf13815154_baseline_chien_r3_rank0_trace.json.gz)
- Raw SHA-256: `328ebde0608cf1bc09d9efaebf24ec7aabc86872308caa235076226b232ee749`
- Compacted SHA-256: `aef9dd573b66c04f168e6d302d5e5cdf695e3fbf4cab2620559c4991e13cbf56`
- Compaction: 242,526 measured GPU slices, 247 replay streams -> 8 lanes.
- CUDA graph launches: 1.

The durable source traces remain at:

```text
manifold://torchtrain_datasets/tree/outputs/MS-Ivan-test-gt-chien-parent-bucket-r1-256-ivankobzarev/profiling/traces/iteration_41/rank0_trace.json.gz
manifold://torchtrain_datasets/tree/outputs/MS-Ivan-baseline-eager-chien-r3-256-ivankobzarev/profiling/traces/iteration_41/rank0_trace.json.gz
```

## Profile findings

The Cat/layout-tail and parent-expert bucketing optimizations are both working
at 256-GPU scale.

| Rank-0 step-41 event | Baseline | Test GraphTrainer parent-bucketing |
| --- | ---: | ---: |
| Selected BF16 FSDP layout `CatArrayBatchedCopy` | 244 | 244 |
| 8-byte contiguous `CatArrayBatchedCopy` | 928 | 928 |
| `memcpy32_post` | 1,527 | 990 |
| NCCL all-gather | 122 | 121 |
| NCCL reduce-scatter | 121 | 121 |
| `cudaGraphLaunch` | 1 | 1 |

Before the fix, the selected layout-copy kernel executed 3,904 times in
GraphTrainer: 244 layout operations in each of 16 microbatches. The pass had
cut the reduction graph at the first collective input, after DTensor split,
padding, concatenation, and layout materialization. Those operations therefore
remained inside every repeated backward graph even though reduce-scatter ran
only on the final microbatch.

The fix moves the FSDP reduce-gradient boundary backward through that
split/pad/concatenate provenance. Each repeated graph accumulates the earlier
local WGrad representation in place, including producer-epilogue fusion where
available. The final reduction graph reconstructs the FSDP layout once and
then reduces it. The 61 MLA gradient concatenations are model computation and
remain in every microbatch.

The Cat-fix-only GraphTrainer profile still contained 179 all-gathers and 179
reduce-scatters. Captured collective nodes identify the routed-expert module as
`layers.N.moe.routed_experts`, but the bucket plan named its `w13` and `w2`
children. The plan therefore did not combine those parameters at the captured
scope and left one extra bucket in every one of the 58 MoE layers.

The `[graph_trainer] Bucket routed experts by parent scope` change updates the
plan to the captured parent scope. The new trace contains 121 all-gathers and
121 reduce-scatters, removing exactly 58 of each while retaining the Cat
optimization. Throughput improves from 5,670.50 to 5,830.50 tokens/s/GPU and
reaches same-cluster eager parity without the experimental dense-overlap
scheduler.

## Reproduction artifacts

Runtime fbpkg used by the parent-bucketing GraphTrainer run:

```text
torchtitan_conda_ivankobzarev_dist_moe_256gpu_sm103_20261005_v4:2765dda411094e02bccca192fbe367d5
```

Launcher fbpkg used by the parent-bucketing GraphTrainer run:

```text
torchtitan_muse_spark_launcher_ivankobzarev_20261005_v1:e14116f730214a42ba830a7ed8548ea7
```

The measured eager baseline used runtime
`torchtitan_conda_ivankobzarev_dist_moe_256gpu_sm103_20261005_v4:4fdd8e5f8f5045bdb43ae1373be41859`
and launcher
`torchtitan_muse_spark_launcher_ivankobzarev_20261005_v1:66d93b3d08c9414bb9545b704fdd7658`.
Both runtime builds contain PyTorch `38cca963`, TorchAO `a701b6a6`, and DistMoE
`18b4f488`.

Packaged source revisions:

| Component | Revision |
| --- | --- |
| TorchTitan package provenance | `38402eb7514e8840f1c7805163d384b0b915f64f` |
| Additional packaged change | `[graph_trainer] Bucket routed experts by parent scope` |
| PyTorch | `38cca96300da024842405ecefa081e4761254922` |
| TorchAO | `a701b6a6058720c21f95908b7ae4a24bf0cae1b6` |
| DistMoE | `18b4f4887ab9a97e35193d0921cff51a249202ee` |
| CUDA architecture | `sm_103` |

The package was assembled before the parent-scope change was committed, so
its provenance records the clean base revision `38402eb75`. The immutable
package contains that one-line source change. The final parent-scope commit is
the subsequently created reviewable change for the exact fix and its
regression tests.

Final launcher checksums:

| File | SHA-256 |
| --- | --- |
| `mast.py` | `4b55622de00a4694f1fe1e275b5ab8dc0e13d0962993c85a4e15e898ee4efd45` |
| `mast_configs.py` | `c29af139ba03915089db5556d3dde379b3fb294e54fbc8f74f93f40aa41a5b6f` |

## Exact launch commands

Fetch the archived launcher and run from its extracted directory. Both
commands request Normal priority through the launcher.

```bash
LAUNCHER_ID=torchtitan_muse_spark_launcher_ivankobzarev_20261005_v1:e14116f730214a42ba830a7ed8548ea7
LAUNCHER_DIR=/absolute/local/path/torchtitan_muse_spark_launcher_20261005_v1
mkdir -p "$LAUNCHER_DIR"
fbpkg fetch --dest "$LAUNCHER_DIR" --extract --verify \
  --unexpected-fails-verify "$LAUNCHER_ID"
cd "$LAUNCHER_DIR"
```

Test GraphTrainer Chien-Chin parent-bucketing:

```bash
torchx run -s mast_conda \
  -cfg workspace_fbpkg_id=torchtitan_muse_spark_launcher_ivankobzarev_20261005_v1:e14116f730214a42ba830a7ed8548ea7 \
  mast.py:train \
  --module_name mast_configs \
  --config_name graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile \
  --name MS-Ivan-test-gt-chien-parent-bucket-repro
```

Baseline Chien-Chin:

```bash
torchx run -s mast_conda \
  -cfg workspace_fbpkg_id=torchtitan_muse_spark_launcher_ivankobzarev_20261005_v1:e14116f730214a42ba830a7ed8548ea7 \
  mast.py:train \
  --module_name mast_configs \
  --config_name deepseek_v3_671b_dist_moe_mxfp8_chien_chin_256gpu_profile \
  --name MS-Ivan-baseline-eager-chien-repro
```

## Source stack and landing boundary

The reproduction branch is organized in application order:

1. `[graph_trainer] Extract FSDP gradient layouts from repeats` - landable Cat/layout-tail elimination and in-place WGrad producer fusion support.
2. `[graph runtime] Restore external state after metadata inference` - landable runtime-state cleanup.
3. `[graph_trainer] Coordinate CUDA graph ownership` - landable outer versus GraphTrainer CUDA-graph ownership.
4. `[docs] Publish DistMoE 256-GPU reproduction report` - documentation only.
5. `[not-for-land] Add DistMoE 256-GPU profiling support` - launch recipes, packaging, capture-preparation shim, and experiment-only DistMoE integration.
6. `[graph_trainer] Bucket routed experts by parent scope` - landable fix for the final performance gap.

The reviewable functional landing stack is the first three implementation
commits plus the final parent-scope bucketing commit. For the latest throughput
improvement alone, only the parent-scope bucketing commit is new. The docs and
profiling-support commits reproduce the experiment but are not core
implementation changes. The experimental dense-overlap commit was disabled in
the successful run and has been removed from this branch.

The launcher still prepares full-step capture before Adam state allocation:
it prewarms lazy kernels and communicators on the graph stream, clears the
prewarm gradients, releases unused cached blocks, and captures before the
first optimizer update. A supported TorchTitan capture-headroom contract is
still required before removing that launcher-local shim.

Exact Sanket parity remains separate work: both the eager baseline and
GraphTrainer test must support one MTP layer with PP2/VPP8, full logits, and
loss scale 0.1. Until that pair succeeds, no Sanket throughput line should be
added to this report.
