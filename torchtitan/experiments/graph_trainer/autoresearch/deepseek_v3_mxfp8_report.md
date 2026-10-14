# DeepSeek V3 MXFP8 MoE auto-fuser run

Seven accepted fusions reduce the production local MoE forward/backward median
from **6.166611 ms to 5.921453 ms**, a **3.9756% latency reduction (1.0414x)** on
GB300. Outputs, loss, gradients, norms, and buffer updates match bitwise across
10 seeded inputs. Peak allocated CUDA memory remains **7,962,782,208 bytes**;
peak reserved memory falls by **20 MiB**. This is a measured improvement under
the skill's best-effort criterion. Absolute device roofline performance has not
been established.

| Whole invocation | Original | Fused |
| --- | ---: | ---: |
| Median forward/backward latency | 6166.611 us | 5921.453 us |
| Minimum / maximum timing sample | 6137.120 / 6206.721 us | 5875.004 / 5965.319 us |
| Graph nodes | 224 | 155 |
| GPU kernel launches per replay | 183 | 96 |
| GPU memset events per replay | 5 | 2 |

The final comparison uses `baseline_parameter_norm/result.json` and
`fused_selected/result.json` under `/tmp/torchtitan_auto_fuser_20261014/`.

This run applies [auto_fuser.md](auto_fuser.md) through parallel implementation
agents for router selection, router weights/auxiliary gradients, and the shared
expert. Integration, independent validation, and performance analysis have
separate owners. The opt-in rewrite lives in [moe_fusion.py](moe_fusion.py), and
[benchmark_moe.py](benchmark_moe.py) constructs and checks the joint graph.

The source recipe is
`graph_trainer_deepseek_v3_671b_dist_moe_mxfp8()`. The measured module is its first
MoE, layer index 3, immediately after parameter unsharding.

| Configuration | Value |
| --- | --- |
| Local tokens | 4096 |
| Model width / expert hidden width | 7168 / 2048 |
| Routed experts / local experts | 256 / 4 |
| Selected experts per token | 8 |
| Expert groups / selected groups | 8 / 4 |
| Expert parallel degree | 64 |
| Shared expert | One, hidden width 2048 |
| Shared and routed GEMMs | Production MXFP8 |
| Router | BF16 operands, FP32 scores, `hi_mid_lo` backward |
| Route scale / normalization epsilon | 2.5 / `1e-20` |
| Auxiliary load-balancing loss | Active, coefficient 0.001 |
| Auxiliary denominator | 62,914,560 production global tokens |
| Routed MXFP8 pipeline | `staged`, `fast_math=True` |
| Production parallelism outside this invocation | DP shard 64, PP 4, EP 64, 240 PP microbatches |

One physical GB300 executes real local expert and shared-expert computation with
a fake process group representing 64 logical EP ranks. Uniform-rank simulation
provides defined fake peer data. The measurement includes the production MoE
auxiliary path and mutable buffers. Its synthetic loss is
`(output.float() * upstream.float()).sum()`, with seeded BF16 inputs and upstream
gradients. Parameter unsharding, the pipeline schedule, optimizer state, and
optimizer updates are outside this one-invocation measurement. These results
measure local computation with fake communication; they do not measure real
network or distributed training throughput.

The baseline is the original canonicalized GraphTrainer joint forward/backward
graph. The candidate replaces closed regions between GEMM and communication
boundaries. The pass checks production shapes, dtypes, devices, contiguity, and
external consumers before matching. GEMM and DistMoE calls retain their original
implementations and order: five router GEMMs, six shared scaled GEMMs, and
DistMoE forward/backward, 13 computational boundary calls in total.

The benchmark uses separate processes on physical GPU 0, 10 warmup calls, 10
distinct validation inputs (seeds 42 through 51), and five timing samples of 50
CUDA-graph replays. Timing excludes validation, profiling, and bandwidth probes.
`config.debug.seed=42` and `config.debug.deterministic=True` apply to both runs.
The repository's `set_determinism` sets
`torch.utils.deterministic.fill_uninitialized_memory=False`; this is the same
production behavior on both sides. Deterministic warning-only mode is never used.

Numerical checks compare eager and traced execution, then compare the baseline
and fused processes on every validation input. They cover the output, scalar
loss, input gradient, all five parameter gradients, parameter gradient norm,
input-gradient norm, and all three buffers (`expert_bias_E`,
`router.tokens_per_expert_E`, and `router.aux_loss.instance_acc`). Tensor evidence
includes shape, dtype, stride, and SHA256 of the exact underlying value bytes.
Signed zero is distinguished. Each validation step preserves the accumulated
buffer updates. Complete per-tensor signatures remain in each run's `result.json`.
For seed 42, both runs produce loss `22.06275177001953`, parameter gradient norm
`54966.50390625`, and input-gradient norm `46.708740234375`. The JSON summary
retains all ten scalar records and matching signature digests.

Memory is measured after warmup and cache cleanup, during CUDA-graph capture and
replay. The acceptance gate compares all six allocated/reserved measurements
with `candidate_bytes <= baseline_bytes`. Both graph memory snapshots and
allocator summaries are retained. Reduced temporary region allocations can leave
the full-layer peak unchanged because another part of the invocation owns that
peak. The comparison does not include optimizer memory.

| PyTorch CUDA allocator metric | Original bytes | Fused bytes |
| --- | ---: | ---: |
| Allocated before capture | 7,069,720,064 | 7,069,720,064 |
| Reserved before capture | 7,472,152,576 | 7,472,152,576 |
| Peak allocated during capture/replay | 7,962,782,208 | 7,962,782,208 |
| Peak reserved during capture/replay | 8,585,740,288 | 8,564,768,768 |
| Allocated after capture/replay | 7,631,233,024 | 7,631,233,024 |
| Reserved after capture/replay | 8,585,740,288 | 8,564,768,768 |

These are allocator measurements taken before signature hashing, profiling, and
bandwidth calibration; they are not NVML total-process/device-memory readings.

The shared-expert plan combines SwiGLU, rowwise and columnwise MXFP8 quantization,
and both scale rearrangements. It retains the production BF16 rounding before
quantization and the saved gate/up activation. The router weight plan preserves
ATen reduction order, rounded division, and the three-piece BF16 residual split.
Joining weight differentiation with scatter, auxiliary-gradient addition,
sigmoid backward, and the split removes the dense router-gradient intermediate.
Auxiliary normalization forward/backward also preserves the original reduction
tree, clamp subgradient, raw/clamped norm storage, token reduction, loss scaling,
and injection into the backward graph.
The final backward plan additionally computes the auxiliary derivative inside
the router backward kernel, eliminating its separate dense gradient allocation.

The following isolated region timings are paired public-API CUDA-graph medians.
They use at least 10 warmups and many measured calls. They quantify individual
plans; their sum is not the full-layer speedup because graph scheduling and cache
state differ. The shared-expert measurements are the conservative paired results
from its final validation process.

| Region | Original us | Fused us | Original peak extra bytes | Fused peak extra bytes |
| --- | ---: | ---: | ---: | ---: |
| Expert selection | 133.659 | 7.979 | 9,076,736 | 262,144 |
| Per-expert token counts | 43.507 | 4.564 | 26,216,960 | 68,608 |
| Selected weights forward | 12.140 | 1.816 | 409,600 | 278,528 |
| Router backward, including auxiliary derivative | 162.452 | 7.350 | 41,943,040 | 6,291,456 |
| Auxiliary normalization forward | 7.535 | 2.990 | 4,227,072 | 4,227,072 |
| Shared SwiGLU + MXFP8 forward | 35.408 | 19.315 | 34,603,008 | 17,301,504 |
| Shared SwiGLU + MXFP8 backward | 57.342 | 31.327 | 69,206,016 | 34,603,008 |

Selected scores and denominator retain the same 147,456 bytes of saved state;
auxiliary raw/clamped norms retain the same 32,768 bytes. Shared-expert output
storage and strides match the original quantizer. Its logical memory traffic
falls from 85,458,944 to 50,855,936 bytes forward and from 154,140,672 to 84,934,656
bytes backward by removing the BF16 and unswizzled-scale write/read pairs.
The token-count plan replaces dense scatter masks and reductions with integer
partial histograms and a final reduction. Its 65,536-byte scratch allocation is
transient; the retained int64/FP32 count vectors remain 3,072 bytes. This rewrite
requires graph-proven top-k indices, because duplicate IDs would have different
dense-scatter semantics.
Combining the two already-fused backward kernels changes their paired median
from 7.428 to 7.350 us, a marginal difference; an earlier measurement was tied.
Its clear additional benefit is removing 4,194,304 bytes of peak temporary
storage. It passes 55 exact-bit cases. The table compares this final plan with
the original eager region.

The implementation agents checked FP64 oracles in addition to exact same-dtype
references. Selected weights and complete router backward passed 41 cases each;
the standalone BF16 split passed 34. The selected-weight constant-upstream case
is ill-conditioned: its exact mathematical derivative cancels, so the
reference-relative FP64 gate is not meaningful. Target and eager still match
every bit. Shared-expert tests cover four shapes, two seeds, zeros, extreme and
tiny values, NaN/Inf, and all output metadata. Its decoded MXFP8 values have the
same oracle error as the production baseline.
Expert selection passed 507 cases across 13 token counts, 13 adversarial
families, and three seeds, including exact unsorted top-k ordering, signed zero,
ties, and NaN payloads. It reproduces this PyTorch build's single-block radix
top-k ordering; another PyTorch implementation requires revalidation. Auxiliary
normalization passed 42 exact-bit cases; its
41 non-degenerate FP64 cases pass, while the uniform expert-gradient case has the
same cancellation limitation. Higher-order derivatives are outside the tested
contract.
Token counting passed 396 cases across 22 token counts, six input families, and
three seeds, plus 10 repeated calls and exact accumulated buffer updates. Its
FP64 oracle error is zero for both implementations. Concentrating every token
on the same eight experts preserves the count kernel's measured improvement.

Several candidate plans were rejected. Deferring both input-gradient additions past
the router GEMMs keeps an extra BF16 `[4096, 7168]` tensor live, increasing memory
by 58,720,256 bytes. Splitting the shared-expert backward into separate gate/up
CTAs increases traffic and regresses timing. Larger shared-expert tiles either
raise register pressure or spill. The retained shared-expert layout uses 32x128
forward tiles and 32x64 backward tiles, four warps, and no global intermediates.
Fusing sigmoid into expert selection passes 507 exact cases but takes 10.239 us
versus 10.142 us for the separate sigmoid plus final selection kernel; the slower
combined plan is not selected. Kernel search covered 99 shared-expert layout and
launch configurations, 20 complete-router-backward configurations, and 48
auxiliary forward/backward configurations.

The final whole-layer traces show where the remaining time goes. The table sums
only GPU kernel and memset event durations over ten replays, without inclusive
CPU profiler entries. Name-based attribution identifies the GEMM, dispatch,
combine, and DistMoE runtime kernels conservatively; the other category also
contains additional work inside opaque DistMoE calls.

| Profiled device duration per replay | Original us | Fused us |
| --- | ---: | ---: |
| Routed GEMM / dispatch / combine | 3987.813 | 4030.511 |
| Shared and router GEMMs | 424.915 | 426.157 |
| Other identified DistMoE runtime kernels | 609.099 | 617.151 |
| New fusion kernels | 0 | 64.252 |
| Other kernels and transfers | 1137.794 | 706.600 |
| Total event duration | 6159.620 | 5844.671 |

Identified fixed-boundary work accounts for 81.5% of baseline and 86.8% of fused
device event time. Accounting for overlap, those intervals occupy 5.014 ms and
5.065 ms per replay respectively. Holding their measured durations and overlap
fixed, even removing all remaining work leaves the fused run with at least
5.065 ms of active GPU intervals. This conditional floor explains the limited
whole-layer benefit from much larger local region speedups. It is not a device
roofline or a prediction for a different schedule. Profiled durations are also
separate from the CUDA-event latency measurements above.

Fresh Nsight Compute 2025.3.1 counters cover all eight final kernels after source
formatting and integration, with 10 warmup calls, 21 profiler passes per kernel,
`--cache-control none`, and `--clock-control none`. The capture is
`/tmp/torchtitan_auto_fuser_20261014/roofline_final.ncu-rep`; the CSV, metadata with
source hashes, and counter summary sit alongside it. The following durations
are profiler measurements of isolated calls and include the profiling context;
they are not the CUDA-graph benchmark medians.

| Final kernel | NCU duration us | SM throughput | Waves per SM | Achieved occupancy |
| --- | ---: | ---: | ---: | ---: |
| Expert selection | 10.11 | 43.42% | 0.84 | 31.33% |
| Selected weights forward | 3.84 | 1.68% | 0.11 | 10.14% |
| Partial token counts | 3.42 | 0.39% | 0.03 | 6.21% |
| Final token-count reduction | 3.20 | 0.08% | <0.01 | 4.25% |
| Auxiliary normalization forward | 5.28 | 11.80% | 0.42 | 18.49% |
| Combined router backward | 10.02 | 32.83% | 0.84 | 36.10% |
| Shared SwiGLU + MXFP8 forward | 14.37 | 57.22% | 1.68 | 40.54% |
| Shared SwiGLU + MXFP8 backward | 22.40 | 68.36% | 3.37 | 44.63% |

All final kernels report zero local spill requests. Router/count/auxiliary grids
have fewer than one full wave across 152 SMs, which limits utilization. Registers
cap theoretical occupancy at 50% for selection, auxiliary forward, combined
backward, and both shared-expert kernels. The shared-expert forward shows latency
constraints; backward uses compute more heavily than memory. These counters
identify remaining limits and do not show absolute roofline saturation.

The bandwidth probe uses `torch.add(source, 0.0, out=destination)` with CUDA graph
replay. Its 512 MiB source and destination exceed cache capacity and provide a
measured streaming comparison; the 4 MiB case measures a small cache-resident
operation, including launch latency. Neither value is a universal roofline for
mixed sorting, reductions, transcendental operations, and quantization. High
cache hit rates also mean `logical_bytes / HBM_bandwidth` is not an absolute
lower bound for these hot region benchmarks.
The final runs measure approximately 7.070 TB/s for the 512 MiB streaming case
and 4.42-4.92 TB/s for the small 4 MiB case.

The environment is PyTorch `2.16.0a0+git5e675f6`, Triton `3.9.0`, TorchAO
`0.19.0.dev20261014+cu130`, and DistMoE `0.1.0` at
`6f0c3d4216aa9d2ef2eec2fa4215fa42b73af314`. GPU 0 is an NVIDIA GB300, compute
capability 10.3, with 152 SMs. The repository base is
`d68dc9e8666eb1c4815a65186a70c622fd25a474`. The isolated Python environment is
`/tmp/torchtitan_auto_fuser_20261014_env/bin/python`.

Validation passes: 10 CPU tests plus 9 subtests with CUDA hidden, 20 GPU tests on
GB300, pre-commit for all implementation/test files, and `git diff --check`.
The full-repository pre-commit run also passes. The
[B200 workflow](../../../../.github/workflows/integration_test_b200.yaml)
explicitly includes the new GPU test file; that workflow was inspected locally,
not executed as a remote CI run.

Reproduce the final pair from the repository root with the environment above
and the same package versions:

```bash
AUTO_FUSER_PYTHON=/tmp/torchtitan_auto_fuser_20261014_env/bin/python
AUTO_FUSER_OUTPUT=/tmp/deepseek_v3_mxfp8_reproduction
CUDA_VISIBLE_DEVICES=0 "$AUTO_FUSER_PYTHON" -m torchtitan.experiments.graph_trainer.autoresearch.benchmark_moe \
  --mode baseline --output-dir "$AUTO_FUSER_OUTPUT/baseline" \
  --warmup-steps 10 --validation-steps 10 --steps 50
CUDA_VISIBLE_DEVICES=0 "$AUTO_FUSER_PYTHON" -m torchtitan.experiments.graph_trainer.autoresearch.benchmark_moe \
  --mode fused --output-dir "$AUTO_FUSER_OUTPUT/fused" \
  --reference "$AUTO_FUSER_OUTPUT/baseline/result.json" \
  --regions selection,weights,gate_backward,counts,aux,swiglu \
  --warmup-steps 10 --validation-steps 10 --steps 50
CUDA_VISIBLE_DEVICES='' "$AUTO_FUSER_PYTHON" -m pytest -q \
  tests/unit_tests/cpu/test_graph_trainer_moe_fusion.py
CUDA_VISIBLE_DEVICES=0 "$AUTO_FUSER_PYTHON" -m pytest -q \
  tests/unit_tests/gpu/models/common/test_graph_trainer_moe_fusion.py
```

The exact profiling command is saved in
`/tmp/torchtitan_auto_fuser_20261014/roofline_final_command.sh`, and its driver is
`profile_regions_final.py` in the same directory. Numerical and timing evidence
for each plan is in `router_selection_report_fused.json`,
`router_counts_report.json`, `router_weights_report.md`, `router_aux_report.md`,
`router_backward_report.md`, and `shared_expert_report.md`. Candidate searches and
rejected implementations are preserved in that artifact directory.

The compact durable measurement summary is
[deepseek_v3_mxfp8_results.json](deepseek_v3_mxfp8_results.json). Large traces,
memory snapshots, numerical sweeps, and kernel search artifacts stay under
`/tmp/torchtitan_auto_fuser_20261014/`.
