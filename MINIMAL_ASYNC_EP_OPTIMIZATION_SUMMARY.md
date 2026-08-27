# MinimalAsyncEP and Kimi K3 Optimization Summary

## Executive summary

This work made `MinimalAsyncEP` runnable with PyTorch's fake process group,
added reproducible DeepSeek V3 and Kimi K3 benchmark launchers, and optimized
the Kimi K3 path from **14.165% to 50.305% median MFU**.

The final Kimi K3 result is:

| Setting | Value |
| --- | --- |
| Branch | `k3-perf` |
| Final commit | `472c526d6` (`moe: compute only forced routing logits`) |
| Model | Kimi K3 full model |
| Precision | BF16 |
| Simulated topology | FSDP256, EP64, fake world size 256 |
| Local batch | 8 sequences |
| Sequence length | 4096 |
| Tokens per rank | 32,768 |
| Activation checkpointing | Full AC |
| CUDA graphs | Disabled |
| Steady-state MFU | **50.305% median**, 50.275% mean |
| Steady-state throughput | **1,955 tokens/s median** |
| Peak GPU memory | **240.00 GiB** |

The steady-state result uses steps 2-9 of a successful 10-step run. Step 1
contains startup/compilation, and step 10 contains profiler overhead.

## Benchmark interpretation

- Fake process groups simulate the 256-rank topology on one physical GPU. The
  run validates model execution, sharding shapes, local kernels, memory use,
  and fake-backend compatibility, but it does not measure real inter-GPU or
  inter-node communication.
- Forced load balance assigns experts deterministically. The final router
  optimization computes only the logits used by that deterministic routing.
  It is applicable to any forced-load-balance run and is not specific to the
  fake process group.
- The standard MFU estimator still describes the full model FLOP count. After
  eliminating unused router logits in forced-balance mode, its MFU should be
  treated as a throughput-oriented benchmark metric for this debug mode, not
  as an apples-to-apples measurement of normal dynamic routing.

## 1. MinimalAsyncEP foundation

### Exact receive-buffer sizing for forced load balance

Commit: `ff096a651` (`minimal_async_ep: bound receive buffer capacity`)

The previous receive allocation used the lossless worst-case routing bound.
That bound is unnecessarily large when round-robin forced load balance is
enabled. The new implementation:

- derives the exact maximum rows received by a rank under round-robin routing;
- allocates dispatch and combine buffers independently;
- propagates the selected capacity through the dispatcher configuration;
- optionally supports a user-provided receive-capacity factor for non-forced
  routing;
- fails asynchronously if real routing exceeds a configured bounded capacity,
  instead of silently dropping tokens;
- includes tests for exact sizing, configuration propagation, and overflow.

This is the change that removes the need for a worst-case receive buffer when
forced balance is active.

### Fake symmetric-memory backend

Commit: `d7064bc27` (`minimal_async_ep: add fake backend benchmark support`)

The fake process group cannot perform a real symmetric-memory rendezvous. A
local fake symmetric-memory handle was added that:

- preserves the symmetric-memory API and expected buffer shapes;
- aliases peer views to the local allocation;
- restricts copy kernels to the simulated local rank;
- zeroes receive buffers where needed so stale local data is not interpreted as
  remote traffic;
- supports MinimalAsyncEP forward and backward in fake-PG tests.

The same commit also fixed 64-bit row-stride arithmetic in the fused SwiGLU
Triton kernels and added regression coverage for strides larger than `2**31`.

## 2. DeepSeek V3 benchmark work

### Reproducible launchers

The following artifacts were added:

- `outputs/dsv3_16b_minimal_async_ep_fake/launch.sh`
- `outputs/dsv3_16b_minimal_async_ep_fake/dsv3_16b_fake_profile_config.py`
- `outputs/dsv3_671b_minimal_async_ep_fake/launch.sh`
- `outputs/dsv3_671b_minimal_async_ep_fake/dsv3_671b_fake_profile_config.py`
- `outputs/dsv3_16b_minimal_async_ep_fake/upload_trace.py`

The 671B launcher supports configurable batch size, sequence length, profiling,
and eager versus graph-trainer execution. It rejects benchmark runs shorter
than 10 steps.

### Retained DSV3 optimizations

- Switched the 671B attention path to varlen attention and forced the FA4 path
  by disabling the competing cuDNN SDPA provider for this benchmark.
- Enabled fused SwiGLU for dense/shared feed-forward layers and fused grouped
  experts for routed experts.
- Enabled the fused MLA override.
- Increased the tested local batch from 8 to 16 at sequence length 4096.
- Kept BF16 as the final precision after evaluating MXFP8.
- Kept profiler collection and per-step metrics enabled.
- Added an optional graph-trainer branch to the launcher.

Notable DSV3-671B observations:

| Run | Local batch | Median MFU | Median tokens/s | Peak memory | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `20260825T200549Z` | 8 | 37.820% | 3,363 | 80.87 GiB | Early seq4096 run |
| `20260825T202906Z` | 8 | 53.410% | 4,749 | 115.80 GiB | Varlen/FA4 iteration |
| `20260825T204532Z` | 16 | 51.260% | 4,558 | 136.13 GiB | Larger batch before the best fused-MLA result |
| `20260825T210528Z` | 16 | **65.255%** | **5,803** | 127.51 GiB | Best completed eager result observed |
| `20260825T234803Z` | 16 | 61.505% | 5,469 | 114.86 GiB | Graph-trainer result |

These DSV3 numbers record the successive experiment states. The current
launcher defaults to local batch 8 and exposes `LOCAL_BATCH_SIZE` for larger
runs.

### DSV3 experiments not retained

- **MXFP8:** conversion reached the MXFP8 linear and grouped-expert kernels,
  but the tested configuration failed before producing a stable benchmark and
  required additional compile integration. The benchmark was restored to
  BF16.
- **CUDA graphs:** attempted, then disabled after capture/varlen compatibility
  issues. The temporary varlen CUDA-graph workaround and the
  `max_num_documents=32` experiment were reverted.
- **Profiler removal:** tested as a possible source of overhead, then reverted;
  final launchers retain traces and per-step metrics.
- **Graph-trainer bucketing:** the joint transformer-block bucketing pass was
  tried. The final graph configuration leaves it disabled, along with the
  CUDA-graph pass.

## 3. Kimi K3 enablement

Commit: `d963aab83` (`kimi_k3: enable minimal async expert parallelism`)

This added the Kimi K3 full-model fake-backend benchmark and the model wiring
needed for MinimalAsyncEP:

- Kimi latent-MoE dispatcher configuration;
- latent expert dimension handling;
- Kimi-specific sharding plans;
- FSDP256/EP64 fake-topology configuration;
- BF16, forced load balance, Full AC, profiling, and per-step metrics;
- a benchmark launcher with configurable batch, sequence length, step count,
  and profiler window.

## 4. Kimi K3 performance optimizations

### Regional compilation around KDA graph breaks

Commit: `00aba2433` (`kimi_k3: compile blocks around KDA graph breaks`)

Kimi K3 could not use the existing full-graph transformer-block compilation
because the external KDA implementation introduces graph breaks. The compile
helper gained a `fullgraph` option, and Kimi applies per-block compilation with
`fullgraph=False`.

This was an enabler rather than an immediate standalone speedup; the first
compile-only run regressed. Later tensor-only compiled helpers provided the
large gains.

### Fused causal convolution

Commit: `5dc32cb59` (`kimi_k3: fuse causal convolution kernels`)

Replaced pad + `Conv1d` + SiLU with FLA's fused `causal_conv1d` path. This
raised the batch-1 median from the initial 14.165% MFU to 15.475% and reduced
peak memory from 137.99 GiB to 133.44 GiB.

### Batch independent recurrent sequences

Commit: `29f1eae60` (`kimi_k3: batch independent recurrent sequences`)

The packed batch previously looked like one long recurrent sequence to KDA.
The new Kimi collator and model plumbing carry sequence offsets and reshape KDA
inputs into independent equal-length sequences. This enabled real local batches
without leaking recurrent state across documents.

Measured progression:

- batch 2: 17.610% MFU;
- batch 4: 19.710% MFU.

### Batched MLA with SDPA

Commit: `b682dae17` (`kimi_k3: batch MLA with scaled dot product attention`)

Added a batched SDPA path for Kimi MLA, using sequence boundaries instead of a
large document mask. The implementation was checked against document-masked
FlexAttention in forward and backward tests.

At batch 4, median MFU increased from 19.710% to 20.240%.

### Compiled SiTU activation

Commit: `6d1beb922` (`kimi_k3: compile SiTU activation`)

Wrapped the tensor-only FP32 SiTU-GLU activation in a full-graph compiled
helper for both dense feed-forward and grouped expert paths.

At batch 4, median MFU increased from 20.240% to 26.530%.

### Compiled attention residual

Commit: `f1c67da27` (`kimi_k3: compile attention residual`)

Refactored Kimi's block-level residual calculation into a tensor-only
full-graph compiled helper, passing parameter tensors rather than modules.

At batch 6, median MFU increased from 27.290% to 37.855%.

### Larger benchmark batch

Commits:

- `31b38e7b8` (`benchmarks: raise Kimi K3 local batch size`)
- `6df48fc0f` (`benchmarks: raise Kimi K3 local batch size to eight`)

The launcher default was raised to batch 8 after memory validation. This
produced 38.600% median MFU at 242.68 GiB peak memory.

### Compiled gated RMSNorm

Commit: `5e07e360e` (`kimi_k3: compile gated RMSNorm`)

Combined FP32 RMSNorm, sigmoid gating, and output conversion into a tensor-only
compiled helper. Median MFU rose from 38.600% to 41.770%, while peak memory fell
to 236.12 GiB.

### Retain KDA intermediates

Commit: `c37e29a6b` (`kimi_k3: optionally retain KDA intermediates`)

Exposed FLA KDA's `disable_recompute` option and enabled it for the benchmark.
The extra retained intermediates trade memory for less backward recomputation.

Median MFU increased from 41.770% to 42.135%.

### Fused attention-residual reduction

Commit: `abce7fca8` (`kimi_k3: fuse attention residual reduction`)

Replaced the small batched matrix multiplication used for the final weighted
residual reduction with an elementwise multiply plus reduction. This gave
Inductor a better fusion opportunity.

Median MFU increased from 42.135% to 44.045%, and peak memory fell from 239.66
GiB to 231.12 GiB.

### KDA chunk and state-layout tuning

Commit: `12132678a` (`kimi_k3: tune KDA state layout`)

Exposed and benchmarked FLA KDA's `chunk_size` and `state_v_first` options. The
winning configuration uses:

```python
disable_recompute = True
chunk_size = 32
state_v_first = True
```

The full-model median reached 44.305% MFU.

### Selected-logit forced router

Commit: `472c526d6` (`moe: compute only forced routing logits`)

The normal forced-load-balance router computed all 896 expert logits, replaced
the selected expert IDs with deterministic round-robin IDs, and then retained
only 16 logits per token. The optimized path:

- constructs the same round-robin expert IDs;
- gathers only the selected gate rows;
- computes the selected logits with grouped batched matrix multiplication;
- preserves sigmoid, route normalization, route scaling, gate bias, and
  gradients;
- computes exact per-expert counts analytically instead of materializing a full
  token-by-expert boolean routing map;
- leaves normal dynamic routing unchanged.

A representative router microbenchmark improved forward + backward from
approximately 19.26 ms to 1.75 ms. The full-model result improved from 44.305%
to **50.305% median MFU**, with throughput rising from 1,722 to 1,955 tokens/s.

## 5. Kimi K3 benchmark progression

Steady-state medians exclude step 1 and the final profiled step.

| Run | Main change | Batch | Median MFU | Median tokens/s | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: |
| `20260826T002253Z` | Initial full-model baseline | 1 | 14.165% | 551 | 137.99 GiB |
| `compile_candidate_graphbreak_b1` | Block compile only | 1 | 12.295% | 478 | 137.99 GiB |
| `fused_causal_conv_b1` | Fused causal conv | 1 | 15.475% | 602 | 133.44 GiB |
| `true_batch2` | Independent KDA sequences | 2 | 17.610% | 685 | 162.21 GiB |
| `true_batch4` | Larger local batch | 4 | 19.710% | 766 | 209.20 GiB |
| `batched_sdpa_batch4` | Batched MLA SDPA | 4 | 20.240% | 787 | 211.14 GiB |
| `compiled_situ_batch4` | Compiled SiTU | 4 | 26.530% | 1,032 | 201.39 GiB |
| `compiled_situ_batch6` | Larger local batch | 6 | 27.290% | 1,061 | 243.19 GiB |
| `compiled_residual_batch6` | Compiled attention residual | 6 | 37.855% | 1,471 | 208.62 GiB |
| `compiled_residual_batch8` | Batch 8 | 8 | 38.600% | 1,500 | 242.68 GiB |
| `compiled_kda_norm_batch8` | Compiled gated RMSNorm | 8 | 41.770% | 1,624 | 236.12 GiB |
| `kda_no_recompute_batch8` | Retain KDA intermediates | 8 | 42.135% | 1,638 | 239.66 GiB |
| `residual_reduction_batch8` | Fused residual reduction | 8 | 44.045% | 1,712 | 231.12 GiB |
| `kda_chunk32_vfirst_batch8` | KDA chunk/state tuning | 8 | 44.305% | 1,722 | 239.00 GiB |
| `selected_router_batch8` | Selected-logit forced router | 8 | **50.305%** | **1,955** | **240.00 GiB** |

Overall, median MFU improved by **3.55x**, and median throughput improved by
**3.55x**, relative to the initial Kimi K3 full-model baseline.

## 6. Kimi experiments not retained

- **Batch 10:** reached about 44.36% MFU but repeatedly triggered allocator OOM
  retries, peaked around 265 GiB, and did not materially outperform batch 8.
  The launcher default remains batch 8.
- **CUDA graphs:** capture failed in the vision encoder at a dynamic
  `grid_thw.tolist()` call. CUDA graphs remain disabled; the earlier varlen
  CUDA-graph workaround was not reintroduced.
- **No activation checkpointing:** rejected because MinimalAsyncEP requires
  Full AC in the eager trainer, or the graph trainer's full memory policy.
- **Q/K/V/output-gate projection fusion:** a local forward/backward
  microbenchmark was slower and used substantially more memory than separate
  projections, so it was not adopted.
- **TF32 router GEMM:** substantially faster in isolation but rejected because
  the router intentionally computes in FP32 for routing stability and TF32 can
  change routing/numerics.
- **KDA `max_num_documents=32` and varlen CUDA-graph fixes:** tested and then
  reverted as requested.

## 7. Validation

The final selected-router change was validated with:

- an exact forward comparison against the original full-logit forced router;
- matching expert IDs;
- input, gate-weight, and gate-bias gradient comparisons;
- CPU MoE tests: 2 passed;
- Kimi K3 unit tests: 11 passed;
- file-scoped pre-commit checks: passed, with repository-wide Pyrefly skipped
  because of unrelated pre-existing failures;
- a completed 10-step full-model benchmark with a final GPU trace.

The KDA, causal convolution, batched attention, compiled SiTU, compiled
residual, and compiled gated-normalization changes also include targeted
forward/backward regression tests in `tests/unit_tests/test_kimi_k3.py`.

## 8. Reproduction

Kimi K3 final benchmark:

```bash
CUDA_VISIBLE_DEVICES=1 \
BENCHMARK_STEPS=10 \
SEQ_LEN=4096 \
LOCAL_BATCH_SIZE=8 \
PROFILE_WARMUP_STEPS=0 \
PROFILE_ACTIVE_STEPS=1 \
RUN_ID=selected_router_batch8 \
bash outputs/kimi_k3_full_minimal_async_ep_fake/launch.sh
```

Relevant artifacts:

- Kimi launcher: `outputs/kimi_k3_full_minimal_async_ep_fake/launch.sh`
- Kimi config: `outputs/kimi_k3_full_minimal_async_ep_fake/kimi_k3_fake_profile_config.py`
- Final log: `outputs/kimi_k3_full_minimal_async_ep_fake/benchmark_runs/selected_router_batch8/console.log`
- Local trace: `outputs/kimi_k3_full_minimal_async_ep_fake/benchmark_runs/selected_router_batch8/gpu_trace/iteration_10/rank0_trace.json.gz`
- DSV3-671B launcher: `outputs/dsv3_671b_minimal_async_ep_fake/launch.sh`
- Trace uploader: `outputs/dsv3_16b_minimal_async_ep_fake/upload_trace.py`

Final Kimi trace:

<https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fbahuang_d9e382a5-3ff7-417a-b640-0f60d8ecf59e_rank0_trace.json.gz>

## 9. Commit sequence

```text
ff096a651 minimal_async_ep: bound receive buffer capacity
d7064bc27 minimal_async_ep: add fake backend benchmark support
d963aab83 kimi_k3: enable minimal async expert parallelism
00aba2433 kimi_k3: compile blocks around KDA graph breaks
5dc32cb59 kimi_k3: fuse causal convolution kernels
29f1eae60 kimi_k3: batch independent recurrent sequences
b682dae17 kimi_k3: batch MLA with scaled dot product attention
6d1beb922 kimi_k3: compile SiTU activation
f1c67da27 kimi_k3: compile attention residual
31b38e7b8 benchmarks: raise Kimi K3 local batch size
6df48fc0f benchmarks: raise Kimi K3 local batch size to eight
5e07e360e kimi_k3: compile gated RMSNorm
c37e29a6b kimi_k3: optionally retain KDA intermediates
abce7fca8 kimi_k3: fuse attention residual reduction
12132678a kimi_k3: tune KDA state layout
472c526d6 moe: compute only forced routing logits
```
