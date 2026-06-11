# DSv3 Scaling — Experiment Results

Validation runs for the `graph_trainer/dsv3_scaling` branch. One section per run.

---

## Run 1 — deepseek_v3 16B baseline (run script + #3419)

**Date:** 2026-06-09
**Branch:** `graph_trainer/dsv3_scaling`
**Branch state at run:** run-script commit + #3419 (`Replace stable_topological_sort
with _move_overlap_nodes`). #3248/#3590/#3561/EP-overlap/#3548 **not yet applied**.
**Launcher:** `./run_graph_trainer_dsv3.sh`

### Setup
| | |
|---|---|
| Hardware | 8× NVIDIA H100 (95.07 GiB each) |
| Model | deepseek_v3 16B — 15,706,484,224 total params (dense 858M, sparse 14.85B, active 2.66B) |
| Parallelism | `pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=4` |
| Compile | `aot_fx_trace`, `memory_policy=full`, cudagraph_pass disabled |
| Batch / seq | local batch 4, global batch 32, grad-accum 1, seq_len 4096 |
| Steps | 20 (c4_test dataset) |
| Peak FLOPS (MFU denom) | 9.890e14 |
| Model-only CUDA mem | 7.83 GiB (8.23%) |

### Results
| step | loss | grad_norm | memory | tps | tflops | mfu |
|-----:|------|-----------|--------|-----|--------|-----|
| 1 | 12.03895 | 1.6740 | 33.03 GiB (34.74%) | 95 | 1.73 | 0.17% |
| 10 | 9.18397 | 7.6505 | 51.24 GiB (53.89%) | 6,493 | 117.55 | 11.89% |
| 20 | 7.53410 | 10.6447 | 51.24 GiB (53.89%) | 5,451 | 98.70 | 9.98% |

Loss converges as expected (12.04 → 9.18 → 7.53). Step 1 reflects graph-capture
/ compile cost; steady-state memory ~51.2 GiB.

### Compile pipeline
All 9 graph passes took **45.884 s**. Notable transforms:
- `joint_transformer_block_bucketing_reordering` (8.95 s): FSDP collectives
  bucketed — `all_gather_into_tensor` 830 → 0 (replaced by 109 bucketed
  `all_gather_into_tensor_out`), `reduce_scatter_tensor` 377 → 55.
- `regional_inductor` (23.50 s): `flex_attention` 54 → 0 and
  `flex_attention_backward` 27 → 0, compiled into fused `inner` regional kernels.

### Artifacts
| artifact | link |
|---|---|
| run log (pastry) | https://www.internalfb.com/intern/paste/P2371489935/ |
| profiler trace (perfetto, rank0 iter 10) | https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fbahuang_cff2050a-09be-4c5b-bc23-ed6e7c0c3ca1_rank0_trace.json.gz |
| tlparse logs (manifold, **temporary** `.tmp` path) | https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/.tmpfcPEdJ/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000 |

#### Per-pass before/after graph diffs
| pass | diff |
|---|---|
| eliminate_dead_code | https://www.internalfb.com/intern/diffing/?before_paste_number=2371487936&after_paste_number=2371488084&selected_tab=plain_diff |
| canonicalize_graph | https://www.internalfb.com/intern/diffing/?before_paste_number=2371487672&after_paste_number=2371487810&selected_tab=plain_diff |
| tag_with_memory_policy | https://www.internalfb.com/intern/diffing/?before_paste_number=2371489378&after_paste_number=2371489494&selected_tab=plain_diff |
| selective_activation_remat | https://www.internalfb.com/intern/diffing/?before_paste_number=2371489120&after_paste_number=2371489248&selected_tab=plain_diff |
| reassign_collective_pgs | https://www.internalfb.com/intern/diffing/?before_paste_number=2371488540&after_paste_number=2371488690&selected_tab=plain_diff |
| joint_transformer_block_bucketing_reordering | https://www.internalfb.com/intern/diffing/?before_paste_number=2371488228&after_paste_number=2371488382&selected_tab=plain_diff |
| annotate_flex_attention_for_regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2371487374&after_paste_number=2371487529&selected_tab=plain_diff |
| regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2371488829&after_paste_number=2371488974&selected_tab=plain_diff |

#### Standalone artifacts
| artifact | paste |
|---|---|
| activation_memory_policy | https://www.internalfb.com/intern/paste/P2371489565/ |
| fx_compute_nodes_runtime_estimation | https://www.internalfb.com/intern/paste/P2371489665/ |

### Notes
- Required copying the `deepseek-moe-16b-base` tokenizer asset into `assets/hf/`
  (missing from this checkout; sourced from `~/local/torchtitan`).
- The flex_attention "unfused implementation" warning at startup is expected —
  it fires during the eager/trace warmup before `regional_inductor` compiles flex.
- tlparse manifold link is a temporary `.tmp...` path and will expire; the pastry
  log and perfetto trace are durable.

---

## Run 2 — deepseek_v3 16B with ChunkedCELoss (run script + #3419 + #3248)

**Date:** 2026-06-09
**Branch:** `graph_trainer/dsv3_scaling`
**Branch state at run:** run-script commit + #3419 + #3248 (Enable ChunkedCELoss).
#3590/#3561/EP-overlap/#3548 **not yet applied**.
**Launcher:** `TORCHINDUCTOR_COMPILE_THREADS=8 ./run_graph_trainer_dsv3.sh`
(thread cap set at launch — not in the committed script — to avoid a cold-cache
compile OOM; see Notes).

> ⚠️ **UNVERIFIED — possible loss divergence; needs further confirmation.**
> I'm concerned the loss may be diverging. These numbers have **not** been
> validated against an eager run, so the chunked-loss numerics are not yet
> trusted. **TODO before relying on this result:** compare against an eager
> `ChunkedCELoss` baseline via `scripts/loss_compare.py` (ideally bitwise with
> `--debug.seed=42 --debug.deterministic`, checking loss *and* grad_norm from
> TensorBoard, not just the 5-digit stdout). Treat the loss curve below as
> **provisional** pending that confirmation.

### Setup
Same as Run 1 (8× H100, `dp_shard=8 ep=4`, `aot_fx_trace`, `memory_policy=full`,
local batch 4 / seq 4096, 20 steps, c4_test) **except the loss**:

| | |
|---|---|
| Loss | **ChunkedCELoss → `ChunkedCELossWithParamGrads`** (`num_chunks=8`) |

#3248 makes `to_graph_trainer_config` swap `ChunkedCELoss` to the tracer-friendly
`ChunkedCELossWithParamGrads` instead of the previous `CrossEntropyLoss` fallback.

### Results
| step | loss | grad_norm | memory | tps | tflops | mfu |
|-----:|------|-----------|--------|-----|--------|-----|
| 1 | 12.02673 | 1.6668 | 29.34 GiB (30.86%) | 88 | 1.59 | 0.16% |
| 10 | 9.04778 | 5.0626 | 35.73 GiB (37.59%) | 7,123 | 128.97 | 13.04% |
| 20 | 7.07054 | 1.4865 | 36.07 GiB (37.93%) | 5,423 | 98.19 | 9.93% |

**Memory vs Run 1 (CrossEntropyLoss): ~36 GiB vs ~51 GiB — ChunkedCELoss saves
~15 GiB**, the intended #3248 benefit (matching eager ChunkedLoss memory). Loss
converges normally.

### Compile pipeline
All 9 graph passes took **113.558 s** (cold-ish Inductor cache):
- `joint_transformer_block_bucketing_reordering` 9.28 s
- `regional_inductor` 89.92 s

### Artifacts
| artifact | link |
|---|---|
| run log (pastry) | https://www.internalfb.com/intern/paste/P2371696211/ |
| profiler trace (perfetto, rank0 iter 10) | https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fbahuang_53437836-36e9-452c-80aa-f9a98b44e4d7_rank0_trace.json.gz |
| tlparse logs (manifold, **temporary** `.tmp` path) | https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/.tmpNACnGq/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000 |

#### Per-pass before/after graph diffs
| pass | diff |
|---|---|
| eliminate_dead_code | https://www.internalfb.com/intern/diffing/?before_paste_number=2371695016&after_paste_number=2371695095&selected_tab=plain_diff |
| canonicalize_graph | https://www.internalfb.com/intern/diffing/?before_paste_number=2371694852&after_paste_number=2371694922&selected_tab=plain_diff |
| tag_with_memory_policy | https://www.internalfb.com/intern/diffing/?before_paste_number=2371695911&after_paste_number=2371695985&selected_tab=plain_diff |
| selective_activation_remat | https://www.internalfb.com/intern/diffing/?before_paste_number=2371695733&after_paste_number=2371695832&selected_tab=plain_diff |
| reassign_collective_pgs | https://www.internalfb.com/intern/diffing/?before_paste_number=2371695378&after_paste_number=2371695465&selected_tab=plain_diff |
| joint_transformer_block_bucketing_reordering | https://www.internalfb.com/intern/diffing/?before_paste_number=2371695203&after_paste_number=2371695291&selected_tab=plain_diff |
| annotate_flex_attention_for_regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2371694655&after_paste_number=2371694746&selected_tab=plain_diff |
| regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2371695568&after_paste_number=2371695645&selected_tab=plain_diff |

#### Standalone artifacts
| artifact | paste |
|---|---|
| activation_memory_policy | https://www.internalfb.com/intern/paste/P2371696021/ |
| fx_compute_nodes_runtime_estimation | https://www.internalfb.com/intern/paste/P2371696070/ |

### Notes
- **ChunkedCELoss confirmed** via `--debug.print-config`: `loss.num_chunks = 8`
  (`CrossEntropyLoss` has no `num_chunks`), and `compile.components` includes
  `"loss"` so the chunked loss is compiled.
- **Cold-cache compile OOM (workaround applied):** the first two attempts were
  SIGKILL'd during `regional_inductor` on a cold Inductor cache. It was *not* a
  steady host-RAM OOM (1 s sampling showed peak ~214 GiB / 2 TiB); fresh
  compilation on this 368-core host spawned a large compile-worker pool whose
  transient spike tripped the OS OOM-killer. Capping
  `TORCHINDUCTOR_COMPILE_THREADS=8` and warming the Inductor cache let it
  complete. Consider baking the cap into the launcher for cold-start reliability.

---

## Run 3 — deepseek_v3 16B with MinimalAsyncEP (run script + #3419 + #3248 + #3561)

**Date:** 2026-06-09
**Branch:** `graph_trainer/dsv3_scaling`
**Branch state at run:** run-script + #3419 + #3248 + #3561 (MinimalAsyncEP).
#3590 / EP-overlap / #3548 **not yet applied**.
**Launcher:** `TORCHINDUCTOR_COMPILE_THREADS=8 ./run_graph_trainer_dsv3.sh` with
`CONFIG=graph_trainer_deepseek_v3_16b_minimal_async_ep` (the committed script's
generic `graph_trainer_deepseek_v3_16b` was temporarily switched to the
MinimalAsyncEP variant for this run).

> ⚠️ **UNVERIFIED — numerics not validated.** Same loss-divergence concern as
> Run 2, and MinimalAsyncEP is a **new sync-free MoE dispatcher** — its numerics
> need an eager comparison (eager `deepseek_v3_16b_minimal_async_ep` via
> `scripts/loss_compare.py`, ideally bitwise with `--debug.seed=42
> --debug.deterministic`, loss *and* grad_norm) before this result can be
> trusted. Treat the loss curve below as **provisional.**

### MinimalAsyncEP confirmed in use
Symmetric-memory buffer init logged at startup (proves the sync-free dispatcher
is active, not a fallback):
`Initializing MinimalAsyncEP buffer: hidden_dim=2048, tokens_per_rank=16384,
top_k=6, num_local_experts=16, ep_size=4, max_routed_tokens=393216`.

### Setup
Same as Run 2 (8× H100, `dp_shard=8 ep=4`, `aot_fx_trace`, `memory_policy=full`,
ChunkedCELoss, B=4 / seq 4096, 20 steps, c4_test) **except the MoE dispatcher**:

| | |
|---|---|
| MoE dispatcher | **MinimalAsyncEPTokenDispatcher** (sync-free EP via symmetric memory) |
| Config | `graph_trainer_deepseek_v3_16b_minimal_async_ep` |

### Results
| step | loss | grad_norm | memory | tps | tflops | mfu |
|-----:|------|-----------|--------|-----|--------|-----|
| 1 | 12.09197 | 1.6856 | 29.34 GiB (30.86%) | 95 | 1.72 | 0.17% |
| 10 | 9.10410 | 6.9499 | 36.06 GiB (37.93%) | 8,221 | 148.84 | 15.05% |
| 20 | 7.21462 | 8.1240 | 36.06 GiB (37.93%) | 5,504 | 99.66 | 10.08% |

Loss converges; memory ~36 GiB (same band as Run 2). step-10 mfu 15.05%
(vs Run 2's 13.04%). **Numerics unverified — see caveat above.**

### Compile pipeline
All 9 graph passes took **94.834 s** (`TORCHINDUCTOR_COMPILE_THREADS=8`; cold
Inductor cache for the new MinimalAsyncEP / offset-aware swiglu kernels):
- `joint_transformer_block_bucketing_reordering` 9.25 s
- `regional_inductor` 74.05 s

### Artifacts
| artifact | link |
|---|---|
| run log (pastry) | https://www.internalfb.com/intern/paste/P2371964742/ |
| profiler trace (perfetto, rank0 iter 10) | https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fbahuang_de493592-0d3f-40da-9376-ab01153b39e9_rank0_trace.json.gz |
| tlparse logs (manifold, **temporary** `.tmp` path) | https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/.tmpd3wvEW/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000 |

#### Per-pass before/after graph diffs
| pass | diff |
|---|---|
| eliminate_dead_code | https://www.internalfb.com/intern/diffing/?before_paste_number=2371964059&after_paste_number=2371964141&selected_tab=plain_diff |
| canonicalize_graph | https://www.internalfb.com/intern/diffing/?before_paste_number=2371963719&after_paste_number=2371963790&selected_tab=plain_diff |
| tag_with_memory_policy | https://www.internalfb.com/intern/diffing/?before_paste_number=2371964490&after_paste_number=2371964578&selected_tab=plain_diff |
| selective_activation_remat | https://www.internalfb.com/intern/diffing/?before_paste_number=2371964352&after_paste_number=2371964417&selected_tab=plain_diff |
| annotate_flex_attention_for_regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2371963554&after_paste_number=2371963641&selected_tab=plain_diff |
| regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2371964223&after_paste_number=2371964281&selected_tab=plain_diff |

#### Standalone artifacts
| artifact | paste |
|---|---|
| activation_memory_policy | https://www.internalfb.com/intern/paste/P2371964600/ |

### Notes
- MinimalAsyncEP parallelism requirements satisfied: `ep=4>1`, `tp=1`,
  `dp_shard=8 % ep=0`, no CP/PP/sequence-parallel, `memory_policy=full`, and
  `spmd_backend=default` (MinimalAsyncEP rejects `full_dtensor`).
- Cold-cache compile mitigation as in Run 2 (`TORCHINDUCTOR_COMPILE_THREADS=8`);
  the MinimalAsyncEP / swiglu kernels are new, so their first compile is cold.
- The committed launcher still uses `graph_trainer_deepseek_v3_16b`; the
  MinimalAsyncEP config was set only for this run.
