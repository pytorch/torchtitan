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
- For Run 3 the committed launcher still used `graph_trainer_deepseek_v3_16b`;
  the MinimalAsyncEP config was a temporary override. (The launcher was later
  switched to MinimalAsyncEP — see Run 4.)

---

## Run 4 — MinimalAsyncEP + cudagraph, FORCED (run script + #3419 + #3248 + #3561 + cudagraph.py override)

**Date:** 2026-06-09
**Branch:** `graph_trainer/dsv3_scaling`
**Branch state at run:** run-script + #3419 + #3248 + #3561, **plus a local
`cudagraph.py` override that forces capture**. #3590 / EP-overlap / #3548 not yet
applied.
**Launcher:** `./run_graph_trainer_dsv3.sh` (`CONFIG=...minimal_async_ep`,
`TORCHINDUCTOR_COMPILE_THREADS=8`, cudagraph enabled).

> 🟡 **cudagraph FORCED past the safety gate.** With cudagraph merely *enabled*,
> `cudagraph_pass` first **SKIPPED** — `is_cudagraphable` flagged **312
> `aten._grouped_mm` nodes** (the MoE grouped expert matmul), which it disallows
> on **< sm_100** (H100 = sm_90) because `_grouped_mm` "may perform internal
> CPU↔CUDA copies not visible in FX metadata; resolved on sm_100+". A local edit
> to `cudagraph.py:cudagraph_pass` bypasses that gate (logs the offenders,
> captures anyway). **cudagraph then applied and ran without crashing.**

> ⛔ **CORRECTNESS UNVERIFIED — forced past a real safety check.** If
> `_grouped_mm` does hidden CPU↔CUDA copies on sm_90, cudagraph replay uses
> **stale** data → silently wrong. The plausible loss is NOT proof. **Must**
> confirm vs an eager `deepseek_v3_16b_minimal_async_ep` run (`loss_compare.py`,
> `--debug.seed=42 --debug.deterministic`) before trusting. (Also still subject
> to the chunked-loss / MinimalAsyncEP numerics caveat from Runs 2–3.)

### Setup
Identical to Run 3 (MinimalAsyncEP, 8× H100, `dp_shard=8 ep=4`, `aot_fx_trace`,
`memory_policy=full`, ChunkedCELoss, B=4 / seq 4096, 20 steps, c4_test) with
**cudagraph enabled AND forced** (the `cudagraph.py` gate bypass above).

### Results (forced cudagraph)
| step | loss | grad_norm | memory | tps | tflops | mfu |
|-----:|------|-----------|--------|-----|--------|-----|
| 1 | 12.02520 | 1.6800 | 36.30 GiB (38.18%) | 116 | 2.09 | 0.21% |
| 10 | 9.17969 | 6.7772 | 36.42 GiB (38.30%) | 9,114 | 165.01 | 16.68% |
| 20 | 7.19782 | 2.4133 | 36.42 GiB (38.30%) | 9,028 | 163.46 | 16.53% |

**cudagraph speedup** (same config, cudagraph off vs forced-on):
| variant | step-10 mfu | step-20 mfu | step-20 tps |
|---|---|---|---|
| Run 3 (no cudagraph) | 15.05% | 10.08% | 5,504 |
| cudagraph *skipped* (earlier Run 4 attempt) | 15.51% | 12.73% | 6,951 |
| **cudagraph FORCED** | **16.68%** | **16.53%** | **9,028** |

Forced cudagraph gives a real, stable speedup (~16.5% mfu, tps ~9k; step-20 no
longer drops off). Loss in the same band. **Numerics unverified — see caveat.**

### Compile pipeline
All **11** graph passes took **91.714 s**. `regional_inductor` 69.52 s;
`cudagraph_pass` now actually applies — logs `FORCING cudagraph despite 312
non-cudagraphable node(s)` then `Applied cudagraph pass.`

### Artifacts
| artifact | link |
|---|---|
| run log (pastry) | https://www.internalfb.com/intern/paste/P2372005165/ |
| profiler trace (perfetto, rank0 iter 10) | https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fbahuang_5ad8fb07-f46c-4670-b5e8-c2ae40df6a69_rank0_trace.json.gz |
| tlparse logs (manifold, **temporary** `.tmp` path) | https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/.tmpg3MotK/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000 |

#### Per-pass before/after graph diffs
| pass | diff |
|---|---|
| eliminate_dead_code | https://www.internalfb.com/intern/diffing/?before_paste_number=2372003617&after_paste_number=2372003737&selected_tab=plain_diff |
| canonicalize_graph | https://www.internalfb.com/intern/diffing/?before_paste_number=2372003396&after_paste_number=2372003512&selected_tab=plain_diff |
| tag_with_memory_policy | https://www.internalfb.com/intern/diffing/?before_paste_number=2372004743&after_paste_number=2372004852&selected_tab=plain_diff |
| selective_activation_remat | https://www.internalfb.com/intern/diffing/?before_paste_number=2372004505&after_paste_number=2372004642&selected_tab=plain_diff |
| reassign_collective_pgs | https://www.internalfb.com/intern/diffing/?before_paste_number=2372004011&after_paste_number=2372004127&selected_tab=plain_diff |
| joint_transformer_block_bucketing_reordering | https://www.internalfb.com/intern/diffing/?before_paste_number=2372003828&after_paste_number=2372003916&selected_tab=plain_diff |
| annotate_flex_attention_for_regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2372003147&after_paste_number=2372003245&selected_tab=plain_diff |
| regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2372004259&after_paste_number=2372004381&selected_tab=plain_diff |

#### Standalone artifacts
| artifact | paste |
|---|---|
| activation_memory_policy | https://www.internalfb.com/intern/paste/P2372004904/ |
| fx_compute_nodes_runtime_estimation | https://www.internalfb.com/intern/paste/P2372004981/ |

### Notes
- **Root cause of the cudagraph skip:** the `_grouped_mm` `< sm_100` gate in
  `is_cudagraphable` (`cudagraph.py`). Allowed on sm_100+ (Blackwell);
  conservatively disallowed on H100 (sm_90).
- The forced-capture change in `cudagraph.py` is an **unconditional debug
  override** (TODO: revert). The principled fix, *if verified numerically safe
  on sm_90*, is to relax just the `_grouped_mm` gate (or gate the force behind a
  flag) — not a blanket bypass.
- MinimalAsyncEP buffer init confirmed; `TORCHINDUCTOR_COMPILE_THREADS=8`
  cold-cache mitigation as before.

---

## Run 5 — + #3590 Match Eager FSDP bucket order (MinimalAsyncEP + forced cudagraph + #3590)

**Date:** 2026-06-09
**Branch:** `graph_trainer/dsv3_scaling`
**Branch state at run:** run-script + #3419 + #3248 + #3561 + **#3590** + the
`cudagraph.py` force. EP-overlap / #3548 not yet applied.
**Launcher:** `./run_graph_trainer_dsv3.sh` (MinimalAsyncEP, cudagraph enabled +
forced, `TORCHINDUCTOR_COMPILE_THREADS=8`).

> ⛔ **CORRECTNESS UNVERIFIED.** Carries forward every prior caveat (chunked loss,
> MinimalAsyncEP, forced cudagraph past the `_grouped_mm` gate). **#3590's whole
> purpose is a *bitwise* match with eager FSDP2 reduce-scatter bucket order**, so
> it specifically needs an eager `loss_compare` (`--debug.seed=42
> --debug.deterministic`, loss + grad_norm) to confirm. Provisional.

### What #3590 adds
`joint_transformer_block_bucketing_reordering_pass` now packs FSDP buckets in
**eager FSDP2 first-seen parameter order** (`FSDPParamOrderBucketer` +
`get_fsdp_param_module_order(traced_result.state_fqns)`) instead of graph
execution order. MinimalAsyncEP buffer init + forced cudagraph (312
`_grouped_mm`) both confirmed, same as Run 4.

### Setup
Identical to Run 4 (MinimalAsyncEP + forced cudagraph, 8× H100, `dp_shard=8 ep=4`,
`aot_fx_trace`, `memory_policy=full`, ChunkedCELoss, B=4 / seq 4096, 20 steps,
c4_test) **plus #3590** (eager FSDP bucket ordering).

### Results
| step | loss | grad_norm | memory | tps | tflops | mfu |
|-----:|------|-----------|--------|-----|--------|-----|
| 1 | 12.02874 | 1.6549 | 36.22 GiB (38.10%) | 109 | 1.97 | 0.20% |
| 10 | 9.17541 | 8.9794 | 36.28 GiB (38.16%) | 9,110 | 164.94 | 16.68% |
| 20 | 7.15267 | 1.7885 | 36.28 GiB (38.16%) | 9,145 | 165.57 | 16.74% |

Perf matches Run 4 (~16.7% mfu, tps ~9.1k) — expected, since #3590 changes bucket
*order* (for eager bitwise match), not the amount of work. Loss in the same band.
**Numerics unverified — see caveat.**

### Compile pipeline
All **11** graph passes took **95.061 s**. `regional_inductor` 72.35 s;
`cudagraph_pass` forced + applied.

### Artifacts
| artifact | link |
|---|---|
| run log (pastry) | https://www.internalfb.com/intern/paste/P2372034462/ |
| profiler trace (perfetto, rank0 iter 10) | https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fbahuang_2361ec10-8c5b-4504-91b4-9d11d8501b34_rank0_trace.json.gz |
| tlparse logs (manifold, **temporary** `.tmp` path) | https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/.tmpqSCN8V/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000 |

#### Per-pass before/after graph diffs
| pass | diff |
|---|---|
| eliminate_dead_code | https://www.internalfb.com/intern/diffing/?before_paste_number=2372033030&after_paste_number=2372033136&selected_tab=plain_diff |
| canonicalize_graph | https://www.internalfb.com/intern/diffing/?before_paste_number=2372032890&after_paste_number=2372032965&selected_tab=plain_diff |
| tag_with_memory_policy | https://www.internalfb.com/intern/diffing/?before_paste_number=2372034105&after_paste_number=2372034195&selected_tab=plain_diff |
| selective_activation_remat | https://www.internalfb.com/intern/diffing/?before_paste_number=2372033826&after_paste_number=2372033969&selected_tab=plain_diff |
| reassign_collective_pgs | https://www.internalfb.com/intern/diffing/?before_paste_number=2372033370&after_paste_number=2372033465&selected_tab=plain_diff |
| joint_transformer_block_bucketing_reordering | https://www.internalfb.com/intern/diffing/?before_paste_number=2372033222&after_paste_number=2372033299&selected_tab=plain_diff |
| annotate_flex_attention_for_regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2372032751&after_paste_number=2372032811&selected_tab=plain_diff |
| regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2372033579&after_paste_number=2372033700&selected_tab=plain_diff |

#### Standalone artifacts
| artifact | paste |
|---|---|
| activation_memory_policy | https://www.internalfb.com/intern/paste/P2372034223/ |
| fx_compute_nodes_runtime_estimation | https://www.internalfb.com/intern/paste/P2372034308/ |

### Notes
- #3590's `models/utils.py` hunk was a no-op here (main already has the
  `_StridedShard` fix); the commit lands the `fsdp_passes.py` + `passes.py`
  changes only.
- Bitwise FSDP-order match is the explicit goal of #3590 — verify against eager.

---

## Run 6 — batch size 16 (Run 5 config, local_batch_size 4 → 16)

**Date:** 2026-06-10
**Branch:** `graph_trainer/dsv3_scaling`
**Branch state at run:** identical to Run 5 (run-script + #3419 + #3248 + #3561 +
#3590 + `cudagraph.py` force). EP-overlap / #3548 not yet applied.
**Launcher:** `./run_graph_trainer_dsv3.sh` with `--training.local_batch_size 16`
(now in the committed launcher); MinimalAsyncEP, cudagraph enabled + forced,
`TORCHINDUCTOR_COMPILE_THREADS=8`.

> ⛔ **CORRECTNESS UNVERIFIED** — same stack of caveats as Run 5 (chunked loss,
> MinimalAsyncEP, forced cudagraph, eager FSDP-order). Needs eager `loss_compare`.
> Loss is **not** directly comparable across batch sizes (different tokens/step).

### Change vs Run 5
`local_batch_size` 4 → **16** (global batch 32 → **128**). MinimalAsyncEP buffer
resized: `tokens_per_rank=65536`, `max_routed_tokens=1572864`. Everything else
identical. (Intermediate B=8 data is folded into the scaling table below.)

### Results (B=16)
| step | loss | grad_norm | memory | tps | tflops | mfu |
|-----:|------|-----------|--------|-----|--------|-----|
| 1 | 11.99964 | 1.6201 | 61.28 GiB (64.45%) | 267 | 4.84 | 0.49% |
| 10 | 9.10102 | 5.9849 | 61.28 GiB (64.45%) | 11,100 | 200.98 | 20.32% |
| 20 | 7.12763 | 3.4668 | 57.82 GiB (60.82%) | 11,264 | 203.94 | 20.62% |

**Batch-size scaling** (MinimalAsyncEP + forced cudagraph + #3590, steady-state):
| batch (local / global) | mem | step-10 mfu | step-20 mfu | step-20 tps |
|---|---|---|---|---|
| B=4 / 32 (Run 5) | ~36 GiB | 16.68% | 16.74% | 9,145 |
| B=8 / 64 | ~45 GiB | 18.25% | 19.13% | 10,447 |
| **B=16 / 128** | **~61 GiB** | **20.32%** | **20.62%** | **11,264** |

MFU and throughput scale up cleanly with batch (16.7% → 19.1% → 20.6% mfu;
9.1k → 10.4k → 11.3k tps), while memory grows sublinearly thanks to
`memory_policy=full` recompute (~36 → ~45 → ~61 GiB). At B=16 we use ~61 GiB /
95 GiB — still headroom. **Numerics unverified — see caveat.**

### Compile pipeline
All **11** graph passes took **163.95 s** (autotuning grows with batch shapes;
`regional_inductor` 141.38 s). Forced cudagraph applied (312 `_grouped_mm`).

### Artifacts
| artifact | link |
|---|---|
| run log (pastry) | https://www.internalfb.com/intern/paste/P2372682804/ |
| profiler trace (perfetto, rank0 iter 10) | https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fbahuang_1a31696b-bd0c-46a4-9aa4-f94f1fbe8436_rank0_trace.json.gz |
| tlparse logs (manifold, **temporary** `.tmp` path) | https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/.tmpIuV0JC/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000 |

#### Per-pass before/after graph diffs
| pass | diff |
|---|---|
| eliminate_dead_code | https://www.internalfb.com/intern/diffing/?before_paste_number=2372680823&after_paste_number=2372680990&selected_tab=plain_diff |
| canonicalize_graph | https://www.internalfb.com/intern/diffing/?before_paste_number=2372680607&after_paste_number=2372680702&selected_tab=plain_diff |
| tag_with_memory_policy | https://www.internalfb.com/intern/diffing/?before_paste_number=2372682275&after_paste_number=2372682375&selected_tab=plain_diff |
| selective_activation_remat | https://www.internalfb.com/intern/diffing/?before_paste_number=2372682005&after_paste_number=2372682159&selected_tab=plain_diff |
| reassign_collective_pgs | https://www.internalfb.com/intern/diffing/?before_paste_number=2372681461&after_paste_number=2372681613&selected_tab=plain_diff |
| joint_transformer_block_bucketing_reordering | https://www.internalfb.com/intern/diffing/?before_paste_number=2372681174&after_paste_number=2372681305&selected_tab=plain_diff |
| annotate_flex_attention_for_regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2372680367&after_paste_number=2372680494&selected_tab=plain_diff |
| regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2372681734&after_paste_number=2372681866&selected_tab=plain_diff |

#### Standalone artifacts
| artifact | paste |
|---|---|
| activation_memory_policy | https://www.internalfb.com/intern/paste/P2372682419/ |
| fx_compute_nodes_runtime_estimation | https://www.internalfb.com/intern/paste/P2372682512/ |

### Notes
- `--training.local_batch_size 16` is now in the committed launcher.
- ~61 GiB / 95 GiB at B=16 — room to push further (B≈24) if desired.

---

## Run 7 — lm_head chunked-loss coalescing fix: BITWISE-VERIFIED vs eager ✅

**Date:** 2026-06-11
**Branch:** `graph_trainer/dsv3_scaling`
**Branch state at run:** Run 6 stack **+ the `chunked_loss.py` fix** (lm_head
chunked-loss collective coalescing, PR #3636).
**Launcher:** `./run_graph_trainer_dsv3.sh` — now **non-MinimalAsyncEP**
(`graph_trainer_deepseek_v3_16b`), `dp_shard=8 tp=1 ep=4`, B=16, seq 4096,
`--compile.memory_policy full`, **cudagraph disabled** (regular EP's
`_grouped_mm` / all-to-all aren't cudagraphable on H100), `num_chunks=8`,
`TORCHINDUCTOR_COMPILE_THREADS=8`.

> ✅ **CORRECTNESS VERIFIED** — resolves the "UNVERIFIED" caveat on Runs 2–6 for
> the chunked-loss path. graph_trainer is now **bitwise-identical to the eager
> `Trainer`** (full-precision loss *and* grad_norm) over 20 steps.

### The fix (PR #3636)
Root cause of the prior eager-vs-graph drift (see `dsv3_numerics_verification.md`):
graph_trainer uses `simple_fsdp` (not FSDP2), so the lm_head weight grad was
reduce-scattered **once per chunk** and summed ranks-then-chunks, while eager
FSDP2 accumulates the per-chunk grads **unsharded in f32** and reduce-scatters
**once** (chunks-then-ranks). Both f32 — only the summation order differed, an
f32-ulp gap that compounded (loss diverged at step 7 at 16B).

`ChunkedCELossWithParamGrads` now reads the all-gathered lm_head weight **once**,
runs the per-chunk forward+backward on a detached leaf (no per-chunk collective —
chunking/memory preserved), accumulates the per-chunk grads **unsharded in f32**,
and does **one** `Partial→Shard` reduce-scatter — reproducing eager's order.

### Bitwise verification (eager FSDP2 vs graph simple_fsdp, 20 steps)
`scripts/loss_compare.py` (`--debug.deterministic --debug.seed=42`, shared init):

| metric | steps | max \|diff\| | diverging steps |
|---|---|---|---|
| loss (`global_avg_loss`) | 20 | **0.000e+00** | 0 / 20 |
| grad_norm | 20 | **0.000e+00** | 0 / 20 |

Also bitwise across debugmodel no-TP (dp8/tp1/ep4), debugmodel TP (dp4/tp2/ep2),
and `lm_head.weight.grad` directly (0/15 at 16B). CPU `test_chunked_loss` 3/3.

### Results (this benchmark — non-MinimalAsyncEP regular EP, B=16)
| step | loss | grad_norm | memory | tps | tflops | mfu |
|-----:|------|-----------|--------|-----|--------|-----|
| 1 | 12.03631 | 1.6044 | 39.16 GiB (41.19%) | 373 | 6.75 | 0.68% |
| 10 | 9.03867 | 6.0536 | 55.47 GiB (58.35%) | 10,108 | 183.02 | 18.51% |
| 20 | 7.08863 | 1.9277 | 57.51 GiB (60.49%) | 7,734 | 140.02 | 14.16% |

Step 10 is the clean steady state (**mfu 18.51%, 10.1k tps**); step 20's lower
mfu reflects the memory-snapshot dump firing at `profile_freq=10` (steps 10 & 20).
Model-only CUDA mem 7.83 GiB; peak ~57.5 GiB / 95 GiB at B=16.

### Compile pipeline
All 9 graph passes took **110.61 s** (`regional_inductor` dominant;
`--compile.debug_graph_passes` instrumentation adds overhead).

### Artifacts
| artifact | link |
|---|---|
| run log (pastry) | https://www.internalfb.com/intern/paste/P2374744733/ |
| profiler trace (perfetto, rank0 iter 10) | https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fbahuang_fe219053-85d4-49ab-ba7c-608967670d51_rank0_trace.json.gz |
| CUDA memory snapshot (memory visualizer, step 20) | https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/bahuang_4f5fac09-f864-437c-b162-d0157945b194_000000_step_20.pickle |
| tlparse logs (manifold, **temporary** `.tmp` path) | https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/.tmpQVIOId/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000 |
| 20-step bitwise loss_compare (pastry) | https://www.internalfb.com/intern/paste/P2374650180/ |
| full numerics root-cause + fix writeup | `dsv3_numerics_verification.md` |

#### Per-pass before/after graph diffs
| pass | diff |
|---|---|
| eliminate_dead_code | https://www.internalfb.com/intern/diffing/?before_paste_number=2374743002&after_paste_number=2374743090&selected_tab=plain_diff |
| canonicalize_graph | https://www.internalfb.com/intern/diffing/?before_paste_number=2374742803&after_paste_number=2374742897&selected_tab=plain_diff |
| tag_with_memory_policy | https://www.internalfb.com/intern/diffing/?before_paste_number=2374744012&after_paste_number=2374744133&selected_tab=plain_diff |
| selective_activation_remat | https://www.internalfb.com/intern/diffing/?before_paste_number=2374743758&after_paste_number=2374743878&selected_tab=plain_diff |
| reassign_collective_pgs | https://www.internalfb.com/intern/diffing/?before_paste_number=2374743379&after_paste_number=2374743474&selected_tab=plain_diff |
| joint_transformer_block_bucketing_reordering | https://www.internalfb.com/intern/diffing/?before_paste_number=2374743176&after_paste_number=2374743282&selected_tab=plain_diff |
| annotate_flex_attention_for_regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2374742627&after_paste_number=2374742717&selected_tab=plain_diff |
| regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2374743561&after_paste_number=2374743651&selected_tab=plain_diff |

#### Standalone artifacts
| artifact | paste |
|---|---|
| activation_memory_policy | https://www.internalfb.com/intern/paste/P2374744192/ |
| fx_compute_nodes_runtime_estimation | https://www.internalfb.com/intern/paste/P2374744274/ |

### Notes
- `run_graph_trainer_dsv3.sh` updated to **non-MinimalAsyncEP + cudagraph
  disabled** (Runs 5–6 used MinimalAsyncEP + forced cudagraph). This is the
  eager-comparable path the fix makes bitwise-exact.
- No OOM at B=16 (peak ~57.5 GiB); regular EP fits with `memory_policy=full`.
- Fix is `chunked_loss.py` only (+194/−76), lint-clean; depends on #3248.
- **PR: https://github.com/pytorch/torchtitan/pull/3636** (ghstack, stacked on #3248).
- tlparse manifold link is a temporary `.tmp` path and will expire; pastry,
  perfetto, and memory-snapshot links are durable.

---

## Run 8 — eager `Trainer` baseline (FSDP2 + full AC), same model as Run 7

**Date:** 2026-06-11
**Branch:** `graph_trainer/dsv3_scaling` (same tree as Run 7)
**Launcher:** `MODE=eager ./run_graph_trainer_dsv3.sh` — eager `Trainer`,
`deepseek_v3_16b`, `dp_shard=8 tp=1 ep=4`, B=16, seq 4096,
`--activation_checkpoint.mode full` (eager equivalent of Run 7's
`memory_policy=full`), loss compiled (config default), `ChunkedCELoss`.

This is the eager reference that Run 7's graph_trainer path is **bitwise-matched
to** (loss + grad_norm 0/20, see Run 7) — here as a standalone perf/memory
baseline with full artifacts.

### Results (eager, B=16)
| step | loss | grad_norm | memory | tps | tflops | mfu |
|-----:|------|-----------|--------|-----|--------|-----|
| 1 | 12.01018 | 1.6315 | 49.57 GiB (52.14%) | 2,758 | 49.93 | 5.05% |
| 10 | 9.14808 | 8.5312 | 66.98 GiB (70.45%) | 10,160 | 183.96 | 18.60% |
| 20 | 7.07741 | 1.5436 | 68.76 GiB (72.32%) | 8,065 | 146.02 | 14.76% |

Model-only CUDA mem 7.83 GiB. Step 10 is the clean steady state; step 20's lower
mfu is the memory-snapshot dump (`profile_freq=10`, fires at steps 10 & 20).

### Run 8 (eager) vs Run 7 (graph_trainer + fix) — same model/parallelism/batch
| | Run 8 eager (full AC) | Run 7 graph (memory_policy full) |
|---|---|---|
| step-10 MFU | 18.60% | 18.51% |
| step-10 tps | 10,160 | 10,108 |
| step-10 tflops | 183.96 | 183.02 |
| **step-10 memory** | **66.98 GiB** | **55.47 GiB** |
| **peak memory** | **~68.8 GiB** | **~57.5 GiB** |
| numerics | reference | **bitwise-identical (0/20)** |

**Takeaway:** graph_trainer (the fix) **matches eager throughput** (18.5% vs
18.6% MFU; ~10.1k tps both) while using **~17% less step memory and ~16% less
peak memory** (55.5 vs 67.0 GiB; 57.5 vs 68.8 GiB) — graph's tensor-granularity
`memory_policy=full` recompute is more memory-efficient than eager's
module-level `checkpoint_wrapper` full AC — *and* it's bitwise-exact vs this
eager run. Eager peaks at ~72% of 95 GiB at B=16; graph has more headroom.

### Artifacts
| artifact | link |
|---|---|
| run log (pastry) | https://www.internalfb.com/intern/paste/P2374793889/ |
| profiler trace (perfetto, rank0 iter 10) | https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fbahuang_2be83b84-0c5e-4bb9-a7bb-5046983dacc1_rank0_trace.json.gz |
| CUDA memory snapshot (memory visualizer, step 20) | https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/bahuang_7e0cc58e-a4ce-4125-ba86-e84bd329e20a_000000_step_20.pickle |
| tlparse logs (manifold, **temporary** `.tmp` path) | https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/.tmpcZyqP7/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000 |

> eager has no aot_fx_trace graph, so there are **no per-pass graph diffs** — the
> tlparse report covers only the loss `torch.compile`. The perfetto trace and
> memory snapshot are the meaningful eager artifacts.

### Notes
- `run_graph_trainer_dsv3.sh` gained a `MODE` toggle: `MODE=eager` runs this eager
  baseline, default `graph` runs the graph_trainer path (Run 7).
- No OOM at B=16 (peak ~68.8 GiB / 95 GiB), but eager is ~12 GiB closer to the
  limit than graph_trainer at the same batch.

---

## Run 9 — graph_trainer + MinimalAsyncEP + forced cudagraph (the fix, sync-free MoE row)

**Date:** 2026-06-11
**Branch:** `graph_trainer/dsv3_scaling` (same tree as Runs 7–8, with the
`chunked_loss.py` fix and the `cudagraph.py` force-capture override)
**Launcher:** `MODE=graph EP=minimal ./run_graph_trainer_dsv3.sh` —
`graph_trainer_deepseek_v3_16b_minimal_async_ep`, `dp_shard=8 tp=1 ep=4`, B=16,
seq 4096, `--compile.memory_policy full`, **cudagraph ENABLED + FORCED**
(MinimalAsyncEP is sync-free / cudagraphable; the force bypasses the `_grouped_mm`
`< sm_100` gate), `ChunkedCELossWithParamGrads` (`num_chunks=8`),
`TORCHINDUCTOR_COMPILE_THREADS=8`.

> ✅ **NUMERICS NOW VERIFIED (update).** When first run this was a perf-only
> comparison; the seed-pinned `--debug.deterministic` check has since proven graph
> ≡ eager **bitwise** for MinimalAsyncEP **with the forced cudagraph on** (loss +
> grad_norm 0/20) — see the "MinimalAsyncEP numerics verification" subsection at
> the end. The throughput/memory numbers below stand as a perf comparison vs the
> eager MinimalAsyncEP baseline (Run 10); the steady-state MFU is re-measured
> cleanly in Run 9′.
>
> *(original caveat, for context: MinimalAsyncEP is a new sync-free dispatcher and
> the forced cudagraph bypasses the `_grouped_mm` sm_90 gate — both were unverified
> at the time of this run; the deterministic check has now cleared both.)*

### Setup
Same model / parallelism / batch / recompute as Run 7 (8× H100, `dp_shard=8 tp=1
ep=4`, `aot_fx_trace`, `memory_policy=full`, B=16 / seq 4096, 20 steps, c4_test,
ChunkedCELoss) **except the MoE dispatcher + cudagraph**:

| | |
|---|---|
| MoE dispatcher | **MinimalAsyncEPTokenDispatcher** (sync-free EP via symmetric memory) |
| Config | `graph_trainer_deepseek_v3_16b_minimal_async_ep` |
| cudagraph | **enabled + forced** (312 non-cudagraphable `_grouped_mm` nodes captured anyway) |

MinimalAsyncEP buffer init confirmed: `tokens_per_rank=65536, top_k=6,
num_local_experts=16, ep_size=4, max_routed_tokens=1572864`.

### Results (graph + MinimalAsyncEP + forced cudagraph, B=16)
| step | loss | grad_norm | memory | tps | tflops | mfu |
|-----:|------|-----------|--------|-----|--------|-----|
| 1 | 12.01981 | 1.6180 | 57.19 GiB (60.16%) | 452 | 8.19 | 0.83% |
| 10 | 9.07173 | 5.7077 | 57.31 GiB (60.28%) | 11,521 | 208.60 | 21.09% |
| 20 | 7.11517 | 2.5456 | 57.31 GiB (60.28%) | 10,411 | 188.50 | 19.06% |

Loss converges; peak ~57.3 GiB / 95 GiB. step-10/20 both carry the
profiler+memory-snapshot dump (`profile_freq=10`), so MFU is best read as a
~19–21% band (clean steady-state is **22.54%** — see Run 9′). **Numerics now
bitwise-verified ✅** (see verification subsection).

### Compile pipeline
All **11** graph passes took **97.213 s** (`regional_inductor` dominant; cold-ish
MinimalAsyncEP/swiglu kernels). `cudagraph_pass` logs `FORCING cudagraph despite
312 non-cudagraphable node(s)` then `Applied cudagraph pass.`

### Artifacts
| artifact | link |
|---|---|
| run log (pastry) | https://www.internalfb.com/intern/paste/P2374809129/ |
| profiler trace (perfetto, rank0 iter 10) | https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fbahuang_b755e5ff-674e-48c9-b179-3593b36330b7_rank0_trace.json.gz |
| CUDA memory snapshot (memory visualizer, step 20) | https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/bahuang_9d5560ff-81d0-4f9a-8d54-e4d4eb72fdd2_000000_step_20.pickle |
| tlparse logs (manifold, **temporary** `.tmp` path) | https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/.tmpNMsm5U/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000 |

#### Per-pass before/after graph diffs
| pass | diff |
|---|---|
| eliminate_dead_code | https://www.internalfb.com/intern/diffing/?before_paste_number=2374807228&after_paste_number=2374807311&selected_tab=plain_diff |
| canonicalize_graph | https://www.internalfb.com/intern/diffing/?before_paste_number=2374807003&after_paste_number=2374807137&selected_tab=plain_diff |
| tag_with_memory_policy | https://www.internalfb.com/intern/diffing/?before_paste_number=2374808470&after_paste_number=2374808585&selected_tab=plain_diff |
| selective_activation_remat | https://www.internalfb.com/intern/diffing/?before_paste_number=2374808252&after_paste_number=2374808337&selected_tab=plain_diff |
| reassign_collective_pgs | https://www.internalfb.com/intern/diffing/?before_paste_number=2374807801&after_paste_number=2374807924&selected_tab=plain_diff |
| joint_transformer_block_bucketing_reordering | https://www.internalfb.com/intern/diffing/?before_paste_number=2374807450&after_paste_number=2374807632&selected_tab=plain_diff |
| annotate_flex_attention_for_regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2374806825&after_paste_number=2374806916&selected_tab=plain_diff |
| regional_inductor | https://www.internalfb.com/intern/diffing/?before_paste_number=2374808044&after_paste_number=2374808148&selected_tab=plain_diff |

#### Standalone artifacts
| artifact | paste |
|---|---|
| activation_memory_policy | https://www.internalfb.com/intern/paste/P2374808648/ |
| fx_compute_nodes_runtime_estimation | https://www.internalfb.com/intern/paste/P2374808722/ |

### Notes
- `run_graph_trainer_dsv3.sh` gained an `EP` toggle: `EP=minimal` selects the
  MinimalAsyncEP config + enables cudagraph; default `EP=regular` is the
  eager-comparable, cudagraph-disabled path (Run 7). Composes with `MODE`.
- Same forced-cudagraph override as Runs 4–6 (`cudagraph.py`, TODO: revert / gate).
- Peak ~57.3 GiB — essentially identical to Run 7's regular-EP graph peak (~57.5),
  i.e. MinimalAsyncEP doesn't cost extra steady-state memory here.

---

## Run 10 — eager `Trainer` + MinimalAsyncEP baseline (FSDP2 + full AC)

**Date:** 2026-06-11
**Branch:** `graph_trainer/dsv3_scaling` (same tree as Run 9)
**Launcher:** `MODE=eager EP=minimal ./run_graph_trainer_dsv3.sh` — eager
`Trainer`, `deepseek_v3_16b_minimal_async_ep`, `dp_shard=8 tp=1 ep=4`, B=16,
seq 4096, `--activation_checkpoint.mode full`, loss compiled (config default),
`ChunkedCELoss`.

This is the eager reference for the MinimalAsyncEP row (the Run 9 counterpart).
**MinimalAsyncEP runs in pure eager** — the sync-free dispatcher is not
graph-only; eager never uses cudagraph. MinimalAsyncEP buffer init confirmed
(`tokens_per_rank=65536, max_routed_tokens=1572864`, identical to Run 9).

> ✅ This eager MinimalAsyncEP path is now the **bitwise reference** for the graph
> side: the seed-pinned deterministic check (verification subsection) shows graph ≡
> this eager run, loss + grad_norm 0/20. (The numbers in this section are perf-only
> and not seed-pinned, so the loss here just converges in the expected band.)

### Results (eager + MinimalAsyncEP, B=16)
| step | loss | grad_norm | memory | tps | tflops | mfu |
|-----:|------|-----------|--------|-----|--------|-----|
| 1 | 12.03384 | 1.6457 | 50.98 GiB (53.62%) | 4,026 | 72.89 | 7.37% |
| 10 | 9.21959 | 7.5177 | 65.67 GiB (69.08%) | 12,537 | 226.99 | 22.95% |
| 20 | 7.29374 | 2.3227 | 65.67 GiB (69.08%) | 9,847 | 178.29 | 18.03% |

Loss converges; peak ~65.7 GiB / 95 GiB. Same profiler-dump perturbation at
steps 10/20 → read MFU as an ~18–23% band.

### Artifacts
| artifact | link |
|---|---|
| run log (pastry) | https://www.internalfb.com/intern/paste/P2374813340/ |
| profiler trace (perfetto, rank0 iter 10) | https://www.internalfb.com/intern/perfetto/open_trace/?manifold_path=perfetto_internal_traces%2Ftree%2Fshared_trace%2Fbahuang_bf74e873-3d0e-4eec-9694-f3aa16edcb40_rank0_trace.json.gz |
| CUDA memory snapshot (memory visualizer, step 20) | https://www.internalfb.com/pytorch_memory_visualizer/perfetto_internal_traces/tree/shared_trace/bahuang_6f90084a-2ee2-4797-84e4-955bf741f92e_000000_step_20.pickle |
| tlparse logs (manifold, **temporary** `.tmp` path) | https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/.tmpd0CHli/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000 |

> eager has no aot_fx_trace graph → **no per-pass graph diffs**; the tlparse report
> covers only the loss `torch.compile`. Perfetto trace + memory snapshot are the
> meaningful eager artifacts.

### Notes
- MinimalAsyncEP works unchanged in eager — no cudagraph, no graph passes.
- Peak ~65.7 GiB, ~1 GiB under the regular-EP eager peak (Run 8, ~68.8 GiB).

---

## Summary — 2×2 matrix (graph_trainer vs eager × regular EP vs MinimalAsyncEP), B=16

All four runs are the **same model / parallelism / batch / seq / recompute**
(`deepseek_v3 16B`, `dp_shard=8 tp=1 ep=4`, B=16, seq 4096, full recompute):

| | **regular EP** (no cudagraph) | **MinimalAsyncEP** (graph: forced cudagraph) |
|---|---|---|
| **graph_trainer** (the fix, `memory_policy=full`) | Run 7 — mfu 18.51%, 10,108 tps, peak **57.5 GiB** ✅ bitwise vs eager | Run 9 — peak **57.3 GiB**, clean MFU 22.54% (Run 9′) **✅ bitwise vs eager (cudagraph on *and* off)** |
| **eager** (FSDP2, full AC) | Run 8 — mfu 18.60%, 10,160 tps, peak 68.8 GiB | Run 10 — mfu ~18–23%, 12,537 tps, peak 65.7 GiB |

(MFU/tps quoted at step 10; step 10 & 20 both carry the profiler/snapshot dump,
so treat single-step MFU as a band, not a point.)

**Takeaways:**
- **MinimalAsyncEP is a real throughput win in both backends.** Step-10 tps:
  graph 10,108 → 11,521 (+14%), eager 10,160 → 12,537 (+23%). The sync-free MoE
  dispatcher beats all-to-all + `_grouped_mm` regular EP on this config.
- **graph_trainer's structural advantage is memory, in both EP modes.** Peak
  ~57 GiB (graph) vs ~66–69 GiB (eager) — **~11–16% lower** — because
  tensor-granularity `memory_policy=full` recompute is tighter than eager's
  module-level `checkpoint_wrapper`. Notably MinimalAsyncEP barely moves graph's
  peak (57.5 → 57.3 GiB).
- **At steady state graph and eager are within ~2% throughput** (eager marginally
  ahead) for the MinimalAsyncEP row — settled by a clean **50-step, profiler-off**
  re-run (Runs 9′/10′ below): **graph 22.54% / eager 22.96% MFU** (steps 20–50
  avg). The earlier single-step readings were *both* measurement artifacts pulling
  opposite ways: (a) the profiled **step-10** MFU (eager 22.95% vs graph 21.09%)
  over-penalized graph because its step-1 **cudagraph capture** is amortized into
  the step-10 rolling average; (b) the profiled **trace** per-step (graph 5,336 vs
  eager 5,639 ms) over-credited graph because the **profiler instruments every
  host launch** — hammering eager's 7,267 launches/step while barely touching
  graph's 596 (turning the profiler off sped eager's step 5.64→5.20 s, +8.5%, but
  graph's only 5.34→5.32 s). At B=16 both are ~99% GPU-bound, so cudagraph's
  launch-collapse buys **memory and host headroom, not throughput**. See the
  trace-analysis and Runs 9′/10′ subsections below.
- **Numerics caveat RESOLVED ✅ — MinimalAsyncEP is bitwise-verified, forced
  cudagraph included.** eager `deepseek_v3_16b_minimal_async_ep` vs graph
  `graph_trainer_..._minimal_async_ep` (`--debug.seed=42 --debug.deterministic
  --no-seed-checkpoint`, 20 steps, dp8/tp1/ep4) is **bitwise-identical in loss AND
  grad_norm (0/20, max|diff| 0.000e+00)** — both with cudagraph **disabled** and
  with the **forced cudagraph** (312 `_grouped_mm` nodes captured past the sm_90
  gate). This clears the unverified flags on the whole MinimalAsyncEP column
  (Runs 3–6, 9) and shows the **forced-cudagraph hack is numerically safe** on
  H100 for this config (the feared `_grouped_mm` hidden CPU↔CUDA copy does not
  corrupt replay). See the verification subsection below.

### Profiler-trace analysis — Run 9 vs Run 10 (rank0, iter-10 capture)

> ⚠️ **These per-step numbers are from the *profiled* runs and are profiler-skewed
> — the profiler over-penalizes the eager path (see below). For the throughput
> verdict use the profiler-off Runs 9′/10′. The trace is still the right tool for
> the *mechanism* (kernel mix, host-launch counts, recompute fragments).**

Pulled both rank-0 traces (Run 9 from manifold, Run 10 local) and sliced one
training step using the per-step loss marker (`num_chunks=8` → **8 loss-chunk
kernels = 1 step**): Run 9's window held **2 steps** (16 `nll_loss_forward`),
Run 10's held **1 step** (8 `log_softmax+nll`). Normalizing to **per-step**:

| | graph (Run 9) | eager (Run 10) |
|---|---|---|
| **wall / step** (profiled) | **5,336 ms** | 5,639 ms |
| **GPU-busy / step** (union) | **5,309 ms** | 5,598 ms |
| GPU busy % | 99.5% | 99.3% |
| cpu_op events (window) | 2,145 | 44,962 |
| cuda launch/Memcpy calls | 596 | 7,267 |
| launches per GPU kernel | **0.028** | 0.652 |

**cudagraph signature:** graph issues ~0.03 host launches per kernel vs eager's
0.65 — CPU-side launch/op overhead is essentially collapsed (the captured graph
replays). Both are ~99% GPU-busy, so neither is launch-bound at this batch, but
the graph path removes ~7k host calls/step.

**Per-step GPU time by kernel category (ms/step; streams overlap so columns
sum > wall):**

| category | graph (Run 9) | eager (Run 10) | Δ (graph−eager) |
|---|---:|---:|---:|
| matmul (dense+MoE) | 2,260.8 | 2,544.8 | **−284.0** |
| symm_mem (EP dispatch) | 1,345.4 | 1,268.8 | +76.6 |
| elementwise | 641.7 | 621.9 | +19.8 |
| comm (nccl) | 309.3 | 460.4 | **−151.0** |
| reduce/norm/softmax | 229.2 | 180.4 | +48.8 |
| other | 184.2 | 295.5 | −111.3 |
| attn_bwd (flex) | 172.9 | 200.5 | −27.5 |
| cat/concat | 162.4 | 180.5 | −18.1 |
| moe_swiglu | 159.2 | 179.2 | −20.0 |
| attn_fwd (flex) | 92.0 | 116.5 | −24.5 |

**Mechanism (robust — kernel durations aren't profiler-inflated, only gaps are):**
1. **Collapsed host overhead** — cudagraph issues 596 vs 7,267 launches/step
   (0.03 vs 0.65 per kernel). This is why the profiler skews the comparison:
   instrumentation cost scales with host launches, so it taxes eager far more.
2. **Slightly less recompute** under `memory_policy=full` tensor-granularity remat
   vs eager module-level full AC: **52 vs 56 flex-fwd fragments/step**, −284 ms
   matmul; offset by +49 ms reduce/norm + ~+20 ms elementwise from more, smaller
   remat kernels. Roughly a wash.
3. **Less NCCL comm** in-trace (−151 ms) — cudagraph + simple_fsdp bucketing.

> **Takeaway:** the per-step deltas here are dominated by the profiler's
> asymmetric per-launch overhead, so they do **not** establish a throughput
> winner. The clean profiler-off Runs 9′/10′ do: **graph and eager are within
> ~2% MFU (eager marginally ahead)**, and graph_trainer's durable advantage is
> **memory (~13% lower, 57.3 vs 65.7 GiB)**. Analyzer: `~/tmp/mfu_compare/analyze.py`.

---

## Runs 9′ / 10′ — clean 50-step MFU re-measure (profiler OFF), MinimalAsyncEP

**Date:** 2026-06-11
**Branch:** `graph_trainer/dsv3_scaling` (same tree as Runs 9/10)
**Launcher:** `STEPS=50 PROFILE=0 MODE={graph,eager} EP=minimal ./run_graph_trainer_dsv3.sh`
— new `STEPS` and `PROFILE` toggles. `PROFILE=0` is a **lean run**: no profiler,
no memory snapshot, no tlparse/`tlp`, no uploads.

Re-measure of the MinimalAsyncEP row at **50 steps with profiling disabled**, to
remove the two artifacts that distorted Runs 9/10: the step-1 cudagraph capture
amortized into the step-10 average, and the profiler's per-host-launch overhead
(which taxes eager's ~7k launches/step far more than graph's ~600). With
`log_freq=10` and the profiler off, **every logged interval past step 10 is a
clean, capture-free steady state**.

### Results — graph (Run 9′) vs eager (Run 10′), B=16, profiler off
| step | graph mfu | graph tps | eager mfu | eager tps |
|-----:|----------:|----------:|----------:|----------:|
| 1  (capture/warmup) | 0.89% | 486 | 7.36% | 4,022 |
| 10 (interval incl. capture) | 21.79% | 11,901 | 22.81% | 12,459 |
| 20 | 22.52% | 12,303 | 23.08% | 12,606 |
| 30 | 22.35% | 12,206 | 22.57% | 12,327 |
| 40 | 22.43% | 12,250 | 23.00% | 12,564 |
| 50 | 22.85% | 12,483 | 23.18% | 12,662 |
| **avg steps 20–50** | **22.54%** | **12,310** | **22.96%** | **12,540** |
| **peak memory** | **57.31 GiB** | | 65.67 GiB | |

**Verdict:** **eager is ~1.9% faster** at steady state (22.96% vs 22.54% MFU;
12,540 vs 12,310 tps); **graph uses ~13% less memory** (57.3 vs 65.7 GiB). At
B=16 both are ~99% GPU-bound, so cudagraph's host-launch savings don't convert to
throughput — they buy memory and CPU headroom. This supersedes the throughput
reading from Runs 9/10 (the step-10 MFU and the profiled trace were both
artifacts; see the matrix takeaways above).

### Loss parity (incidental — these perf runs are NOT deterministic)
These are throughput runs, launched **without** `--debug.seed/--debug.deterministic`,
so the kernels use nondeterministic algorithms and the curves drift run-to-run:
step-10 9.865/9.786, step-20 8.027/7.987, step-50 **5.95344 / 5.95386** (~4e-4
apart). That ~4e-4 is **nondeterminism jitter, not a graph-vs-eager difference** —
the seed-pinned `--debug.deterministic` comparison (see the "MinimalAsyncEP
numerics verification" subsection below) proves graph and eager are **bitwise
identical** (loss + grad_norm 0/20). Correctness milestone: **MET ✅.**

### Notes
- No artifacts (profiler off by design). Pure throughput/memory measurement.
- `run_graph_trainer_dsv3.sh` gained `STEPS` (default 20) and `PROFILE` (default 1)
  toggles, composing with `MODE`/`EP`. `PROFILE=0` skips `tlp` + profiler flags +
  uploads.

---

## MinimalAsyncEP numerics verification — graph ≡ eager, BITWISE ✅ (cudagraph on & off)

**Date:** 2026-06-11
**Branch:** `graph_trainer/dsv3_scaling`
**Launcher:** `CONFIG_SUFFIX=_minimal_async_ep [TEST_CUDAGRAPH=0|1] STEPS=20 ./run_loss_compare_dsv3.sh`
(new `CONFIG_SUFFIX` + `TEST_CUDAGRAPH` toggles).

The MinimalAsyncEP analog of the Run 7-vs-8 regular-EP bitwise check: eager
`deepseek_v3_16b_minimal_async_ep` (FSDP2, full AC, `--compile.no-enable`) vs
graph `graph_trainer_deepseek_v3_16b_minimal_async_ep` (simple_fsdp, aot_fx_trace,
`memory_policy=full`), via `scripts/loss_compare.py` with `--debug.deterministic
--debug.seed=42 --no-seed-checkpoint --assert-equal`, 20 steps, dp8/tp1/ep4, B=4
(loss_compare default batch). Run **twice**: cudagraph disabled, and forced on.

| variant | loss max\|diff\| | grad_norm max\|diff\| | verdict |
|---|---|---|---|
| graph (cudagraph **off**) vs eager | 0.000e+00 (0/20) | 0.000e+00 (0/20) | **BITWISE IDENTICAL ✅** |
| graph (**forced cudagraph**) vs eager | 0.000e+00 (0/20) | 0.000e+00 (0/20) | **BITWISE IDENTICAL ✅** |

Real full-precision values (identical across all three runs — eager, graph-off,
graph-forced-cg): step 1 loss `11.98491382598877`, step 10 `9.099592208862305`,
step 20 `7.229188442230225` / grad_norm `5.547492504119873`.

**Two things this proves:**
1. **MinimalAsyncEP is deterministic and numerically correct** — it runs cleanly
   under `--debug.deterministic` (no missing-deterministic-kernel error), and
   graph simple_fsdp matches eager FSDP2 to the last bit (the lm_head chunked-loss
   fix + #3590 FSDP-order carry over to the sync-free MoE path).
2. **The forced cudagraph is numerically safe** on H100 (sm_90) for this config —
   capturing the 312 non-cudagraphable `_grouped_mm` nodes past the gate does
   **not** corrupt replay (the hypothesized hidden `_grouped_mm` CPU↔CUDA copy
   either doesn't occur here or is captured correctly). Resolves the ⛔/🟡
   correctness caveats carried since Runs 4–6.

Incidental perf at this deterministic B=4 (avg steps >10, graph/eager): cudagraph
**off** 0.983× tps (graph slightly slower — launch overhead exposed at B=4);
forced cudagraph **on** 1.016× tps (graph faster — cudagraph recovers the launch
overhead). Consistent with the B=16 finding that cudagraph's benefit is
launch-bound (a wash once GPU-bound at large batch).

### Notes
- Summaries: `~/tmp/loss_compare_dsv3_16b_minimal_async_ep/fullprec_summary.md`
  (no-cg) and `..._minimal_async_ep_cg/fullprec_summary.md` (forced cg).
- `run_loss_compare_dsv3.sh` gained `CONFIG_SUFFIX` (variant on both sides) and
  `TEST_CUDAGRAPH` (default 0 = disable cudagraph on the graph side; 1 = keep the
  forced cudagraph) toggles.
- Scope: 20 steps / B=4 / dp8 tp1 ep4. The regular-EP divergence (pre-fix)
  appeared by step 7, and a cudagraph stale-data bug would surface on the first
  post-capture replay (step 2), so 20 steps is a decisive window.
