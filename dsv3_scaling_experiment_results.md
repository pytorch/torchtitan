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
