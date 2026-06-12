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
