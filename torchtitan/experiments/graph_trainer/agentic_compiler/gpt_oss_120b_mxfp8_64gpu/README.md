# Agentic compiler loop: GPT-OSS 120B MXFP8, FSDP 64 x EP 16

This loop has four stages:

1. fusion_finder proposes a fusion plan.
2. `substitute_with_kernels_pass` (#5838) turns each lifted region into a kernel-cache problem.
3. The autokernel swarm (#5839) writes CuTe DSL kernels, and an independent verifier checks each one.
4. Verified kernels replace the regions in training.

Model numerics must stay bitwise identical. The recipe is `graph_trainer_gpt_oss_120b_mxfp8_64gpu`
(`torchtitan_recipes/graph_trainer/megatron_bridge_benchmarks.py`).

## This directory

| Path | Contents |
|---|---|
| `loop_recipe.py` | configs `full` / `full_det` (benchmark recipe), `capture`, `lifted`, `full_lifted` and `lifted_inputs` (sec. 3) |
| `plans/plan_iterN.txt` | the fusion plan (lift passes, in order) of each iteration; `plan_iter3.txt` is current |
| `fusion_regions/` | the 17 regions of the current plan: `region.json` and the generated `lift_pass.py` each (G-031's matcher edited by hand, sec. 9) |
| `kernel_cache/` | the substitute-kernels cache: 20 kernel files across 19 patterns. Each pattern has `reference.py`, `specialization_*.json` and the promoted kernel `<kernel_name>.py`. The files are byte-identical to the verified candidates (sha256 in sec. 7) |
| `regions.json` | the finder's full region ranking for the baseline capture |
| `results.json` | per run: GPU time per microbatch or tokens/s, substitute hits, and the logged loss / grad_norm values, compared with the baseline |
| `autokernel/board.md`, `autokernel/harness.py`, `autokernel/gen/` | the swarm's task board (every handoff, verdict and promotion), the correctness/timing harness, and the input generators for index-valued inputs |

**Kept outside the repo.** These stay in the local loop workspace `outputs/agentic_compiler/gpt_oss_120b_mxfp8_64gpu/`:

- real captured inputs;
- profiler traces and graph dumps;
- writer candidates;
- the verifiers' scripts and logs (`autokernel/run1/verify/`).

Paths in the report below that start with `autokernel/run1/`, `real_inputs/`, a run name (`final/`,
`full_final.log`, ...), `tbcmp.py`, `gpu_time.py` or `categories.py` refer to that directory.

The generated lift passes and the kernels are committed as produced and verified. They are not reformatted,
since reformatting a kernel would change the file the verifier checked.

## Run

Benchmark recipe (5 microbatches per step, 10 steps) with the plan and the kernels:

```
S=torchtitan/experiments/graph_trainer/.claude/skills/fusion_finder/scripts
D=torchtitan/experiments/graph_trainer/agentic_compiler/gpt_oss_120b_mxfp8_64gpu
env FUSION_PLAN=$(cat $D/plans/plan_iter3.txt) LIFT_OPTIONAL=1 RUN=full_final PYTHONPATH=$S:$D:. \
    NGPU=64 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -m torchtitan.train --module loop_recipe --config full_lifted --comm-backend fake
```

- **Profiled run:** `--config lifted` (with `RUN=final`) runs one microbatch per step and profiles step 10.
- **Baseline:** `--config full_det` is the bitwise baseline.
- **Outputs:** TensorBoard, profiler trace and graph dumps go to `$D/$RUN/`.
- **Cache misses:** a miss runs the region's original ops, so numerics are unchanged either way.

# Audit report

## 1. Summary

| | Value |
|---|---|
| Benchmark recipe tokens/s per GPU at step 10, same session, exclusive GPU 1 | **25,532** vs **7,840** deterministic baseline (**3.26x**) and 15,239 non-deterministic recipe (**+67.5%**) |
| GPU time per microbatch, excluding fake-PG collectives, capture -> final (profiled 1-microbatch runs) | **2000.4 ms -> 534.7 ms (-1465.7 ms, 3.74x)**; kernel launches 23,167 -> 12,669 |
| Numerics | loss, grad_norm and load-balance loss bitwise identical to the deterministic baseline in every run (sec. 6) |
| Kernels | 20 kernel files across 19 patterns. Each was written by one agent, checked by a different, freshly spawned verifier, and promoted only on PASS. 19 hit at every site in the benchmark graph; the 20th (G-046) hits only in the 1-microbatch graph (sec. 8) |
| Verified per-call savings x calls per microbatch | ~1441 ms, consistent with the measured -1466 ms (CUDA-graph timings, sec. 7) |
| Biggest single win | G-005: the deterministic `index_add` for the expert-bias gradients took ~1030 ms of the 2000 ms microbatch; the kernel takes ~7 ms |
| Caveat before any real multi-GPU use | 5 kernels use whole-GPU cooperative launches (sec. 11) |

**How the deterministic recipe overtook the non-deterministic one.** The deterministic baseline is about 2x
slower than the non-deterministic recipe. In the capture, the deterministic `index_put` / `index_add` path
(radix sort, then ordered accumulation) takes 1094 ms of the 2000 ms microbatch. The kernels compute exactly
what that path computes, in about 1% of the time. The non-deterministic recipe was not profiled, so how much of
the gap this path explains is not measured.

## 2. Goal and constraints

- **Goal (user, 2026-11-05):** optimize `graph_trainer_gpt_oss_120b_mxfp8_64gpu` with the fake PG, using the
  agentic_compiler workflow.
- **Hard constraint (user):** model numerics must not change. A training run counts as passing only if loss
  and grad_norm are bitwise identical to the deterministic baseline.
- **Other rules followed:**
  - nothing is submitted to KernelGen;
  - no GitHub activity;
  - other users' processes on the shared host are never touched.

## 3. Setup

- **Branch:** `agentic_compiler` at `765bc29a4` plus the `substitute_kernels` changes of sec. 9 (committed as
  `4f955c1c8`).
- **Environment:** NVIDIA GB300 (sm_103), CUDA 13.0, torch 2.16.0a0+git3c53c6f, nvidia-cutlass-dsl 4.8.0.
- **Recipe:** `graph_trainer_gpt_oss_120b_mxfp8_64gpu`.
  - Parallelism: FSDP 64, EP 16, no PP, fake process group on one GPU (`NGPU=64 LOCAL_RANK=0`).
  - Batch: GBS 1280, MBS 4, sequence length 4096, so 5 microbatches of 16,384 tokens per step.
  - Precision: MXFP8 attention projections and lm_head; experts stay BF16 behind the all-to-all dispatcher.
  - No CUDA graphs: the dispatcher host-syncs on its split sizes.
  - Seed 42, deterministic (torchtitan's `set_determinism` also sets `fill_uninitialized_memory=False`).
- **Two kinds of runs** (configs in `loop_recipe.py`):
  - **1 microbatch (`capture`, `lifted`).** These run the fusion finder's pass pipeline
    (`enable_fusion_finder`) with `num_tokens_per_train_step=-1`, so the profiled step 10 is exactly one
    forward and backward. GPU time is the sum of kernel durations in that step (`gpu_time.py`). Kernels that
    only emulate collectives under the fake PG (`all_gather_into_tensor`, `reduce_scatter_tensor`,
    `all_to_all`, about 35 ms) are excluded.
  - **Benchmark (`full_det`, `full_lifted`).** This is the recipe as shipped, plus seed and determinism, for 10
    steps. Tokens/s is the value logged at step 10. Its graph goes through the gradient-accumulation (GA) graph
    builder, so `full_lifted` also needs `unsubstituted_to_call_function` and `LIFT_OPTIONAL=1` (sec. 9).
- **Bitwise check:** `python tbcmp.py <baseline>/tb <run>/tb` compares grad_norm, global_avg_loss,
  global_max_loss and the load-balance loss for exact float equality.

## 4. The loop

```
 fusion_finder (profile + graph dump of the 1-microbatch capture)
   -> regions.json, one folder per region with a generated topology-matching lift pass
   -> a plan of lift passes that replays without overlapping sites
 training run with the plan (loop_recipe lifted / lifted_inputs): each region becomes call_module fusion_G<id>
   -> #5838 substitute_kernels: a cache miss writes kernel_cache/pat_<l1>/{reference.py, specialization_<l2>.json};
      the real arguments of each region are saved to real_inputs/
 #5839 autokernel swarm (board: autokernel/run1/board.md, harness: autokernel/run1/harness.py)
   writer (one GPU lease) -> candidate + its own checks
   -> fresh verifier (never the writer) -> PASS/FAIL with evidence under autokernel/run1/verify/<name>/
   -> coordinator promotes on PASS: copy to a temp file, os.replace onto pat_<l1>/<kernel_name>
 1-microbatch run with the plan + cache -> bitwise check vs the capture -> profile
 benchmark run (full_lifted) -> bitwise check vs full_det -> tokens/s
```

## 5. Results by iteration

**1-microbatch profiled runs.** "Excl." excludes the fake-PG collective emulation.

| Run | Plan | Kernels added | GPU | Total ms | Excl. ms | Kernels launched | Hits | Loss / grad_norm |
|---|---|---|---|---|---|---|---|---|
| capture | - | - | 0 | 2035.8 | 2000.4 | 23,167 | - | reference |
| iter1 | iter1 (12 passes) | none (lift only) | 0 | 2046.5 | 2011.1 | 23,581 | 0 | bitwise |
| iter2a | iter1 | G-007, G-017 | 0 | 1903.7 | 1868.3 | 22,861 | 72 | bitwise |
| iter2b | iter1 | + G-019 x2, G-020 x2, G-028, G-037 | 0 | 1822.0 | 1786.6 | 21,425 | 260 | bitwise |
| trial_g005 | iter1 | + G-005 x2 from a trial cache, before verification | 0 | 765.9 | 730.6 | 19,293 | 332 | bitwise |
| iter3a | iter1 | + G-005 x2 (verified) | 1 | 763.7 | 728.3 | 19,293 | 332 | bitwise |
| iter3b | iter3 (17 passes) | + G-031 v1, G-039, G-048, G-041, G-046 | 0 | 732.8 | 697.4 | 16,376 | 512 | bitwise |
| iter3d | iter3 | + G-023, G-025, G-036, G-014, G-031 v2 | 1 | 625.0 | 589.6 | 13,564 | 656 | bitwise |
| **final** | **iter3** | **+ G-011** | **1** | **570.2** | **534.7** | **12,669** | **692** | **bitwise** |

**Benchmark recipe, tokens/s per GPU at step 10.** Hits are graph sites substituted at compile time.

| Run | Kernels | GPU | tokens/s | Hits | Loss / grad_norm |
|---|---|---|---|---|---|
| full_det | baseline (deterministic) | not recorded | 7,651 | - | reference |
| full | non-deterministic recipe | not recorded | 13,343 | - | (differs, not deterministic) |
| full_iter2b | 8 | 0 | 8,852 | 260 | bitwise |
| full_trial_g005 | 10, G-005 before verification | 0 | 19,392 | 332 | bitwise |
| full_iter3a | 10 | 1 | 18,526 | 332 | bitwise |
| full_det_gpu1 / full_gpu1 | baselines | 1 | 7,767 / 14,929 | - | reference / - |
| full_iter3b | 13 | 0 | 18,892 | 440 | bitwise |
| full_iter3d | 18 | 1 | 22,319 | 620 | bitwise |
| **full_final** | **19 (all but G-046)** | **1** | **25,532** | **656** | **bitwise** |
| full_det_final / full_nondet_final | baselines, same session as full_final | 1 | 7,840 / 15,239 | - | reference / - |

**Caveats:**
- **Lifting cost:** lift only (iter1) costs +10.7 ms and +414 kernel launches over the capture. The cause was not
  investigated; every lifted site is substituted in the final runs.
- **GPU differences:** GPUs 0 and 1 differ by a few percent on the same run (iter3a vs trial_g005, same kernels:
  728.3 vs 730.6 ms). Compare benchmark tokens/s only within one GPU and session. The headline compares
  full_final with full_det_final and full_nondet_final, run back to back on GPU 1.
- **Contended runs:** full_iter2a (7,612 tokens/s, not in `results.json`) ran while verifiers shared GPU 0; it is
  not a clean measurement.
- **Memory:** peak memory is 202.62 GiB in every benchmark run, baseline and final. In the 1-microbatch runs it is
  184.20 GiB (capture) and 182.48 GiB (final).

**Where the GPU time went** (1-microbatch, ms; `categories.py` over `capture/analysis/` and `final/analysis/`):

| Category | capture | final |
|---|---:|---:|
| deterministic `index_put` / `index_add` (radix sort + `indexing_backward_kernel`) | 1094.0 | 5.2 |
| other aten (elementwise, reductions, gather, cat, softmax, ...) | 515.1 | 82.1 |
| substituted kernels | - | 83.9 |
| expert grouped GEMM (BF16, CUTLASS) | 202.3 | 200.8 |
| flash attention forward / backward | 71.6 | 71.6 |
| MXFP8 GEMM (nvjet) + quantize | 60.6 | 59.5 |
| memcpy / memset | 24.9 | 5.6 |
| optimizer (`multi_tensor_apply`) | 24.7 | 18.9 |
| regional Inductor / other CuTe kernels | 7.3 | 7.2 |
| **total, excluding fake-PG collective emulation (35.3 / 35.4)** | **2000.4** | **534.7** |

**Optimizer caveat.** The optimizer's work does not change with fusion. Its 5.8 ms drop is the capture's GPU 0
against the final run's GPU 1.

## 6. Numerics evidence

- **What is compared:**
  - each 1-microbatch run's TensorBoard against the capture's;
  - each benchmark run's TensorBoard against `full_det`.

  TensorBoard logs steps 1 and 10. Both steps are bitwise equal in every run, for grad_norm, global_avg_loss,
  global_max_loss and the load-balance loss.
  - Benchmark: grad_norm at step 1 = 13.711431503295898 and at step 10 = 12.714406967163086; global_avg_loss at
    step 10 = 7.925948143005371. full_det, full_det_gpu1, full_det_final and every full_* kernel run match these.
  - 1-microbatch: grad_norm at step 10 = 12.357583045959473 and loss = 7.877079963684082, in the capture and
    every iteration.
- **Limitation:** steps 2-9 are not logged. Step 10 depends on every earlier optimizer update, so an exact match
  at step 10 is strong evidence, but not a proof, that steps 2-9 match.
- **Per-kernel parity** is checked bytewise against the immutable `reference.py` in three ways:
  - the harness, with random seeds, real captured inputs and poisoned output memory;
  - for index-valued inputs, the generators in `autokernel/gen/`;
  - each verifier's own adversarial suite (sec. 7).
- **Same results, deterministic or not:** the 1-microbatch and benchmark graphs differ (one microbatch vs GA),
  so their loss values differ. Each is compared only with its own baseline.

## 7. Kernels in the cache

**Where the times come from:**
- **Reference and kernel times:** the verifier's interleaved A/B measurements on real inputs, per call. The
  eager time includes host overhead; the graph time is one call per CUDA-graph replay, unless noted. The
  reference graph times come from the harness.
- **Saving:** (reference graph - kernel graph) x calls per microbatch.
- **In training:** GPU time per microbatch in the final 1-microbatch profile.

| Pattern (region) | What | Calls / microbatch | Ref us eager / graph | Kernel us eager / graph | Saving ms | In training ms | Writer -> verifier | sha256 prefix | Evidence (`autokernel/run1/verify/`) |
|---|---|---|---|---|---|---|---|---|---|
| seox (G-005 v1) | w13 expert-bias grad: deterministic `index_add` [65536,5760] -> [8,5760] | 36 | 14895 / 14724 | 115.1 / 108.0 | 526.2 | 4.4 | ivy t9 -> ora | 3e0149f53aa41e8b | G005_ora |
| seqs (G-005 v0) | w2 expert-bias grad, [65536,2880] -> [8,2880] | 36 | 14138 / 13878 | 65.9 / 58.2 | 497.5 | 2.4 | ivy t9 -> ora (same file) | 3e0149f53aa41e8b | G005_ora |
| yjxz (G-007) | SwiGLU-with-clamp backward | 36 | 3371 / 3301 | 552 / 534 | 99.6 | 17.0 | sol t2 -> vera | b8643e626c0fbebd | G007_vera |
| jvof (G-011) | MoE combine backward: gather + mul + row sum + `index_put` | 36 | 1787 / 1702 | 142.9 / 145.2 | 56.0 | 4.9 | mox f1 + pip f3 -> joy (f1 FAIL fay, f2 FAIL hal) | 5a3efd0808d5121d | G011f3_joy, G011_fay, G011f2_hal |
| ubtn (G-014) | MoE combine forward: index + mul + `deterministic_scatter_add` | 36 | 1622 / 1504 | 86.5 / 86.7 | 51.0 | 3.0 | pip t1 -> eva | ea36c1b27dfd1668 | G014_eva |
| w7az (G-017) | SwiGLU-with-clamp forward | 36 | 1454 / 1381 | 415 / 384 | 35.9 | 13.6 | sol t2 -> vito | 25f385c386489072 | G017_vito |
| ojkx (G-019 q) | RoPE backward, q [16384,64,64] | 36 | 993 / 940 | 57.9 / 46.3 | 32.2 | 1.5 | sol t2 -> una | 3a5408ff05f1f418 | G019q_una |
| mv2l atcl (G-020 q) | RoPE forward, q | 36 | 954 / 915 | 62.5 / 48.4 | 31.2 | 1.5 | sol t2 -> rho | f9a089f558d1cb7c | G020q_rho |
| 5r57 (G-023) | `new_zeros` + `index_put` + add, [16384,2880] | 36 | 843 / 718 | 95.6 / 97.0 | 22.4 | 3.5 | pip t3 -> bea | fa84dd63ba30ca56 | G023_bea |
| jn7t (G-025) | `new_zeros` + `index_put`, [65536,2880] | 36 | 758 / 686 | 136 / 133 | 19.9 | 4.6 | pip t3 -> cal | 9d7ef43d1ff3e2d4 | G025_cal |
| cuks (G-028) | attention-sink backward (row/column sums + `sigmoid_backward`) | 36 | 481 / 438 | 91.5 / 71.7 | 13.2 | 2.3 | sol t2 -> ana | fc038532b4526002 | G028_ana |
| jofj (G-031 v2) | w13 expert-bias expand (`repeat_interleave` + `index_select` + add) | 36 | 780 / 526 | 216.6 / 213.9 | 11.2 | 7.5 | rex t1 -> gus | 17e2fdc7831abed2 | G031v2_gus |
| 53aq (G-037) | loss backward (`log_softmax` + `nll`), [2048,201088] | 8 | 1630 / 1586 | 371 / 346 | 9.9 | 2.9 | sol t2 -> eli | 858b51b32058a8b1 | G037_eli |
| 2uey (G-036) | `new_empty` + `index_put_`, [65536,2880] | 36 | 481 / 343 | 134.5 / 136.5 | 7.4 | 4.5 | pip t3 -> dov | 2f351c8ba0c90cb2 | G036_dov |
| s7of (G-039) | w2 expert-bias expand | 36 | 522 / 300 | 111.7 / 123.4 | 6.4 | 3.7 | rex t2 -> lux | 1e4d821d07cb4f2d | G039_lux |
| 4bre (G-041) | router backward (softmax backward + `scatter_add`) | 36 | 576 / 145 | 17 / 11 | 4.8 | 0.3 | rex t2 -> noa | 59f22a7d5e6fb1ae | G041_noa |
| f6c7 (G-048) | routed-experts `mul.Scalar` + `mul`, [65536,2880] | 36 | 294 / 271 | 161 / 158 | 4.1 | 5.5 | rex t1 -> kit | deb4a0590629ab15 | G048_kit |
| wzsj (G-046) | aux-loss forward (`scatter_` + bit ops + sums) | 36 | 377 / 124 | 45 / 12.8 | 4.0 | 0.1 | rex t2 -> ivo | 509e84947a0b7a64 | G046_ivo |
| mloe (G-019 k) | RoPE backward, k [16384,8,64] | 36 | 260 / 118 | 25.7 / 10.1 | 3.9 | 0.2 | sol t2 -> zed | a3cc09aa97ff5a97 | G019k_zed |
| mv2l ajqa (G-020 k) | RoPE forward, k | 36 | 176 / 116 | 24.8 / 9.8 | 3.8 | 0.2 | sol t2 -> kai | 620ad80410f4fd7d | G020k_kai |

**Notes on the table:**
- **G-046** is not lifted in the benchmark graph (sec. 8).
- **Candidates:** sources are under `autokernel/run1/candidates/<pattern>/<writer>/<trial>/`.
- **Promoted copies:** `kernel_cache/pat_<key>*/<kernel_name>`. The sha256 prefixes above were taken from the
  files in `kernel_cache/` and match the verified candidates on the board.
- **Superseded:** G-031 v1 (`pat_i5a5...`, rex t2, verified by tam) was replaced by v2 (sec. 8) and is not in
  the cache.

**What every verifier checked** (details in each evidence directory):
- **Contract:**
  - sha256 and the signature against the spec;
  - the official harness gate (seeds plus real inputs, every case EXACT);
  - output metadata and alias structure against the reference;
  - inputs unmutated, and repeated calls identical.
- **Memory:** allocator poisoning, proving every output byte is written, and compute-sanitizer
  (memcheck/initcheck/racecheck/synccheck).
- **CUDA graphs:** many replays with inputs changed in place.
- **Adversarial inputs:** each verifier's own suite, with negative controls (deliberately broken copies that must
  fail).
  - Pointwise kernels (G-007, G-017, G-019, G-020, G-048) were swept over all 2^32 input bit-pattern pairs, a
    complete elementwise proof.
  - Reduction orders were checked with order-sensitive inputs: near-ties, cancellation, and direct probes of
    sums that only feed a scalar.
- **Index semantics:** the `index_put` / `index_add` kernels were derived from and checked against PyTorch
  3c53c6f's deterministic path (`index_put_with_sort`). That path wraps negative indices, does a stable radix
  sort, then adds duplicates in ascending source-row order, rounding each add to bf16 from +0.
  - Input families: training-like routing, arbitrary duplicates, heavy duplicates, empty destinations and
    negative indices (`autokernel/gen/gen_mox.py`, `gen_index_add.py`).
- **PDL:** kernels using programmatic dependent launch were tested with an early-triggering, small-grid
  predecessor.

**Dependencies to know about:**
- **G-028:** reproduces aten's row and column reduction orders for this GPU and PyTorch build (row order from
  PyTorch #164790; columns split 128 ways). There is no runtime check. `PYTORCH_SUM_INNER_TREE=1` changes the
  reference and breaks parity (sec. 11).
- **Rounding points:** the SwiGLU kernels (G-007, G-017) reproduce aten's bf16 rounding points op by op.
  `addcmul` is fp32 `fma` then round, which is not a bf16x2 `fma`. This was measured exhaustively by sol
  (board, 17:55).

## 8. Plans and what was not promoted

- **Plans:**
  - `plan_iter1.txt`: G-005, G-007, G-011, G-014, G-019, G-020, G-017, G-023, G-025, G-028, G-036, G-037.
  - `plan_iter3.txt`: those, plus G-031, G-039, G-048, G-041 and G-046.
- **Dropped from the plan:**
  - **G-033, G-045, G-047:** their sites overlap regions lifted earlier in the plan, so they find "no instance to
    lift".
- **Not active in the benchmark graph:**
  - **G-046:** its region absorbed the `all_reduce` bucket-pack `cat` + `view`, which exists only after FSDP
    bucketing. The GA builder runs bucketing after the extra passes, so the region does not match there.
    `LIFT_OPTIONAL=1` skips it (`full_final.log`: "skipping lift_fusion_region_g046"). It is active in the
    1-microbatch runs.
  - **G-031:** had the same problem: its region started at the FSDP unshard `view` of the flat bias. The
    matcher was edited to start at the `[E, 2, O]` bias (sec. 9). The problem was regenerated (`pat_jofj...`),
    rewritten (rex t1) and re-verified (gus); it now matches in both graphs.
- **Superseded:**
  - **G-031 v1** (`pat_i5a5...`): verified, but keyed to the old region; not in the cache.
- **Remaining time** (sec. 5, final):
  - About 333 ms of the 535 ms are GEMMs and attention, which no kernel here can reproduce bit-exactly.
  - 82 ms is other aten glue spread over many small regions.
  - The benchmark's wall time is 642 ms per microbatch (81,920 tokens per step at 25,532 tokens/s). That is
    roughly 70 ms above the 570 ms of kernel time, including the fake-PG copies: launch overhead (12,669
    launches per microbatch) and the dispatcher's host syncs.

## 9. Code changes

- **`torchtitan/experiments/graph_trainer/substitute_kernels.py`** (#5838 file), with tests in
  `tests/test_substitute_kernels.py`:
  - **Symbolic sizes.** Inputs with symbolic sizes are keyed by their description, with the symbols renamed
    `N0`, `N1`, ... GPT-OSS's routed-row count is a new unbacked symbol in every layer. The AOTAutograd key
    includes the enclosing graph's symbol names, so every site got its own specialization, and no kernel could
    cover all 36 layers.
  - **One module per kernel file.** Each kernel file is loaded once per content hash. Before, `_load_run`
    re-executed the module at every call site, which cost about 0.16 s and a device sync each (found by
    verifier lux). The substitute pass went from 98 s to 7.5 s.
- **`loop_recipe.py`** (this directory):
  - **`unsubstituted_to_call_function`.** The GA builder (`split_fsdp_collectives.extract_fsdp_reduce_grad_graph`)
    deep-copies the graph and runs DCE on the copy, which has no owning module. A remaining `call_module` node
    fails that DCE's purity check ("owning_module not set"). The pass calls such submodules as functions,
    keeping each call site's meta. In `full_final` every lifted site is substituted, so it has nothing left to
    convert, but it is needed whenever a kernel is missing.
  - **`LIFT_OPTIONAL=1`:** skips a lift pass whose region has no instance in the graph (G-046 in the GA graph).
- **`fusion_regions/g031_w13/lift_pass.py`:** edited by the coordinator. The region now starts at the
  `[8, 2, 2880]` bias instead of the FSDP unshard `view` (`x2 = n9`; `n9` removed from the members).
- **`autokernel/harness.py`:**
  - **Inputs:** symbolic sizes from `--sym` or the real inputs; `--gen` input generators for index-valued
    inputs.
  - **Matching training:** deterministic algorithms with `fill_uninitialized_memory=False`, and poisoned output
    memory.

## 10. Incidents and corrections

1. **GA builder crash on `call_module`:**
   - **First benchmark run:** it failed in the GA builder's DCE (sec. 9).
   - **Two failed fixes:** both tried to inline the submodules instead.
     - The shared submodule was deleted while other sites still used it.
     - Copying `meta["val"]` from the first site gave the k RoPE site the q shape.
   - **Final fix:** `unsubstituted_to_call_function` keeps each site's own meta.
2. **Contended benchmark:** full_iter2a ran while verifiers used GPU 0 and logged 7,612 tokens/s, below the
   baseline. All later headline numbers come from an exclusive GPU.
3. **Unverified kernel in a training run (by design, isolated):** G-005 (ivy t9) ran from a separate
   `kernel_cache_trial` (trial_g005, full_trial_g005) before its verdict, to size the win. It was promoted only
   after verifier ora's PASS, then re-measured from `kernel_cache` (iter3a).
4. **G-011 took three candidates:**
   - **Writer stopped:** writer mox stopped responding and was stopped; the coordinator found its f1 on disk.
   - **f1 FAIL (fay):** when the first call happens inside CUDA-graph capture, the grid-barrier word is cached
     un-zeroed. With a dirty memory pool the next call hung for 207 s.
   - **f2 FAIL (hal):** pip's fix introduced a two-thread race that freed a barrier word still in use.
   - **f3 PASS (joy):** adds a module lock.
   - **Status:** neither defect is reachable in this recipe (no CUDA graphs, one thread per device), but neither
     candidate was promoted.
5. **Regions absorbing bucketing nodes:** G-031 and G-046 did not match in the GA graph (sec. 8). Found in the
   first plan_iter3 benchmark (full_iter3b).
6. **Wait loops watched the wrong PID** (the bash wrapper instead of the trainer). Later loops match
   `pgrep -f "^python -m torchtitan.train ..."`.
7. **Board times:** some coordinator rows had guessed times; they were corrected to `date` output.
8. **Misattributed GPU process:** the coordinator blamed writer ivy for a GPU 0 process, from an `nvidia-smi`
   query that covered all GPUs. This was corrected; no result was affected.

## 11. Open items and decisions for you

- **Done:** final runs with all 20 kernels: 25,532 tokens/s, 3.26x the same-session deterministic baseline and
  +67.5% over the non-deterministic recipe, bitwise identical.
- **Before using these kernels in a real multi-GPU run:**
  - **Cooperative kernels:** G-011, G-014, G-023, G-025 and G-036 use whole-GPU cooperative grids with a
    persistent grid barrier.
    - Their launch waits for all 152 SMs. Under the fake PG there is one stream, and this is fine.
    - With real NCCL kernels overlapping, they would serialize against communication. They could deadlock if
      an SM-holding kernel waits on work queued after them (shown by verifiers bea and dov with side-stream
      hogs).
    - They need a real multi-GPU test, or non-cooperative variants.
  - **G-028:** the hardcoded reduction order has no runtime guard (sec. 7).
  - **G-005:** raises `ValueError` on misaligned or non-contiguous inputs (no fallback), and silently drops
    out-of-range indices where the reference device-asserts.
  - **Import-time warm-up and first call under capture:**
    - G-031, G-039, G-041, G-046 and G-048 synchronize the device in an import-time warm-up.
    - G-014 and G-023 cannot make their first call under CUDA-graph capture.
    - Both are fine here (substitution runs in real mode, no CUDA graphs), but not under CUDA graphs.
- **Upstream fixes, needs your decision:**
  - **GA builder:** don't require every node to have an owning module, so `unsubstituted_to_call_function` is
    unnecessary.
  - **Finder:** regions should not absorb FSDP or collective bucketing nodes. That would make G-046 match in the
    GA graph (~4 ms per microbatch).
  - **#5838:** report the two `substitute_kernels` changes (sec. 9) to the PR author (not posted anywhere).
- **Beyond kernel fusion:**
  - **MXFP8 experts:** the recipe's TODO, about 200 ms of BF16 grouped GEMM per microbatch.
  - **CUDA graphs:** currently blocked by the dispatcher's host sync.
  - **Fewer launches:** about 70 ms per microbatch of the benchmark's wall time is not covered by kernels.
- **Not yet done:** a pull request.

## 12. How to audit

- **Re-check numerics:**
  - benchmark: `python tbcmp.py full_det/tb full_final/tb`;
  - 1-microbatch: `python tbcmp.py capture/tb final/tb`.
- **Re-gate one kernel:**
  ```
  CUDA_VISIBLE_DEVICES=<gpu> python autokernel/harness.py \
      kernel_cache/pat_.../specialization_<l2>.json kernel_cache/pat_.../<l2>.py \
      --seeds 0 1 2 3 4 --real <workspace>/real_inputs/fusion_G<id>.pt --time [--gen autokernel/gen/<gen>.py]
  ```
  The index-valued problems need `--gen`:
  - G-005: `gen_index_add.py`;
  - G-011, G-014, G-023, G-025 and G-036: `gen_mox.py`;
  - G-031, G-039, G-041 and G-046: `gen_rex.py`.
- **Verify cache contents:** `sha256sum kernel_cache/pat_*/[a-z0-9]*.py | grep -v reference.py`, against sec. 7.
- **Read the record:**
  - each verifier's directory under `autokernel/run1/verify/` has its checker scripts, negative controls, logs
    and the verdict summary;
  - `autokernel/board.md` has the full history: problem rows, handoffs, verifications, promotions.
- **Reproduce:**
  - the final benchmark: the command under "Run", about 3 minutes on one GB300 with a warm kernel cache;
  - the deterministic baseline: `RUN=full_det ... --config full_det`.
