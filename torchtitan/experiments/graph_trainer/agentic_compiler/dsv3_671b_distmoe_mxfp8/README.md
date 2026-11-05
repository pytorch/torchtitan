# Agentic compiler loop: DSv3 671B DistMoE MXFP8

fusion_region_finder proposes a fusion plan; `substitute_with_kernels_pass` (#5838) turns each lifted region
into a kernel-cache problem; the autokernel swarm (#5839) writes CuTe DSL kernels; an independent verifier
checks each one; verified kernels replace the regions in training. Model numerics must stay bitwise
identical.

## This directory

| Path | Contents |
|---|---|
| `loop_recipe.py` | recipes `lifted` (plan + kernel substitution) and `capture` (plan + saving each region's real inputs) |
| `plans/plan_iterN.txt` | the fusion plan (lift passes, in order) of each iteration; `plan_iter9.txt` is current |
| `fusion_regions/` | the 15 regions of the current plan: `region.json` and the generated `lift_pass.py` each |
| `kernel_cache/` | the substitute-kernels cache: per pattern `reference.py`, `specialization_*.json`, and the promoted kernel `<kernel_name>.py` where one exists; files are byte-identical to the verified candidates (sha256 in sec. 7) |
| `regions.json` | the finder's full region ranking for the baseline capture |
| `results.json` | per run: GPU time per step and the logged loss / grad_norm values, compared with the baseline |
| `autokernel/board.md`, `autokernel/harness.py` | the swarm's task board (every handoff, verdict and promotion) and the correctness/timing harness |

Not committed, kept in the local capture directory `outputs/fusion_finder/dsv3_671b_distmoe_mxfp8/`:
real captured inputs (7 GB), profiler traces and graph dumps, writer candidates, and the verifiers' scripts and
logs (`autokernel/run1/verify/`, about 1 GB). Paths in the report below that start with `autokernel/run1/`,
`real_inputs/`, `iterN/` or `tb/` refer to that directory.

The generated lift passes and the kernels are committed as produced and verified: they are not reformatted by
the repo's pre-commit hooks, since reformatting a kernel would change the file the verifier checked.

## Run

```
S=torchtitan/experiments/graph_trainer/.claude/skills/fusion_region_finder/scripts
D=torchtitan/experiments/graph_trainer/agentic_compiler/dsv3_671b_distmoe_mxfp8
env FUSION_PLAN=$(cat $D/plans/plan_iter9.txt) LOOP_ITER=iter9 PYTHONPATH=$S:$D:. NGPU=256 LOCAL_RANK=0 \
    CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -m torchtitan.train --module loop_recipe --config lifted --comm-backend fake
```

Outputs (TensorBoard, profiler trace, graph dumps) go to `$D/$LOOP_ITER/`. Requires torchao and dist_moe
(DistMoE recipe dependencies). A kernel cache miss runs the region's original ops, so numerics are unchanged
either way.

# Audit report

## 1. Summary

| | Value |
|---|---|
| GPU time per step, excluding fake-PG collectives, baseline -> iter9 (both on GPU 0) | **583.9 ms -> 503.4 ms (-80.5 ms, -13.8%)** |
| Same, including fake-PG collectives | 758.8 ms -> 678.4 ms |
| Numerics | loss and grad_norm bitwise identical to the baseline in every iteration (sec. 6) |
| Kernels in the cache | 20 kernel files across 18 patterns, each written by one agent, checked by a different freshly spawned verifier, and promoted only on PASS (all 20 hit at every site in iter9) |
| Verified per-call savings x calls per step | ~76 ms; consistent with the measured -80.5 ms (iter9 also includes ~2-3 ms of run-to-run variation) |
| Pending | nothing in the loop; remaining options need a decision (sec. 11) |

## 2. Goal and constraints

- **Goal (user, 2026-11-04):** improve DSv3 DistMoE MXFP8 performance in a loop. fusion_region_finder proposes a
  fusion plan; #5838 (`substitute_with_kernels_pass`) turns the lifted regions into kernel-cache problems; #5839
  (autokernel skill) generates kernels.
- **Hard constraint (user):** model numerics must not change. A training run counts as passing only if loss and
  grad_norm are bitwise identical to the baseline.
- **Other rules followed:**
  - nothing is submitted to KernelGen;
  - nothing is committed or pushed;
  - no GitHub activity;
  - the shared host's other user's processes are never touched.

## 3. Setup

- **Branch:** local `fusion-region-finder` = main `d0ab95e46` + `b0101d3d9` (#5838) + `2259a7df1` (#5839),
  cherry-picked locally.
- **Uncommitted local patch:** `torchtitan/experiments/graph_trainer/substitute_kernels.py`, `_op_modules`
  (sec. 9).
- **Recipe:**
  - base: `graph_trainer_deepseek_v3_671b_dist_moe_mxfp8`, i.e. DistMoE + MXFP8 + fused_mla + fused_swiglu;
  - parallelism: FSDP 256, EP 64, PP 1, fake process group, one GB300 (`NGPU=256 LOCAL_RANK=0`);
  - `fsdp_defer_gradient_reduction=False`, because GraphTrainer rejects True;
  - `scratch_capacity_factor=64`;
  - seed 42, deterministic. torchtitan's `set_determinism` also sets `fill_uninitialized_memory=False`.
  - Files: `torchtitan/experiments/graph_trainer/.claude/skills/fusion_region_finder/scripts/example_recipe.py`,
    `loop_recipe.py` (configs `lifted`, `capture`).
- **Run one iteration:**
  ```
  env FUSION_PLAN=$(cat plan_iterN.txt) LOOP_ITER=iterN PYTHONPATH=$S:$OUT:. NGPU=256 LOCAL_RANK=0 \
      CUDA_VISIBLE_DEVICES=<gpu> PYTORCH_ALLOC_CONF=expandable_segments:True \
      python -m torchtitan.train --module loop_recipe --config lifted --comm-backend fake
  ```
  `S` = the skill's `scripts/` directory; `OUT` = this directory. 10 steps, profiler on step 10, TensorBoard to
  `iterN/tb`.
- **Bitwise check:** `python /tmp/tbcmp.py tb iterN/tb` compares grad_norm, global_avg_loss, global_max_loss and
  the MoE load-balance loss for exact float equality.
- **GPU time:**
  - `parse_trace.py iterN`, then sum `analysis/kernels.pkl` durations;
  - `all_gather_into_tensor_out` / `reduce_scatter_tensor` / `all_to_all` are excluded: about 165-175 ms of copies
    that only emulate collectives under the fake PG and are not production compute.

## 4. The loop

```
 fusion_region_finder (profile + graph dump of the baseline capture)
   -> regions.json, one folder per region with a generated topology-matching lift pass
   -> plan_passes.py: a plan of lift passes that replays consistently (no overlapping sites)
 training run with the plan (loop_recipe capture): each region becomes call_module fusion_G<id>
   -> #5838 substitute_kernels: cache miss writes kernel_cache/pat_<l1>/{reference.py, specialization_<l2>.json};
      real arguments of each region are saved to real_inputs/
 #5839 autokernel swarm (board: autokernel/run1/board.md, harness: autokernel/run1/harness.py)
   writer (one GPU lease) -> candidate + its own checks
   -> fresh verifier (never the writer) -> PASS/FAIL with evidence under autokernel/run1/verify/<name>/
   -> coordinator promotes on PASS: copy to temp file, os.replace onto pat_<l1>/<kernel_name>
 training run with the plan + cache -> bitwise check vs baseline TensorBoard -> profile -> next plan
```

All lift passes are generated from the baseline capture's `pre_inductor` graph and run before substitution. The
iteration-7 profile was used to re-rank what was left (sec. 8).

## 5. Results by iteration

| Run | Plan (passes) | New kernels | GPU | Total ms | Excl. fake comm ms | vs baseline | Hits | Loss / grad_norm |
|---|---|---|---|---|---|---|---|---|
| baseline | - | - | 0 | 758.8 | 583.9 | - | - | reference |
| iter2 | 7 | none (lift only) | 0 | 758.9 | 583.4 | -0.5 | - | bitwise |
| iter3 | 7 | router bwd | 0 | 746.2 | 572.9 | -11.0 (-1.9%) | 58 | bitwise |
| iter4 | 7 | + router recompute | 0 | 740.6 | 567.4 | -16.5 (-2.8%) | 116 | bitwise |
| iter5 | 12 | + MLA kv-bwd/quant, 5 norm+quant | 0 | 730.6 | 554.7 | -29.2 (-5.0%) | all sites | bitwise |
| iter6 | 12 | + router fwd | 3 | 702.0 | 535.4 | -48.5 (-8.3%) | all sites | bitwise |
| iter7 | 12 | + bucket pack, 2 norm-bwd+quant, GEMM-output permute | 3 | 681.2 | 515.9 | -68.0 (-11.6%) | all sites | bitwise |
| iter8 | 15 | + routed-experts pack, 4 quant-before-GEMM | 0 | 688.2 | 514.2 | -69.7 (-11.9%) | all sites | bitwise |
| **iter9** | **15** | + prepare_mxfp8_weight w13, w2 | **0** | **678.4** | **503.4** | **-80.5 (-13.8%)** | **all sites** | **bitwise** |

**Caveats:**
- **GPU 3 runs faster:** iter6 and iter7 ran on GPU 3, which is faster on fake-comm copies (~9 ms) and GEMMs
  (~1.6%, ~2 ms). Compare iter8 / iter9 against the baseline (all on GPU 0) for the clean number.
- **Label shift:** GEMMs inside substituted kernels (5nj2, ponk, lfu5, bqhy) appear under the label `run`, not
  `_scaled_mm_v2`. Per-category GEMM totals are not comparable across iterations; the step totals are.
- **Memory:** peak memory is flat at 138.9-139.5 GiB in all runs.
- **Step-10 values:** every run logs loss 12.47923 and grad_norm 4.8495.

## 6. Numerics evidence

- **What is compared:** each iteration's TensorBoard against the baseline `tb/`. TensorBoard logs steps 1 and 10;
  both are bitwise equal in every run, for grad_norm, global_avg_loss, global_max_loss and the load-balance loss.
  Example from iter9: grad_norm step 1 = 63.61985778808594, step 10 = 4.849545478820801, identical to the baseline.
- **Limitation:** steps 2-9 are not logged. Step 10 depends on every earlier optimizer update, so an exact match at
  step 10 is strong evidence but not a proof that steps 2-9 match.
- **Per-kernel parity:** checked bytewise against the immutable `reference.py`, on the harness (random seeds, real
  captured inputs, poisoned output memory) and on each verifier's own adversarial suite (sec. 7).

## 7. Kernels in the cache

Reference and kernel times are the verifier's amortized measurements: several calls per CUDA graph, interleaved
with the reference. "In training" is per call in the iter8 profile; for lfu5 and bqhy it includes the unchanged
GEMM.

| Pattern (region) | What | Calls/step | Ref us | Kernel us | Saving ms/step | Writer -> verifier | Candidate (sha256 prefix) | Evidence |
|---|---|---|---|---|---|---|---|---|
| gjn6 (G-001) | router forward: group-limited top-8, aux loss | 58 | 244.9 | 6.22 | 13.8 | cyd t23 -> vic | 9aff7655dcd58285 | verify/router_fwd_vic (t22 failed vee) |
| erxi (G-006) | router backward | 58 | 201.1 | 5.84 | 11.3 | ada t2 -> vera | 71b49b7f27a00aee | verify/router_bwd_vera |
| h4xd (G-020) | router recompute | 58 | 86.7 | 6.23 | 4.7 | bix t8 -> vik | 6e7d1dfbf01e2537 | verify/router_recompute_vik |
| k6pn (G-035) | fused_mla_kv_backward + mxfp8 row/col quant | 61 | 193 | 101 | 5.6 | dex t12 -> vox | 9b86f99de39130c5 | verify/mla_kv_bwd_quant_vox |
| z47c (G-047 v0) | rmsnorm + 2x rowwise quant | 61 | 52.2 | 18.2 | 2.1 | eve g4 -> val | 5fddb4600bb4dcc4 | verify/norm_quant_val |
| use7 (G-047 v1) | rmsnorm -> y, rstd + 2x quant | 61 | 54.1 | 26.4 | 1.7 | eve g4 -> val | a1355f8434cd18aa | verify/norm_quant_val |
| dre3 (G-082 v2) | add + rmsnorm + quant | 61 | 65.1 | 40.8 | 1.5 | eve g4 -> val | 6a8757ee290e3c57 | verify/norm_quant_val |
| c2qd (G-082 v0) | add + rmsnorm + quant | 58 | 65.2 | 40.0 | 1.5 | eve g4 -> val | fdb868bd34c83521 | verify/norm_quant_val |
| 5yin (G-082 v1) | add + rmsnorm + quant | 3 | 65.0 | 32.7 | 0.1 | eve g4 -> val | bd6453efc96f4a73 | verify/norm_quant_val |
| a6rv (G-049 v1) | rmsnorm bwd + add + row/col quant | 61 | 92.7 | 56.6 | 2.2 | fay t16 -> vin | 576abac161b1ca23 | verify/norm_bwd_quant_vin |
| 3ybb (G-049 v0) | same, other output order | 60 | 92.9 | 56.6 | 2.2 | fay t16 -> vin | 3b78bb0a31b1856c | verify/norm_bwd_quant_vin |
| 5nj2 (G-007) | GEMM (unchanged) + 8-row-block permutation replacing a slow cat | 58 | 232.8 | 76.8 | 9.0 | gil t5 -> vlad | d61766f2c01319da | verify/gemm_permute_vlad |
| ponk (G-003) | GEMM (unchanged) + FSDP reduce-scatter bucket pack | 58 | 494.4 | 396.7 | 5.7 | gil t4 -> vera2 | a5ab4066053b41b7 | verify/bucket_pack_vera2 |
| czaf (G-028) | routed-experts reduce-scatter bucket pack | 58 | 173.6 | 102.5 | 4.1 | gil t1 -> vex | dc7661b015fcb641 | verify/routed_pack_vex |
| lfu5 a2jgmo (G-015) | fused quantize + scale layout, then unchanged GEMM | 61 | 30.4 | 26.0 | 0.3 | eve f1 -> vola | f48e45c866b5455a | verify/quant_gemm_vola |
| lfu5 adm7po (G-015) | same | 61 | 44.8 | 38.5 | 0.4 | eve f1 -> vola | 6a3be1cfa65e7a11 | verify/quant_gemm_vola |
| lfu5 autudf (G-015) | same | 61 | 187.2 | 168.1 | 1.2 | eve f1 -> vola | 6918f0a46c7edb20 | verify/quant_gemm_vola |
| bqhy a42i3e (G-015) | same, other output order | 58 | 89.7 | 81.7 | 0.5 | eve f1 -> vola | cd1656916fc4da0e | verify/quant_gemm_vola |
| x6t6 (G-021) | `dist_moe.prepare_mxfp8_weight`, w13 [4,4096,7168] | 116 | 102.0 | 51.4 | 5.9 | hana f2 -> vita | 88c5a967f5814cf4 | verify/prep_weight_vita |
| 5uzo (G-061) | `dist_moe.prepare_mxfp8_weight`, w2 [4,7168,2048] | 116 | 45.1 | 28.3 | 1.9 | hana f2 -> vita | 5673845eb1483e64 | verify/prep_weight_vita; writer claimed 23.8 us, measured with L2-resident outputs |

Candidate sources are under `autokernel/run1/candidates/<problem>/<writer>/...`; the board's "promote" rows list
exact paths. Promoted copies are `kernel_cache/pat_<key>*/<kernel_name>`. The sha256 prefixes above were taken
from the files in `kernel_cache/` and match the verified candidates.

**Current keys.** Cache keys no longer depend on the device index (substitute_kernels, `d89f7f53e`). erxi's
region contains a device literal, so its entry is now
`kernel_cache/pat_uu33hfreyzn473dzqocdqrbpqz2g3dyzp3xqfqvslrw66qjxcud/aja7qubrcdovq6ft262x33yxnf34l3sxh4ti6uxukos6kqjbb7ei.py`.
It is the same file (sha256 `71b49b7f27a00aee`); its `reference.py` differs only in `device(type='cuda')`. The
lift passes in `fusion_regions/` are regenerated with shape- and rank-agnostic matchers (`ae98eaac8`).

**What every verifier checked** (details in each evidence directory):
- **Contract:**
  - sha256 and the signature against the spec;
  - the official harness gate: seeds 0-7 plus real inputs, every case EXACT;
  - output metadata and alias structure against the reference;
  - inputs unmutated (whole storages) and repeated calls identical.
- **Memory:** allocator poisoning, proving every output byte is written, and compute-sanitizer
  (memcheck/initcheck/racecheck/synccheck).
- **CUDA graphs:** neighbour kernels writing the inputs and reading the outputs around the call, many replays with
  inputs changed in place, and two calls in one graph.
- **Adversarial inputs:** each verifier's own suite with negative controls (deliberately broken copies that must
  fail). Reduction orders were checked directly where an output hides them, e.g. a column sum that only feeds a
  scalar.
- **GEMM-adjacent kernels:** proof that the `_scaled_mm_v2` call is identical (same op, same arguments, same operand
  metadata and alignment, same cuBLAS kernel) and gives bitwise-equal output.
- **PDL:** kernels using programmatic dependent launch were tested with an early-triggering, small-grid predecessor.

**Dependency to know about:** the norm kernels (z47c, use7, dre3, c2qd, 5yin, a6rv, 3ybb) reproduce the
reduction order of the vendored quack CuTe RMSNorm. On this PyTorch build, `torch/_native/ops/norm/rmsnorm_impl.py`
dispatches `aten._fused_rms_norm` and its backward there, unless `TORCH_DISABLE_NATIVE_JIT` is set. If that
override is disabled, the reference's own numerics change, and these kernels would no longer match it.

## 8. Plans and what was not promoted

- **Plans:**
  - `plan_iter2.txt` .. `plan_iter4.txt`: G-006, G-020, G-001, G-047, G-082, G-049, G-035.
  - `plan_iter5.txt` .. `plan_iter7.txt`: those plus G-007, G-003, G-015, G-031, G-073.
  - `plan_iter8.txt`: those plus G-028, G-021, G-061.
  - `plan_iter9.txt`: the same passes as `plan_iter8.txt`; iter9 adds the 2 `prepare_mxfp8_weight` kernels.
- **Not promoted, or not attempted, and why:**
  - **t6rh (G-031, colwise quant before GEMM, 6 specializations):** torchao's quantize is already ~7 TB/s; the
    gain is about 1-3 us per call, partly within noise. Not promoted.
  - **Rare specializations** (lfu5 aglgyz, bqhy afd5ld; 1-3 calls/step): timing swings with GEMM clock drift.
    Not promoted.
  - **l4cu (G-073, FSDP all-gather bucket):** the output contains uninitialized bytes outside this rank's slot, so
    a bytewise gate is ill-defined. Skipped; this is a #5838 contract gap (sec. 11).
  - **`triton_to_mxfp8_32x32_swizzle` + GEMM regions** (G-025 family) and **G-062** (GEMM + rmsnorm bwd): the
    non-GEMM part is already at bandwidth, or nothing is fusable without changing the GEMM. Dropped from the plan;
    cache entries moved to `kernel_cache_stale/`.
  - **GEMM epilogues writing straight into the bucket** (ponk): would need a batched GEMM with a different cuBLAS
    algorithm, so numerics risk. Only the pack was replaced.
- **x6t6 / 5uzo (G-021 / G-061):** `dist_moe.prepare_mxfp8_weight` for w13 and w2. Verified PASS by vita and
  promoted after iter8. vita measured w2 at 1.59x (28.3 us), not the writer's 1.90x: the writer's number came from
  a graph whose outputs stayed L2-resident.
- **Iteration-7 re-ranking** (`iter7/analysis/`):
  - Remaining memory-bound glue is 191 ms measured against 148 ms ideal per node, so most of it is near bandwidth.
  - Remaining time is GEMMs, attention and DistMoE grouped GEMMs, which no kernel here can reproduce bit-exactly.

## 9. Code changes (all local, uncommitted)

- **`torchtitan/experiments/graph_trainer/substitute_kernels.py`** (#5838 file), `_op_modules`. Without this
  patch:
  - torchao ops (e.g. `torchao.mxfp8_quantize`) raised "not a torch.library.custom_op";
  - bucketing ops raised "not defined in an importable module".

  The patch resolves an op's module from `torch._library.simple_registry` fake-impl source files. All 18 tests in
  `tests/test_substitute_kernels.py` pass.
- **Skill: fusion_region_finder** (`torchtitan/experiments/graph_trainer/.claude/skills/fusion_region_finder/`,
  untracked):
  - **`lift_regions.py`:** `_plain_outputs` in `find_matches`. Per matched site, a tuple-valued member takes its
    `getitem` users, and a view of external inputs that is also used outside stays outside, so lifted submodules
    output plain tensors at every site. Also `writes_outside` and recorded-schema support.
  - **`curate_regions.py`:** the generated matchers come from one example instance, and the per-site output rules
    live in `lift_regions.py` only.
  - **`plan_passes.py`** (new): replays a wish list of passes and emits a consistent plan.
  - **`graph_lib.py`, `build_node_table.py`, `find_regions.py`, `calibrate_roofline.py`:** MXFP8 GEMM peak and
    `fused_moe` category.
  - **`SKILL.md`:** documents the above.
- **Run-local files** (in this directory):
  - `loop_recipe.py`: `KERNEL_CACHE` override, atomic real-input saves.
  - `autokernel/run1/harness.py`, matching training: deterministic with `fill_uninitialized_memory=False`; poisons
    candidate output memory; flattens nested outputs; fp8 input dtypes; position-index int64 inputs; no bytecode
    writes under `kernel_cache/`.
- **Untouched:** your pre-existing modified files (numerical_tests_kimi.py, video.py, flux hf_embedder.py,
  tokenizer.py).

## 10. Incidents and corrections

1. **Inflated per-kernel speedups (fixed):** the harness ran with `fill_uninitialized_memory=True`, so the
   references paid for NaN-filling `torch.empty` outputs that training never pays. Example: the k6pn reference was
   346 us with the fill, 199 us without. The harness was fixed and all references re-timed. End-to-end numbers come
   from training and were never affected.
2. **Harness blind spot: unwritten zero bytes (fixed):** found by writer gil. A pack kernel with its zero-padding
   stores removed passed, because fresh memory reads as zero. The harness now poisons output memory; that broken
   copy fails, and all promoted kernels still pass. All verifiers had already used poisoning in their own checks.
3. **PDL ordering bug in router forward t22 (caught):** verifier vee found the main kernel triggered its dependent
   tail before waiting on its own predecessor. With a legal early-triggering predecessor, the aux loss was wrong in
   20/20 trials. Fixed in t23 (wait, then trigger) and re-verified by vic. Promoted kernels were audited for the
   same pattern; router recompute reads its input before its wait, but is safe because its main kernel launches
   without PDL.
4. **Wrong pass lists from region overlap (fixed earlier):** G-047's matcher took G-067's sites; `plan_passes.py`
   now replays plans.
5. **Inconsistent submodule outputs** (tuples and input views leaking at some sites, fixed): this changed the cache
   keys of four patterns, which were regenerated; stale entries are in `kernel_cache_stale/`.
6. **Measurement noise from the shared host:** another user's `benchmark_probe` periodically holds memory on all
   four GPUs, at 0% utilization when sampled. Profiled training runs waited for the GPU to be free. One verifier
   (vita) blocked for hours waiting for an empty GPU and was told to time on idle utilization instead.
7. **A verifier's `pkill` may have hit another agent's sanitizer run** (val). No verdict was affected.
8. **A waiter loop matched its own command line** (`pgrep -f`) and never exited. It was stopped; later waiters
   poll `nvidia-smi` instead.

## 11. Open items and decisions for you

- **Done:** iter9 with all 20 kernels: -80.5 ms (-13.8%) vs the baseline on the same GPU, bitwise identical.
- **Beyond kernel fusion, needs your decision:**
  - **Save, don't recompute:** stop recomputing `prepare_mxfp8_weight`; about 8.5 ms/step, identical numerics,
    more memory.
  - **FSDP unpack copies:** remove the all-gather `split_with_sizes_copy` copies; about 31.5 ms/step, already at
    bandwidth, so this needs a different parameter layout in the bucket.
  - **Pre-permuted GEMM inputs (gil's idea):** permute the rows of 5nj2's GEMM inputs so it writes rows in output
    order; about 0.75 ms. This changes the GEMM operands, so it needs approval and a bitwise check.
- **Worth reporting to the #5838 author (not posted anywhere):**
  - the `_op_modules` patch (sec. 9);
  - outputs with uninitialized regions (FSDP all-gather bucket) cannot be expressed in the cache contract;
  - first-call-under-capture: 5nj2 builds a small table lazily, which fails loudly if the very first call happens
    under CUDA-graph capture. GraphTrainer always runs eagerly first, so training never hits this.
- **Not yet done:** commits, a PR, or folding the loop scripts (substitute registration, real-input capture,
  harness) into the skills.

## 12. How to audit

- **Re-check numerics of a run:** `python /tmp/tbcmp.py tb iter8/tb`.
- **Re-gate one kernel:**
  ```
  cd autokernel/run1
  CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=/home/bahuang/local/torchtitan \
      python harness.py <kernel_cache/pat_.../specialization_*.json> \
      <kernel_cache/pat_.../<kernel_name>.py> --seeds 0 1 2 3 4 --real ../../real_inputs/<file> --time
  ```
- **Verify cache contents:** `sha256sum kernel_cache/pat_*/a*.py` (exclude `reference.py`) against sec. 7.
- **Read the record:**
  - each verifier's directory under `autokernel/run1/verify/` has its checker scripts, negative controls, logs and
    the verdict summary;
  - `autokernel/run1/board.md` has the full history: problem rows, handoffs, verifications, promotions.
- **Reproduce an iteration:** the command in sec. 3 with `plan_iterN.txt`. iter8 takes about 15 minutes on one
  GB300.
