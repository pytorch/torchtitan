# Agentic compiler loop: Qwen3 30B-A3B DistMoE MXFP8

The same loop as `../dsv3_671b_distmoe_mxfp8/`, applied to Qwen3 30B-A3B:
- fusion_region_finder proposes a fusion plan;
- `substitute_with_kernels_pass` (`graph_trainer/substitute_kernels.py`) turns each lifted region into a
  kernel-cache problem;
- the autokernel swarm writes CuTe DSL kernels;
- an independent verifier checks each one;
- verified kernels replace the regions in training.

Model numerics must stay bitwise identical.

## This directory

| Path | Contents |
|---|---|
| `loop_recipe.py` | recipes `baseline` (finder capture), `lifted` (plan + kernel substitution) and `capture` (plan + saving each region's real inputs) |
| `plans/plan_iterN.txt` | the fusion plan (lift passes, in order); `plan_iter4.txt` is current |
| `fusion_regions/` | the 16 regions of the current plan: `region.json` and the generated `lift_pass.py` each (G-90 and G-91 hand-built, sec. 8) |
| `kernel_cache/` | the substitute-kernels cache: per pattern `reference.py`, `specialization_*.json`, and the promoted kernel `<kernel_name>.py` where one exists; files are byte-identical to the verified candidates (sha256 in sec. 7) |
| `regions.json` | the finder's full region ranking for the baseline capture (taken with cuDNN attention, sec. 3) |
| `results.json` | per run: GPU time per step and the logged loss / grad_norm values, compared with the baseline |
| `autokernel/board.md`, `autokernel/harness.py` | the swarm's task board (every handoff, verdict and promotion) and the correctness/timing harness |

Not committed, kept in the local capture directory `outputs/fusion_finder/qwen3_30b_a3b_distmoe_mxfp8/`:
- real captured inputs (`real_inputs/`, 9 GB);
- profiler traces, graph dumps and TensorBoard of every run (`baseline_fa4/`, `iterN/`);
- the frozen kernel caches iter5-7 used (`cache_snapshots/iterN/`);
- writer candidates and the verifiers' scripts and logs (`autokernel/run1/`);
- the helper scripts used for the numbers below (`tools/`).

Paths in the report below that start with `autokernel/run1/`, `real_inputs/`, `iterN/`, `baseline_fa4/` or
`tools/` refer to that directory.

The generated lift passes and the kernels are committed as produced and verified: they are not reformatted by
the repo's pre-commit hooks, since reformatting a kernel would change the file the verifier checked.

## Run

```
S=torchtitan/experiments/graph_trainer/.claude/skills/fusion_region_finder/scripts
D=torchtitan/experiments/graph_trainer/agentic_compiler/qwen3_30b_a3b_distmoe_mxfp8
ENV="PYTHONPATH=$S:$D:. NGPU=8 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True"

# One iteration: the plan's regions lifted, each replaced by its kernel where the cache has one.
env $ENV FUSION_PLAN=$(cat $D/plans/plan_iter4.txt) LOOP_ITER=iter7 \
    python -m torchtitan.train --module loop_recipe --config lifted --comm-backend fake

# Baseline (profile, graph dumps, TensorBoard; the reference for the bitwise check).
env $ENV FUSION_FINDER_OUT=$PWD/outputs/fusion_finder/qwen3_30b_a3b_distmoe_mxfp8/baseline_fa4 \
    python -m torchtitan.train --module loop_recipe --config baseline --comm-backend fake

# One step that writes cache problems for regions without a kernel and saves their first real inputs.
env $ENV FUSION_PLAN=$(cat $D/plans/plan_iter4.txt) \
    python -m torchtitan.train --module loop_recipe --config capture --comm-backend fake
```

Outputs (TensorBoard, profiler trace, graph dumps) go to `$D/$LOOP_ITER/`; captured inputs to `$D/real_inputs/`
(both gitignored). `KERNEL_CACHE=<dir>` points the run at a frozen copy of the cache. A kernel cache miss runs the
region's original ops, so numerics are unchanged either way. Requires torchao and dist_moe (DistMoE recipe
dependencies) and FA4 (`flash_attn.cute`).

# Audit report

## 1. Summary

| | Value |
|---|---|
| GPU busy time per step, excluding fake-PG collectives, baseline -> iter7 (both on GPU 0) | **967.9 ms -> 679.6 ms (-288.3 ms, -29.8%)** |
| Same, as a sum of kernel durations | 968.4 ms -> 688.4 ms (-280.0 ms) |
| Numerics | loss and grad_norm bitwise identical to the baseline in every iteration (sec. 6) |
| Kernels in the cache | 15 kernel files across 15 patterns, each written by one agent, checked by a different freshly spawned verifier, and promoted only on PASS (all 15 hit at every site in iter7) |
| Verified per-call savings x calls per step | ~282 ms; consistent with the measured -288.3 ms |
| Roofline estimate | the finder's baseline analysis puts the total fusion gain at ~289 ms (sec. 8); what is left of glue headroom is ~12 ms |
| Pending | nothing in the loop; remaining options need a decision (sec. 11) |

## 2. Goal and constraints

- **Goal (user, 2026-11-05):** follow the DSv3 workflow (`../dsv3_671b_distmoe_mxfp8/README.md`) to optimize Qwen3
  30B-A3B. The user chose a new run directory and the same rules as DSv3.
- **Hard constraint:** model numerics must not change. A training run counts as passing only if loss and grad_norm
  are bitwise identical to the baseline.
- **Other rules followed:**
  - nothing is submitted to KernelGen;
  - nothing is committed or pushed unless the user asks;
  - no GitHub activity;
  - other users' processes on the shared host are never touched.

## 3. Setup

- **Branch:** `agentic_compiler` at `d89f7f53e`, plus the two fusion_region_finder skill fixes in sec. 9.
- **Software:** torch 2.16.0.dev20261105+cu132, CUDA 13.2, nvidia-cutlass-dsl 4.8.1, torchao
  0.19.0.dev20261105; one GB300 (sm_103, 152 SMs).
- **Recipe** (`loop_recipe.py`, `qwen3_30b_base`):
  - base: `graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu` (FSDP 8 x EP 8, 32K-token microbatches, DistMoE +
    MXFP8), fake process group on one GPU (`NGPU=8 LOCAL_RANK=0`);
  - `num_tokens_per_train_step = -1`: one microbatch per step, so the profiled step is one forward and backward;
  - **FA4 varlen attention instead of cuDNN** (`sdpa_backends = ["FLASH_ATTENTION"]`), see below;
  - seed 42, deterministic; 10 steps, profiler on step 10, TensorBoard to `iterN/tb`.
- **Why FA4: cuDNN's varlen attention backward is nondeterministic** under `debug.deterministic`, even with
  `use_deterministic_algorithms(True)` and `cudnn.deterministic=True`:
  - Two identical unlifted 3-step runs with cuDNN differed already at step 1: grad_norm 68.36235809 vs
    68.36019135 (`tools/det_recipe.py`, `tools/det/det_g2`, `det_g3`). The first lifted run (`iter1_cudnn/`) also
    differed from the baseline in grad_norm and step-10 loss.
  - With FA4 (`flash_attn.cute`), the same pair of runs is bitwise identical at every step
    (`tools/det/detfa4_g2`, `detfa4_g3`).
  - So the bitwise gate would be meaningless with cuDNN. The loop's baseline is `baseline_fa4/` (`--config
    baseline` with the FA4 pin). The finder's capture behind `regions.json` was taken before the pin, with cuDNN.
  - FA4 costs about 28 ms/step more attention time than cuDNN (157 vs 129 ms). The lifted regions are glue outside
    attention, so their problems are the same with either backend.
- **Bitwise check:** `python tools/tbcmp.py baseline_fa4/tb iterN/tb` compares grad_norm, global_avg_loss,
  global_max_loss and the MoE load-balance loss (steps 1 and 10) for exact float equality.
- **GPU time:**
  - `parse_trace.py iterN`, then `tools/busy_time.py`: the union of kernel intervals in the profiled step,
    excluding the fake-PG collective copies (~3 ms). Kernels launched with programmatic dependent launch (PDL)
    overlap their predecessor, so a plain sum of durations double-counts; `results.json` records both.
  - `tools/record_run.py` writes a run's row into `results.json`.

## 4. The loop

```
 fusion_region_finder (profile + graph dump of the baseline capture, cuDNN; 84 regions, 64 with a lift pass)
   -> regions.json, one folder per region with a generated topology-matching lift pass
   -> plan_passes.py: a plan of lift passes that replays consistently (no overlapping sites)
 training run with the plan (loop_recipe capture): each region becomes call_module fusion_G<id>
   -> substitute_kernels: cache miss writes kernel_cache/pat_<l1>/{reference.py, specialization_<l2>.json};
      real arguments of each region are saved to real_inputs/
 autokernel swarm (board: autokernel/run1/board.md, harness: autokernel/run1/harness.py)
   writer (one GPU lease) -> candidate + its own checks
   -> fresh verifier (never the writer) -> PASS/FAIL with evidence under autokernel/run1/verify/<name>/
   -> coordinator promotes on PASS: back up the old file to rollback/, copy to a temp file, os.replace
 training run with the plan + a frozen snapshot of the cache -> bitwise check vs baseline_fa4 -> profile -> next plan
```

The iteration-4 profile was re-analyzed to find what was left (G-91 came from there), and the iteration-6 profile
to measure the remaining headroom (sec. 8).

## 5. Results by iteration

All runs on GPU 0. "Busy" is GPU busy time per step excluding fake-PG collectives; "sum" is the sum of kernel
durations, same exclusion.

| Run | Plan (passes) | New kernels | Sum ms | Busy ms | Busy vs baseline | Hits | Loss / grad_norm |
|---|---|---|---|---|---|---|---|
| baseline_fa4 | - | - | 968.4 | 967.9 | - | - | reference |
| iter1 | 6 | none (lift only) | 968.4 | 967.8 | -0.1 | - | bitwise |
| iter2_lift | 14 | none (lift only) | 969.4 | 968.8 | +0.9 | - | bitwise |
| iter2 | 14 | G02, G07, G33 t1, G42 t1 | 845.4 | 844.9 | -123.0 (-12.7%) | all sites | bitwise |
| iter3 | 15 | + G19 t5, G28, G41, G45; G33/G42 t1 -> t3 | 839.1 | 834.3 | -133.6 (-13.8%) | all sites, G17 24/48 (sec. 10) | bitwise |
| iter4 | 15 | + G01 bix t5, G17 t5 | 726.7 | 722.1 | -245.8 (-25.4%) | all sites | bitwise |
| iter5 | 16 | + G06, G15, G-90 eve t2; G19 t5 -> t6 | 697.5 | 689.2 | -278.7 (-28.8%) | all sites | bitwise |
| iter6 | 16 | + G47; G17 t5 -> t6; G-90 eve t2 -> dex t3 | 695.7 | 686.6 | -281.3 (-29.1%) | all sites | bitwise |
| **iter7** | **16** | + G-91; G01 bix t5 -> ada t30 | **688.4** | **679.6** | **-288.3 (-29.8%)** | **all sites** | **bitwise** |

**Notes:**
- **Memory:** peak memory at step 10 is 236.0-236.2 GiB in the baseline and the lift-only runs, and 231.3-233.6 GiB
  with kernels.
- **Problems:** the kernel-cache problems were written by the cache misses of `capture` runs (one per plan; real
  inputs are saved in the same run): 6 patterns for `plan_iter1`, 8 more for `plan_iter2` (G24 has two
  specializations), G-90 for `plan_iter3`, G-91 for `plan_iter4`: 16 patterns, 17 specializations.
- **Step-10 values:** every run logs loss 5.71725 and grad_norm 9.8121.
- **Label shift:** GEMMs inside substituted kernels (G15, G28) appear under the label `run`, not `_scaled_mm_v2`.
  Per-category GEMM totals are not comparable across iterations; the step totals are.
- **Plans:** iter1 `plan_iter1.txt`; iter2 `plan_iter2.txt`; iter3-4 `plan_iter3.txt`; iter5-7 `plan_iter4.txt`.

## 6. Numerics evidence

- **What is compared:** each iteration's TensorBoard against `baseline_fa4/tb`. TensorBoard logs steps 1 and 10;
  both are bitwise equal in every run, for grad_norm, global_avg_loss, global_max_loss and the load-balance loss.
  Example from iter7: grad_norm step 1 = 68.36530303955078, step 10 = 9.812052726745605, loss step 10 =
  5.717247009277344, identical to the baseline.
- **Limitation:** steps 2-9 are not logged. Step 10 depends on every earlier optimizer update, so an exact match at
  step 10 is strong evidence but not a proof that steps 2-9 match.
- **Per-kernel parity:** checked bytewise against the immutable `reference.py`, on the harness (random seeds, real
  captured inputs, poisoned output memory) and on each verifier's own adversarial suite (sec. 7).

## 7. Kernels in the cache

Reference and kernel times are the verifier's measurements: CUDA-graph replays with rotating inputs and live
outputs, interleaved with the reference. Saving = (reference - kernel) x calls per step.

| Pattern (region) | What | Calls/step | Ref us | Kernel us | Saving ms/step | Writer -> verifier | sha256 prefix | Evidence |
|---|---|---|---|---|---|---|---|---|
| qnfe (G01) | attention bwd glue: RoPE bwd q/k, q/k RMSNorm bwd (incl. dw), cat with v grad, MXFP8 row+col quant, rearrange | 48 | 2480.6 | 184.5 | 110.2 | ada t30_final -> viggo | 8127cc3d06dac097 | verify/qknorm_rope_bwd_t30_viggo; replaced bix t5 c664e22d (260.7 us, vlad) |
| 4dzo (G02) | attention fwd glue: split of the wqkv output, q/k RMSNorm, RoPE with position gather | 48 | 2252.4 | 157.1 | 100.6 | ada t5_dyn_final -> vera | bbc34dfe16e1e024 | verify/qknorm_rope_fwd_vera |
| s3of (G06) | router + aux-loss backward | 48 | 381.8 | 12.05 | 17.7 | cyd t14 -> vala | 6a848bf318ba1f29 | verify/router_bwd_t14_vala (t13 failed vee) |
| 53aq (G07) | loss bwd: nll_loss_backward + log_softmax_backward, [4096,151936] fp32 -> bf16 | 8 | 2556.1 | 518.5 | 16.3 | dex t5 -> vox | b08035523e8e8253 | verify/loss_bwd_vox |
| bme7 (G-90) | loss fwd per chunk: bf16 -> fp32 cast, log_softmax, nll_loss, div | 8 | 2040.5 | 544.2 | 12.0 | dex t3 -> vika | 76a0781ef523133f | verify/loss_fwd_t3_vika; replaced eve t2 (719.6 us, vela) |
| bp6b (G17) | router fwd: per-expert token counts (int64 + fp32) | 48 | 125.75 | 3.59 | 5.9 | cyd t6 -> vojta | f129ebaa50c067c0 | verify/router_counts_t6_vojta; replaced t5 (vince) |
| zfby (G19) | add + RMSNorm bwd (N=2048) + add + MXFP8 row+col quant + rearrange | 48 | 258.7 | 146.9 | 5.4 | dex t6 -> vesna | 4c7b3a3220825f18 | verify/norm_bwd_quant_t6_vesna; t5 verified by vex |
| bj3b (G-91) | router aux-loss fwd: row L1 norm, clamp_min, div, masked column sum | 48 | 78.54 | 7.01 | 3.4 | bix t9 -> vilma | 47e58732566c7da3 | verify/aux_loss_vilma |
| tl5t (G15) | wqkv dgrad GEMM (call unchanged) + attention_norm RMSNorm bwd + add | 48 | 348.0 | 282.7 | 3.1 | dex t4 -> vada | e22b4f26e7fa68ed | verify/gemm_norm_bwd_t4_vada (t3 failed vanya) |
| gri3 (G33) | routed-experts w13 [16,1536,2048] fp32 -> `dist_moe.prepare_mxfp8_weight` | 48 | 80.31 | 37.30 | 2.1 | eve t3 -> vola | 1f34a5814fc27cff | verify/prep_weight_t3_vola; t1 verified by vik |
| adoe (G42) | same for w2 [16,2048,768] | 48 | 44.97 | 20.15 | 1.2 | eve t3 -> vola | 8165daeb5e2ee8a4 | verify/prep_weight_t3_vola; t1 verified by vik |
| kbxm (G41) | attention_norm RMSNorm + MXFP8 quant + rearrange | 48 | 76.53 | 52.09 | 1.2 | ada t6_final -> vin | 534045e564d5cebb | verify/norm_quant_vin |
| na2k (G45) | residual add + ffn_norm RMSNorm | 48 | 98.26 | 77.25 | 1.0 | ada t7_final -> vic | 95b15574b32ee529 | verify/add_norm_vic |
| vf5c (G28) | wo fwd: MXFP8 quant + rearrange, then the unchanged GEMM | 48 | 241.5 | 221.8 | 0.9 | eve t2 -> vita | 5cfcdb9d12354430 | verify/quant_gemm_vita |
| 5ifh (G47) | router softmax, top-k score gather, normalization | 48 | 25.23 | 7.95 | 0.8 | cyd t8 -> vlasta | 53dbbbfeeb3feb18 | verify/router_softmax_vlasta |

Candidate sources are under `autokernel/run1/candidates/<region>/<writer>/<trial>/`. Promoted copies are
`kernel_cache/pat_<key>*/<kernel_name>`. The sha256 prefixes above were taken from the files in `kernel_cache/`
and match the verified candidates. Replaced kernels are backed up in `autokernel/run1/rollback/`.

**What every verifier checked** (details in each evidence directory):
- **Contract:**
  - sha256 and the signature against the spec;
  - the official harness gate: seeds 0-7 plus real inputs, every case EXACT;
  - output metadata, storage offsets and alias structure against the reference;
  - inputs unmutated and repeated calls identical.
- **Memory:** allocator poisoning, proving every output byte is written, and compute-sanitizer
  (memcheck/initcheck/racecheck/synccheck) with planted-bug controls that the filters must flag.
- **CUDA graphs:** neighbour kernels writing the inputs and reading the outputs, many replays with inputs changed in
  place, two or more calls in one graph, and the first call inside capture.
- **Adversarial inputs:** each verifier's own suite with negative controls (deliberately broken copies that must
  fail). Reduction orders were checked directly with exactly representable cancelling terms where an output hides
  them (the RMSNorm dw reductions, column sums that only feed a scalar).
- **GEMM-adjacent kernels (G15, G28):** proof that the `_scaled_mm_v2` call is identical (same op, arguments,
  operand metadata and alignment) and gives bitwise-equal output.
- **PDL:** kernels using programmatic dependent launch were tested with an early-triggering predecessor that writes
  the inputs, and their SASS was checked for non-coherent (`.nc`) loads of data written during the call chain.

**Multi-device check (coordinator, `tools/multidev_check.py`, logs `tools/multidev_logs/`):** all 15 promoted
kernels are EXACT on cuda:1 when imported with cuda:0 current and then run on cuda:1, when imported after
`set_device(1)`, and when run on cuda:1 inputs while cuda:0 stays current.

**Behaviour outside the spec, by design or found by verifiers.** None of it is reached in this training run: the
specialization key fixes shape, stride and dtype, and the value cases below (invalid labels, out-of-range positions
or expert indices) do not occur with valid data.
- **G01 t30:**
  - asserts a 152-SM device at import;
  - strided weights or misaligned inputs raise a clear error or give the quack result; t5 used to fall back to the
    reference op sequence;
  - after importing with cuda:0 current, a first call on cuda:1 inside graph capture raises, because the static
    tables are uploaded from pageable memory. GraphTrainer runs eagerly before capture.
- **G07:** invalid labels (>= vocab, or < 0 other than -100) no longer device-assert.
- **G02:** out-of-range positions read outside the RoPE table instead of raising.
- **G47:** out-of-range expert indices wrap (`idx & 127`) where the reference asserts.
- **G-91:** a fast division path is used only inside a proven input range. The range check must stay exactly as it
  is; vilma verified the formula on 1.07e9 random and 3.2e6 hard-case pairs.
- **G-91:** non-contiguous or misaligned inputs fall back to `reference.py`.

**Dependency to know about:** the norm kernels (G01, G15, G19, G41, G45) reproduce the reduction order of the
vendored quack CuTe RMSNorm on sm_103 with 152 SMs (its CTA count is 16 x num_SMs = 2432). On this PyTorch build,
`torch/_native/ops/norm/` dispatches `aten._fused_rms_norm` and its backward there. A torch or quack upgrade,
`TORCH_DISABLE_NATIVE_JIT`, or another GPU model can change the reference's own numerics, and these kernels must
then be re-verified.

## 8. Plans and what was not promoted

- **Plans:**
  - `plan_iter1.txt`: G01, G02, G07, G06, G17, G19 (the six largest gaps among liftable regions).
  - `plan_iter2.txt`: those plus G15, G33, G42, G41, G45, G24, G28, G47.
  - `plan_iter3.txt`: plus G-90.
  - `plan_iter4.txt`: plus G-91.
- **Hand-built regions:**
  - **G-90 (loss forward):** the finder's G-09 spans all 8 cross-entropy chunks as one instance, which is unsafe to
    lift (chunk 1's log-softmax is consumed before chunk 8's logits exist). G-90 lifts each chunk separately
    (`tools/make_loss_fwd_region.py`; 8 sites, all safe).
  - **G-91 (aux loss forward):** found by re-running the finder on the iteration-4 profile (its G-09); the lift pass
    is that capture's generated pass, renamed.
- **Not promoted, or not attempted, and why:**
  - **G24 (pat t6rh, colwise MXFP8 quant before the wo wgrad GEMM, 2 specializations, 96 calls/step):** eve's t1 was
    exact, but the gain was about 2 us per call (torchao's quantize is near bandwidth). Stopped; the region still
    runs its reference ops (23.5 ms/step, mostly GEMM).
  - **Superseded, verified kernels:** G01 bix t5 (260.7 us) by ada t30 (184.5 us); G-90 eve t2 (719.6 us) by dex
    t3 (544.2 us); G17 t5 by the stateless t6; G33/G42 t1 by t3.
  - **Within noise, not submitted:** G02 t7 (ada), G47 t11 (cyd).
  - **G19 cluster-fused version** (colwise quant fused into the norm pass, dex): placement makes it issue-bound;
    dropped.
- **Roofline view:**
  - **Baseline** (finder analysis of the cuDNN capture, 942 ms GPU busy): memory-bound glue was 356.2 ms. Fusing
    35 non-overlapping regions, each at its fused roofline, gives 61.2 ms, plus 6.7 ms of glue in no region, so
    fusion could recover ~289 ms. Each op at bandwidth without fusion would only recover 118 ms.
  - **iter7** recovered 288.3 ms. Some kernels beat the "ideal", which assumes the calibrated 6.92 TB/s copy rate:
    G01 t30 moves 1.40 GB in 184.5 us, about 7.6 TB/s. Others fall short (G19, and the kernels with a GEMM
    inside), so the estimate is soft by about 10%.
  - **Remaining glue headroom** (iteration-6 re-analysis, adjusted for G01 t30 and G-91): ~12 ms. That is G19's
    kernel (11.2 ms measured vs 5.7 ms ideal), ~5.5 ms of scattered small ops (FSDP bucket packs, embedding
    backward, aux-loss pieces, casts), and ~1.6 ms in G01.
  - **The rest of the step** is DistMoE grouped GEMMs (304 ms), attention (157 ms), other GEMMs (52 ms), optimizer
    (34 ms) and other kernels outside the graph (32 ms).

## 9. Code changes

- **Skill: fusion_region_finder** (`torchtitan/experiments/graph_trainer/.claude/skills/fusion_region_finder/`),
  two small fixes, needed to re-analyze profiles of runs that already substitute kernels (a separate change from
  this directory):
  - **`lift_regions.py`, `op_schema`:** targets that are Python callables (substitute_kernels'
    `_substitute_kernels_pat_*` wrappers) have no op schema; it used to try `ns.op.overload` parsing and crash.
  - **`curate_regions.py`:** a region whose instance is empty once outside reads and writes stay outside gets a
    "not liftable" reason instead of crashing.
- **Run-local files** (in this directory):
  - `loop_recipe.py`: the FA4 pin (sec. 3); `KERNEL_CACHE` override; atomic real-input saves; `capture` keeps
    existing real-input files, so the inputs behind earlier kernel evidence stay the same.
  - `autokernel/harness.py`, matching training: deterministic with `fill_uninitialized_memory=False`; poisons
    candidate output memory; position-index int64 inputs; fp8 dtypes; no bytecode writes under `kernel_cache/`.
- **Untouched:** `substitute_kernels.py` and everything else in the repo.

## 10. Incidents and corrections

1. **cuDNN nondeterminism (sec. 3):** the first lifted run was not bitwise. The cause was the attention backward,
   not the lifting; the loop moved to FA4 and the baseline was re-run (`baseline_fa4/`).
2. **Promotion during a compile (fixed):** a kernel promoted while a training run was compiling was picked up by
   only some sites: G17 hit 24/48 sites in iter3, and the first iter5 attempt (stopped) mixed G06 hits and misses.
   Numerics stayed bitwise, since a miss runs the reference. From iter5 on, every run uses a frozen copy of the
   cache (`cache_snapshots/iterN`, via `KERNEL_CACHE`).
3. **Sum-of-durations understated gains (fixed):** PDL kernels overlap their predecessor, so summed durations count
   that overlap twice. The step metric is now the busy-time union; `results.json` records both.
4. **Bugs caught by verifiers (not promoted until fixed):**
   - **G06 t13 (vee):**
     - the softmax-backward sum lacked the `+0` seed, giving -0 where the reference gives +0;
     - a misaligned stride-0 input path was wrong.

     Fixed in t14 and re-verified by vala.
   - **G15 t3 (vanya) and G19 t5:** non-coherent (`ld.global.nc`) loads of partials written by an earlier kernel in
     the same PDL chain. Fixed with plain loads in G15 t4 (vada) and G19 t6 (vesna).
   - **G17 t5 (vince, passed with a caveat):** persistent per-stream state would be wrong if two graphs ran
     concurrently; replaced by the stateless and faster t6 (vojta).
   - **G01 t28 (writer-internal):** racecheck flagged a one-lane mbarrier arrive; t29/t30 use per-thread arrivals.
   - **G-91 timing (vilma):** the claimed 5.3 us held only with an L2-resident input; with rotating inputs it is
     7.0 us.
5. **Agent coordination:**
   - **Stalled agents:** several writers and verifiers ended their turn waiting for a background notification that
     never came. They were resumed by message; one verifier (vito) never started and was replaced (vanya).
   - **Board races:** one board update overwrote another agent's row (restored); board writes now always keep a
     trailing newline and no blank lines inside tables.
   - **`pkill`-style kill:** around 22:30 a writer (bix) killed compute-sanitizer processes by pattern, which ended
     one of vlasta's initcheck runs. vlasta re-ran it, and no verdict was affected. All agents now kill only PIDs
     they started.
   - **Lock deadlock:** a writer (eve) held `gpu0.lock` while waiting on it again, for about 10 minutes; fixed in its
     scripts.
   - **Stray bytecode:** agents importing references wrote `__pycache__` into `kernel_cache/`; removed, and the
     harness sets `sys.dont_write_bytecode`.
6. **Shared host:** other users' jobs ran on all four GPUs at times: a capture job on GPUs 0-3, and a large probe on
   GPU 2. They were never touched. Timings taken under contention were redone on an idle GPU, and verifiers checked
   GPU utilization before their final timings.

## 11. Open items and decisions for you

- **Done:** iter7 with all 15 kernels: -288.3 ms (-29.8%) vs the baseline on the same GPU, bitwise identical.
- **cuDNN vs FA4:** the bitwise gate needs FA4, which costs ~28 ms/step against cuDNN's nondeterministic backward.
  The kernels do not depend on the attention backend.
- **Beyond glue fusion, needs your decision:** what is left is DistMoE grouped GEMMs, attention, GEMMs and the
  optimizer (sec. 8). The remaining glue headroom (~12 ms) is mostly G19, which a verified-exact rewrite has not
  beaten.
- **Worth reporting to the substitute_kernels / fusion_region_finder authors (not posted anywhere):**
  - the two skill patches (sec. 9);
  - the cache contract has no way to say a kernel is only valid for one GPU model or SM count. G01 t30 asserts 152
    SMs, and the quack-order kernels assume it.
- **Not yet done:** a PR, or folding the loop scripts (`tools/`: TensorBoard comparison, busy time, multi-device
  check) into the skills.

## 12. How to audit

- **Re-check numerics of a run:** `python tools/tbcmp.py baseline_fa4/tb iter7/tb`.
- **Re-gate one kernel:**
  ```
  cd autokernel/run1
  CUDA_VISIBLE_DEVICES=<gpu> python harness.py <kernel_cache/pat_.../specialization_*.json> \
      <kernel_cache/pat_.../<kernel_name>.py> --seeds 0 1 2 3 4 --real ../../real_inputs/fusion_G<id>.pt --time
  ```
- **Verify cache contents:** `sha256sum kernel_cache/pat_*/a*.py` (excludes `reference.py`) against sec. 7.
- **Read the record:**
  - each verifier's directory under `autokernel/run1/verify/` has its checker scripts, negative controls, logs and
    the verdict summary;
  - `autokernel/board.md` has the full history: problem rows, handoffs, verifications, promotions.
- **Reproduce an iteration:** the command in "Run" with `plan_iterN.txt`. One iteration takes about 10 minutes on one
  GB300, plus a few minutes for the trace analysis.
