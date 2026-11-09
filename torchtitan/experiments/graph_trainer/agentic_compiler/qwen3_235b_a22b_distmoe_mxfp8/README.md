# Qwen3 235B-A22B DistMoE MXFP8: fusion loop seeded from the Qwen3 30B-A3B results

The fusion_region_finder -> substitute_kernels -> autokernel loop for Qwen3 235B-A22B, started from the regions and
kernels of the Qwen3 30B-A3B loop (`../qwen3_30b_a3b_distmoe_mxfp8/`). This is the experiment for item 2 of
`../TODO.md`. `reuse_report.md` covers what reuse changed; this file is the audit report of the run.

## This directory

| Path | Contents |
|---|---|
| `loop_recipe.py` | recipes: `baseline`/`lifted` (profiled step, fusion_finder pipeline), `capture` (cache problems and real inputs), `timed_baseline`/`timed` (20 unprofiled steps of the recipe as is) |
| `plans/` | `plan_iter1.txt`: the 13 30B regions that still match; `plan_iter2.txt`: the current plan, 20 regions |
| `fusion_regions/` | the lift passes of the plan: 13 copied unchanged from the 30B loop, 7 from this run's finder (G-1xx) |
| `regions.json` | the finder's 97 regions of the 235B baseline |
| `kernel_cache/` | substitute-kernels cache: reference and specialization per pattern, 19 promoted kernels |
| `autokernel/` | swarm board, harness, writer and verifier briefs, `promotions.log`, and each verifier's `verdicts/` |
| `results.json` | per run: loss, grad_norm, load-balance loss (TensorBoard), bitwise flag, GPU busy time or tokens/s |
| `timeline.log` | the coordinator's wall-clock log of every phase |
| `tools/` | `coverage.py` (which finder regions a plan covers), `busy_time.py`, `promote.py`, `try_as_is.sh` |

Generated lift passes, kernels and the harness are committed exactly as produced and run, not reformatted. Paths
under `outputs/agentic_compiler/qwen3_235b_a22b_distmoe_mxfp8/` in the board, briefs and verdicts refer to the
run's working directory (candidates, evidence, captured inputs); they are not committed.

## Run

```
S=torchtitan/experiments/graph_trainer/.claude/skills/fusion_region_finder/scripts
D=torchtitan/experiments/graph_trainer/agentic_compiler/qwen3_235b_a22b_distmoe_mxfp8
ENV="PYTHONPATH=$S:$D:. NGPU=256 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True"

# 20 steps of the recipe as is (default pipeline, CUDA graphs on), without and with the kernels:
env $ENV python -m torchtitan.train --module loop_recipe --config timed_baseline --comm-backend fake
env $ENV FUSION_PLAN=$(cat $D/plans/plan_iter2.txt) LOOP_ITER=timed_final \
    python -m torchtitan.train --module loop_recipe --config timed --comm-backend fake

# Profiled step (fusion_finder pipeline, CUDA graphs off), for GPU busy time and graph dumps:
env $ENV python -m torchtitan.train --module loop_recipe --config baseline --comm-backend fake
env $ENV FUSION_PLAN=$(cat $D/plans/plan_iter2.txt) LOOP_ITER=iter_final \
    python -m torchtitan.train --module loop_recipe --config lifted --comm-backend fake
```

Outputs go to `$D/<run>/` (gitignored). A cache miss runs the region's original ops, so numerics are the same
either way. Requires torchao, dist_moe and FA4 (`flash_attn.cute`).

# Audit report

## 1. Summary

| | Value |
|---|---|
| Wall clock, start to validated result | **2 h 41 min** (2026-11-06 15:46 -> 18:27) |
| tokens/s, 20 steps of the recipe as is (CUDA graphs on), GPU 3 | **7,698 -> 10,631 (+38.1%)**; step 1.064 -> 0.771 s |
| GPU busy time per step, profiled step, GPU 0 | 1082.2 -> 783.5 ms (-27.6%); excluding fake-PG collectives 1036.4 -> 737.7 ms (-28.8%) |
| Numerics | loss, grad_norm and load-balance loss bitwise identical to the baseline in every run, on both pipelines |
| Kernels | 19, each written by one agent and passed by a different, freshly spawned verifier; G24 not attempted |
| Verified per-kernel savings, summed | ~292 ms/step, matching the measured step saving |
| Agents | 6 writers, 19 verifiers; ~6.3M subagent tokens |

## 2. Goal and constraints

- **Goal (user, 2026-11-06):** redo the end-to-end loop for Qwen3 235B-A22B, reusing the Qwen3 30B-A3B kernel
  pack, and measure how long it takes.
- **Hard constraint:** model numerics must not change; a run passes only if loss and grad_norm are bitwise
  identical to the baseline.
- **Other rules:** nothing submitted to KernelGen; no other users' processes touched.

## 3. Setup

- **Recipe:** `graph_trainer_qwen3_235b_a22b_dist_moe_mxfp8_256gpu`, FSDP 256 x EP 32, 94 layers, 128 experts
  top-8, MXFP8 attention linears, Dist-MoE MXFP8 experts. One 8192-token microbatch per step.
- **Determinism:** FA4 varlen attention, because cuDNN's varlen backward is nondeterministic even with
  `debug.deterministic`. `debug.deterministic`, seed 42.
- **Hardware:** fake PG with NGPU=256 on one GB300 (152 SMs).
- **Stack:** torch 2.16.0a0+git3c53c6f, nvcc 13.0, nvidia-cutlass-dsl 4.8.0, torchao 0.19.0.dev20261103.
- **Roofline (`calibrate_roofline.py`):** HBM 6.93 TB/s, MXFP8 GEMM 3.09 PFLOP/s.

Shape changes from 30B-A3B:

| | 30B-A3B | 235B-A22B |
|---|---|---|
| Tokens per microbatch | 32768 | 8192 |
| Hidden size | 2048 | 4096 |
| q heads (kv heads 4, head_dim 128) | 32 | 64 |
| wqkv output width | 5120 | 9216 |
| Loss chunk (vocab 151936) | 4096 rows | 1024 rows |
| Local experts, expert sharding | 16, not FSDP-sharded | 4, FSDP-sharded 8 ways |

## 4. The loop

1. **Baseline capture and roofline** (15:49-15:54).
   - Baseline step: 1082 ms GPU busy, of which 392 ms is memory-bound glue.
2. **Replaying the 30B plan on the 235B graph dump.**
   - 13 of 16 lift passes lift their expected sites: 94 per layer region, 188 for G24, 8 for the loss regions.
   - G02 misses because the graph decomposes the k/v split as `reshape` where 30B had `clone` + `view`.
   - G33 and G42 miss because the expert weights now arrive from the FSDP all-gather (`cat` of uneven shards,
     bf16) instead of an fp32 `_to_copy` (R1 in `../TODO.md`).
3. **Finder on the 235B profile** (16:04-16:08): 97 regions. `tools/coverage.py` shows which of them the 30B plan
   already covers. The plan adds:
   - G-102, G-115 and G-129, replacing G02, G33 and G42;
   - four regions that only the 235B layout has:
     - G-105: reduce-scatter bucket pack of the expert gradients;
     - G-111: LM-head gradient padding;
     - G-125: attention/norm/router gradient bucket;
     - G-133: all-gather bucket prep.

   Together they cover every liftable region with a gap of 1.3 ms or more.
4. **Problem capture:** 20 problems with real inputs. G24 has no kernel in 30B and is not assigned.
5. **Writers** (6, from 16:16): each started from the corresponding 30B kernel, its board row and its verifier's
   checklist (`autokernel/WRITER_BRIEF.md`). 18 of 19 candidates were handed off within 26 minutes.
6. **Verification:** one fresh verifier per candidate (`autokernel/VERIFIER_BRIEF.md`). The coordinator promoted
   each PASS with `tools/promote.py` (temporary file + `os.replace`).
7. **Validation:**
   - iter1 (all lifted, no kernels): bitwise.
   - iter2 (7 kernels): bitwise.
   - timed_iter3 (17 kernels, recipe as is): bitwise, +19.6% tokens/s.
   - final runs (all 19 kernels, frozen cache snapshot).

## 5. Results by run

From `results.json`. Profiled runs use the fusion_finder pipeline on GPU 0. Timed runs are 20 steps of the
recipe as is on GPU 3.

| Run | Kernels | GPU busy ms (excl. fake comm) | tokens/s (step 20) | Bitwise |
|---|---|---|---:|---|
| baseline | none | 1082.2 (1036.4) | | reference |
| iter1 | none, 20 regions lifted | 1081.8 (1036.1) | | yes |
| iter2 | 7 | (GPU shared with verifiers) | | yes |
| iter_final | 19 | 783.5 (737.7) | | yes |
| timed_baseline_run1 / timed_baseline | none | | 7,698 / 7,698 | reference |
| timed_iter3 | 17 (not G01, G06) | | 9,204 | yes |
| timed_final | 19 | | 10,631 | yes |

The second timed baseline ran right after timed_final on the same GPU and reproduced the first.

timed_iter3 is a cross-check: the verified savings of its 17 kernels predict -172 ms per step, and the measured
change was -174 ms.

## 6. Kernels

Times are the verifiers' CUDA-graph measurements, interleaved with the reference. Saving = (reference - kernel) x
calls per step. Start point = the 30B kernel the writer adapted ("new" = no 30B counterpart).

| Region | Pattern | What | Calls | Ref us | Kernel us | Saving ms/step | Start point | Writer -> verifier |
|---|---|---|---:|---:|---:|---:|---|---|
| G01 | bjai | attention bwd glue: RoPE bwd, q/k RMSNorm bwd + dw, cat, MXFP8 row+col quant | 94 | 1196.9 | 112.7 | 101.9 | qnfe t30 | ada -> viggo |
| G-102 | xfyh | attention fwd glue: split, q/k RMSNorm, RoPE | 94 | 1062.1 | 74.9 | 92.8 | 4dzo t5 | bix -> vika |
| G-105 | r5yf | expert-grad reduce-scatter bucket pack | 94 | 384.1 | 46.5 | 31.7 | new | dex -> vera |
| G06 | llq3 | router + aux-loss bwd | 94 | 196.2 | 5.4 | 17.9 | s3of t14 | cyd -> vala |
| G19 | 2nu2 | add + RMSNorm bwd + add + MXFP8 row+col quant | 94 | 144.4 | 78.4 | 6.2 | zfby t6 | fay -> vesta |
| G-111 | 5dmq | LM-head grad padding | 8 | 1100.8 | 354.7 | 6.0 | new | dex -> vito |
| G17 | bp6b | router per-expert counts | 94 | 53.3 | 2.9 | 4.7 | bp6b t6 | cyd -> vince |
| G-115 | pqle | w13 `prepare_mxfp8_weight` (bf16, unshard permutation) | 94 | 68.8 | 23.7 | 4.2 | gri3 t3 | eve -> vesna |
| G-125 | 3h2b | attention/norm/router grads reduce-scatter bucket | 94 | 88.1 | 44.4 | 4.1 | new | dex -> vanya |
| G07 | 53aq | loss bwd per chunk [1024,151936] | 8 | 645.7 | 131.4 | 4.1 | 53aq t5 | dex -> vox |
| G15 | 4j6e | wqkv dgrad GEMM (unchanged) + RMSNorm bwd + add | 94 | ~270 | ~227 | 3.6-4.0 | tl5t t4 | fay -> vanda |
| G90 | bme7 | loss fwd per chunk | 8 | 554.6 | 143.7 | 3.3 | bme7 t3 | dex -> vela |
| G91 | nh2i | router aux-loss fwd | 94 | 31.6 | 4.4 | 2.6 | bj3b t9 | cyd -> vilma |
| G-129 | gbfx | w2 `prepare_mxfp8_weight` | 94 | 39.0 | 13.5 | 2.4 | adoe t3 | eve -> vola |
| G-133 | 4a7h | FSDP all-gather bucket prep | 94 | 17.0 | 1.6 | 1.5 | new | eve -> vada |
| G47 | 5ifh | router softmax + top-k gather | 94 | 17.8 | 3.7 | 1.3 | 5ifh t8 | cyd -> vlasta |
| G41 | xf3v | RMSNorm + MXFP8 quant | 94 | 42.3 | 28.7 | 1.3 | kbxm t6 | fay -> vin |
| G45 | pnge | residual add + RMSNorm | 94 | 53.4 | 40.0 | 1.3 | na2k t7 | fay -> vic |
| G28 | vf5c | MXFP8 quant + wo GEMM (unchanged) | 94 | 185.3 | 175.7 | 0.9 | vf5c t2 | eve -> vita |

**What every verifier checked** (details in `autokernel/verdicts/`):
- the harness gate on seeds 0-7 plus the real inputs;
- output metadata, storage offsets and aliasing; inputs unmutated; repeated calls identical;
- poisoned outputs;
- compute-sanitizer memcheck, racecheck, synccheck and initcheck, each with a planted-bug control;
- CUDA graphs: first call in capture, neighbour writers and readers, in-place replays, several calls per graph;
- an adversarial suite with direct reduction-order probes and 8-27 negative controls;
- interleaved timing.

Every order that changed with the new shapes was derived independently by its verifier, and the 30B order is
among the negative controls each verifier catches:
- the RMSNorm order at N=4096;
- G91's column sum (128 CTAs at T=8192);
- G90's nll reduction (64 threads for 1024 rows);
- G01's dw ownership.

## 7. Conditions and caveats

- **G01** needs its calls ordered on one stream: two calls in flight on two streams can hang, because the
  cross-CTA wait has no timeout. It also needs fewer than 134 of the 152 SMs held by kernels that wait on later
  work. One compute stream and a full-step CUDA graph satisfy both.
- **G01 and the RMSNorm kernels** (G15, G19, G41, G45) reproduce quack's launch schedule for sm_103 with 152 SMs on
  this torch build. Another build or GPU needs re-verification.
- **G-133:** the reference writes only rank 0's slot of a `torch.empty` bucket; the all-gather fills the rest.
  The harness therefore fails even a copy of the reference. The gate was:
  - slot bytes and the second output exact;
  - one identical allocation;
  - nothing written outside the slot (allocator-baited tail identical).

  Under the fake PG the unwritten tail reaches the model; all runs with G-133 are bitwise identical.
- **Inputs outside the contract** (never produced by the recipe):
  - misaligned or non-contiguous inputs raise `ValueError` in most kernels;
  - out-of-range expert indices are skipped (G06, G17) or wrap (G47) where the reference asserts.
- Kernels compile for the device that is current at import. Several assert a 152-SM GPU.

## 8. Incidents

1. **A single GPU 0 lock serialized the verifiers.** Verifiers held the lock through CuTe JIT compiles; at the
   worst point 23 commands were queued with GPU 0 at 0% utilization. The fix: no lock for correctness work,
   timing-only locks, and verifiers split over GPUs 0 and 2. About 30-40 min lost.
2. **The 20-agent session cap** delayed G01's verifier by 17 minutes, on the critical path.
3. **The 30B G07/G90 kernels "passed" on the 235B specs** because they fall back to a copy of the reference ops
   for other shapes; their timing equalled the reference. This run required kernels to raise instead.
4. **One timed run** lost its PYTHONPATH in a backgrounded subshell and was relaunched (coordinator error).

## 9. How to audit

- **Kernel files:** `autokernel/promotions.log` gives each promoted kernel's sha256. The files in `kernel_cache/`
  match it.
- **Reproducing a verdict:**

  ```
  python autokernel/harness.py kernel_cache/pat_<key>*/specialization_<l2>.json \
      kernel_cache/pat_<key>*/<kernel_name> --seeds 0 1 2 3 4 --time
  ```

- **Numerics:** compare the TensorBoard scalars in `results.json` with a rerun of the commands under Run.
