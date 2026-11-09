# G06 router + aux-loss backward -- cyd t1 -- verifier vala

**Verdict: PASS**

- Candidate: `candidates/G06/cyd/t1/router_bwd.py`, sha256 `37a2c59de0c9cc2789563c56e9fce88c361f24cd24b2179b63552e1f587d8cd7`.
  The handoff's hash matches; the copy under test is `cand.py`.
- Problem: `pat_llq372ctlfcwkfrs73so3hgzgo3w2npf2s6k76tsg6v5aa2sm75` / `specialization_ahb246mgbvjy2evvlmjhnnowroynp4moipjfl2wemj2k7q5idv75`.
  reference.py sha256 `bdf0af67e559dbb5...` matches the board.
- Timing (CUDA graph, 8 calls per graph, 8 rotating input sets at the spec layout, live outputs, idle GPU 2,
  two runs): reference 196.1 / 196.3 us/call, candidate 5.38 / 5.50 us/call. That is about 36x and saves
  **~17.9 ms/step** at 94 calls/step.

Environment: GB300 (sm_103), torch 2.16.0a0+git3c53c6f, nvidia-cutlass-dsl 4.8.0, compute-sanitizer 13.0.
The harness gate ran on GPU 0 under gpu0.lock, before the 16:58 GPU rule change. Everything else ran on GPU 2,
with timing under gpu2.lock. Scripts are in `scripts/`, logs in `logs/`, `sanitizer/` and `negctl/`. Commands
run from /tmp, e.g. `CUDA_VISIBLE_DEVICES=2 python -I scripts/suite.py cand.py`.

## 1. Contract

- **sha256:** matches the handoff.
- **Signature:** `run` is `def run(in0, ..., in12)`, identical to the spec (`scripts/contract.py`, `logs/contract.log`).
- **Standalone import:** a fresh `python -I` process imports it with only torch and cutlass.
- **Diff vs the verified 30B kernel** (Q/kernel_cache/pat_s3of.../ax4qsz6...py, sha256 6a848bf3): only
  `ROWS = 32768 -> 8192` plus two docstring lines ("235B-A22B, 8192 tokens", "2048 CTAs"). The 235B
  reference.py equals the 30B one with 32768 -> 8192 substituted. The writer's claim holds.
- **Row-count dependence**, reviewed line by line. ROWS feeds:
  - NGROUPS = 2048, so the grid is 2048 one-warp CTAs. 8192 % 4 == 0, so there is no tail group.
  - The Uint32 view of in5, whose 2048 words are exactly the 8192 mask bytes.
  - The output allocation [8192, 384].
  - The realigned column for a misaligned broadcast in7.

  Every reduction is per row (sum over 128 experts, sum over 8 slots, softmax backward over 128). There is no
  reduction across rows and no other row-dependent constant.
- **Note:** run() does not check input shapes. Another T would silently misbehave, but the specialization key
  fixes T.

## 2. Official gate and output contract

- **Harness:** `python harness.py <spec> cand.py --seeds 0 1 2 3 4 5 6 7 --real real_inputs/fusion_G06.pt --time`
  gives seed0-7 and real all EXACT, PASS (`logs/harness_gate.log`).
  - Harness timing (GPU 0, 1 call/graph): reference 198.3 us, candidate 9.9 us.
  - Harness eager: 1114.5 vs 74.2 us.
- **Real input layout:** `fusion_G06.pt` stores in7 contiguous (stride (128, 1)), not at the spec's (1, 0).
  So the harness "real" case runs the contiguous-in7 variant. My suite and the timing run the real values at the
  spec's broadcast layout as well.
- **Checks in every suite case** (`scripts/common.py`):
  - shape, stride, dtype, device and storage offset of each output: out0 at offset 0; out1/out2/out3 are views
    at offsets 0/128/256 with stride (1, 384);
  - alias structure (all four outputs share one storage) and storage size;
  - byte equality;
  - inputs unmutated (bytes and strides);
  - a repeated call is byte-identical.

## 3. Memory

**Poisoning** (`scripts/poison.py`, `logs/poison.log`): 60/60 EXACT.
- 5 cases x 3 in7 layouts x poison bytes 0xFF/0x5A/0x00/0x7F.
- In every run the candidate's output was verified to be the freed, poisoned block itself (same data_ptr).

**compute-sanitizer** (`sanitizer/`, `logs/sanitizer_summary.log`). Setup:
- Driver `scripts/san_driver2.py`: 9 calls covering broadcast, contiguous and misaligned in7, with fast-path,
  mixed slow/fast and duplicate-index groups.
- Options: `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, `--kernel-name kns=RouterBwd`, `--report-api-errors no`.

| tool | candidate | planted bug (copy in negctl/) | control result |
| --- | --- | --- | --- |
| memcheck --padding 256 | 0 errors | S1: output row index +4 (out-of-bounds write) | 32 errors, "Invalid __global__ write of size 8" |
| racecheck --racecheck-report all | 0 hazards (0 errors, 0 warnings) | S2: sync_warp before the P4 transposed reads removed | 10,000,016 RAW hazard warnings. Values stay 158/158 EXACT, so only racecheck catches this |
| synccheck | 0 errors | S3b: sync_warp(mask=0x0000FFFF) called by all 32 lanes | 65,536 "Barrier error detected. Invalid arguments" |
| initcheck | 0 errors | uninitialized in10 (torch.empty) | 589,824 "Uninitialized __global__ memory read" |

Notes:
- A first sanitizer pass (`sanitizer/v1_with_api_errors/`) counted 34 memcheck "errors". All of them are host
  `cuGetProcAddress_v2` CUDA_ERROR_INVALID_VALUE calls from the CuTe DSL compile path; none are kernel errors.
- My first synccheck plant (S3: sync_warp inside `if lane < 16`) is legal for bar.warp.sync on sm_70+, so
  synccheck stays silent. It shifts barrier pairing, though, and the value suite catches it (155/158 MISMATCH).
- initcheck may not track the cp.async.bulk loads of in6/in12/in7. Poisoning proves that every output byte is
  written.

## 4. CUDA graphs

**Graph test** (`scripts/graph_test.py`, `logs/graph_tests.log`). Each graph contains:
- in-graph producer kernels (bitwise copies) writing all 13 inputs, including the base column of the
  broadcast in7;
- 2 candidate calls with different input sets;
- consumer kernels reading all four output views.

It is replayed 30 times. Between replays the staged inputs are rewritten in place, and a matmul loop keeps a
side stream busy. Ten input sets are used: realistic, harness-like, x1e30, negzero, mixed slow/fast, specials,
duplicate order, softmax cancel, n7 boundaries and slow n7.

Results, 60 calls per run, all GRAPH PASS:

| in7 layout | warm-up first | first call of the process inside capture (compile during capture) |
| --- | --- | --- |
| bcast | PASS | PASS |
| contig | PASS | PASS |
| mis4 | not run | PASS |
| mis12 | PASS | not run |

Checker controls:
- A wrapper that returns its first outputs forever: GRAPH FAIL, 54/60.
- Mutant M04 (first call inside capture): GRAPH FAIL, 54/60.

**Fresh-process layout orders** (`scripts/layout_order.py`, `logs/layout_order.log`): 5 orders, 21 calls, all
EXACT. Orders include misaligned first (mis4, mis12, mis8 first), contiguous first and broadcast first. The
compile cache never holds more than 2 variants. Control M25 (realignment removed) raises a "Misaligned Tensor
data" ValueError.

## 5. Adversarial values

**Suite** (`scripts/suite.py`, `logs/suite_cand.log`): **158/158 EXACT**. Each case runs at the spec layout,
with contiguous in7, and for selected cases with misaligned (+4 B) broadcast in7. Cases:
- **Random data:** harness-like seeds 10-15; realistic seeds (softmax probabilities, L1 norms, top-k) x4 and
  x1e30; real.
- **Slow division path:** every operand scaled out of range (x6, n7, d10, a9, a0, s3, tiny and huge), plus
  mixed slow/fast rows.
- **Exact fast-path range boundaries** (2^-60/2^60, 2^-40/2^40, 2^-20/2^20, each +-1 ulp and +-sign), one
  operand per 4-row group, for x6, n7, d10, a9, a0 and w4 (inside, and just outside, which forces all groups
  slow). Also all operands at the range extremes at once.
- **Special values:** NaN, +-inf, +-0, subnormals, +-FLT_MAX, FLT_MIN, range bounds and 1e-12, injected into
  each float input separately and all together; s3 set to each special; whole special rows.
- **in8 threshold:** 0x2B8CBCCB / CC / CD, 1e-12, +-0, -1, NaN, +-inf, subnormal.
- **Masks:** in1 all/none, in5 none, both all, and all 16 in5 byte patterns within a 4-row group.
- **Negative-zero tmp rows:** all three constructions (masked row with out <= -0; v44 < 0 with out = +0;
  mixed per column) and all rows.
- **Indices:** duplicates at every slot pair, 8-way duplicates, 3-expert duplicates with order-sensitive values
  on both the slow and fast paths; edge indices 0 and 127; all slots 127; all slots 0.
- **Reduction-order probes:**
  - sum over 128 (v26): power-of-two terms, and exactly representable cancelling terms 2^24, 1, -2^24, 2^25, ...
  - softmax-backward sum: power-of-two terms, the same cancelling terms, and non-power-of-two values for FMA
    contraction.
  - sum over 8 (v39): power-of-two terms.

**Negative controls** (`scripts/make_mutants.py`, `negctl/*.log`, `logs/negctl_summary.log`): all 21 value
mutants are caught, as MISMATCH count out of 158 or an error:

| mutant | caught |
| --- | --- |
| M01 butterfly order 1,2,4 | 143 |
| M02 lane sum pairwise | 105 |
| M03 sum8 order 1,2,4 | 121 |
| M04 softmax sum reversed | 140 |
| M05 softmax sum unseeded (the 30B t13 bug) | 14 (negzero cases + special rows) |
| M06 mul+sub instead of fma in output | 130 |
| M07a threshold -1 ulp | 4 |
| M07b threshold >= | 4 |
| M08a lo cvt.rz | 15 |
| M08b hi cvt.rz | 143 |
| M09 always fast division | 22 |
| M11 duplicate scatter order reversed | 10 |
| M12 index guard drops 127 | 116 |
| M13 slow-path sgn by xorsign | 4 |
| M14 in5 byte order | 116 |
| M15 in8 row off by one | 84 |
| M18 v44 add association | 88 |
| M24 tree4 rows swapped | 142 |
| M25 no misaligned in7 realign | ValueError |
| M26 z = +0 instead of 0*v29 | 2 |
| S1 out-of-bounds write | device assert |
| S3 divergent sync | 155 |

Positive control P10 (always the exact div.rn path): 158/158 EXACT.

**Division fast path vs div.rn** (`scripts/divcheck.py`, `logs/divcheck.log`). This independent Triton check
uses the same PTX sequence, both scalar and packed f32x2 with distinct lanes. 0 mismatches on:
- 3.44e10 random pairs over the admitted ranges (dividends +-[2^-60, 2^60), divisors +-[2^-20, 2^20));
- 3.29e10 structured pairs (every divisor mantissa in 40 binades x 98 edge dividends);
- +-2^60 against every in-range divisor.

Just-outside controls (2.7e8 pairs each) give 0.9M to 35M mismatches, so the range check is needed and
discriminates.

## 6. Special cases

- **GEMM-adjacent:** not applicable.
- **PDL:** not used (no griddepcontrol or launch_dependents in the source).
- **Multi-device:**
  - Assumptions in the code: the compile cache key includes `in6.device.index`; the output is allocated on
    `in6.device`; the launch uses the TVM-FFI environment stream.
  - Coverage: only cuda:0 was tested (one visible GPU). The coordinator's 30B multi-device check covered the
    identical code path.

## 7. Timing

Command: `flock gpu2.lock python scripts/timing.py cand.py --n {8,16} --rounds 12 --reps 20`. Reference and
candidate graphs are interleaved, and each graph holds N calls on N rotating input sets at the spec layout (set 0 is
the real input with broadcast in7). Every output stays live. Logs: `logs/timing_run2.log`, taken with GPU 2
idle before each run.

| config | reference us/call median (p10-p90) | candidate us/call median (p10-p90) |
| --- | --- | --- |
| 8 calls/graph, run A | 196.08 (195.78-196.37) | 5.38 (5.30-5.52) |
| 8 calls/graph, run B | 196.26 (196.09-196.54) | 5.50 (5.39-5.68) |
| 16 calls/graph (sets > L2), run A | 194.44 (194.17-194.69) | 5.10 (5.04-5.17) |
| 16 calls/graph, run B | 222.02 (221.68-222.34) | 5.53 (5.44-5.59) |

- Saving: (196.2 - 5.45) us x 94 calls = **17.9 ms/step**.
- `logs/timing_run1.log` was taken while other verifiers' processes were running on GPU 2. Its 16-call
  reference reading (483 us) is contaminated; ignore it.
- The writer measured 217.4 vs 6.01 us. Same ratio regime.

## Findings for the writer (none block the verdict)

1. **adv06.py never ran the spec's broadcast layout.** It clones its inputs with `x.clone()`, which turns the
   stride-(1, 0) in7 into a contiguous tensor (`expand(...).clone().stride() == (4, 1)`). So its "base",
   "in7_misaligned", "negzero" and other cases ran the contiguous-in7 variant. The harness "real" case is also
   contiguous, because the saved real input has in7 at (128, 1). In the writer's evidence only the harness's
   random seeds exercised the broadcast path. My suite covers every case at the broadcast layout, including
   misaligned broadcast; all are EXACT.
2. **Out-of-contract behaviour, unchanged from the 30B verdicts.** Indices outside [0, 128) are skipped where
   the reference device-asserts. run() does not assert the input shapes (T is fixed by the specialization key).
