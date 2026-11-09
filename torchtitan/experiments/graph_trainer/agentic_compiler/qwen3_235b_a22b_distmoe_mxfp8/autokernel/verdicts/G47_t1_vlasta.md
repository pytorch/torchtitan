# Verdict: G47 cyd t1 (router softmax + top-k gather + normalization, 235B) -- PASS

- Candidate: `candidates/G47/cyd/t1/router_softmax.py`, sha256 `cf429c74994ea35e2c0c65e2ca4142a04e845283b4b21578a1d931b46259f984`
  (equals the handoff's; copied to `cand.py` here; all runs use the copy).
- Contract: `kernel_cache/pat_5ifhra3qgpmtsp3al5ub453lyxraxj74ripto26nbr23wvwplzt/specialization_apwkvc4t...json` +
  `reference.py` (sha256 prefix 3def67cfc99109f5, matches the board; copied to `spec_copy.json`, `reference.py`).
- Diff vs the verified 30B kernel (53dbbbfe): only `ROWS 32768 -> 8192`, `NGW 2 -> 1` and docstrings (checked with diff).
- GPU 0 (CUDA_VISIBLE_DEVICES=0). Correctness runs without lock (brief rev. 16:58); timing under gpu0.lock (fcntl,
  only around the timed loops).
- Verifier: vlasta. All scripts in `scripts/`, logs in `logs/`, `sanitizer/`, `negctl/`.

## 1. Contract -- PASS

`python scripts/contract.py` (logs/contract.log), fresh process, imports only torch + cutlass:
signature `def run(in0, in1)` == spec; real input metadata == spec (contiguous, idx in [0, 127], distinct per row);
on real and random inputs: all 5 outputs byte-exact, shape/stride/dtype/device == spec and reference, storage offsets
all 0, alias groups `[0, 0, 2, 3, 4]` (out1 is a view of out0) == reference, no output aliases an input, inputs not
mutated, repeated calls (with and without the torch.empty NaN fill) identical.

## 2. Official gate -- PASS

`python W/autokernel/harness.py spec_copy.json cand.py --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G47.pt`
(logs/harness_gate.log): seed0-7 EXACT, real EXACT, PASS.

## 3. Memory -- PASS

- Poisoning (`scripts/poison.py`, logs/poison.log): output blocks pre-filled with 0x00 / 0x7F / 0xFF / 0x5A (fill
  kept: NaN fill off) and once with the torch.empty NaN fill; every output pointer verified inside a poisoned block;
  real input and a random x20 input with -inf columns, all -inf rows and all-zero rows: all EXACT. Grid: 2048 one-warp
  CTAs x 4 rows = 8192 rows, no tail; v2 = one 2 KB bulk store per CTA, v4/v8 one slot per lane, v6 lane t == 0.
- compute-sanitizer, `--kernel-name kns=RouterFwd`, `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (per-tensor allocations),
  driver `scripts/sanit_driver.py` (H2D inputs, real + random x30 inputs incl. -inf, 2 calls each):
  - memcheck `--report-api-errors no`: 0 errors (sanitizer/clean_memcheck_noapi.log). Without that flag it shows 34
    host API errors, all `cuGetProcAddress_v2` CUDA_ERROR_INVALID_VALUE from the CuTe DSL JIT path
    (`get_compute_capability_major_minor`), not kernel accesses (sanitizer/clean_memcheck.log).
  - racecheck: 0 hazards (0 errors, 0 warnings); synccheck: 0 errors.
  - initcheck: 0 errors with v4/v6/v8 copied D2H (sanitizer/clean_initcheck_nov2host.log). The D2H of v2 is reported
    as uninitialized (exactly the 4,194,304-byte v2 allocation): v2 is written by cp.async.bulk, which initcheck does
    not track (as found in the 30B run); every v2 byte is proven written by the poisoning gate above.
  - Planted bugs (sanitizer/plant_*.py, scripts/run_sanitizer.sh, sanitizer/summary.txt): far OOB store ->
    memcheck "Invalid __global__ write" (1 error); no syncwarp before the smem gather -> racecheck 4 hazards (~57k
    each, gather read vs v2 smem writes); `sync_warp(mask=0xFFFF)` -> synccheck 65536 "Barrier error ... Invalid
    arguments"; uninitialized in0 -> initcheck 524288 "Uninitialized __global__ memory read". The filter and tools see
    this kernel.
- Code read: each lane fences (`fence.proxy.async.shared::cta`) before the warp barrier, lane 0 issues the bulk
  store, lane 0 waits `cp.async.bulk.wait_group.read 0` before exit; the smem slot is never rewritten (NGW = 1).

## 4. CUDA graphs and streams -- PASS

`python scripts/graph_test.py` (logs/graph_test.log), fresh process:
- the very first candidate call (JIT compile) inside capture; replay + 4 replays with inputs rewritten in place: EXACT;
- one graph with neighbours writing in0/in1 (copy_), 3 candidate calls (call 2's in0 = 40 * call 1's v2 - 20, i.e. it
  reads a candidate output; call 3 reuses in0), a reduction reading outputs of all 3 calls, and clones of all 15
  outputs; 6 replays with new inputs (scales 1..46) EXACT vs eager reference (live outputs, clones, reduction);
  20 more replays bitwise identical;
- side stream with a ~0.1 s sleeping producer writing in0, candidate and consumer on that stream: EXACT (the kernel
  launches on the current stream).

## 5. Adversarial values, reduction orders, exactness sweeps -- PASS

- `scripts/adv_suite.py` (logs/adv_suite.log): 305/305 cases byte-exact on all 5 outputs + metadata/alias, inputs
  unmutated: randn x1e-3..x1e30; real x0.5..x300 and real with random idx; 27 special row types (NaN payloads incl.
  sNaN/negative at col 0/127/random, +inf, +-inf, -inf half rows, all -inf, all -inf but one, +0/-0/mixed rows,
  positive and mixed-sign denormals, constant +-FLT_MAX, FLT_MAX with -FLT_MAX, one-hot vs -1e30/-100/-41, duplicate
  max x2..x128, 1e-38 constant, +-1e-30, randn x1e38, d in [-110, -80]) with random idx and idx on the special
  columns; fast/slow boundary min - max = -41 + k ulp (k = -3..3) for max 0 / 3 / -7 with all 16 fast/slow mixes in a
  4-row vote group (+ idx on the min/max columns); below-gate dividends (d in [-100|-87|-60, -41) with a non-trivial
  row sum); idx all 0, all 127, row-constant, 0/127 alternating, random with duplicates, real idx; tiny gathered
  probabilities (v4 = 0 -> v6 = 1e-20, denormal v4, d in [-50, -40] and [-49, -43] so v5 ~ eps); out-of-spec layouts
  (transposed in0, in0 offset 1/2/3 floats, strided in1, in1 offset 1): values exact.
- Reduction-order probes (exactly representable, order-revealing):
  - softmax row sum: every triple (i: e = 1, j < k: e = exp(-17) ~ 2^-24.5, rest e = 0 (slow path) or exp(-41)
    (fast path)); S = 1 or 1 + 2^-23 depending on which pair is added first, read back as v2[i] = 1/S: all
    1,040,384 triples EXACT on both paths; the order changes S in 341,376 of them (rooted triples determine the
    whole tree: lane sums over it = 0..3, then xor 16, 8, 4, 2, 1).
  - gather sum over 8 slots: slot i -> v2 = 1, slots j < k -> exp(-17), others 0 or exp(-41): all 168 (i, j, k)
    configurations EXACT on both paths; v6 takes both 1.0 and 1 + 2^-23, so the probe is sensitive.
- Exhaustive exp sweep (`scripts/exp_sweep.py`, logs/exp_sweep.log, 2074 calls): every sign-bit-set fp32 (-0 ..
  -inf, all negative denormals) as d = x - max (max +0 at col row % 128, 127 consecutive patterns per row; this
  includes every d in [-41, 0] on the fast path, 8,745,120 fast rows) and every NaN payload (16,777,214):
  end to end 0 mismatching calls; instrumented copy (instr/cand_e.py, v2 <- e) e == torch.exp(d) on every element;
  reference softmax's exp == torch.exp on all 8,170,659 rows with S == 1.
- NaN in fast-path rows (`scripts/nan_fastpath.py`, logs/nan_fastpath.log): all 16,777,214 NaN payloads, one per
  otherwise fast row (all 2048 groups fast): 0 mismatches.
- Division fast path (`scripts/div_unit.py`, logs/div_unit.log): the candidate's own frcp_refined + p_mul + 2 p_fma
  vs div.rn: every fp32 a in [2^-60, 1] (503,316,481) x 116 divisors in [1, 128] (boundaries + log-uniform) =
  5.84e10 pairs, 0 mismatches; 4.29e9 random pairs, 0 mismatches. Control: a in [2^-126, 2^-60) is wrong in 0.257%,
  so the gate is needed; min - max >= -41 gives e >= expf(-41) = 1.56e-18 > 2^-60.
- Negative controls (`scripts/run_negctl.sh`, negctl/summary.txt, *.adv*.log): all 15 broken copies FAIL --
  n01 lane-sum order, n02 butterfly s-bit order, n03 butterfly t-bit order, n04 exp without log2e_lo, n05 log2e_hi
  +1 ulp, n06 gather-sum order, n07 eps -1 ulp, n08 always-fast division (5 cases), n09 gate threshold -100 (3 cases),
  n10 grid - 1 (all cases), n11 row off-by-one, n12 idx mask 63, n13 slow path as rcp-multiply, n14 exp fma.rm -> rn,
  n15 gather from the wrong smem row. Positive control: always-slow division passes everything (as it should).

## 6. Special cases

Not GEMM-adjacent, no PDL (no griddepcontrol). Multi-device: run on cuda:0 only. Device-dependent code: the compiled
function is cached per `in0.device.index`, outputs are allocated on `in0.device`, the launch uses the TVM-FFI env
(current) stream; same launch code as the 30B kernel, which passed the coordinator's multi-device check.

## 7. Timing -- candidate 4.9x faster

`python scripts/timing.py` (logs/timing2.log, timing3.log; timing1.log ran during my own background jobs and agrees):
32 calls per graph on 32 rotating input sets (real input, row-permuted: every row is fast path, as in the real
input), all outputs live (288 MiB per replay, more than L2), reference and candidate graphs interleaved, 30 rounds.
Other verifiers' correctness jobs were on GPU 0 (they take no lock); medians were stable across the three runs.

| | reference us/call (p10-p90) | candidate us/call (p10-p90) |
|---|---|---|
| 32-call rotating graph, run 2 | 17.79 (17.76-17.85) | 3.63 (3.59-3.72) |
| 32-call rotating graph, run 3 | 17.76 (17.74-17.79) | 3.66 (3.64-3.68) |
| 1-call graph (L2-hot), run 3 | 18.68 (18.66-18.73) | 6.21 (6.17-6.28) |

No `<kernel_name>` file exists in the pattern dir. Saving: (17.78 - 3.65) us x 94 calls/step = **1.33 ms/step**.
The writer's 8-call graph numbers (16.75 -> 3.31) are consistent; my larger rotating working set adds ~0.3 us.

## Notes (not blocking; same as the 30B kernel)

- Out-of-range or negative indices wrap (`idx & 127`); the reference device-asserts. Not reachable with router top-k.
- `run` hard-codes ROWS = 8192 and does not check shapes; the specialization key fixes the shape.
- v2 is a bulk (async-proxy) store: initcheck cannot see it; poisoning covers it.
