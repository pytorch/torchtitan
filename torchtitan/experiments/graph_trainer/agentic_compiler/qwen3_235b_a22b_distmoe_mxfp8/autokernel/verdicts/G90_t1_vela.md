# G90 dex t1 -- verifier vela -- verdict: PASS

- Candidate: W/autokernel/run1/candidates/G90/dex/t1/kernel.py, copied to cand.py,
  sha256 792a342e864fa2b19d64da8a0c9d1694628576c03f7beda1c9e3261aa0c4e2d5 (equals the handoff's).
- Problem: pat_bme7z4siizkq.../specialization_amyhw33k... (bf16 logits [1024, 151936] -> fp32 -> _log_softmax ->
  nll_loss_forward(sum, ignore -100) -> div by int64 in2). reference.py sha256 57b3bd0b255dc39d (= board).
  No kernel_name file exists in the pattern dir: compared against reference.py only.
- GPU 0 (GB300 sm_103, 152 SMs), torch 2.16.0a0+git3c53c6f, CuTe DSL 4.8.0, compute-sanitizer 2025.4.1 (CUDA 13.1).
- W = /home/bahuang/local/torchtitan/outputs/agentic_compiler/qwen3_235b_a22b_distmoe_mxfp8, E = this directory.
  Scripts in scripts/, logs in logs/, mutants in negctl/, sanitizer/, pdlctl/.

## What changed vs the verified 30B kernel (dex t3, vika)

`diff` against Q/kernel_cache/pat_bme7.../axoccr6q...py: M 4096 -> 1024, NLL_T 256 -> 64 (with an import-time assert
of torch's formula), the nll smem tree loop bound derived from NLL_T, docstrings, and the out-of-spec fallback to
the reference replaced by `ValueError`. Kernel 1 per-row code is byte-identical.
torch source at 3c53c6fe (aten/src/ATen/native/cuda/Loss.cu): `nll_loss_threads(1024)` =
clamp(1 << round(log2(1024 / 16)), 32, 1024) = 64; the profiler shows the reference launching
nll_loss_forward_reduce_cuda_kernel_2d with block [64] and cunn_SoftMaxForward<4, float, ...> with grid [1024],
block [1024] (logs/contract.log), i.e. the orders the candidate reproduces.

## 1. Contract -- PASS

- `sha256sum` (logs/sha256.log): candidate copy == handoff hash; reference.py == board hash.
- `python -I scripts/standalone.py cand.py` in a fresh process (logs/standalone.log): imports standalone (cutlass,
  torch, tvm_ffi only; no torchtitan).
- `python scripts/contract.py` (logs/contract.log): signature `run(in0, in1, in2)` == spec; output shapes, strides,
  dtypes, devices == spec; storage offsets 0 and storage sizes == reference; 3 distinct output storages, none
  aliasing an input (same alias structure as the reference); inputs unmutated; 5 repeated calls identical.
- `python scripts/layouts.py` (logs/layouts.log): out-of-spec inputs (misaligned in0 view, non-contiguous in0,
  fp32 in0, int32 / strided in1, CPU in2, wrong shape) raise ValueError (never return values).

## 2. Official gate -- PASS

`CUDA_VISIBLE_DEVICES=0 python W/autokernel/harness.py E/spec.json E/cand.py --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G90.pt`
(spec.json and reference.py are byte copies in E, so nothing runs from cache_dir): seed0-7 EXACT, real EXACT, PASS
(logs/harness_gate.log). Note: the harness's random in1 is arange(1024) and the real in1 only has 38 distinct
targets in [44, 121] with no -100, so targets/ignore were covered by my suite (sec. 5).

## 3. Memory -- PASS

- Poisoned outputs: harness 0xFF / 0x5A + NaN-fill (gate above); contract.py 0x00 / 0xFF / 0x5A / 0x7F freed bait
  blocks of every output size + `fill_uninitialized_memory=True`: all EXACT.
- `compute-sanitizer --tool <t> --report-api-errors no python scripts/san_run.py run cand.py` (host-generated
  inputs, targets in the first 4096-column chunk, whole vocab, V-1 and -100; output sha256 == reference):
  memcheck 0 errors (with and without PYTORCH_NO_CUDA_MEMORY_CACHING=1), initcheck 0 (both), racecheck
  0 hazards, synccheck 0 errors; every run `san_run result: EXACT` (sanitizer/*_cand.log). (A first memcheck run
  without `--report-api-errors no` listed 34 host-side CUDA_ERROR_INVALID_VALUE from cuGetProcAddress_v2 in
  cuda-python's driver binding at import, no device errors.)
- Tool controls (planted bugs, sanitizer/SC*.py, scripts/make_mutants.py): memcheck flags SC1 (tail store 384
  elements further: "Invalid __global__ write ... out of bounds" in candLogSoftmaxFwd); initcheck flags SC2
  (chunk 0 never stored: "Uninitialized __global__ memory read" in candNllFwd; run stopped after 100 reports);
  racecheck flags SC3 (no barrier between the smem partial stores and the tree: 258 RAW hazards); synccheck flags
  SC4b (bar.warp.sync with a mask that excludes the caller: 64 "Barrier error ... Invalid arguments"). SC4 (a
  bar.sync moved inside the `tid < stride` branch) is not flagged: legal on sm_70+ (the non-arriving warp exits),
  so it is not a usable control; SC4b replaces it.

## 4. CUDA graphs and streams -- PASS

`python scripts/graphs.py first cand.py` (logs/graph_first_in_capture.log): the candidate's very first launch in the
process is inside capture; neighbour copies write its inputs, neighbour copies read its 3 outputs; 9 replays over 3
input sets changed in place: 9/9 EXACT.
`python scripts/graphs.py warm cand.py` (logs/graph_warm.log, 48/48 EXACT): 2 calls per graph captured on a side
stream with distinct static inputs, 30 replays with in-place source changes (+ in-place logit negation and -100
edits); 2 calls on the same input buffers in one graph; eager on a side stream behind a slow GEMM producer that
writes the inputs; 12 back-to-back eager calls with live outputs; early-triggering writer tests (sec. 6).

## 5. Adversarial values and reduction orders -- PASS (124/124 EXACT, every order probe discriminating)

`python scripts/suite.py cand.py` (logs/suite_cand.log): 124 cases, all byte-exact on all 3 outputs (+ metadata,
alias, inputs unmutated, second call identical).
- nll order, probed directly (86 probes with in2 = 1 so v8 == v6; reference checked against an fp32 emulation of
  nll_loss_forward_reduce_cuda_kernel_2d with 64 threads: 86/86 equal). Designed rows have exact log-softmax at the
  target (x[i,0] = 0, x[i,col] = -D, rest -inf -> y = -D exactly); losses B = 2^32 and s = 256 (half an fp32 ulp of
  B) show whether two s rows meet before B. D1 {0,64: s; 128: B} (T64 chain joins rows 0,64 first, the 30B T256 /
  T128 orders do not), D2 (vs T32), D3 (chain direction), D4 (tree direction), D5 {0,256: s; 64: B} (the reverse
  case for T256), D6/D7 (far chain positions, last thread), D8 odd partial sums, D9 inf loss, D10 ignored row inside
  a chain, D11 each tree level stride 32..1, D12 each chain step 1..14; 48 random sparse absorption probes with up
  to 40 ignored rows; 6 dense order-sensitive inputs; T1 real logits with whole-vocab targets. Alternative orders
  differ from the reference on: T32 4, T128 20, T256 (30B) 19, T512 19, T1024 19, reversed chain 21, stride-1-first
  tree 20, contiguous rows per thread 41, sequential 31, numpy pairwise 41, fp64 19 probes.
- Targets: whole vocab, row-boundary columns (CTA slices, register/smem chunk 10/11, mbarrier groups, chunk 36,
  tail 151552..151935, V-1), all 0, all V-1, all ignored (in2 8192 -> 0, in2 0 -> 0/0 NaN, in2 -7 -> -0), a single
  counted row (row 0; row 1023 col V-1 = last output element).
- in2: 1, 0, -1, -7, 3, 7, 8192, 1024, 2^24+1, 2^24+3, 2^25+2, 2^25+6, 2^40, 2^53+1, 2^62+2^39+1, 2^63-1, -2^63.
- Views: in1 = slice of a larger tensor (offset 1024), in2 = in1[7] (aliasing view), in2 with storage offset 1,
  in0 = rows [2:1026] of a larger buffer with NaN neighbours.
- Special rows (S1-S4): 28 kinds x 26 boundary positions -- all 0 / 1 / bf16 max / -bf16 max, one-hot (incl. d
  overflow), +-0 tie and lone -0 maxima (L = 0), 7 NaN payloads (q/s, +-), +inf, scattered and all -inf, single
  finite in -inf, all NaN, all +inf, subnormal rows (with/without zeros), 2- and 3-way ties at boundaries, exact
  k-ties k = 1..V, random bf16 bits, per-element exponents 2^-120..2^120, absorption rows (0 among -17), max at the
  last element, x1e4 and x1e-3 scales; targets at the special position / argmax / random / 0 / V-1 / -100; S2 hides
  every non-finite target behind -100 (NaN rows only in ignored rows, emulation checked), S3 all ignored.
  W0-W5: randn x 10^U(-3..4.5), all zeros (sum = V), all bf16 max.
- Bad targets (V, -1, -101, 2^63-1; scripts/trap.py, logs/trap.log): the reference device-asserts, the candidate
  traps (unspecified launch failure); both fail the call.
- Negative controls (`python scripts/suite.py negctl/<m>.py --stop-first`, logs/negctl_summary.log): 20 of 21 caught.
  nll: N01 T256 (the 30B order; caught by D1), N02 T32 (D2), N03 T128 (D1), N04 reversed chain (D1), N05 stride-1-first
  tree (D4), N06 contiguous rows per thread (D1), N07 div via rcp.rn * (Dv 3), N08 cvt.rz.f32.s64 (Dv 2^24+3), N09 ignored
  rows in total_weight (D10), N10 ignored rows add their column-0 value (Y0). Kernel 1: N11 smem chunks before
  register chunks, N12 tail element first, N13 warp tree offsets ascending, N14 cross-warp tree ascending, N15
  x - (max + L), N16 lg2.approx log, N17 exp without the log2e_lo term (all Y0), N19 tail owner off by one (D1),
  N20b warp sums read before the mbarrier wait (D1), N20 no warp-sum wait (crash). Not caught: N18 (max without the
  -FLT_MAX floor) -- output-equivalent: it only changes an all -inf (or -inf + NaN) row, where both versions produce
  canonical NaN everywhere (reference: (-inf + FLT_MAX) - log(0) = -inf + inf; mutant: d = -inf + inf).

## 6. Special cases -- PASS

- Not GEMM-adjacent (no _scaled_mm).
- PDL: kernel 1 is a normal launch (no PDL attribute) that calls griddepcontrol.launch_dependents at entry; kernel 2
  is a PDL secondary. SASS (sass/*.sass, CUTE_DSL_KEEP=ptx,cubin): K1 64 regs, 0 local; its only non-coherent loads are
  11 `LDG.E.NA.64.CONSTANT` of in0 (an input never written in the call chain) plus bulk copies of in0. K2 42 regs, no
  `.CONSTANT` loads; the 16 target loads (in1, not written by K1; its producers complete before K1 starts) precede
  `ACQBULK` (griddepcontrol.wait); the 16 coherent `LDG.E` of K1's output and the in2 load follow it.
- Early-triggering predecessor (graphs.py, a CuTe kernel that executes griddepcontrol.launch_dependents at entry,
  sleeps, then writes in0/in1/in2), eager x6 and in a graph x8, three writer shapes (1216x256 / 152x128 / 16x256
  CTAs): candidate 14/14 EXACT in every shape (logs/graph_warm*.log). Controls: PDL1 (K1 launched with use_pdl=True)
  is caught 14/14 with the 16-CTA writer (logs/graph_warm_16ctawriter_PDL1.log; with full-GPU writers the SMs stay
  occupied and it is not exposed), PDL2 (K2 reads K1's output before griddepcontrol.wait) fails 46/48 tests.
- Multi-device: only cuda:0 tested (lease). The kernels are compiled at import for the current device (sm_103a);
  run() requires all inputs on one CUDA device but sets no device guard and launches on the TVM-FFI environment
  stream, so a call with inputs on another device than the current one is untested.

## 7. Timing (CUDA graph, interleaved, gpu0.lock held only around the timed loops)

`python scripts/bench.py cand.py 3 30` (logs/bench.log); 3 rotating input sets (real + 2 randn), 3 live outputs per
graph, reference and candidate replays interleaved over 30 rounds (order alternating); graph outputs bytewise equal.

| measurement | reference.py | candidate |
| --- | ---: | ---: |
| graph, 3 rotating calls, per call: median (p10-p90) | 554.6 us (554.0-555.4) | 143.7 us (143.4-144.4) |
| graph, 1 call (real input) | 559.6 us (557.8-561.9) | 148.0 us (146.4-148.4) |
| eager, rotating | 567.3 us (565.4-569.0) | 146.2 us (145.1-147.8) |

Speedup 3.86x; saving (554.6 - 143.7) us x 8 calls/step = 3.29 ms/step. (Other verifiers' unlocked correctness jobs
shared GPU 0; spreads stayed within 1%.)

## Findings for the writer

None blocking. Minor: the import-time assert duplicates torch's formula in a one-liner with `__import__('math')`;
the `use_tvm_ffi_env_stream` launch path has no device guard (see sec. 6).
