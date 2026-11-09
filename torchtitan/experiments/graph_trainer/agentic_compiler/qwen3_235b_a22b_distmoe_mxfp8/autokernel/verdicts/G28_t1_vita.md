# G28 t1 (eve) verification by vita: PASS

- Problem: pat_vf5ce5t5225w4kyike7yviltbinryn55fucpm6bgbn6rnfdgvwg / specialization azuccjslhqonqyruqx4odhkgwznipdkplstzopb3osj3sxm5n234
  (rowwise MXFP8 quantize of in0 [8192, 8192] bf16 + triton_mx_block_rearrange, then the wo `_scaled_mm_v2`, 94 calls/step)
- Candidate: candidates/G28/eve/t1/kernel.py, sha256 48db3dea35e0652eed2c3fc904bc09fa2bed3e1979bd37235a503e48ff464957
  (copied to `kernel.py` here; reference.py copied, sha256 8cd8c5ce66c1f2b9..., the same file as the 30B pattern dir)
- No `kernel_name` file exists for this specialization, so the comparison is against reference.py.
- GPU 0 (CUDA_VISIBLE_DEVICES=0). Correctness, sanitizer and graph runs used no lock (brief revised 16:58). Timing
  loops held gpu0.lock inside the process.
- Scripts: `scripts/`. Logs: `logs/`. Planted-bug copies: `negctl/`.

## 1. Contract: PASS
- The sha256 of the copied file equals the handoff's. Compared with the promoted 30B file (Q/kernel_cache/pat_vf5c.../a5ofsyjq...py,
  5cfcdb9d), only `R = 8192` and `C = 8192` changed (`diff`).
- `inspect.signature(run)` is `(in0, in1, in2)`, matching the spec's `def run(in0, in1, in2)`.
- A fresh `python -I` process imports the file standalone. It needs only torch and cutlass (`logs/import_standalone.log`).
- No fallback to the reference ops. The TorchDispatchMode log of `run()` shows only `aten.empty` x2, views and
  `aten._scaled_mm_v2` (`logs/gemm_identity.log`).

## 2. Official gate: PASS
`python W/autokernel/harness.py <spec> kernel.py --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G28.pt --time`
gives seed0-7 EXACT, real EXACT, PASS (`logs/harness.log`; harness graph timing: baseline 192.3 us, candidate 183.0 us).
Further checks in `scripts/adv.py` on every family:
- Output metadata matches: shape (8192, 4096), stride (4096, 1), bf16, cuda:0, storage offset 0.
- The output storage is fresh and does not alias any input.
- in0, in1 and in2 are unmutated.
- Repeated calls (3 poisoned reruns) are identical.

## 3. Memory: PASS
**Allocator poisoning** (`adv.py`):
- Before each `_quantize`, blocks of the q (64 MiB) and scale (2 MiB) sizes are filled with 0xFF, 0x7F and 0x5A, then
  freed. The script asserts that the candidate's q and s pointers land in those blocks. When the candidate clones an
  input, the clone's 128 MiB block is baited as well.
- The `run()` output is poisoned the same way with 0xFF and 0x5A.
- Every case is byte-equal to the reference.

**compute-sanitizer** (`--report-api-errors no`). A first try reported 34 host-side `cuGetProcAddress` API errors from
the cuda-python bindings and 0 device errors (`logs/sanitizer_try1_api_errors/`). Final runs are in
`scripts/run_sanitizers*.sh`, `logs/sanitizer/`, `logs/sanitizer_summary.log`:

| tool | ptr%32 == 0 (32 B loads) | ptr%32 == 16 (16 B loads) | planted control |
| --- | --- | --- | --- |
| memcheck (no-caching allocator) | 0 errors (also with the GEMM) | 0 errors | n10 (scale atoms shifted by 16 B): `Invalid __global__ write of size 16 bytes` |
| initcheck (no-caching allocator, H2D input, consumer kernels read every q/s byte) | 0 errors | 0 errors | n9 (even 16 B scale chunks never written): `Uninitialized __global__ memory read` in the consumer; never-written input: `Uninitialized __global__ memory read of size 32 bytes` in the candidate (both stopped after flagging) |
| racecheck (filter RowQuant) | 0 hazards | 0 hazards | n5 (barrier removed): `Race reported` (1044480 hazards) |
| synccheck (filter RowQuant) | 0 errors | 0 errors | n13 (bar.warp.sync mask excluding lanes 16-31): `Barrier error detected. Invalid arguments.` |

Synccheck controls that did not flag, recorded for completeness:
- n11: shfl.sync mask excludes the executing lanes.
- n14: CTA barrier reached by warps 0-7 only, which is legal on Volta+ when the other warps exit.
- n12: predicated-off lanes in a full-mask shfl. This one deadlocked and was killed.

Host guard: calling the 32 B-load function on a ptr%32 == 16 input raises the TVM-FFI alignment ValueError before any
launch.

## 4. CUDA graphs: PASS (`scripts/graphs.py`, `scripts/cold_gemm_only.py`, `logs/graphs.log`)
- **cold_quant.** The very first candidate call in the process (`_quantize`) runs inside capture. 12 replays with the
  input rewritten in place (randn, all-amax, specials, bits, wide exponents, conversion grid, dynamic range, real) are
  q and s EXACT against torchao.
- **full.** cuBLAS is warmed by reference.run only, so the candidate kernel is first launched inside capture. The
  graph, captured on a user stream:
  1. writers of in0, in1 and in2;
  2. run();
  3. a reader;
  4. a writer that rewrites in0;
  5. a second run();
  6. a reader;
  7. three `_quantize` calls with writers between them.

  24 replays rewrite the sources in place, using all families as in0 and real or random in1/in2. Both calls and all
  three quantizes are EXACT against eager reference.run / torchao in every replay. An eager run on a side stream (a
  writer, then run(), no sync) is also EXACT.
- **Cold first run() inside capture** raises `operation failed due to a previous error during capture` for the
  candidate, for reference.run, and for a bare cold `_scaled_mm_v2` on precomputed operands. The failure belongs to
  cuBLAS, not the candidate (as at 30B). GraphTrainer runs eagerly before capture.

## 5. Adversarial values: PASS, 61/61 EXACT (`scripts/adv.py`, `scripts/common.py`, `logs/adv_candidate.log`)
Each case compares two things byte for byte, including metadata:
- the intermediate q + blocked scales against torchao `mxfp8_quantize` (rowwise, e4m3, rceil) + `triton_mx_block_rearrange`;
- `run()` against `reference.run`.

Families:
- **amax_exhaustive.** Every one of the 65536 bf16 patterns (both signs, inf, every NaN payload) is the block amax at
  every one of the 32 positions (65536 x 32 = all 2097152 blocks). The other values are random bits <= amax. This
  covers 249 distinct e8m0 bytes, 0..247 and 255, including the rceil carry boundary (mantissa 96/97) at every exponent.
- **conv_exhaustive.** Block amax is 448 * 2^s. The pre-conversion values cover all 17377 bf16 magnitudes in
  [0, 0x43E0] (verified). That includes every e4m3 rounding midpoint +-1 bf16 ulp, the e4m3 subnormal range and the
  448 saturation edge, at s in [-8, 8] plus random s in [-126, 119]. 254 distinct q bytes.
- **bits, wide_exp, randn, real.** Random 16-bit patterns, per-block exponents from -140 to 128, randn, and the
  captured training input.
- **dyn_range.** Huge amax (exponent field 200..254) with tiny or subnormal others. This exercises x * recip underflow
  and the sign of zero.
- **special.** 24 block kinds:
  - NaN: all canonical, all negative NaN, random payloads at each position;
  - +-inf, inf together with NaN, alternating +-inf, inf with zeros;
  - zeros: +0, -0, mixed signs;
  - subnormal-only and one-hot subnormal blocks;
  - bf16 max +- (0x7F7F);
  - rceil boundary amax mantissa 96 and 97;
  - tiny amax (e8m0 clamps to byte 0);
  - byte 247;
  - all-negative, all-equal and one-hot blocks.
- **rows_equal, rows_onehot, zeros, neg_zeros.**

Offsets:
- **ptr%32 == 16** (the 16 B-load path) on amax_exhaustive, special, conv_exhaustive and bits is EXACT against the
  reference on the same view.
- **ptr%32 in {2, 4, 8, 24} and a non-contiguous view.** The reference raises RuntimeError
  (cuTensorMapEncodeTiled: TMA needs 16 B alignment). The candidate clones the input and is EXACT against the
  reference on an aligned contiguous copy. This is outside the spec.

Reference-side oracles, checked on every family:
- torchao's triton rearrange equals torchao's pure-PyTorch `to_blocked` and equals the explicit index formula
  `(r//128)*128*NCB + (c//4)*512 + (r%32)*16 + ((r%128)//32)*4 + c%4`.
- torchao's row-major scales equal an independent rceil emulation from the input bits:
  `255 if a >= 0x7f80 else max(((a+31)>>7)-8, 0)`.

Reduction order: the only reduction is a NaN-propagating max |x| over each 1x32 block. It is exactly order-independent;
its NaN and inf handling is probed at all 32 positions with all payloads. No output depends on the CTA shape.

Negative controls: 10/10 broken copies are caught by the suite (`logs/negctl_adv_*.log`):

| control | failing cases |
| --- | ---: |
| n1 rceil threshold off by one | 42 |
| n2 max without .NaN | 21 |
| n3 rearrange column off by one | 46 |
| n4 sticky bits dropped before cvt.rn (wrong rounding) | 48 |
| n5 barrier removed | 50 |
| n6 e8m0 clamp removed | 23 |
| n7 16 B halves swapped, 16 B path only | 6 (all of them ptr%32 == 16 cases, which shows that path is exercised) |
| n8 half-block amax | 46 |
| n9 scale chunks skipped | 52 |
| n10 scale OOB | 52 |

n11 (bad shfl mask) changes no values. It is only a synccheck candidate.

## 6. GEMM identity: PASS (`scripts/gemm_identity.py`, `logs/gemm_identity.log`, `scripts/layout_semantics.py`, `logs/layout_semantics.log`)
- **Dispatch log (real and randn in0).** Both implementations call `aten._scaled_mm_v2.default` with identical records:
  - non-tensor arguments `[3], [1], [3], [1], None, torch.bfloat16`;
  - for every tensor: shape, stride, dtype, device, storage offset (0), contiguity, storage nbytes, and pointer
    alignment (>= 2 MiB in these runs; both are fresh caching-allocator blocks of the same size);
  - the same output metadata.

  All four operands are byte-identical (q, in1, the A scales [8192, 256], in2).
- **cuBLAS kernel (profiler trace).** Both launch the same kernel with the same configuration:
  `nvjet_sm103_qqtst_128x256_128x6_2x1_2cta_v_bz_Avec32UE8M0_Bvec32UE8M0_TNT`, grid [1024, 1, 1], block [256, 1, 1],
  255 registers, 222592 B shared memory.
- **Output** is bitwise equal.
- **Blocked layout for the new shape, checked independently of torchao's triton kernel.**
  - The candidate's A scales, read as 64 x 64 atoms of 32x16 and un-blocked with the explicit formula, equal torchao's
    row-major scales.
  - in2's shape [32, 64, 32, 16] is [N/128, (K/32)/4, 32, 16] for N = 4096, K = 8192, the same convention.
  - The real in2 is constant (byte 111), which cannot expose a layout error, so this check uses a random in2. run()
    is still bitwise equal to the reference.
  - An fp32 dequantized GEMM using the formula-unblocked A and B scales reproduces cuBLAS's output to bf16 rounding
    (relative L2 error 1.66e-3).
  - Wrong layouts do not: A read row-major 0.70, B read row-major 1.28, transposed A atom 0.53.

## 7. Special cases
- **PDL:** not used. The launch has no PDL attribute and there is no griddepcontrol. The `.nc` loads read only in0,
  which nothing writes during the call. Writers immediately before the call in the same stream or graph are covered
  by section 4.
- **Multi-device:** tested on cuda:0 only (GPU lease). `_quantize` allocates on `in0.device` and launches on the
  TVM-FFI environment stream. The code has no device guard and no device-index assumption; the identical 30B code
  passed the coordinator's cuda:1 check.

## 8. Timing (`scripts/bench.py`, `logs/bench_ncall3_run{1,2}.log`, `logs/bench_run1.log`, `logs/bench_diag.log`)
Setup:
- CUDA graphs with 3 rotating input sets: distinct 128 MiB in0 buffers, distinct in1/in2, all outputs live.
- reference.py and the candidate are interleaved, with the order alternating each round.
- Capture and warm-up ran unlocked; gpu0.lock was held for the timed loop.

GPU 0 was shared with other verifiers' unlocked runs (100% utilization). With 12 calls per graph, the e2e replays
(about 2.3 ms) were preempted once per replay by time-slicing: the profiler shows one GEMM per replay at 2.78 ms
instead of 148 us. So e2e timing uses 3 calls per graph (about 0.55 ms per replay):

| run (3 calls/graph, 300 rounds) | reference us/call median (p10-p90) | candidate us/call median (p10-p90) | paired saving us/call median (p10-p90) | rounds won |
| --- | ---: | ---: | ---: | ---: |
| e2e, run 1 | 185.30 (183.96-186.49) | 175.73 (174.36-177.03) | 9.49 (7.89-11.34) | 300/300 |
| e2e, run 2 | 185.26 (184.35-186.40) | 175.79 (174.85-177.14) | 9.41 (7.94-10.91) | 300/300 |
| quant stage, run 1 (torchao quantize + rearrange vs `_quantize`) | 37.88 (37.66-38.47) | 30.84 (30.23-31.08) | 7.05 (6.63-8.20) | 300/300 |
| quant stage, run 2 | 37.86 | 30.81 | 7.11 (6.62-8.19) | 300/300 |
| quant stage, 12 calls/graph (unaffected by preemption) | 36.40 (36.22-36.55) | 29.74 (29.59-29.91) | 6.64 (6.46-6.87) | 100/100 |

- GEMM alone: 150.3 us.
- The e2e saving (9.4 us) exceeds the quantize-kernel saving (~7 us). The reference launches 3 kernels per call
  (quantize, swizzle, GEMM) against the candidate's 2, so it pays one more kernel boundary.

**Saving: 9.45 us/call x 94 calls/step = 0.89 ms/step (p10-p90 0.74-1.07). It is measurable well beyond noise: the
candidate wins all 600 rounds.**

## Verdict: PASS
Notes for the writer and coordinator (none blocks promotion):
1. A cold first run() inside graph capture fails, but identically for the reference (cuBLAS).
2. Misaligned or non-contiguous in0 is cloned where the reference raises. This is outside the spec.
3. The real in2 is constant, so the harness's real case alone would not detect a B-scale layout change. The candidate
   passes in2 through untouched, and the random-in2 run() check is EXACT.
