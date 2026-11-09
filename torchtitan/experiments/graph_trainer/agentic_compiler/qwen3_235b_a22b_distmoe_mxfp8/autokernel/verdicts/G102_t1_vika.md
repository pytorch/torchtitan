# Verdict: G102 (pat_xfyh.../azk7g7) bix t1 -- PASS

- Verifier: vika (independent; did not write or review the candidate)
- Candidate: `kernel.py` (copy of candidates/G102/bix/t1/kernel.py), sha256
  `910393a8b3c58d05a3551127d3400b5fed411217609dbc3a8c113b03f6642c96` (= handoff)
- Reference: pat_xfyh.../reference.py sha256 `0ff9f3f280c74d1b...` (copied to `reference_copy.py`; spec copied to
  `spec_copy.json`); no `<kernel_name>` file exists, so the candidate is compared with the reference.
- Hardware: NVIDIA GB300 (sm_103, 152 SMs), torch 2.16.0a0+git3c53c6f, nvidia-cutlass-dsl 4.8.0, compute-sanitizer
  2025.3.1. Checks before 16:58 ran on GPU 0 under gpu0.lock; after the coordinator's rule change, correctness and
  sanitizer runs used GPU 2 without a lock and timing used GPU 2 under gpu2.lock.
- All scripts and logs are in this directory. Commands run from here (`E`), with `W` = the experiment dir.

## 1. Contract -- PASS
`python contract.py` -> `contract.log`
- sha256 matches the handoff. A fresh process imports the file standalone from this dir; no relative imports;
  imports are only torch and cutlass.
- `def run(in0, in1, in2, in3, in4)`, which matches the spec signature.

## 2. Official gate and output contract -- PASS
`CUDA_VISIBLE_DEVICES=0 python W/autokernel/harness.py <spec> kernel.py --seeds 0 1 2 3 4 5 6 7 --real
W/real_inputs/fusion_G102.pt` -> `harness_gate.log`: seed0-7 EXACT, real EXACT, PASS. The harness poisons outputs
with 0xFF and 0x5A, reruns, and checks metadata and aliasing.

`contract.py` (seeds 11, 12, spec layout):
- **Values:** EXACT.
- **Metadata:** shape, stride, dtype and device of all 9 outputs equal the spec and the reference.
- **Storage:** storage offsets, storage sizes and alias structure equal the reference. out0/out1 are views of the
  in0/in1 storage at offset 0. out2..out8 are separate fresh storages of the reference's sizes.
- **Inputs and determinism:** inputs are bytewise unmutated, and 3 repeated calls are bitwise identical.
- **Note (informational):** reference out2/out4/out7 are autograd views (`_is_view()` True: a reshape of the clone
  and views of quack's rstd buffer). The candidate returns base tensors with the same storage, size, offset and
  strides. This is not observable through the contract.

## 3. Memory -- PASS
- **Allocator poisoning:**
  - The harness poisons with 0xFF and 0x5A.
  - `adv.py` runs every one of its 18 cases twice, once with the NaN fill from `fill_uninitialized_memory` and once
    with freed 0x5A bait blocks.
  - All EXACT. N18 (one rstd group unwritten) and N20 (last unit skipped) prove that unwritten bytes are caught.
- **Sanitizers:** run on the full-size spec layout, 2 calls plus a bitwise compare with the reference under the
  tool, with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (`run_san.sh`, `san.py`).
  - In `san.py`, the cos/sin row gaps (elements 128..255 of every 256-stride row) and the in0/in1 tails
    (128..255) are never initialized. A clean initcheck therefore also proves that the kernel reads only the bytes
    the op uses.

| tool | candidate | planted bug (copy in negctl_san/ or flag) | control result |
| --- | --- | --- | --- |
| memcheck (`--kernel-name kns=AttnGlueFwd --report-api-errors no`) | 0 errors, EXACT (`sanitizer/memcheck.log`) | M1: v stores shifted +512 B | 32 x "Invalid __global__ write of size 16 bytes ... out of bounds" |
| racecheck | 0 hazards (0 errors, 0 warnings), EXACT (`sanitizer/racecheck.log`) | R1: no `syncwarp` before the stage refill | 5 hazards (race between the write and the reads of the stage) |
| synccheck | 0 errors, EXACT (`sanitizer/synccheck.log`) | S1: `bar.warp.sync 0x0000FFFF` | 13440 "Barrier error detected. Invalid arguments" |
| initcheck (unfiltered, see below) | 0 errors (`sanitizer/initcheck.log`) | I1: `--uninit-cos`, a cos table never written | 32768 "Uninitialized __global__ memory read of size 512 bytes" in AttnGlueFwd |

- **Notes on the sanitizer runs:**
  - The first GPU 0 memcheck reported "34 errors" (`memcheck_with_api_errors.log`). Every printed record is a
    host-side `CUDA_ERROR_INVALID_VALUE` from `cuGetProcAddress_v2`: cuda-python probing symbols during
    `cute.compile`. None is a device error. With `--report-api-errors no` the result is 0.
  - initcheck must not use a kernel filter. Writes by uninstrumented kernels (randn, copy_) are then not tracked,
    so every input read looks uninitialized. Those runs are kept as `*_kernelfilter_invalid.log` and are not
    evidence.
  - R2 (no `syncwarp` between lane 0's smem write of the next unit id and the other lanes' read) is not flagged by
    racecheck: an intra-warp hazard in a converged warp. The tool does not model it. R1 is the racecheck control
    that is flagged.

## 4. CUDA graphs and the work-claim counter -- PASS
- **First call inside capture:** `python graphs.py kernel.py first` -> `graphs_first.log`.
  - In a fresh process, the very first `run()` is inside capture, after in-graph producers that write all 5
    inputs, and before a consumer that clones the outputs.
  - 6 replays with the producer source rewritten in place (4 input sets): EXACT, aliasing intact.
- **Main stress:** `python graphs.py kernel.py main` -> `graphs_main.log`, all EXACT:
  - **3 calls per graph** (A, B, A again), with producers writing all inputs and consumers reading the outputs:
    50 replays rotating 4 input sets. Weights, x and cos/sin all differ between sets.
  - **2 calls per graph on the same inputs**, inputs rewritten in place outside the graph: 12 replays.
  - **Eager:** 8 back-to-back calls with no sync between them.
  - **2 concurrent streams:** 24 iterations, with an 8192^3 GEMM on a third stream that delays CTA residency.
  - **2 graphs replayed concurrently** on 2 streams: 20 iterations.
  - **Forked-stream capture:** 2 calls run concurrently inside one graph, 10 replays.
- **Counter negative controls** (`negctl_counter_summary.log`):
  - C01, a module-level counter that is never re-zeroed: caught on every graph replay.
  - C02, a shared counter zeroed per call: safe on one stream, wrong across streams. Caught by the concurrent
    streams, concurrent graphs and forked-capture tests.
  - So the stress tests do exercise the counter. The candidate allocates a fresh `torch.zeros(1)` counter per call
    (a memset node in graphs).

## 5. Adversarial values -- PASS
`python adv.py kernel.py --ties` -> `adv.log`: 18/18 EXACT.

**Value cases:**
- **rand_wide:** x is 2^[-40,40) with random mantissas; weights 2^[-8,8); cos/sin 2^[-30,30).
- **specials_x:** 1/64 of x set to bf16 specials:
  - NaN payloads 7FC0/7FFF/7F81/7FA5/FFC0/FFC1/FF81, and +-inf, +-0;
  - subnormals 0001/007F/8001/8040, min normal, +-max;
  - and others.
- **rows_struct:** per-128-row patterns:
  - all-equal rows of 21 magnitudes (sumsq underflow, subnormal squares, 2^60..2^65, 3e38, +-0);
  - one-hot rows at every position;
  - k-hot rows of 2^62 crossing the sumsq overflow boundary;
  - half-zero rows, tiny rows with subnormal squares.
- **order_probe and order_probe2:** sums whose fp32 result depends on the order:
  - a two-level exponent mix;
  - chunk-structured rows with one 2^12 element plus +-1 elements at a random slot, where a sequential chain,
    a tree and a reversed chain give 2^24, 2^24+6 and 2^24+8 (big element in slot 0).
- **weights_special:**
  - NaN payloads, +-inf, +-0, subnormals, min normal, huge values;
  - in0/in1 elements 128..255 set to NaN, which must not be read;
  - w1[:64] = w0[64:].
- **weights_tiny_y:** w ~ 2^-140..2^-120 and 2^100..2^127, so y is fp32-subnormal or overflows.
- **cs_special:**
  - fp32 NaN payloads (7FC00001, FFC12345, sNaN 7F800001), +-inf, +-0, subnormals;
  - whole tokens with cos/sin = +-3e38, 1e-40, 1e-44.
- **cancel:** x[d] == x[d+64] and w[d] == w[d+64], with cos == +-sin, so the RoPE outputs cancel exactly.
- **ties:** rows with +-8192 pairs, so sumsq = 2^27 and rstd = 2^-10, with w = 1+2^-7. This gives bf16 midpoints
  for y, and cos in {1+2^-8, ...} with sin = 0 gives midpoints for the RoPE output.

**Tie coverage:** emulation counts the fp32 values that sit exactly on a bf16 midpoint. In the ties case there are
41.3M for y and 1.03M for q_rope; every other case has hundreds to 720k.

**Layout cases:**
- cos/sin row stride 128 (the real capture's layout), and the two halves of one [T,1,256] buffer;
- strides 256/384 with a 16 B storage offset, and stride 132 with odd size-1-dim strides;
- misaligned cos/sin (4 B offset), which go through the wrapper copy;
- broadcast cos/sin (row stride 0, out of spec);
- non-contiguous x (row stride 9232) with in0 misaligned and in1 the same tensor as in0;
- misaligned x, with cos the same tensor as sin.

**Negative controls** (`make_negctl.py` -> negctl/, `negctl_summary.log`): all 20 non-equivalent broken copies
FAIL, each caught by 4..18 of the 18 cases.

| control | change |
| --- | --- |
| N01 | in-thread chain order reversed |
| N02 | butterfly offsets 1,2,4 |
| N03 | chunk A+B add moved after the butterfly |
| N04 | x*(r*w) |
| N05 | RoPE fma-contracted |
| N06 | no bf16 rounding of y |
| N07 | cos/sin row of the neighbouring token |
| N08 | weights read from elements 128..255 |
| N09 | rstd via sqrt.rn+rcp.rn |
| N10 | sin sign |
| N11 | eps 1e-5 |
| N12 | output cvt.rz |
| N13 | q head slot swap |
| N14 | ftz in the norm multiplies (caught only by the subnormal/special cases) |
| N15 | ftz in the RoPE multiplies (caught only by the subnormal/special cases) |
| N16 | y cvt.rz |
| N17 | v copied from the k head |
| N18 | rstd of one group unwritten |
| N19 | k normalized with the q weight |
| N20 | last unit skipped |

N21 swaps the operands of a commutative add, which is provably equivalent: it stays EXACT, as expected.

**Reduction order:** the N=128 sumsq order is pinned by N01, N02 and N03. The reference dispatches
`aten._fused_rms_norm` to the vendored quack RMSNorm, whose N=128 config is `RmsNormFwdConfig(num_threads=128,
threads_per_row=16)` on this build. The candidate matches it on every order probe.

## 6. Special cases -- n/a / PASS
- **Not GEMM-adjacent:** no `_scaled_mm`.
- **No PDL:** there is no griddepcontrol and the launch has no PDL attribute. `ld.global.nc` is used only for the
  in0/in1 weight loads at kernel start. Those weights are written by earlier kernels, and the in-graph producer
  test rewrote them on every replay: EXACT.
- **Devices:** ran only on visible-index-0 devices (physical GPU 0, then GPU 2).
- **Device assumptions:**
  - The grid is the SM count of the current device at import.
  - Correctness does not depend on the grid size: static units 0..3*nwarps-1 plus atomic claims from 3*nwarps
    cover all 16384 units for any grid of up to 1365 SMs.
  - The counter is allocated on `in2.device`, and the launch uses the tvm-ffi env stream.

## 7. Timing -- candidate 74.9 us vs reference 1062.1 us per call
`run_timing.sh` waits for GPU 2 to have no compute processes, then holds gpu2.lock and runs `python timing.py
kernel.py 60`:
- One CUDA graph per implementation, with 8 calls rotating over 4 spec-layout input sets; all outputs stay live.
- Reference and candidate graph replays are interleaved: 60 each, CUDA events, per-call = replay / 8.
- Graph outputs are EXACT against each other.

`timing_run2.log` (GPU 2 had 0 other apps before and after):

| | median us/call | p10 - p90 |
| --- | ---: | --- |
| reference.py | 1062.1 | 1061.5 - 1062.8 |
| candidate t1 | 74.9 | 74.5 - 75.4 |

- **Speedup and saving:** 14.2x; (1062.1 - 74.9) x 94 calls/step = **92.8 ms/step** saved (isolated kernel).
- **Writer's numbers:** 1066.5 -> 74.6 us in the harness 1-call graph; consistent.
- **Discarded run:** `timing_run1.log` (reference median 2242 us, p10 1054) overlapped other verifiers'
  processes that started on GPU 2 during the run (5 apps afterwards).

## Findings for the writer
Nothing blocking. Informational only:
- The 3 reference outputs that are autograd views are returned as base tensors. Storage, offsets and strides are
  identical.
- Out-of-spec layouts go through wrapper copies: misaligned or non-contiguous x/in0/in1, and cos/sin rows that are
  not 16 B aligned. These are still exact.
