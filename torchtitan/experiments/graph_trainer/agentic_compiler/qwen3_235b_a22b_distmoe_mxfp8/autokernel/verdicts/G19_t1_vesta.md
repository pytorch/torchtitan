# G19 / fay t1 -- verifier vesta -- PASS

Candidate `candidates/G19/fay/t1/g19_t1.py`, copied to `cand.py`, sha256
`7cfee6ae11639b7044f11d86b51859279ac30717e0dd73ff025a11da751ef684` (matches the handoff).
Problem: pat_2nu2j6h3mmjcl7st3hqaz4gnvaeekauugf6vgivyz4ocb3uglmc / amxdvm3hplarcd3reirn2o4wreq53rb7okjyyzxice6kmgsessgp
(reference.py sha256 b0e4d6bc5487c54e..., copy in this dir). There is no current kernel_name file, so the comparison is
against reference.py. GPU 0 (GB300, sm_103, 152 SMs). torch 2.16.0a0+git3c53c6f, nvidia-cutlass-dsl 4.8.0.
W = outputs/agentic_compiler/qwen3_235b_a22b_distmoe_mxfp8. Scripts are in `scripts/`, logs in `logs/`.

## 0. Reference order, derived independently (I did not use fay_common/)

- Dispatch on this build: `aten._fused_rms_norm_backward` is overridden by `torch/_native/ops/norm/rmsnorm_impl.py`,
  which calls `norms.quack_rmsnorm_bwd` and then vendored quack `RMSNormBackward`. The device has cc (10, 3), which
  is not in `((9,0),(10,0))`, so `blocks = min(M, get_sm_count(4096)) = 2 x 152 = 304`. The config is
  `RmsNormBwdConfig(num_threads=128, threads_per_row=128, cluster_n=1, reload_wdy='smem', reload_x=None,
  use_tma=False, smem_stages=2)` (printed in `logs/ref_profile.log`). The profiler shows the RMSNormBackward kernel at
  grid 304 x 128 and `reduce_rows(..., 8, 256)` at grid 512 x 256.
- Sources read: `reduction_base._get_tiled_copy` gives thread t the cols 8t + 1024k + j. In `quack/reduce.row_reduce`,
  `TensorSSA.reduce` is followed by `cute.arch.warp_reduction`, a butterfly with offsets 16, 8, 4, 2, 1 (cutlass
  nvvm_wrappers.py:792). `block_reduce` loads lanes < 4 and fills the other lanes with init 0. `reduce_rows` is in
  `torch/_native/cutedsl/reduce.py`.
- SASS (`sass/ref/ref_bwd.sass`, my own dump of the cubin compiled in-process with QUACK_CACHE_ENABLED=0
  CUTE_DSL_KEEP=cubin):
  - w is passed through HADD2/HFMA2.BF16 with RZ, so w' = bf16(w + 0).
  - x_hat = FMUL2(x, rstd), wdy = FMUL2(d, w').
  - Each thread computes an FFMA chain from RZ in the order (k, j).
  - SHFL.BFLY 16..1, then STS, BAR, lanes <= 3 LDS (others hold +0), then SHFL.BFLY 16..1 again. The result is
    S = ((W0+W2)+(W1+W3)) with +0 added to each W.
  - mean = FMUL by 2^-12.
  - dx = F2FP(FMUL2(FFMA2(d, w', -FMUL2(x_hat, mean)), rstd)).
  - The dw partial is an FFMA2 chain fma(x_hat, d, acc) over the CTA's rows.
  - reduce_rows: sequential FADD from 0 over rows m, m+32, ..., then SHFL.BFLY 8, then 16, then sequential W0..W7,
    then F2FP.
- Empirical confirmation with my bit-exact numpy emulator (`scripts/emu.py`). It reproduces the reference g/dw
  bitwise in every case:
  - all 8192 rows of 3 row-order probes, 3 dw-order probes x 4096 columns, 2 dx-formula probes and 2 FMA probes;
  - 1024 full random rows (randn_101..103).
  - The alternative orders change many probe outputs (`logs/adv_full.log`, `adv_fma.log`):
    - 30B-style 256 thr/row: ~1410 rows;
    - chain j-major: ~650 rows;
    - cross-warp (W0+W1)+(W2+W3): ~1840 rows;
    - ascending butterfly: ~2900 rows;
    - N=2048 dx form fma(-x_hat, mean, wdy): ~7850 / 8192 rows of the dx-form probe (4 / 4M elements on randn);
    - dw with the 30B ownership B=608: ~1400 / 4096 columns;
    - reversed CTA row chain: ~730 columns;
    - reversed A_m: ~350 columns;
    - bfly (A0+A2)+(A1+A3): ~130 columns;
    - W tree: ~390 columns.
- This agrees with the writer's HANDOFF description of the order.

## 1. Contract -- PASS (`scripts/contract.py`, `logs/contract.log`)

- sha256 matches the handoff. `run(in0, in1, in2, in3, in4, in5)` matches the spec signature.
- A fresh `python -I` process in /tmp imports the evidence copy standalone. It does not load torchao, triton or quack.
- Output metadata, storage_offset (0), storage nbytes and contiguity match the reference and the spec for all 6
  outputs. Alias structure: none between outputs, none with inputs.
- Inputs are unmutated after 3 calls, and the 3 calls give identical outputs.
- Each call returns fresh storages.

## 2. Official gate -- PASS (`logs/harness_gate.log`)

`python W/autokernel/harness.py <spec> cand.py --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G19.pt`:
seed0-7 and real are all EXACT (the harness also reruns with allocator poisoning at 0xFF and 0x5A plus NaN-fill),
result PASS.

## 3. Memory -- PASS

- Poisoning: every adversarial case reruns the candidate with freed 0xA5 bait blocks and NaN-filled `torch.empty`,
  and compares against the first run. Controls m01 (g row 0 not written) and m02 (last 16 B of each 512 B scol
  block not written) are caught.
- compute-sanitizer 2025.4.1 (`scripts/san_driver.py`, `PYTORCH_NO_CUDA_MEMORY_CACHING=1`,
  `--kernel-name kns=kernel_cutlass_kernel_c` = K0/K1/KC; inputs are built on the CPU and copied H2D):

  | Tool | Candidate | Planted control |
  | --- | --- | --- |
  | memcheck `--padding 512` | 0 errors | `sanitizer/san_memcheck_oob.py` (dw store 16 B past the end): "Invalid __global__ write", exit 9 |
  | racecheck `--racecheck-report all` | 0 hazards (0 errors, 0 warnings); output EXACT under the tool | `san_racecheck_nobar.py` (KC barrier between smem tile writes and reads removed): "Potential RAW hazard" |
  | synccheck | 0 errors (EXACT) | `san_synccheck_divbar.py` (barrier under `tid < 16`): "Barrier error detected. Divergent thread(s) in warp", exit 9 |
  | initcheck | 0 errors | `san_initcheck_noreset.py` (K0 removed): "Uninitialized __global__ memory read of size 4", exit 9 |

  Without `--padding`, memcheck did not flag the 16-byte OOB control. With padding it did, and the candidate stays
  at 0. The earlier initcheck "cudaMemcpy source" reports were false positives on memory written by kernels the
  filter excluded. They went away once the inputs were built with H2D copies and the torch.equal checks were
  dropped from that run.

## 4. CUDA graphs / PDL / concurrency -- PASS (`scripts/graph_stress.py`, `logs/graph_stress.log`)

In total 1316 output sets were checked, with 0 failures:
- T1: the first-ever `cand.run` is inside capture. The graph is: PDL early-triggering writer kernels that write all
  6 inputs, then cand.run, then a PDL consumer (griddepcontrol.wait first) that reads all 6 outputs, plus torch
  copy_ readers. It was replayed 200 times with the sources rewritten in place over 4 rotating sets (randn, randn
  with negative rstd, real, NaN/inf rows).
- T2: 3 calls in one graph with in-place input rewrites between them (counter reuse), 100 replays.
- T3: 320 eager back-to-back calls.
- T4: 216 calls (eager and graph) racing an SM occupier on another stream. The occupier swept 38-608 CTAs, smem 0,
  100 KB and 200 KB, and spins of 20 us and 150 us.
- T5: 80 calls on 2 concurrent streams.

Sync negative controls (`logs/negctl_sync.log`):
- s01 (no counter wait): 1215 failures. s04 (no counter reset): 1231 failures.
- s02 (wait for 120 instead of 128): 18 failures (occupier). s03 (no griddepcontrol.wait before dw): 18 failures
  (occupier).
- s06 (row signaled before its own g stores): 4 failures, caught weakly.
- s05 (final CTA barrier removed): NOT caught, as with the 30B controls. Ordering there rests on the PTX argument:
  bar.sync -> fence.acq_rel.gpu -> red.relaxed.gpu on the K1 side, and ld.acquire.gpu -> bar.sync -> ld.global.cg
  on the KC side. I checked this in SASS: K1 has BAR.SYNC -> MEMBAR.ALL.GPU -> REDG.E.ADD.STRONG.GPU, both in the
  loop and at the end.

## 5. Adversarial suite -- PASS (`scripts/adv.py`, `scripts/probes.py`)

52 cases, all EXACT on all 6 outputs. Every case also checks metadata, that inputs are unmutated, and a poisoned
rerun (`logs/adv_full.log` 44, `adv_extra.log` 4, `adv_fma.log` 4).

General cases:
- randn x3, plus harness-style with negative rstd x2.
- Heavy-tailed 2^+-20 and 2^+-60 (partial NaN), overflow, per-row scale 2^+-30 with rstd perturbed by +-1 ulp.
- Specials (NaN payloads 0x7FC0/0x7F81/0xFFC1, +-inf, +-0, +-subnormal, max) in in0, in1, x, w and in5, both
  scattered and row-confined, and in rstd (NaN/inf/0/-0/subnormal/3e38/negative).
- Finite-only specials in w, and specials confined to 256 rows with finite w.
- Subnormal inputs and subnormal products.
- Signed zeros (w = -0, d = -0 from -0 + -0, in5 +-0, rstd +-0), and zero d with w <= -0.
- Add edges for d = in0 + in1: overflow, inf - inf, exact cancel, bf16 ties to even and to up.
- dx/g ties and cancellation, all-equal rows, one-hot rows, random bits (full range and finite).
- real, real with rstd +-2 ulp, real with in5 = 0 and w negated.

Order probes:
- Row-sum probes x3: an exactly representable +-2^25 / {1, 3} / -+2^25 triple per row, read back through a helper
  column whose dx = -S. Families: random cols, same thread chain, same warp, three warps.
- dw probes x3: the same triples per column. Families: random rows, same K1 CTA chain, same reduce_rows lane m,
  same warp group, different warp groups, and r/r+304/r+608/r+32 mixes that separate B=304 from B=608.
- dx-formula probes x2: every partial sum is exact, mean = 1 + k 2^-23, and x_hat * mean is inexact.
- FMA-contraction probes x2: +b / -b at two positions of one thread chain, and two rows of one CTA for dw. The fused
  chain leaves the exact rounding error: the reference dw is nonzero in 3848 / 3952 columns.

MXFP8 amax sweeps x4 (rows and cols): g = in5 exactly (d = 0, checked). Every positive finite bf16 amax pattern
(32640), +-inf and NaN blocks appear about 32 times in both orientations. That covers 249 distinct scale bytes and
about 11M elements on exact e4m3 midpoints per sweep (`logs/amax_stats.log`).

Numeric negative controls (`logs/negctl_numeric.log`, first catching case, run with --stop): 20 of 21 are caught.
- n01 B=608 (30B partial ownership), n02 30B dx form, n03 2^-11, n04 pair-major chain, n05 cross-warp 01/23,
  n06 ascending butterfly, n07 reversed A_m, n08 W tree, n09 bfly 02/13.
- n10 no w+0 (signed_zeros), n11 row sum without fma, n12 dw rz, n13 dw partial without fma.
- n14 scale floor, n15/n16 tie bias in qrow/qcol, n17 amax dropping NaN, n19 byte 255 -> inf recip, n20 srow
  swizzle off by one.
- n18 (fl(1/448) - 1 ulp) is not caught, and it is provably output-equivalent. amax * c lands within 2^-23 of a power
  of two only for amax = 7 * 2^j. There, c gives exactly 2^(j-6) and c' gives 2^(j-6)(1 - 2^-24), and both have the
  same ceil. In the subnormal range the rounding differs from a .5 boundary by at least 1/14.
- Probe-only rerun (`logs/negctl_probes.log`, row/dw/dxform probes only, no random data): n01 (about 2100 dw bytes
  per probe), n02, n04, n05, n06, n07, n08 and n09 are each caught by the targeted probes. n11 and n13 are caught by
  the FMA probes (`logs/negctl_fma.log`: about 16K g bytes and 3848 dw columns).

## 6. Special cases

- No GEMM in this region.
- PDL. K1 calls launch_dependents first (PREEXIT at 0x00c0). KC is launched with use_pdl and executes ACQBULK
  (griddepcontrol.wait) before all 20 dw-partial loads, which are plain LDG.E with no .nc. Every KC exit comes after
  ACQBULK. KC loads g with LDG.E.64.STRONG.GPU (ld.cg) and the counters with LDG.E.STRONG.GPU + CCTL.IVALL
  (acquire).
- K1's only .nc (LDG.CONSTANT) loads are w (4 x 128-bit) and rstd. Nothing in the call chain writes them, and K1 is
  a normal (non-PDL) launch, so the preceding writers have completed. The early-trigger writer tests in T1/T2 cover
  this (`sass/cand/*.sass`).
- Multi-device: tested on cuda:0 only.
  - `_B` and `_NSM` are taken from `torch.cuda.current_device()` at import. run() re-checks `_num_blocks(in0.device)`
    but does not check that all inputs share a device.
  - Kernels use the tvm-ffi env stream.
  - Import raises ValueError on any GPU other than sm_103.
- Input validation (`logs/misc.log`): misaligned bf16 inputs and wrong strides raise ValueError (the 30B kernel fell
  back to the reference instead). rstd at a 4-byte offset is EXACT. The reference is deterministic: two calls
  gave identical output.

## 7. Timing (`scripts/bench.py`, `logs/bench.log`)

CUDA-graph replays, reference and candidate interleaved (order alternating per round), 3 rotating input sets, live
outputs, 60 rounds, gpu0.lock held for the timed loop only, GPU idle before each run. 3 runs:

| Mode | reference us (p10-p90) | candidate us (p10-p90) | paired saving us/call |
| --- | --- | --- | --- |
| 1 call/graph | 144.4 / 144.4 / 144.5 (144.1-145.0) | 78.5 / 78.4 / 78.4 (78.0-78.9) | 66.0 / 66.1 / 66.1 |
| 3 calls/graph (per call) | 141.9 / 141.8 / 141.8 (141.5-142.2) | 75.2 / 75.1 / 75.2 (74.9-75.6) | 66.7 / 66.6 / 66.5 |

Speedup is 1.84x. Saving = (144.4 - 78.4) us x 94 calls/step = 6.2 ms/step (6.3 ms/step from the 3-call graph).

## Verdict: PASS

The candidate is bitwise exact against reference.py on everything I tested, including order probes of both
reductions at the N=4096 / M=8192 orders I derived from this build (B = 304 partial rows). Wrong orders, including
the 30B N=2048 ones, are caught. Notes for the coordinator (none blocks promotion):
1. The s05 race (final barrier) is not empirically observable; the protocol is unchanged from the 30B kernel and
   correct by the PTX argument.
2. The kernel reproduces quack's sm_103 / 152-SM schedule (B = 2 x SMs, reduce_rows(8, 256)). A torch/quack upgrade,
   a different GPU, or adding (10, 3) to the AOT table needs re-verification. With the current
   `backward_launch(4096, dw)` rule the AOT path would also give 304 blocks.
3. g is stored with L2 evict_last. The whole-workload L2 effect is not measured here.
