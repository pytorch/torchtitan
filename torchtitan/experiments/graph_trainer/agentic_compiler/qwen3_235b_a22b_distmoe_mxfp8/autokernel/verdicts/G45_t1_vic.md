# G45 / fay t1 -- verifier vic -- verdict: PASS

Problem: pat_pngey6czyzs7m6vmf2f5zl7hkxxu6xfsjw7ttvwqrhti6c6jfsq / specialization aet66m7n... (residual add +
ffn_norm RMSNorm forward, N=4096, 94 calls/step). Reference: `reference.py` (sha256 b6640baf661c8085...), no
current `<kernel_name>` file. Candidate: `g45_t1.py`, sha256
`7d582eed3ce2b087c0d70071db060e77041f77af6a0527b7bea88427ad432349` (matches HANDOFF.md; copy in this dir).
GPU 0 (GB300, sm_103, 152 SMs), torch 2.16.0a0+git3c53c6f, nvidia-cutlass-dsl 4.8.0.

All scripts are in `scripts/`, logs in `logs/`, SASS/PTX in `sass/`. Commands are run from `scripts/` with
`CUDA_VISIBLE_DEVICES=0 python -I <script>`; only `bench.py` takes `gpu0.lock` (in-process, around the timed loop).

## 1. Contract -- PASS

- sha256 equals the handoff (`sha256.log`). `run(in0, in1, in2)` == spec signature. A fresh `python -I` process
  imports the copied file standalone (`sass/cand` dump run). Reference copied to `reference_copy.py`, same hash.
- `contract_meta.py` (`logs/contract_meta.log`): outputs tuple[3]; shape/stride/dtype/device == spec;
  storage_offset 0 and storage nbytes == reference (64 MiB, 64 MiB, 32 KiB); 3 distinct storages, no byte overlap
  with each other or with the inputs (same alias structure as the reference); requires_grad False; inputs bitwise
  unmutated; 5 repeated calls identical; side stream with a busy queue EXACT; `in0 is in1` EXACT.
- Off-spec inputs (misaligned, non-contiguous, wrong dtype/shape) raise ValueError. Not reachable in this run:
  in the baseline graph all 189 N=4096 `_fused_rms_norm` calls take x from an aten add and w = `view(empty_*)`
  (fresh, 512 B aligned allocations). Note: the reference itself would clone/fall back for such inputs; the 30B
  kernel cloned, t1 raises (loud, not silent).

## 2. Official gate -- PASS

`python W/autokernel/harness.py <spec> g45_t1.py --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G45.pt`
-> seed0..seed7 EXACT, real EXACT, PASS (`logs/harness.log`; harness includes two poisoned-allocator reruns per case).

## 3. Memory -- PASS

- Allocator poisoning: harness (0xFF, 0x5A) + `contract_meta.py` (0x00, 0xFF, 0x5A, 0x7F, 0x80, with NaN-filled
  torch.empty) + every adv case rerun with 0xA5: all EXACT -> every output byte is written.
- compute-sanitizer (`scripts/run_san.sh`, `run_initcheck_unfiltered.sh`, `logs/san/`, `logs/san_summary.log`),
  2 candidate calls on random inputs, PYTORCH_NO_CUDA_MEMORY_CACHING=1:
  - memcheck 0 errors; racecheck 0 hazards; synccheck 0 errors; initcheck (unfiltered) 0 errors.
  - Planted-bug controls, each flagged: memcheck <- rstd store out of bounds (610 invalid __global__ writes);
    racecheck <- segment barrier removed (shared WAR/RAW hazards on the partial slots); synccheck <- lanes 16..31
    executing `bar.warp.sync 0x0000FFFF` (180064 "Barrier error ... Invalid arguments"); initcheck (unfiltered) <-
    w read from a fresh torch.empty (uninitialized 16 B reads at the w offsets).
  - Caveats: initcheck with `--kernel-name` reports every input read (writes of the uninstrumented torch kernels
    are not tracked), so only the unfiltered run counts. A first synccheck control (`__syncthreads` skipped by
    half the CTA, which then exits) was NOT flagged by synccheck (legal on sm_70+ since exited threads count),
    hence the warp-mask control. The racecheck and initcheck controls were stopped after their reports were
    printed (error counting over 8192 CTAs ran >10 min).

## 4. CUDA graphs -- PASS (`graphs.py`, `logs/graphs.log`)

- Graph A, fresh process, the candidate's very first launch is inside capture; in-graph neighbours write the
  inputs (copy_, neg_/neg_ round trip), consumers read h, y, rstd, a 2nd candidate call in the same graph consumes
  call 1's h and y, then a consumer of call 2. 24 replays with the staging buffers rewritten in place (random,
  row scales 2^-70..2^64, specials, real): all 6 outputs + 4 consumers bitwise == eager reference every replay.
- Graph B (warm), 3 chained calls per graph, the graph's own input tensors rewritten in place between 16 replays:
  all 9 outputs EXACT every replay.

## 5. Adversarial values and reduction orders -- PASS (`adv.py`, `logs/adv_candidate.log`, 43/43)

Reference backend derived independently: profiler shows `at::native::vectorized_elementwise_kernel<8,
CUDAFunctor_add<BFloat16>>` + `torch_vendor.quack RMSNorm` (torch/_native override, `_fused_rms_norm_cond` True;
flydsl override is gfx950-only). `RmsNormFwdConfig.from_analytical_heuristic(4096, 16, 10)` = 128 threads,
threads_per_row 64, cluster 1, no reload. Reference SASS dumped myself (QUACK_CACHE_ENABLED=0,
CUTE_DSL_KEEP=cubin,ptx; `sass/ref_fwd.sass`): 8 x LDGSTS.128 at +0x400 (cols 8t + 512k + j); one FHFMA.BF16
chain from RZ in (k, j) order, lo half first (fma.rn.f32.bf16, no ftz); SHFL.BFLY 16,8,4,2,1 + FADD -> W_p;
lane 0 STS, BAR, lanes 0..1 LDS (others +0), butterfly again -> fl(W0 + W1); FFMA(S, 2^-12, fl32(1e-6)) ->
MUFU.RSQ (PTX rsqrt.approx.ftz.f32); rstd stored by the col-0 thread; FMUL2 (x*rstd) then *w; F2FP.BF16 RN.
Candidate SASS (`sass/cand/cand.sass`) has the same instruction forms (FHFMA.BF16 chain continued through smem per
lane, same butterfly, FADD W0+W1, FFMA 2^-12/eps, MUFU.RSQ, FMUL2 x2, F2FP; add = HADD2.BF16 / HFMA2.BF16 x*1+y).

Value families (all three outputs bitwise, plain + poisoned rerun):
- exhaustive add: all 2^32 bf16 (a, b) pairs through the candidate (128 calls), h, y and rstd EXACT;
- specials in in0 / in1 / both (NaN payloads 0x7FC0/0x7F81/0xFFFF..., +-inf, +-0, subnormals, max, 0x7F7E),
  12 targeted pair classes (inf + -inf, NaN + x, -0 + -0, -0 + +0, +0 + -0, max + max overflow, max + half-ulp,
  exact cancellation, subnormal sums) sprinkled and as whole rows;
- add rounding ties (b = +-half-ulp(a), quarter-ulp, just off the tie);
- full-range and +-30 exponents, row scales 2^-75..2^70 (sumsq overflow -> rstd 0, underflow, subnormal
  squares), sumsq near fp32 overflow, eps regime;
- y rounding ties (dominant 2^20 element -> power-of-2 rstd, exact x*w products), subnormal y;
- special weights, w all +0 / all -0 / subnormal / huge;
- zero rows (+-0 mixes), constant rows, one-hot rows at every column, one-hot wide magnitudes, in0 is in1;
- 4 seeds of wide-mantissa order rows, 8 spikes per row, real inputs, real inputs with spikes.

Direct order probes (8192 designed rows; in1 = 0, w = 1; squares and partial sums exact fp32 by construction;
sums of squares cannot cancel, so the probes use exactly representable big (2^28, 2.25 * 2^28) and small
(4..144) squares whose absorption depends on where the small terms join): families chain (one thread's 64-term
chain), bfly (one term per lane of one warp), join (big in one warp, smalls in the other / same lane of the
other warp), segment (big at segment q of a chain), mixed, random. An fp32 emulation of the derived order
(chain (k, j) from +0 -> butterfly 16..1 per warp -> fl(W0 + W1)) passed through rsqrt.approx.ftz(fma.rn(S,
2^-12, fl32(1e-6))) (my own Triton inline-PTX kernel) reproduces the reference rstd on all 8192 rows; candidate
EXACT on all. Alternative orders differ from the reference rstd on (rows): 30B N=2048 order (tpr 32, one warp,
cols 256k + 8c + j) 1875; single sequential chain 4532; butterfly 1..16 868; chain j-outer 1549; interleaved warps
(t = 2c + p) 1028; lane-wise cross-warp first 1028; per-segment restart 1472; the real aten fallback kernel
(override deregistered) 2423.

Negative controls (`make_negctl.py` -> `negctl/`, `run_negctl.sh`, `logs/negctl_summary.log`, `logs/negctl/`):
15/15 FAIL the suite: n01 butterfly 1..16 (22 cases), n02 30B N=2048 order kernel (22), n03 single-chain kernel
(23), n04 segment restart (22), n05 hi/lo swap in the chain (22), n06 eps + 1 ulp (9), n07 (x*w)*r (21), n08 cvt.rz
for y (27), n09 chain partial taken from lane c^1 (off by one lane, 22), n10 no segment barrier (31), n11 IEEE
rsqrt (28), n12 scale 2^-11 (28), n13 skip all-zero y stores (16, caught by poisoning), n14 add via fp32 ftz
(6, incl. exhaustive add), n15 rstd only for even rows (33). On the order probes the n01/n02/n03/n04 kernels
differ from the reference in exactly 868/1875/4532/1472 rows, the counts their emulations predict.

## 6. Special cases

Not GEMM-adjacent. No PDL (no griddepcontrol in PTX/SASS). The one `.nc` load (LDG.E.128.CONSTANT) reads w,
written only by earlier kernels (no overlap without PDL; in-graph writer of w tested in graph A). Multi-device:
cuda:0 only (per brief); outputs are allocated on `in0.device` and the launch uses the TVM-FFI environment
stream; no hard-coded device index.

## 7. Timing (`bench.py`, `logs/bench1.log`, `logs/bench2.log`)

CUDA graphs of 8 calls rotating 4 input sets, live outputs, reference and candidate interleaved per round
(order alternating), gpu0.lock held for the timed loop; GPU 0 showed one other process (1.2 GiB, user bahuang, another verifier) at start and end of both timed loops.

| run | reference us/call median (p10-p90) | candidate us/call median (p10-p90) | speedup |
| --- | --- | --- | --- |
| bench1, 60 rounds | 53.35 (53.09-53.58) | 40.04 (39.98-40.09) | 1.332 |
| bench2, 40 rounds | 53.38 (53.14-53.66) | 40.08 (40.03-40.28) | 1.332 |
| single-call graph (bench1) | 58.24 (57.60-59.77) | 44.02 (43.36-45.69) | 1.324 |

Saving = (53.35 - 40.04) us x 94 calls/step = **1.25 ms/step**. Candidate moves 268.5 MB/call -> 6.7 TB/s
(reference 5.0 TB/s).

## Findings for the writer (non-blocking)

- Off-spec (misaligned / non-contiguous) inputs raise ValueError where the reference computes (quack clones);
  unreachable in this graph (all inputs are fresh allocations), and a failure would be loud.
- The handoff's "n5 eps add not fused passes: output-equivalent" claim is consistent with my analysis
  (S * 2^-12 is exact unless it underflows, where eps dominates).
