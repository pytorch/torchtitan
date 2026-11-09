# G15 / fay t1 -- verdict: PASS

- Verifier: vanda (independent; did not reuse the writer's probes or negctls)
- Candidate: `W/autokernel/run1/candidates/G15/fay/t1/g15_t1.py`, sha256
  `ca33ae7e7807d47c2658a2968b34dc989303e52ee68f4403d3b4f8f9550fc402` (= handoff; copy `cand.py` here)
- Problem: pat_4j6e.../aihl3cdzkg64 (wqkv dgrad `_scaled_mm_v2` [8192,9216]x[9216,4096] + attention_norm RMSNorm
  backward N=4096 + residual add), 94 calls/step. No current kernel_name: compared against reference.py (sha256
  prefix d52926e98352020b, = board).
- GPU 0, torch 2.16.0a0+git3c53c6f, nvidia-cutlass-dsl 4.8.0, compute-sanitizer 2025.4.1. All scripts in
  `scripts/`, logs in `logs/`, sanitizer logs in `san/`, SASS/PTX in `sass/`, mutants in `negctl/`.

## Timing (CUDA graph, interleaved with reference.py, 3 rotating ~250 MB input sets, live outputs)

`scripts/bench.py` (gpu0.lock held only for the timed loops, about 0.5 s; clocks settled under the lock first):

| run | GPU 0 at lock | reference 1-call median (p10-p90) | candidate 1-call median (p10-p90) | 3-call graph per call ref / cand | saving |
| --- | --- | --- | --- | --- | --- |
| 4 | only this process | 269.6 (262.9-277.5) | 227.4 (222.3-233.1) | 270.3 / 227.9 | 42.2 us = 3.96 ms/step |
| 5 | only this process | 270.3 (264.5-277.1) | 227.2 (222.7-233.2) | 269.0 / 226.3 | 43.1 us = 4.05 ms/step |
| 1 | other processes active | 252.7 (248.4-255.1) | 214.0 (211.1-217.8) | 247.4 / 210.5 | 38.7 us = 3.63 ms/step |
| 2 | other processes active | 252.2 (248.4-254.9) | 214.2 (210.1-217.6) | 248.0 / 211.1 | 38.0 us = 3.58 ms/step |

Speedup 1.18-1.19x in every run. The paired per-round saving (ref - cand) was always positive: median 37-43 us,
minimum 30.5 us. Saving at 94 calls/step: **3.6-4.0 ms/step**. The writer's harness figure (253.3 -> 215.8 us)
falls inside this range. Absolute times shift by about 15 us between an idle and a shared GPU 0; the ratio does not.

## Checks

### 1. Contract -- PASS
- sha256 matches the handoff (`logs/sha256.log`).
- The signature `run(in0, in1, in2, in3, in4, in5, in6, in7)` matches the spec (`scripts/contract.py`, `logs/contract.log`).
- A fresh `python -I` process imports the copy standalone from /tmp. Its only imports are cutlass and torch
  (`logs/standalone.log`).

### 2. Official gate -- PASS
`CUDA_VISIBLE_DEVICES=0 python W/autokernel/harness.py <spec> cand.py --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G15.pt`
-> seed0-7 and real all EXACT, PASS (`logs/harness.log`).

`contract.py` checked further, on seeds 0-7, the real inputs and 9 special GEMM input sets:
- output shape, stride, dtype, device and storage offset (0) match the reference;
- the two outputs do not alias each other or any input;
- inputs are bitwise unmutated;
- candidate reruns with NaN-filled `torch.empty`, 0xFF bait and 0x5A bait are all bitwise equal to the reference;
- all 18 cases are EXACT.

### 3. Memory -- PASS
**Poisoning.**
- Harness: 0xFF and 0x5A allocator bait, plus NaN fill of `torch.empty`.
- Every adversarial case: NaN fill and 0xFF bait, with the 1.2 MB partials buffer baited too.
- Result: every output byte is written.

**compute-sanitizer** (`scripts/run_san.sh`, `san_driver.py`, `logs/san_summary.log`):
- Settings: `PYTORCH_NO_CUDA_MEMORY_CACHING=1` so that tensor bounds are exact. "full" = real GEMM; "tail" = run()
  with a torch-written d.
- **memcheck:** full 0 errors, tail 0 errors (`--leak-check no`). With leak-check on, the only reports are leaks at
  exit: the cuBLAS handle workspace and a 4 B pinned host buffer. No invalid accesses.
- **racecheck:**
  - tail: 0 hazards.
  - Full path with the GEMM also instrumented (`kns=nvjet,NormBwdAdd,ReduceRows`): 0 hazards.
  - Tool limitation: racecheck hits "Internal Sanitizer Error: failed to handle a hardware exception" whenever the
    cuBLAS nvjet GEMM runs un-instrumented under racecheck. That happens with our kernels selected, with no kernel
    selected, and for reference.py with only the quack kernels selected. It is not candidate-specific.
- **synccheck:** full 0 errors, tail 0 errors.
- **initcheck:**
  - tail, unfiltered: 0 errors.
  - On the full path, every report is a 16 B read of d. This happens in our K1 and, identically, in quack's
    RMSNormBackward for reference.py, because initcheck does not track the cuBLAS GEMM's stores (same as the 30B
    note).
  - Note: kernel filters must not be used with initcheck. Un-instrumented kernels' writes then look
    uninitialized, so those runs (`san/old/`) are invalid.

**Planted-bug controls** (`negctl/san/`), each flagged by its tool:

| control | planted bug | tool | result |
| --- | --- | --- | --- |
| s01 | K2 stores dw one vector past its end | memcheck | 4096 invalid __global__ writes |
| s02 | no barrier before the cross-warp smem loads | racecheck | 258962 hazards |
| s03 | warp-divergent bar.sync in K2 | synccheck | 53472 "Divergent thread(s) in warp" |
| s04 | CTA 0 never stores its partial | initcheck | 8192 uninitialized reads in K2 |

### 4. CUDA graphs -- PASS
`scripts/graphs.py`, `logs/stress_candidate.log`: **369/369 EXACT** against eager reference.run on the same inputs.
There are 7 input sets: 3 seeds, real, NaN e8m0, extreme e8m0, and tail specials.
- **t1:** a graph with in-graph copies that write all 8 inputs, then run(), then in-graph copies that read both
  outputs. 42 replays rotate the copy sources in place.
- **t2:** live outputs, with inputs rewritten in place between replays.
- **t3:** three calls in one graph (sets A, B, A), which exercises reuse of the partials buffer inside the graph.
- **t4:** an early-triggering PDL predecessor (`griddepcontrol.launch_dependents` first, then spins 0/20/100 us,
  then writes all inputs) before run(). PDL consumers (`griddepcontrol.wait`) of dw and g run after it. Tested
  eagerly and in a graph.
- **t5:** SM occupier spin kernels on a side stream during run(): 76-456 CTAs, 30/150 us, 0/100 KB smem. Tested
  eagerly and with graph replay.
- **t6:** back-to-back eager calls with rotating inputs and live outputs.

**First call inside capture** (`logs/stress_first_in_capture.log`):
- The very first run() in a fresh process, captured, then replayed and rewritten in place: 3/3 EXACT. Only cuBLAS
  had been initialised beforehand, by an unrelated 64x64 matmul.
- Cold (no prior cuBLAS use), capture fails in `cublasCreate`. reference.run fails identically, because this is the
  shared `_scaled_mm_v2` call.

**Stress controls:**
- n20 (K2 without `griddepcontrol.wait`): 128/138 checks fail in `graphs.py --short`.
- n23 (cp.async ring waited too early): 132/138 fail.

### 5. Adversarial values and reduction orders -- PASS
**Reference order, derived independently on this build.**
- Dispatch (`logs/dispatch.log`, profiler): nvjet GEMM, then quack `RMSNormBackward` with grid 304 x 128, then
  `reduce_rows(8, 256)` with grid 512, then the aten bf16 add.
- Why grid 304: sm_103 is not in the (9,0)/(10,0) launch table of `torch/_native/ops/norm/norms.py`, so
  blocks = `min(M, get_sm_count(4096))` = 2 x 152.
- Config: `_for_blackwell_bwd(4096, bf16)` gives 128 threads/row, cluster 1, cp.async with 2 stages,
  reload_wdy="smem".
- I dumped the quack cubin myself (`QUACK_CACHE_ENABLED=0 CUTE_DSL_KEEP=cubin`; `sass/quack_bwd.sass`) and read it:
  - **Weights and per-element inputs:**
    - w' = bf16(w + 0) (HADD2/HFMA2.BF16);
    - x_hat = FMUL2(x, rstd);
    - wdy = FMUL2(d, w').
  - **Row sum S:**
    - per thread: an FFMA chain from RZ over i = 8k + j, for cols 8t + 1024k + j;
    - across the warp: SHFL.BFLY 16, 8, 4, 2, 1;
    - across warps: lanes 0-3 load W0-W3 and the other lanes hold 0 (IMAD.MOV R33 = RZ), then butterfly
      16..1. The result is ((W0+W2) + (W1+W3)) after the +0.
  - **Mean and dx:**
    - mean = S * 0.000244140625;
    - dx = F2FP.BF16(FMUL2(FFMA2(d, w', -FMUL2(x_hat, mean)), rstd)), using the exact d*w' product inside the FFMA.
  - **dw:**
    - partial: FFMA2(x_hat, d, acc) from CS2R zero, in row order;
    - reduce_rows: `0 + p[m] + p[m+32] + ...`, then SHFL.BFLY 8 then 16, then a sequential W1..W7 sum and
      F2FP.BF16.
  - The candidate's SASS (`sass/cand_k1.sass`, `cand_k2.sass`) has the same instruction sequence. One detail
    differs: wdy is computed with FHFMA.BF16(d, w', -0), which equals rn(d*w').

**Order model checked against the reference's own outputs.**
- I wrote a numpy emulator of this order (`cases.py emu_rowsum` / `emu_dw`).
- I probed with exactly representable cancelling terms +2^25, -2^25, +1: the result is 1 iff the +1 joins after the
  cancellation.
- Coverage: 10 row-sum probes over 81,920 rows (levels: thread chain, warp butterfly, cross-warp, any, mix) and 14
  dw probes over 57,344 columns (levels: CTA row chain, K2 chain, bfly8, bfly16, W sum, any, mix).
- The model predicts the reference's revealed S (dx at the 2^25 column) and dw with **0 mismatches**.
- The probes discriminate the alternatives:
  - j-major chain: differs in about 2,690 rows per thread-level probe;
  - butterfly 1..16: about 4,000 rows per warp-level probe;
  - cross-warp (01)(23): about 5,400 rows;
  - 30B grid B=608: about 1,400 columns per dw probe;
  - K2 butterfly 16-then-8: about 1,000 columns per bfly16 probe.

**Adversarial suite** (`scripts/adv.py`, `cases.py`; `logs/adv_candidate.log`): **67/67 EXACT**, plus negzero_S 2/2
(`logs/negzero_S.log`). Each case goes through run() with the GEMM replaced by a crafted d, against the verbatim
reference tail. Poisoned outputs, a deterministic reference, unmutated inputs and correct metadata are checked
every time. Cases:
- **Numerics:**
  - randn, heavy-tailed;
  - real (GEMM d), and real with sign/noise perturbation;
  - row scale (2^+-60), column scale;
  - all-equal and one-hot rows;
  - exact S cancellation pairs.
- **Specials:**
  - NaN payloads in d, x, w, residual and rstd;
  - +-inf;
  - d*w overflow, which distinguishes the exact product inside the FFMA from the rounded one;
  - subnormal x_hat, wdy, mean, dx and g (no FTZ);
  - signed zeros, including w=-0 and residual -0;
  - row sum -0 before the cross-warp +0 (negzero_S);
  - rstd 0/-0/negative/subnormal/huge/inf/NaN;
  - bf16 rounding ties for dx and g;
  - mean extremes (underflow to +-0, subnormal, huge);
  - random bit patterns.
- **Order probes:** as above.

**Negative controls** (`scripts/make_negctl.py`, `negctl/`, `logs/negctl/`; counts are from `adv.py --quick`, one seed
per case, 29-33 cases): all 23 mutants change behaviour and
**all 23 are caught**. Each is listed with the cases that catch it:
- **30B orders:**
  - n01 30B dx formula fma(x_hat, -mean, wdy): 9 cases;
  - n02 30B grid B=608: 17;
  - n03 30B 2^-11: 18.
- **Order changes:**
  - n04 q-major chain: 12;
  - n05 chunks reversed: 12;
  - n06 butterfly 1..16: 12;
  - n07 cross-warp (01)(23): 12;
  - n08 K2 butterfly 16-8: 8;
  - n09 K2 W sum reversed: 13.
- **Rounding and signed zeros:**
  - n10 unfused dw accumulation: 4;
  - n11 no w+0: signed_zero;
  - n12 no cross-warp +0: negzero_S, 16.25M elements differ;
  - n13 rounded d*w in dx: wdy_overflow;
  - n14 dx cvt.rz: 19;
  - n15 g via fp32 add + cvt.rz: 13.
- **Off-by-one and synchronisation:**
  - n16 K2 skips the last partial row: 22;
  - n17 K1 one row fewer: 29;
  - n18 rstd of the wrong row: 25;
  - n19 missing K1 barrier: 28;
  - n20 K2 without PDL wait: 26, plus the stress test;
  - n23 cp.async ring waited too early: 27, plus the stress test.
- **FTZ:**
  - n21 FTZ on x_hat: sub_xhat;
  - n22 FTZ in the row-sum chain: sub_wdy, sub_mean_dx.

Two mutants needed extra work:
- **n12:** at first I thought it was output-equivalent. I then built negzero_S, where every thread's first product
  underflows to -0, which shows the difference.
- **n15:** the first version (`add.rz.bf16x2`) is not valid PTX and failed to compile
  (`logs/negctl/invalid_n15_...`); it was replaced.

### 6. Special cases -- PASS
**GEMM identity** (`contract.py` TorchDispatchMode recorder):
- Candidate and reference each issue exactly one `aten._scaled_mm_v2.default`.
- The arguments are identical: the same input tensor objects (same data_ptr, shape, stride, dtype, storage offset 0,
  1 KiB-aligned), `[in2], [3], [1], [in3], [3], [1], None, torch.bfloat16`.
- The output metadata is identical: (8192, 4096), (4096, 1), bf16, offset 0, 1 KiB-aligned.
- The d bits are equal on seeds 0-7, the real inputs, and the special sets:
  - NaN e4m3 in A and in B;
  - NaN e8m0 in the A scales and in the B scales;
  - extreme e8m0 (mix of 0x00/0x01/0xFD/0xFE; hi only; lo only);
  - NaN plus extreme e8m0;
  - -0 A rows;
  - e4m3 max with large scales.
- Profiler: the same cuBLAS kernel `nvjet_sm103_qqtst_128x256_128x6_2x1_2cta_v_bz_Avec32UE8M0_Bvec32UE8M0_NNT`
  (grid 1024, block 256, smem 222592, 255 regs) in both.

**PDL / non-coherent loads** (`logs/nc_inventory.log`):
- **K2:** ACQBULK (griddepcontrol.wait) is at 0x30, before all 10 partial loads, which are plain `LDG.E`. There are
  0 `.CONSTANT` loads and 0 `ld.global.nc` in the PTX, and no EXIT before the wait.
- **K1:**
  - PREEXIT (launch_dependents) is at 0x0a0, and K1 is a normal launch.
  - Its `.nc` loads are only w (4 x 128-bit) and rstd (2 x 32-bit). Both are inputs that are never written during
    the call chain.
  - d, x and in7 go through `LDGSTS.BYPASS` (cp.async.cg).

**Multi-device:** only cuda:0 inputs were run (the GPU 0 lease). The code's device assumptions are listed under
the next heading.

### Notes for the writer and coordinator (non-blocking; none affects the spec contract)
1. The candidate has no fallback. Inputs with a storage offset that keeps 16 B alignment are EXACT
   (`logs/offsets.log`). A misaligned x, residual or w (offset of 1 bf16 element) raises
   `ValueError("unsupported inputs ...")`, where the reference computes normally. This is outside the spec (shape,
   stride and dtype only) and explicit rather than silent.
2. Device assumptions:
   - `_B` and the compiled kernels come from `torch.cuda.current_device()` at import;
   - run() requires an sm_103 device with the same SM count;
   - kernels launch on the tvm-ffi env stream.
   Not exercised on cuda:1 here; the coordinator's multidev_check covers this.
3. A cold first call inside capture fails in `cublasCreate`, identically for reference.py.
4. Like the 30B kernel, numerics depend on quack's sm_103 schedule (B = 2 x SMs, which is not in the AOT table) and
   on reload_wdy="smem" at N=4096. A torch or quack update, or `TORCH_DISABLE_NATIVE_JIT`, requires re-verification.
