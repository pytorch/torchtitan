# G91 router aux-loss forward, cyd t1 -- verifier vilma

**Verdict: PASS**

- Candidate: `candidates/G91/cyd/t1/aux_fwd.py`, sha256 `51bb469f2376ffd5c04616acbe0d82705741bc87bbef838c2fcd06d2e7e7e9b7`.
  The copy `cand.py` in this directory has the same hash, which also matches the handoff.
- Problem: `pat_nh2i7pxvf2zq7l2vjitrfx5mtruso335mumcpmo5ytxvmcwto6c` / `specialization_a67qwv7tvw2auv2wpoywzvo7mgp2xlcpllrqx5upvx3ll6lf4dud`.
  The reference.py sha256 is `326d90e76779c77f...`, which matches the run contract. There is no current kernel_name file, so the candidate was compared against reference.py.
- Environment: GB300 (sm_103, 152 SMs), torch 2.16.0a0+git3c53c6f, nvidia-cutlass-dsl 4.8.0, compute-sanitizer 2025.3.1.
  - Settings: deterministic algorithms, fill_uninitialized_memory=False, allow_tf32=False.
  - GPUs: checks 1-2 ran on GPU 0 under gpu0.lock, before the 16:58 rule change. Everything after that ran on GPU 2, with only the timed loops holding gpu2.lock.
- Scripts are in this directory and logs are in `logs/`, `negctl/` and `sanitizer/`. Every number below comes from a rerun; none is taken from the handoff.

## 1. Contract (`contract.py`, `logs/contract.log`)

- **Signature and imports.** sha256 of the copy matches the claim. `run(in0, in1)` matches the spec. A fresh process imports the file standalone; it needs only torch, cutlass and tvm_ffi.
- **Output metadata.** Every output matches the reference: shape, stride, dtype, device, storage offset 0 and storage nbytes.
  - v4 is `[8192,128]` with stride `(1,0)`, a view of a 32768-byte storage holding clamp_min(v2). Its storage bytes match the reference.
  - The three outputs use three distinct storages, as in the reference, and none of them aliases an input.
- **Inputs and reruns.** Inputs are not mutated (checked on the full storage bytes). Five repeated calls give identical results.
- **Reference launch configuration.** The torch profiler shows the configuration the candidate relies on:
  - the norm runs `reduce_kernel<512,1,ReduceOp<NormOneOps,...,4,4>>` with grid 512 and block (32,16);
  - the column sum runs `reduce_kernel<128,4,...sum...>` with grid (1,128,1) and block (32,4), i.e. **128 CTAs**;
  - the candidate launches K1 with grid 128 x 128 threads (102 regs, 3072 B smem) and K2 with grid 32 x 128 threads (PDL).
- **Reduction order in the source.** Reduce.cuh at the build commit 3c53c6f confirms the orders:
  - `ctas_per_output = clamp(2432, 8, 128)`;
  - the CUDA intra-warp path uses decreasing shuffle offsets (`dim_x >> 1 .. 1`);
  - thread_reduce_impl uses vt0 = 4 strided accumulators combined in order;
  - block_y_reduce combines (T0+T2)+(T1+T3);
  - in the last CTA, warp y walks CTAs y, y+4, ... starting from ident.
- **Real input.** The captured real input is degenerate: every in0 element is 1/128 and every mask value is true. All its sums are exact, so the real case says nothing about order. Reduction order is covered by section 5.

## 2. Official gate (`logs/gate.log`)

`python harness.py spec.json cand.py --seeds 0 1 2 3 4 5 6 7 --real real_inputs/fusion_G91.pt`

- seed0 through seed7: EXACT.
- real: EXACT.
- Result: **PASS**. The harness checks bytes, metadata and alias structure, with poisoned outputs (0xFF and 0x5A) and a rerun.

## 3. Memory

**Every output byte is written** (`adv.py`):
- Every adversarial case ran twice, after freeing bait blocks of every candidate allocation size filled with 0xFF and then with 0x5A.
- In addition, K1/K2 were launched directly with v2, v3, the `part` scratch buffer and v7 pre-filled with 0xFF, 0x5A, 0x00 and NaN. All four results are EXACT.

**compute-sanitizer** (`run_sanitizer.sh`, `sanitizer/summary.log`):
- Setup: kernel filter `kns=kernel_cutlass`; inputs written H2D; memcheck and initcheck run with `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, and memcheck adds `--padding 512`. The driver compares results on the host and printed EXACT under every tool.
- Candidate results: memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors, initcheck 0 errors.
- Planted bugs, each flagged by its tool:

| Tool | Planted bug | Result |
| --- | --- | --- |
| memcheck | K2 writes v7 at index +4, out of bounds | 4 errors |
| racecheck | K1 without its CTA barrier | 10 hazards |
| synccheck | K2 barrier executed by even threads only | 1728 "divergent thread(s) in warp" errors |
| initcheck | CTA 77 partial left unwritten | 128 uninitialized reads |

- A half-warp shfl membermask control was **not** flagged by synccheck, so it is not counted as a control.

## 4. CUDA graphs, streams, PDL (`graphs.py`, `logs/graphs.log`)

| Test | Result |
| --- | --- |
| First ever run() call inside graph capture, then 5 replays with new inputs written in place | EXACT |
| Producer kernels write in0/in1 in place, run(), consumers read all three outputs; 100 replays with changed sources | EXACT |
| Three calls in one graph, the second call's input computed in-graph from the first call's outputs; 30 replays | EXACT |
| Early-trigger predecessor (CuTe kernel: griddepcontrol.launch_dependents first, ~20k dependent-op spin, then writes in0 and the mask over NaN/False stale values), eager | 200/200 EXACT |
| Same predecessor, in graph | 50/50 EXACT |
| Side stream: slow writer, then run() on the same non-default stream | 50/50 EXACT |

PDL controls:
- **P1 (K1 launched with use_pdl=True): caught.** Eager gives 0/200 EXACT and in-graph 0/50 EXACT.
- **P2 (K2 without griddepcontrol.wait): not caught**, as in the 30B verification, because K1 triggers only after its last store. SASS confirms the real guarantee instead (section 6).

## 5. Adversarial values and reduction orders

**Order probes** (`order.py`, `logs/order.log`): 50 inputs, candidate EXACT on all.
- **Emulator.** An fp32 emulator of the reference's derived orders (row L1 norm and the 128-CTA column sum) matches the reference bitwise on all 50 inputs.
- **Column-sum probes.** These use exactly representable cancelling terms (B = 0.25, -B, s = 2^-27), with v6 = p exactly because each probe row's L1 is exactly 1, held in one lane. There are 15 structured probes, one per join of the reference tree:
  - accumulator chain, and membership by k mod 4;
  - thread combine ((a0+a1)+a2)+a3;
  - CTA combine (T0+T2)+(T1+T3);
  - last-CTA chain b = y, y+4, ... (pair-first, s-first, ends 0/120/124) and membership by b mod 4;
  - final combine (U0+U2)+(U1+U3).
- **Probe variants.** Each structured probe runs on 32 columns, with random CTA/warp/slab positions per column. On top of these come 5 random-row probe sets. Everything runs in two variants: zero fill (probe warps on the div.rn path) and 2^-60 fill (fast path everywhere).
- **Expected outcomes.** Every structured probe gives its predicted outcome: all s, or all 0.
- **Other inputs.** Norm probes put a 1 and two 2^-24 values (a rounding tie) at random columns, 8192 rows per input. There are also 4 wide-exponent random inputs.
- **Controls.** Every alternative order disagrees with the reference on at least one input, so the probes discriminate:
  - different CTA splits: 64, 256 and 512 (the 30B split) CTAs;
  - sequential column sum, contiguous accumulators, pairwise thread combine, sequential CTA combine;
  - contiguous last-CTA split, sequential final combine;
  - norm: ascending shuffle (ROCm order), pairwise lane combine, sequential.

**Adversarial suite** (`adv.py`, `logs/adv.log`): 53 cases plus 4 prefilled direct launches, ALL EXACT. Each case is checked for values, metadata, storage bytes, aliasing, rerun identity and unmutated inputs. The cases cover:
- **masks:** all, none, none with all-negative x (every v6 = -0), alternating, one row, whole warps, k slabs, p=0.01 and p=0.99, and mask bytes 2/0x10/0x40/0x80/0xFF (torch reads byte != 0);
- **signed zeros:** +0 and -0 rows, a -0 column, an all -0 tensor, an all +0 tensor;
- **fast/slow division paths:** all-negative fast-path rows half masked, mixed fast/slow warps, a single zero element forcing a div.rn warp;
- **NaN and inf:** NaN payloads (+-qNaN, sNaN, custom payloads, some in masked rows), an all-NaN tensor, +-inf elements and rows;
- **overflow:** L1 overflow (2^121 x 128), 3e38 and FLT_MAX rows;
- **subnormals:** an all-subnormal tensor, rows of min and max subnormals, one large element with the rest subnormal (subnormal quotients summed);
- **clamp boundary:** L1 == fl32(1e-12) +-3 ulp exactly, and L1 near eps;
- **fast-path range edges:** d == 2^-20 and 2^20 exactly and +-ulp, d near 2^+-20, |x| == 2^-60 and +-1 ulp with d ~ 2^20, quotients ~2^-80;
- **just outside the fast range:** |x| < 2^-60, and d between 2^21 and 2^40;
- **wide ranges:** exponents over 110 binades, row scales of 2^+-30;
- **real-input variants:** the real input with random masks, with specials, and scaled by 2^-70;
- **layouts:** in0 at a 16 B-aligned offset (direct kernel path), a misaligned offset, row stride 256; in1 with stride 2 and with byte offset 3.
- **Out-of-contract inputs** (fp16, shape [8192,64], uint8 mask, CPU tensors) raise ValueError.

**Fast division** (`divcheck.py`, `logs/divcheck.log`). The candidate's own device code (frcp_refined + p_div_fast) was run in a standalone kernel and compared with torch div.rn.
- **In range** (2^-20 <= d <= 2^20 and 2^-60 <= |x| <= d), 0 wrong quotients in 1.61e9:
  - 1.07e9 random-bit pairs (including d = 2^+-20 exactly);
  - 2.7e8 log-uniform pairs;
  - 2.7e8 near-midpoint quotients.
- **Outside the range:**
  - |x| < 2^-60 gives 2.9M wrong in 134M, so **the x bound is load-bearing**;
  - d in (2^20, 2^40] and d in [2^-40, 2^-20) give 0 wrong. `logs/dbound.log` shows 0 wrong for d up to 2^120, and failures only from d >= 2^120 (3.9M in 33.5M). The d bounds are therefore conservative.
- **End to end:** 1200 single-unmasked-row calls, so that v7 equals that row's quotients exactly. That is 153,600 quotients, half of them in hard rows (x0 = d*, near-midpoint x_j, L1 == d* exactly). Each row ran with its warp on the fast path and forced onto div.rn, and all calls are EXACT.
- **Bound probe** (`boundprobe.py`, `logs/boundprobe.log`): 60 single-row calls whose quotients are 6224 (x, d) pairs where the fast formula is wrong (|x| < 2^-60). The candidate is EXACT on all 60; mutant M14 (x bound removed) fails all 60.

**Negative controls** (`make_mutants.py`, `run_negctl.sh`, `negctl/summary.log`): 27 value mutants; 26 caught by order.py, adv.py and (for M14) boundprobe.py, and one (M15) is equivalent in practice.
- **Norm order:**
  - M01: lane chain made pairwise;
  - M02: butterfly levels 8 and 16 swapped;
  - M03: wrong shuffle offset.
- **Column-sum order and splits:**
  - M04: contiguous accumulators;
  - M05: pairwise thread combine;
  - M06: sequential CTA combine;
  - M07: contiguous last-CTA split;
  - M08: sequential final combine;
  - M09: **different CTA split**, rows b + 128w + 512k, with v2/v3 still correct.
- **K2 bugs:**
  - M10: seeded with the first value, which leaks -0 (caught by the all-masked negative case);
  - M11: last CTA skipped;
  - M12: CTA index off by one.
- **Division and rounding:**
  - M13: no Newton step;
  - M14: x bound removed (caught by the bound probe);
  - M15b: d upper bound removed (caught by the d = inf rows);
  - M16: per-lane check without the vote;
  - M17: div.full instead of div.rn.
- **Clamp:** M18 (eps + 1 ulp) and M19 (no NaN propagation).
- **Mask handling:**
  - M20: only bit 0 of the mask byte;
  - M21: wrong row;
  - M22: fast path ignores the mask;
  - M23: slow path ignores the mask.
- **Memory and synchronization:**
  - M24: no CTA barrier;
  - M25: one CTA partial unwritten;
  - M26: one v2 group unwritten.
- **Not caught:** M15 (d upper bound widened to 2^40). It is equivalent in practice, because no wrong quotient exists below 2^120 (see the fast-division section above).

## 6. Special cases

- **No GEMM.**
- **SASS** (`logs/sass.log`, `dump/`):
  - `.nc` loads appear only for in0 (16x `LDG.E.NA.128.CONSTANT`) and the mask (1x `LDG.E.U8.CONSTANT`). Neither is written during the K1 -> K2 chain, and K1 is a normal launch, so it waits for its producers.
  - K1's PDL trigger (`PREEXIT` at 0x6590) comes after both `STG.E.128`.
  - K2 executes `ACQBULK` (griddepcontrol.wait) at 0x30, before its only global load, `LDG.E.128.STRONG.GPU`, which is not `.nc`.
  - FTZ and directed-rounding instructions appear only inside ptxas's div.rn.f32 subroutine (0x65b0-0x6c30), which the slow path calls. The main datapath has no FTZ.
- **Multi-device:** tested on one device only (GPU 0/2 leases; `cuda:0` inside the process).
  - The code hard-codes no device index. It reads `get_device_properties(in0.device)`, launches on the TVM-FFI env stream, and its cubins are compiled at import for the import-time GPU (sm_103a).
  - It raises ValueError if the GPU would give the reference column sum fewer than 128 CTAs.
  - Calls on a non-current device were not tested.

## 7. Timing (`timing.py`, `logs/timing.log`; GPU 2, gpu2.lock held for the timed loop)

Method:
- CUDA graphs hold K calls on K rotating input sets (set 0 = real, the rest router-like positive rows with random masks), and every call's outputs stay alive.
- 300 interleaved rounds, each timing one replay of the reference graph and one of the candidate graph.
- GPU 2 was at 100% utilization from other verifiers' unlocked correctness and sanitizer runs. With 20-replay windows the reference absorbed other contexts' time slices (88.5 us/call, see `logs/timing_contended_nrep20.log`), so the single-replay medians below are the numbers to use.

| Graph | reference.py median (p10-p90) | candidate median (p10-p90) | speedup | saving at 94 calls/step |
| --- | --- | --- | --- | --- |
| 8 calls, 8 sets | 31.64 us (31.37-32.04) | 4.43 us (4.37-4.48) | 7.1x | **2.56 ms/step** |
| 32 calls, 32 sets | 32.42 us (32.28-32.55) | 4.25 us (4.17-4.32) | 7.6x | 2.65 ms/step |
| 1 call | 35.78 us (35.14-37.69) | 8.99 us (8.58-9.44) | 4.0x | 2.52 ms/step |

The candidate was faster in 300/300 rounds in every configuration. The writer's claim (31.97 -> 4.02 us/call) is consistent with this; the extra ~0.4 us/call here is the graph launch, amortised over 8 calls in single-replay samples.

## Findings for the writer (non-blocking)

1. **Transposed in0 gives silently different results** (`layout_probe.py`, `logs/layout_probe.log`). This is out of spec: the spec fixes stride (128,1), and stride is part of the specialization key.
   - For a transposed in0 (stride (1, 8192)), reference.py itself changes its reduction orders.
   - `_prepare` copies the tensor to contiguous and returns the contiguous-order result. It differs from reference.py in v2/v4 and v7, with no error.
   - Row-strided in0 (stride(1) == 1) matches reference.py.
   - Suggested fix: raise ValueError when `in0.stride(1) != 1`, instead of silently computing a different order. The 30B t9 fell back to reference.py here.
2. **The fast-division d bounds are conservative** (exact up to d ~ 2^120). Only the |x| >= 2^-60 bound is load-bearing. Keep the check as it is.
3. **The real captured input cannot detect reduction-order bugs** (all 1/128, mask all true). Order coverage comes from the probes in section 5.
