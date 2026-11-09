# G41 fay t1 -- verifier vin -- PASS

Candidate: `candidates/G41/fay/t1/g41_t1.py`, sha256 `37c4fbc1c4861b74e13a586913be59cd7ed7adfc4ed4dcdc32d49bdb7e9d9763`
(matches HANDOFF.md; re-checked at the end; evidence copy `g41_t1.py` identical).
Problem: pat_xf3v.../af5md3 (attention_norm `aten._fused_rms_norm` N=4096 + `torchao.mxfp8_quantize` rowwise 1x32
e4m3 rceil + `triton_mx_block_rearrange`), outputs (y, q, s, rstd), 94 calls/step. No current `<kernel_name>`; compared
against `reference.py` (sha256 5428a7cc..., run from the copy `reference_copy.py`). GPU 0 (GB300), torch
2.16.0a0+git3c53c6f, cutlass-dsl 4.8.0, torchao 0.19.0.dev20261103. All commands run from this directory with
`CUDA_VISIBLE_DEVICES=0`; logs under `logs/`.

## 1. Contract -- PASS
- `sha256sum` (logs/sha.log): candidate == handoff hash.
- Fresh-process standalone import of the copy (logs/import.log): `run(in0, in1)` == spec signature; imports only
  torch + cutlass; compiles at import (no launch).
- `check_contract.py` (logs/contract.log), seeds 0, 11 and real: shape/stride/dtype/device/storage_offset/storage
  nbytes of all 4 outputs == reference (y 64 MiB, q 32 MiB, s [8192,128] 1 MiB, rstd [8192,1] stride (1,1)); no
  output aliases an input or another output (same as reference); inputs unmutated; 3 reruns identical; reference
  deterministic.
- Reference dispatch on this build (profiler): `kernel_cutlass_kernel_torch_vendorquackrmsnormRMSNorm...`,
  `mxfp8_quantize_kernel<bf16, e4m3, 1, 32, (ScaleCalculationMode)1>`, `triton_scale_swizzle`.

## 2. Official gate -- PASS
`python W/autokernel/harness.py <spec> g41_t1.py --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G41.pt`
(logs/harness_gate.log): seed0-7 EXACT, real EXACT (poisoned 0xFF / 0x5A + NaN-filled output memory, rerun equal).

## Reference order (derived independently)
From the source on this build: `torch/_native/ops/norm/rmsnorm_impl.py` overrides `aten._fused_rms_norm` ->
`torch/_vendor/quack/rmsnorm.py` with `RmsNormFwdConfig.from_analytical_heuristic(4096, 16)`: num_threads 128,
threads_per_row 64 (N <= 6144), cluster_n 1, vecsize 8; `tiled_copy_2d` thread layout (2, 64) row-major -> thread
l = 32p + c of a row owns cols 8l + 512k + j; `row_reduce`: TensorSSA reduce (per-lane chain), `warp_reduction`
butterfly offsets 16, 8, 4, 2, 1 (cutlass source), `block_reduce`: lane 0 -> smem, barrier, second butterfly
over {W0, W1, 0, ...} = fl(W0 + W1); rstd = rsqrt(sum / 4096 + eps, fastmath).
Behavioural confirmation, `check_orders.py` (logs/orders.log): exact torch-fp32 emulations (emu.py) of H (above)
and 8 alternatives (30B order B30 = 1 warp/row with lane c owning 8c+256k+j for k<16; butterfly reversed;
chain (j,k) / reversed; lanes l and l+32 joined first; contiguous lanes; per-vector tree; warp halves), with rstd
from the real `fma.rn` + `rsqrt.approx.ftz` PTX (Triton). On 8 probe sets x 8192 rows the reference matches H on
every row (H:0) while every alternative mismatches 72-4663 rows per set. Targeted join-level probes (probes.py):
one big b (B = b^2, even mantissa) + two smalls with s^2 = ulp(B)/2 exactly (each absorbed by a tie unless the
two meet first), b chosen so rsqrt.approx separates B and B + ulp: per-lane chain (272 rows/set), all 20 ordered
butterfly-offset pairs (32 rows each; BFLYREV wrong on 32/32 of every pair), cross-warp join (1000 rows) -- all
match H. The 30B order is wrong on 2692 targeted rows. The candidate equals the reference on all 4 outputs on
every set (unit and random w). The negative-control kernels NC01 / NC04 reproduce the BFLYREV / B30 emulations
exactly (0 rows differ), validating the emulator.

## Blocked-scale layout -- PASS
`check_layout.py` (logs/layout.log): torchao `triton_mx_block_rearrange` on 3 synthetic [8192,128] tensors whose
bytes encode position == offset (c//4)*512 + (r//128)*16384 + (r%32)*16 + ((r%128)//32)*4 + c%4 (a bijection);
candidate s == rearrange(raw torchao scales of candidate y) == reference s on 3 per-block wide-exponent inputs
(85 distinct bytes, 98.8% of positions differ from each layout neighbour r+1, c+1, r+32, r+128).

## 3. Memory -- PASS
- Allocator poisoning: harness (0xFF, 0x5A, NaN-fill; seeds 0-7 + real) all bytes written.
- compute-sanitizer on `san_run.py` (randn + NaN/inf/subnormal rows, 2 calls, then every output byte read),
  `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (logs/san/): memcheck 0 errors (also with `--padding 65536`), racecheck
  (`--racecheck-report all`) 0 hazards, synccheck 0, initcheck 0.
- Tool negative controls (negctl/): NT1 w read 8 KiB past its end -> memcheck --padding 89892 errors; NC19 /
  NC20 missing barrier -> racecheck 65538 / 1835010 errors; NT2 barrier in a lane-0-only branch -> synccheck 497
  errors ("Divergent thread(s) in warp"); NT3 one 16-byte scale unit per CTA unwritten -> initcheck 65536 errors.
  (Without --padding, memcheck did not flag NT1: the OOB reads landed in a neighbouring allocation; the padded run
  is the meaningful one.)

## 4. CUDA graphs / streams -- PASS
`check_graphs.py` (logs/graphs.log), fresh process, first candidate launch inside `torch.cuda.graph` capture
(side stream). Graph: producer copies into x/w, call 1, consumer clones of all outputs, WAR `x.mul_(-2)`, call 2,
consumer clones: 48/48 replays EXACT vs eager reference (both calls + live outputs), sources rewritten in place
(randn, heavy-tailed, specials, in-place arithmetic; w changed every 3rd replay). Side-stream eager behind a
`_sleep` + producer write: 6/6 EXACT. Negative control NT4 (launch on the default stream): side-stream 1/6, graph
0/48 -> caught (logs/graphs_NT4_*.log).

## 5. Adversarial values -- PASS
`adv.py --coverage` (logs/adv.log): 47 cases, all EXACT on y, q, s, rstd: randn; real x with random w; real x
scaled 2^+-60; heavy-tailed 2^+-20; order-targeted and big/small rows; row scales 2^-150..2^130 (sumsq underflow,
eps-dominated, overflow -> rstd 0, inf*0 NaN); eps regime; per-block wide exponents (w random / w = 1); 8 scale
sweeps (w = 2^e per block column, e over -133..127, +- random mantissas); special weights (+-0, subnormal, huge,
+-inf, NaN payloads), w all -0, w all subnormal, w huge; NaN payloads / +-inf / +-0 / subnormals / bf16 max in x
(sprinkled, dense rows, no-NaN variant); subnormal-x rows and rows where x*rstd is fp32-subnormal (FTZ probes);
whole-row kinds (all NaN/inf/-inf/+-0/subnormal/max, alternating +-inf, all-equal rows at 14 exponents, one-hot
rows at lane/vector/warp/row edges with finite/huge/tiny/inf/NaN); controlled rows x = +-2^e (rstd = 2^-e exactly,
y = +-w exactly) with block amax exactly at 448*2^E boundaries (mantissas 0x5F/0x60/0x61) and powers of 2 for all E;
e4m3 midpoint (tie) cases normal and subnormal; all +0 / all -0.
Coverage (from reference outputs): all 249 reachable e8m0 bytes (0..247, 255); all 32640 finite bf16 block-amax
bit patterns 0x0000..0x7F7F plus inf and NaN; 175.8M normal and 22.0M subnormal e4m3 rounding ties; 22.5M
subnormal e4m3 outputs.
Negative controls (`make_negctl.py`, `run_negctl.sh`, logs/negctl/): all 22 FAIL the suite:
NC01 butterfly 1..16 (27 cases), NC02 chain pair order (27), NC03 chain k reversed (27), NC04 30B order (27),
NC05 join W0+W0 (29), NC06 2^-11 (45), NC07 eps 1e-5 (44), NC08 rcp(sqrt) rstd (31), NC09 (x*w)*r (28), NC10 cvt.rz
(28), NC11 .ftz x*r (7), NC12 30B scale row-block stride (43), NC13 row-group swap (27), NC14 / NC15 rceil off by one
each way (37 / 34), NC16 amax without NaN (5), NC17 recip 0 for byte 255 (10), NC18 amax over 2 lanes (43), NC19 /
NC20 missing barriers (45 / 47), NC21 scale packing (39), NC22 recip + 1 ulp (44). Order mutations NC01-NC05 also
fail check_orders.

## 6. Special cases
- No GEMM, no PDL (no griddepcontrol / programmatic launch). x and w are read with `ld.global.nc`; safe here since
  nothing writes them during the kernel and outputs are fresh allocations.
- Device: allocations on `x.device`, launch on the TVM-FFI env stream (= torch current stream; verified by the
  stream test); tested on cuda:0 only.
- Out of spec (`check_misc.py`, logs/misc.log): EXACT with deterministic algorithms off as well; 16-byte aligned
  views with storage offsets EXACT; misaligned / non-contiguous / wrong dtype / wrong shape raise ValueError (the
  reference would clone or handle them; the spec fixes contiguous inputs, the 30B kernel cloned misaligned ones).

## 7. Timing (CUDA graph, gpu0.lock held only around the timed loop)
`check_timing.py` (logs/timing_run1.log, timing_run2.log): 8 calls per graph on 8 rotating input sets, outputs
live, reference and candidate interleaved over 30 rounds x 10 replays; two runs agree.

| | reference us/call (p10-p90) | candidate us/call (p10-p90) | speedup |
| --- | --- | --- | --- |
| 8 rotating inputs/graph, run 1 | 42.29 (42.18-42.64) | 28.65 (28.58-28.87) | 1.476x (per round 1.468-1.503) |
| 8 rotating inputs/graph, run 2 | 42.34 (42.26-42.41) | 28.68 (28.62-28.74) | 1.476x (per round 1.471-1.482) |
| 1 input/graph, run 1 | 45.99 (45.67-46.69) | 30.98 (30.78-31.19) | 1.484x |

Saving: (42.29 - 28.65) us x 94 calls = **1.28 ms/step**. Other verifiers' unlocked correctness jobs shared GPU 0
(utilization 0-9% sampled); interleaving and the agreement of both runs make the ratio robust.

## Verdict: PASS
No findings the writer must fix. Note for the record: misaligned or non-contiguous inputs raise ValueError instead
of matching the reference (outside the spec).
