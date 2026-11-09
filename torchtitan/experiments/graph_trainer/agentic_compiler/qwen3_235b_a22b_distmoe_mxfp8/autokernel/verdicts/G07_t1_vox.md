# G07 dex t1 verification (vox): PASS

- Candidate: `W/autokernel/run1/candidates/G07/dex/t1/kernel.py`, sha256
  `58fde907ec918c30331f2b6bb8f8d735165bc7cd4e638ae22397d5b6ab65c38b` (= handoff; copied to `cand.py` here)
- Problem: `pat_53aq.../specialization_a7jgyh...json` (loss backward per 1024-row chunk, [1024,151936] fp32 -> bf16),
  reference.py sha256 prefix 145e8f3eee52d9fd (= board). No `a7jgyh...py` kernel exists in the pattern dir, so the
  comparison is against reference.py.
- GPU: harness gate on GPU 0 under gpu0.lock (before the 16:58 rule change); everything else on GPU 2
  (`CUDA_VISIBLE_DEVICES=2`, device cuda:0 in-process), timing under gpu2.lock.
- Diff against the verified 30B kernel (Q/kernel_cache/pat_53aq.../aw7vap...py): `M = 4096 -> 1024`, NG comment,
  docstring, and the reference fallback replaced by `raise ValueError`. Nothing else.

## 1. Contract -- PASS (`logs/contract.log`, `scripts/contract.py`)

- sha256 matches the handoff; `def run(in0, in1, in2, in3, in4, in5)` == spec signature.
- Fresh `python -I` process in /tmp imports `cand.py` standalone (compiles at import), no cache_dir access.
- Output: shape [1024,151936], stride [151936,1], bf16, cuda:0, storage offset 0, own storage of exactly
  311164928 B, aliases no input (same as reference); requires_grad matches.
- Inputs unmutated (bytewise snapshot after reference + 7 candidate calls); 6 repeated calls identical and distinct.
- No fallback path: a TorchDispatchMode log of `cand.run` shows only `aten.empty.memory_format` and
  `aten.view.default`; the profiler shows exactly one CUDA kernel (LossBwd). The source has no reference path:
  unsupported inputs raise `ValueError` (tested: non-contiguous in5, in5 misaligned by 4 B, bf16 in5, int32 in3,
  short in3, int32 in1, CPU in0, wrong in2 shape). Row-chunk views of a [2048,V] / [2048] buffer (storage offset,
  still 32 B / 8 B aligned, as a chunk of the full [8192,V] logits would be: 1024*V*4 % 32 == 0) are EXACT.
- in2 and in4 are not read, proven on the reference itself: in2 = NaN / -inf / in5, in4 = NaN / 0 / -3.5 leave the
  reference output bitwise unchanged, and the candidate matches.

## 2. Official gate -- PASS (`logs/harness.log`)

```
flock W/autokernel/run1/gpu0.lock env CUDA_VISIBLE_DEVICES=0 python W/autokernel/harness.py spec.json cand.py \
  --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G07.pt --time
seed0..seed7: EXACT; real: EXACT; PASS
baseline: eager 696.4 us, cuda-graph 650.9 us; candidate: eager 159.5 us, cuda-graph 135.1 us
```
(The harness also checks metadata and alias structure and reruns the candidate with 0xFF / 0x5A poisoned output
memory.) The harness's seeds never produce -100 targets and use in1 in [1, 2^20); the suites below cover the rest.

## 3. Memory -- PASS

- Poisoning (`scripts/poison.py`, `logs/poison.log`): the block the output gets is pre-filled with 0x00, 0xFF, 0x5A,
  0x80, 0x7F, the bitwise NOT of the correct output (every stale byte wrong), and the NaN fill of
  `fill_uninitialized_memory`; block reuse verified by pointer equality each time; all EXACT.
  Control N13 (last CTA not launched): all 7 cases FAIL (4096 elements = 512 groups x 8).
- compute-sanitizer 2025.3.1 with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (each tensor its own cudaMalloc, exact bounds),
  `scripts/run_san.sh` + `scripts/san_driver.py` (2 candidate calls, NaN/-inf in in5, -100 rows, targets at columns
  0 and V-1 and the last element, then torch reads every output byte for the compare): `sanitizer/cand_*.log`

  | Tool | Filter | Candidate | Planted-bug control | Control result |
  | --- | --- | --- | --- | --- |
  | memcheck | kns=LossBwd | 0 errors, EXACT | N15 one extra CTA (reads past in5/in3, writes past out) | 226 errors ("Invalid __global__ read of size 32 bytes ... 1,025 bytes after ... allocation of size 622,329,856") |
  | racecheck (report all) | kns=LossBwd | 0 hazards | N16 all threads st/ld one smem word, no barrier | WAW hazards reported (20 printed; run stopped after that, it was flagging every CTA) |
  | synccheck | kns=LossBwd | 0 errors | N17 `bar.warp.sync 0x1` (mask excludes executing lanes) | 1632 "Barrier error detected. Invalid arguments" |
  | initcheck | none (torch's compare kernel reads all output bytes) | 0 errors | N13 last CTA missing | 1024 "Uninitialized __global__ memory read" in torch's compare kernel |

  The kernel uses no shared memory (cubin `SHARED:0`, no LDS/STS in SASS), so racecheck is structurally clean.
- Out of contract, memory safety only: invalid labels (V, V+7, -1, -101, 2^31, 2^40, 2^32+5, 2^32-100, INT64_MIN/MAX)
  under memcheck: 0 errors (`sanitizer/invalid_targets_memcheck.log`).

## 4. CUDA graphs and streams -- PASS (`scripts/graphs.py`, `logs/graphs.log`)

- Fresh process; the first candidate call of the process (first launch / module load) is inside capture.
- One graph: neighbour copies write in0/in1/in3/in5 -> call 1 -> consumer copies the output -> neighbours overwrite
  in5/in3 after the call (write-after-read) -> call 2 on a second input set -> call 3 on the refilled first set.
- 24 replays (17 with every source buffer changed in place: random in5 incl. NaN/inf/-0/subnormal entries, new
  targets with -100 rows, new in0 and in1 incl. |in1| up to 2^40 and negative), 3 calls each, consumers and the
  graph-pool outputs EXACT vs eager reference.py every replay; 10 back-to-back replays EXACT.
- Side stream with a 0.1 s spin before the producer and an overwrite after the call: EXACT (kernel follows torch's
  current stream).
- Controls: N13 (last CTA missing) fails all replays; N14 (launch on an unrelated stream) fails all replays and the
  side-stream test (the side-stream test needed a warm-up call and NaN-prefilled buffers to be sensitive; both are in
  the script).

## 5. Adversarial values -- PASS (`scripts/adversarial.py`, `logs/adversarial_{A,B,CDEF}.log`)

Every case is a full [1024,151936] call compared bytewise with reference.py on identical inputs.

| Section | Content | Calls | Result |
| --- | --- | ---: | --- |
| A | all 2^32 fp32 bit patterns of in5 (28 calls each) x 24 (in0, in1) settings: 1/8192 (real), 1/1024, +0/7, -0/3, +0/-3, 3.5/-2, 1e30/1, FLT_MAX/-1, qNaN, sNaN 0x7f800001, -NaN 0xffc12345, +-inf, 2/0, -2/0, 0/0, 1/INT64_MAX, 1/INT64_MIN, 1/(2^24+3), min-subnormal/1, FLT_MIN/3, 10/3, -7.25/123456789, 1/1; random targets, -100 every 8th row | 672 | EXACT |
| B | target path: 5,111,808 target-column values: stride-1024 sweep of all 2^32 patterns (4.19M), near-zero window \|x\| in [2^-40, 1) (524K), every pattern within +-2^16 ulps of exp overflow 0x42b17218, the normal-min boundary 0xc2aeac50 and the zero boundary 0xc2cff1b5; 12 finite g settings | 4992 | EXACT |
| C | every one of the 151936 columns as a target (37-stride sweep + 20 calls for the columns hidden by -100 rows), all 4096 (u, tid, residue) thread slots, all -100 (also with g = +0 and -0), all 0, all V-1, alternating -100/V-1, residues 0..7 in the first and last group, the last CTA's region | 180 | EXACT |
| D | 32 specials at the target (NaN payloads, +-inf, +-0, +-subnormal min/max, +-FLT_MIN/MAX, exp thresholds +-1 ulp, +-1, +-2^-24, +-0.5) x 24 g settings; all-equal rows (0, -11.931, -1e-30, 88, -104, -inf, inf, NaN); one-hot rows (0 at one column, -inf elsewhere) with the target on and off the hot column | 48 | EXACT |
| E | 64 random (in0, in1): random finite in0 bit patterns over all exponents, in1 of random magnitude 2^0..2^63, a third negative | 64 | EXACT |
| F | sum-order probe (below) | 24 + probe | EXACT |

Sum order (the writer's claim, re-derived from torch source at git 3c53c6fe and checked on the device):
- The reference has one reduction. `div` is scalar; `nll_loss_backward` (reduction=2, Sum) zero-fills grad_input and
  writes `grad_input[i, t_i] = -grad_output` for `t_i != -100` (`Loss.cu` nll_loss_backward_reduce_cuda_kernel_2d;
  total_weight only read for Mean); `_to_copy` is elementwise. `_log_softmax_backward_data` runs
  `cunn_SoftMaxBackward` (inner_size 1, dim 151936 > 1024, 607 KB > sharedMemPerBlock so not the Smem variant):
  per row `sum = blockReduce(ilpReduce(gradOutput, AddFloat, +0), Add, +0)`, epilogue `gO - exp(o) * sum`.
- That row sum is over v7 only. Checked on the device for all 24 settings: every v7 element except the valid targets
  is bitwise +0 and the target is bitwise -g (rows with -100 are all +0). A sum of one term x and +0s seeded with +0
  is x + 0 in any order (x = -0 -> +0; NaN -> canonical NaN), and torch's own row sum of v7 equals (-g) + 0 bitwise
  in every setting. So the output does not depend on the order, and the row count M does not enter any reduction.
- The only observable of the sum is the zero sign: with g = +0 the reference's target outputs are 0x8000 in every
  row and the candidate matches. Controls N01 (drop the +0 term) and N02 (seed with -0) are caught by A, B, D, F.

Negative controls (`negctl/*.diff`, `logs/negctl_matrix.log`; adversarial suite with `--stop-first`; all caught):
N01 sum without +0, N02 sum seeded -0, N03 target column +1, N04 unfused target epilogue, N05 cvt.rz in the main
path, N06 exp without the log2e low part, N07 ignore_index -101, N08 fixup only for the first group per thread,
N09 division via rcp*mul, N10 cvt.rz.f32.s64, N11 target read from the next row, N12 fixup before the group store
(program-order bug). Equivalence check X01 (main path uses the reference's `-(j - 12583039)` form): EXACT on the
full suite, confirming the writer's claim that the two forms are interchangeable.

## 6. Special cases

- Not GEMM-adjacent. No PDL: no griddepcontrol in the PTX or SASS (`sass/`). The `.nc` loads (LDG.CONSTANT) read only
  inputs that are complete before the launch (coherent across kernel boundaries); the graph and side-stream tests
  write them right before the call.
- Device: tested on cuda:0 only. The kernel is compiled at import for the current device, `out` is allocated on
  `in5.device`, the launch uses torch's current stream (via the tvm-ffi env stream). Inputs on a device other than
  the current one are untested here (the coordinator's multidev check covers that class).

## 7. Timing

`scripts/timing.py` (GPU 2, gpu2.lock held only around the timed loop, after compile, capture and warm-up): one
CUDA graph per implementation with 4 calls on 4 distinct input sets (set 0 = the real captured input), all 4
outputs alive in the graph pool, 80 interleaved replay rounds (order flipped every round), us per call = replay
time / 4. Outputs bitwise equal before and after the timed loop.

| Run | reference.py median (p10-p90) | candidate median (p10-p90) | speedup | saving at 8 calls/step |
| --- | ---: | ---: | ---: | ---: |
| `logs/timing_quiet.log` (GPU 2 idle, no other process) | 645.7 us (644.9-646.5) | 131.4 us (131.0-132.0) | 4.91x | 4.11 ms/step |
| `logs/timing_run2.log` (other processes present, idle) | 645.7 us (644.5-646.9) | 131.4 us (131.0-131.9) | 4.91x | 4.11 ms/step |
| harness 1-call graph, static input (GPU 0) | 650.9 | 135.1 | 4.82x | - |

Candidate effective bandwidth 7.10 TB/s (622 MB read + 311 MB write per call), i.e. at the HBM floor (writer's
floor estimate 134.5 us at 6.94 TB/s). No current kernel_name exists, so there is no third column. A first attempt
while other verifiers' sanitizer jobs kept GPU 2 at 100% (`logs/timing_contended_run0.log`) inflated only the
reference (1241 us; the 2.6 ms reference graph spans time slices, the 0.5 ms candidate graph does not) and is
discarded.

## Notes for the coordinator (none blocks promotion)

- Invalid labels (out of contract; the reference device-asserts): the kernel reads only the low 32 bits of each
  int64 target. Targets >= V or negative other than -100 write no target element (row still uses sum = -g);
  targets whose low 32 bits are a valid column (2^40, 2^32+5, INT64_MIN) are treated as that column, and
  2^32-100 as ignored. Memory-safe in all cases (memcheck 0). Same class as the 30B note.
- Misaligned in5 (data_ptr % 32 != 0) or in3 (% 8) raise ValueError instead of running; the specialization key does
  not include alignment, but chunk views of the logits/labels stay aligned.
