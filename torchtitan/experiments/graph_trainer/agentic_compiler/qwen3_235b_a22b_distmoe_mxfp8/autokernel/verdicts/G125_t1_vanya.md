# G125 dex t1 -- verifier vanya -- PASS

- Problem: pat_3h2bcwq5ae3gotghbz545xeht2dwfsw36sp5bs7iwqrjztw3cev / specialization azl3vp... (G125 bwd,
  `bucketing._pre_bucket_reduce_scatter` of 7 grads into a 256-chunk bucket, 94 calls/step)
- Candidate: candidates/G125/dex/t1/kernel.py, sha256 fbc5449c92ac7f7004b9ca0ca70def1e51dc1e6871d55a9389733fc762a155cf
  (copied to cand.py; all runs used the copy)
- Reference: reference.py sha256 a9efa9f4e30e07a7... (matches the board's immutable hash). The eager op is
  `torch._chunk_cat(ins, 0, 256).flatten()`; `unpadded_input_indices` only affects the fake impl.
- No current kernel_name file in the pattern dir: compared against reference.py only.
- GPU: checks 1-2 and the CUDA-graph check ran on GPU 0 under gpu0.lock (original brief); after the 16:58 revision,
  the rest ran on GPU 2 unlocked, timing on GPU 2 with gpu2.lock held only around the timed loop.

## 1. Contract -- PASS (logs/contract.log, logs/check_main.log)
- sha256 of the candidate == handoff's claim. `inspect.signature(run)` == spec `(in0, ..., in6)`.
- Fresh process (`python -I`, cwd /tmp) imports cand.py standalone (no GPU work at import).
- Output vs reference: shape [72360448], stride (1,), bf16, cuda:0, storage_offset 0, storage nbytes 144720896,
  contiguous, requires_grad False, strided -- identical. The output overlaps no input storage; each call returns a
  fresh storage (as the reference). Inputs byte-identical after every call (checked in every case). Repeated calls
  (plain, plain, poisoned) identical.

## 2. Official gate -- PASS (logs/harness_seeds0-7_real.log)
`CUDA_VISIBLE_DEVICES=0 flock gpu0.lock python -I harness.py spec.json cand.py --seeds 0 1 2 3 4 5 6 7 --real real_inputs/fusion_G125.pt`
-> seed0..seed7 EXACT, real EXACT, PASS (harness also runs two poisoned candidate calls, 0xFF and 0x5A, per case).

## 3. Memory -- PASS
- Poisoning (scripts/check_main.py): every candidate call in the suite runs on an output block pre-filled with
  0xFF / 0x5A / 0x80 / 0xA5 (cycled) and asserts the allocator reused that exact block. Padding (128 chunks x
  (in2 + in3 + in4 row) = 524544 elements) checked to be 0x0000 (+0) in every case. Positive control: nc2/nc3
  (padding stores removed) give the right answer on fresh zeroed memory but fail every poisoned case and the harness.
- compute-sanitizer (scripts/sanitize.sh, logs/sanitizer_summary.log, sanitizer/*.raw), GPU 2,
  `--report-api-errors no` (see note):
  | tool | candidate | planted bug (copy) | planted result |
  | --- | --- | --- | --- |
  | memcheck, PYTORCH_NO_CUDA_MEMORY_CACHING=1 | 0 errors | nc5: in6 source index +1 word (reads past in6 for chunk 255) | 1 error, Invalid __global__ read of size 4 |
  | memcheck, caching allocator | 0 errors | - | - |
  | initcheck, no caching | 0 errors | san_initread: in2/in3 padding word read back from the uninitialized output | 128 errors (one per padding chunk); output still bitwise equal on zeroed fresh memory, so only initcheck/poisoning see it |
  | racecheck | 0 hazards | san_race: tail CTA smem write/read across warps without barrier | 2 hazards |
  | synccheck | 0 errors | san_sync: CTA barrier from half of every warp in the tail CTA | 21184 errors, Divergent thread(s) in warp |
  Note: without `--report-api-errors no`, memcheck lists 34 "errors" that are all host-side
  `cuGetProcAddress_v2 -> CUDA_ERROR_INVALID_VALUE` from cuda-python symbol probing inside `cute.compile`
  (cand.py:141 -> dsl.py _prepare_compilation -> get_compute_capability_major_minor); no device error
  (sanitizer/cand_memcheck_nocache_apierrors.raw). The candidate kernel uses no shared memory and no barriers, so
  racecheck/synccheck are trivially clean; the planted controls show the tools work on this kernel's code path.

## 4. CUDA graphs -- PASS (scripts/check_graph.py, logs/check_graph.log; GPU 0)
Fresh process; the very first candidate call (incl. the CuTe compile) is inside `torch.cuda.graph` capture: OK.
Graph: producers copy sources -> inputs, call 1, consumer copies output, producers build call-2 inputs from call 1's
output (bitwise NOT of slices), call 2, consumer copies output. 24 replays with the sources rewritten in place
(random 16-bit patterns, all -0, specials): 24/24 bitwise equal to the eager reference for both calls and both
consumer copies. Eager call on a side stream behind a ~25 ms `torch.cuda._sleep` producer: exact (kernel runs on the
current stream; tvm-ffi env stream).

## 5. Adversarial values and layout proof -- PASS (scripts/check_main.py, logs/check_main.log: 40/40 OK)
- Independent layout model (no `_chunk_cat`): for each input, chunk c = flat [c*w, (c+1)*w) with w = ceil(d0/256)
  x row size, beyond-the-end = padding; outputs = concat over inputs per chunk. The reference itself is checked
  against this model in every case (equal everywhere, incl. NaN payloads).
- Position encoding: pass A = low 16 bits of each element's flat index (all 65536 bf16 patterns: every NaN payload,
  +-inf, -0, subnormals), pass B = 0x8000 | input id << 10 | index >> 16. Decoding (A, B) from the candidate's
  output gives the (input, flat index) of every one of the 72360448 output elements: 0 mismatches vs the model;
  every source element appears exactly once; padding decodes as (0, 0) = +0 bytes.
- Small grads explicitly: chunks 0, 1, 126, 127, 128, 129, 254, 255 printed for in2, in3, in4 first/last, in5
  first/last, in6 first/last: in2/in3/in4 present through chunk 127 (in2[c] low half, in3[c] high half of the shared
  word), +0 from chunk 128; in5/in6 16 elements per chunk through chunk 255 (in6[4080..4095] at the last 16 output
  elements).
- Values: hashed patterns, all -0 (padding must stay +0), all +0, all 0xFFFF, all 1.0, NaN payload sweep, +-inf /
  max / min subnormal / -0 cycle, one-hot rows, 6 seeds uniform 16-bit, 6 seeds randn x 2^U(-130,130), real inputs:
  all bitwise equal to reference and model.
- Placement: 4-byte-aligned views at nonzero storage offsets (in2/in3 at odd element offsets), aliased inputs
  (in3 is in2, in6 is in5): exact. 2-byte-misaligned in0/in1/in4/in5/in6 and non-contiguous in0 raise ValueError
  (the reference accepts them; see findings).
- No reductions (pure copy): no order probes apply.
- Negative controls (negctl/, logs/negctl_suite.log): all 10 caught by my suite AND by the harness:
  nc1 in2/in3 halves swapped (256 elems), nc2 in4 padding not written, nc3 in2/in3 padding word not written,
  nc4 chunk boundary 127 instead of 128, nc5 in6 source off by one word, nc6 in5 destination one word early,
  nc7 store pair swap within each tile, nc8 bf16 halves rotated for in5, nc9 chunk stride one word short,
  nc10 last in1 CTA dropped.

## 6. Special cases
- Not GEMM-adjacent, no PDL (plain launch; `ld.global.nc` only on inputs, which nothing writes during the kernel).
- Multi-device: ran on cuda:0 / single visible device only. Code: compile cache keyed by `device.index`, launch
  inside `torch.cuda.device(in0.device)` on that device's current stream, all inputs required on in0's device, no
  SM-count or device-0 assumption found.

## 7. Timing (scripts/time_graph.py, scripts/time_when_idle.sh, logs/timing_idle.log, logs/timing.log)
8-call CUDA graph per implementation rotating over 4 input sets (145 MB each, > L2), outputs live, ref/cand replays
interleaved, 60 rounds, GPU 2, gpu2.lock held only around the timed loop (compile/capture before it, unlocked).
Final numbers are idle-gated: each run started only after GPU 2 read 0 % utilization in 5 samples over 2.5 s
(other verifiers' correctness runs on GPU 2 take no lock).
| run | reference us/call median (p10-p90) | candidate us/call median (p10-p90) | saving ms/step (x94) |
| --- | --- | --- | --- |
| idle 1 | 88.13 (87.54-90.56) | 44.35 (44.16-44.59) | 4.12 |
| idle 2 | 87.97 (87.53-89.32) | 44.35 (44.12-44.60) | 4.10 |
| idle 3 | 88.10 (87.65-88.54) | 44.32 (44.15-44.58) | 4.12 |
Result: reference 88.1 us vs candidate 44.35 us per call (1.99x); saving 43.8 us x 94 calls/step = 4.11 ms/step.
Earlier contended runs (logs/timing.log, GPU 2 at 100 % from other processes): candidate 44.3-44.4 every time,
reference 87.9 / 93.0 / 2425 (the last discarded: reference replays preempted by another context).
Writer's harness single-call graph 89.0 -> 48.9 us (3.8 ms/step) is consistent (single-call graphs include more
launch/drain overhead). Candidate effective bandwidth: 289 MB / 44.35 us = 6.5 TB/s.

## Findings (none blocking)
- 2-byte-misaligned in0/in1/in4/in5/in6 (and non-contiguous inputs) raise ValueError where the reference works.
  The specialization key does not include alignment; real captured inputs are separate storages at offset 0, and the
  writer documented the limit. A misaligned grad in training would fail loudly, not silently.
