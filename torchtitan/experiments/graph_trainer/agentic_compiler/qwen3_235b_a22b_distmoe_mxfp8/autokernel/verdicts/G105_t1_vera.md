# G105 dex t1 verification (vera): PASS

- Candidate: W/autokernel/run1/candidates/G105/dex/t1/kernel.py, sha256
  3e8f5000c7b987619ff72db61624c0b4c8b54369aef04c69fd407851df896172 (matches HANDOFF.md; copied here as kernel.py)
- Spec: pat_r5yf.../specialization_acsrbg...json (copied here as spec.json, byte-identical); reference.py sha256
  b8129b929a501d1d... (matches the board's immutable hash). No current kernel_name file exists in the pattern dir.
- Op: out[75497472] bf16 = torch._chunk_cat([cat(split(w2g,512,1)), cat(split(w13g.view(4,2,1536,4096),192,2))], 0, 8)
  (bucketing._pre_bucket_reduce_scatter on this build). The candidate is a pure 32 B-vector copy of 96 contiguous
  1.5 MiB blocks (grid 48 x 96, 256 threads, 4 x LDG.E.NA.ENL2.256.CONSTANT then 4 x STG.E.ENL2.256; SASS in
  dump/sass.txt; no PDL / griddepcontrol, no shared memory, no barriers).
- GPU: checks 1-2 and the first sanitizer pass ran on GPU 0 under gpu0.lock (before the 16:58 rule change); the
  rest on GPU 2 (CUDA_VISIBLE_DEVICES=2), timing under gpu2.lock. GB300 sm_103, torch 2.16.0a0+git3c53c6f.

All commands are run from /tmp with E = this directory, W = the experiment root.

## 1. Contract: PASS
- `sha256sum kernel.py` -> 3e8f5000...6172 (= handoff).
- Fresh process, `python -I -B`, no GPU visible, importlib from E/kernel.py: imports standalone; `run(in0, in1)`
  matches the spec signature `def run(in0, in1)` (import_standalone.log).

## 2. Official gate and contract details: PASS
- `python W/autokernel/harness.py <cache spec> E/kernel.py --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G105.pt`
  -> seeds 0-7 EXACT, real EXACT, PASS (harness_gate_cachespec.log; same with E/spec.json: harness_gate.log).
- `python E/contract.py` (contract.log), all OK:
  - output metadata equal to the reference and spec: shape [75497472], stride [1], bf16, cuda:0, storage_offset 0,
    storage nbytes 150994944, contiguous; output aliases neither input (same as reference);
  - inputs bitwise unmutated; 3 repeated calls bitwise identical and each returns a fresh allocation;
  - 32 B-aligned storage-offset views of either input (offset 16 and 256 elements): EXACT;
  - out-of-spec inputs raise ValueError (16 B- and 2 B-misaligned views, transposed, fp32/fp16, 4-D view, CPU,
    swapped inputs) -- never a silent wrong answer;
  - non-default stream: inputs rewritten on a side stream behind a ~100 ms `_sleep`, candidate called on that
    stream sees the new values, and its output read on that stream is complete (runs on torch's current stream);
  - real input: metadata = spec, storage offset 0, EXACT.

## 3. Memory: PASS
- Poisoning: every adversarial case (sec. 5) runs the candidate twice with its output block pre-filled with 0xFF and
  then 0x5A (bait reuse verified by data_ptr equality every time; no "bait not reused" warning), compared bytewise
  to the reference: 0 differing bytes in all 17 cases, so every output byte is written. The decode test also shows
  the output is an exact permutation of all 75497472 source elements (each written once). The harness gate
  poisons too.
- compute-sanitizer (driver: sanitize_driver.py, PYTORCH_NO_CUDA_MEMORY_CACHING=1, `--report-api-errors no`
  because cuda-python's cuGetProcAddress probing raises 34 host-side INVALID_VALUE API errors that are unrelated
  to device memory; logs in sanitizer/):

  | Tool | Candidate | Planted-bug copy (negative control) | Control result |
  |---|---|---|---|
  | memcheck `--padding 4096` | 0 errors (cand_memcheck_padding4096.log; also 0 without padding) | nc/nc_offbyone_src_vec.py (32 B over-read) | 2 errors: Invalid __global__ read of size 32 (end of in0 and of in1) |
  | memcheck `--padding 4096` | | nc_san/nc_mem_offbyone_dst_vec.py (32 B over-write) | 1 error: Invalid __global__ write of size 32 |
  | racecheck `--racecheck-report all` | 0 hazards | nc_san/nc_race_smem_no_barrier.py (smem write/read across warps, no barrier) | Potential RAW hazards reported (stopped after the reports; >10 min run) |
  | synccheck | 0 errors | nc_san/nc_sync_bad_warp_mask.py (`bar.warp.sync 0xffff` from all 32 lanes) | 168288 "Barrier error detected. Invalid arguments." |
  | initcheck | 0 errors | nc_san/nc_init_v8_load_from_output.py (the kernel's own 256-bit nc loads read the unwritten output) | Uninitialized __global__ memory read of size 32 |
  | initcheck | | nc_san/nc_init_b32_volatile_from_output.py | 2359296 uninitialized 4 B reads |

  The memcheck/initcheck controls use the candidate's exact 256-bit LDG/STG instructions, so the tools do
  instrument them. First attempts that did not exercise the tools are kept in sanitizer/superseded/ (README.txt).
  Racecheck/synccheck are vacuous for the candidate itself (no shared memory, no barriers) but were run.

## 4. CUDA graphs: PASS
`python E/graphs.py --first-in-capture` and `python E/graphs.py` (graphs.log). One graph: neighbour copies write
both inputs from static sources -> call 1 -> neighbour copies the output to a sink -> neighbour xors in0 in place
(int16 ^ 0x5555) -> call 2 -> neighbour copies call 2's output to a second sink. 40 replays with the static sources
rewritten in place between replays (random 16-bit patterns, position patterns, all-sNaN / all -0 / qNaN / min
subnormal): 0 mismatches against the eager reference for both sinks and both graph-owned outputs (distinct
buffers). With --first-in-capture the process's very first candidate call, including the CuTe JIT compile, happens
inside the capture and works. Eager calls after graph use are EXACT.

## 5. Adversarial layout probes: PASS (all negative controls caught)
`python E/adversarial.py E/kernel.py` (adversarial.log), each case with the two-poison check and input-mutation check:
- pos_lo / pos_hi / pos_mix: element bit pattern = f(global source index). pos_lo covers all 65536 bf16 bit
  patterns (every NaN payload incl. signalling NaNs, +-inf, -0, every subnormal, max finite) at ~1152 positions
  each: EXACT, so bytes are moved, not converted.
- decode: (hi << 16 | lo) of the candidate output equals an independently derived source-index map (plain
  split/cat + explicit chunk-major concatenation; the reference's decode equals the same map): 0 wrong positions;
  the output is a permutation of all source elements.
- rand_bits seeds 0-3 (uniform 16-bit patterns), const zero / -0 / qNaN / sNaN / negative NaN payload / -inf /
  min subnormal / max finite / 0xFFFF, block_edges (first/last element of every 1.5 MiB source block set to
  +-1, sNaN, -0): all EXACT.
- No reductions in this op, so section 5's order probes reduce to the layout probes above.

Negative controls (`python E/make_ncs.py`, then `adversarial.py nc/<file> --expect-fail`, negative_controls.log):

| Broken copy | Caught by |
|---|---|
| order_w13_h_major (w13 blocks h-major) | pos_hi, pos_mix, decode, rand_bits |
| order_w2_shard_xor1 (w2 blocks from the wrong shard) | pos_hi, pos_mix, decode, rand_bits |
| order_words_reversed (word order inside each 32 B vector) | pos_lo, pos_mix, decode, rand_bits, block_edges |
| offbyone_rotate_vec (thread t copies thread t+1's vector) | pos_lo, pos_mix, decode, rand_bits, block_edges |
| offbyone_src_vec (32 B source over-read) | illegal memory access (also memcheck, sec. 3) |
| missing_last_cta (last 32 KiB of each block unwritten) | every case incl. const_* (poisoning), decode, perm |
| missing_last_unroll_store (1/4 of the stores skipped) | every case (poisoning), decode, perm |
| convert_mul_one (each bf16x2 through mul.rn.bf16x2 by 1.0: NaN canonicalization) | pos_lo, pos_mix, rand_bits, const_qnan/snan/nan payload/0xFFFF, block_edges, decode, perm |

Note: the harness's randn inputs contain no NaNs, so the gate alone would not catch the NaN-canonicalizing copy;
pos_lo / rand_bits do.

## 6. Special cases
- Not GEMM-adjacent, no PDL. The 256-bit `.nc` (LDG ...CONSTANT) loads read only the inputs, which are written by
  earlier kernels on the same stream (in training: dist_moe.block_scaled_backward outputs [2], [3]); nothing writes
  them during the kernel, and the output is a fresh allocation that cannot alias them.
- Multi-device: run on cuda:0 inputs only (one visible GPU). The code has no hard-coded device index: it compiles
  once per `in0.device.index` (`_CACHE`), launches under `torch.cuda.device(in0.device)` on the tvm-ffi env stream
  (torch's current stream, verified on a side stream). Not exercised: inputs on a device other than the current one.
- Domain: requires contiguous, 32 B-aligned cuda bf16 inputs of the spec shapes, otherwise ValueError (the reference
  accepts any). The real input and the training graph give contiguous standard-stride tensors from a custom op,
  so it is not reached in this run.

## 7. Timing: CUDA graph, rotating inputs, live outputs, interleaved with the reference (GPU 2, gpu2.lock)
`flock W/autokernel/run1/gpu2.lock env CUDA_VISIBLE_DEVICES=2 python E/timing.py`, 2 runs (timing.log). Graph of 4
calls on 4 distinct 144 MiB input sets with all outputs alive; 200 replays alternating with the reference graph:

| | median us/call | p10 | p90 |
|---|---:|---:|---:|
| reference (4-call graph) | 384.11 / 383.86 | 383.38 | 384.87 |
| candidate (4-call graph) | 46.46 / 46.46 | 46.42 | 46.97 |
| reference (1-call graph, harness-style) | 388.54 / 388.51 | 386.43 | 390.66 |
| candidate (1-call graph) | 50.62 / 50.59 | 48.61 | 50.75 |
| torch `copy_` of 144 MiB, same 4-call setup | 47.01 / 47.17 | 46.70 | 47.45 |

Speedup 8.27x; saving (384.1 - 46.5) us x 94 calls/step = **31.7 ms/step**. The candidate moves 288 MiB in 46.5 us
(6.5 TB/s) -- at the rate of torch's own device copy on this GPU, i.e. at the copy roofline. One other verifier's
correctness process was resident on GPU 2 during timing (allowed by the revised rules); the two runs agree to 0.1%.

## Findings for the writer
None blocking. Behaviour outside the spec, by design: misaligned or non-contiguous inputs raise ValueError instead
of computing the reference result.
