# G111 dex t1 verification (vito): PASS

- Candidate: `W/autokernel/run1/candidates/G111/dex/t1/kernel.py`, copied to `cand.py`.
  sha256 `09f4ac990f7fb75cd56ffd545352f9294d34bf3cb7308b03792601b0b1b52970`, which matches HANDOFF.md.
- Problem: `pat_5dmq.../specialization_av3635...json` (reference.py sha256 prefix a86cc87ba4ba2b7d, matching the board).
  There is no current kernel_name file, so the candidate is compared against reference.py.
- Op, read from reference.py: `split(in0 [151936,4096] bf16, 594)` gives 256 shards. 255 of them go through
  `constant_pad_nd(.., [0,0,0,0], 0.0)` and the last (466 rows) through `[0,0,0,128]`. The shards are then
  concatenated on dim 0 into [152064,4096]. Net effect: the in0 bytes followed by 128 rows of +0.0. Each suite case
  checks this statement directly, so the reference itself is checked too.
- W = /home/bahuang/local/torchtitan/outputs/agentic_compiler/qwen3_235b_a22b_distmoe_mxfp8. E = this directory.
  All runs used a single visible device (cuda:0). The gate, suite, negative controls, graph tests, the first four
  sanitizer runs and the SASS dump ran on GPU 0 under gpu0.lock. After the 16:58 rule change, the
  synccheck/initcheck controls and the graph negative controls ran on GPU 2 without a lock. Timing ran on GPU 2
  under gpu2.lock.

## 1. Contract: OK
`python E/scripts/contract.py E` -> logs/contract.log
- `def run(in0)` matches the spec signature.
- A fresh process imports cand.py standalone. It uses only torch, cutlass and numpy. It imports nothing from
  kernel_cache and has no reference fallback: the words "reference" and "constant_pad" occur only in the docstring.
- It has no PDL (no griddepcontrol and no use_pdl). It raises ValueError for any input outside the spec.

## 2. Official gate: PASS
`CUDA_VISIBLE_DEVICES=0 python W/autokernel/harness.py <spec> E/cand.py --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G111.pt --time`
-> logs/harness.log
- seed0-7 and real are all EXACT.
- Harness timing: baseline cuda-graph 1199.8 us, candidate 357.8 us.

The suite `python E/scripts/check_g111.py E/cand.py` (logs/check_g111_cand.log) gave **171/171 OK**. It checks the
following for 11 input kinds:
- Output metadata equals the reference's: type, shape, stride, dtype, device, storage_offset 0, untyped-storage
  nbytes, contiguity, requires_grad, `_base`, layout.
- The output storage does not overlap the input.
- The input is unmutated.
- 2 repeated calls are identical and equal to the reference.

## 3. Memory: OK
**Poisoning.** The suite poisons every case 4 ways before the candidate runs:
- the output-sized block is filled with u8 0xFF;
- filled with u8 0x5A;
- filled with bf16 -0 (0x8000; this hides an unwritten zero tail from float compares, but not from byte compares);
- `fill_uninitialized_memory` NaN fill.

For the first 3 the suite checks that the candidate's output pointer equals the poisoned block's pointer, i.e. the
block was really reused. Reuse was True in all 33 checked cases (11 kinds x 3 fills).

**compute-sanitizer** (`PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool T python E/scripts/san_run.py E/cand.py`).
san_run.py writes the input by host-to-device memcpy and checks the result on the host, so the candidate's kernel
is the only device kernel in the process.

| Tool | Result on cand.py | Log |
| --- | --- | --- |
| memcheck | 0 errors, EXACT | sanitizer/cand_memcheck_noapi.log |
| racecheck | 0 hazards | sanitizer/cand_racecheck.log |
| synccheck | 0 errors | sanitizer/cand_synccheck.log |
| initcheck | 0 errors | sanitizer/cand_initcheck.log |

The first memcheck run without `--report-api-errors no` (sanitizer/cand_memcheck.log) lists 34 host-side
`cuGetProcAddress_v2` invalid-argument return codes. They come from cuda-python symbol probing during the CuTe
compile. None of them is a device memory error.

**Planted-bug controls** (copies built by scripts/make_sanctl.py). Each tool flags its bug:

| Control | Planted bug | What the tool reported |
| --- | --- | --- |
| ctl_memcheck_oob_read | src + 32 B | memcheck: "Invalid __global__ read of size 32 bytes by thread (255,0,0) in block (37983,0,0)", i.e. exactly the planted byte range |
| ctl_memcheck_oob_write | grid + 1 CTA | memcheck: invalid __global__ write in block 38016, 160 errors |
| ctl_racecheck_smem_race | smem cross-warp RAW with no barrier | racecheck: 32769 hazards |
| ctl_synccheck_divergent_barrier | bar.sync on even lanes only | synccheck: "Barrier error detected. Divergent thread(s) in warp", 3616 errors |
| ctl_initcheck_uninit_read | zero CTAs read the uninitialized output | initcheck: 8192 uninitialized 32-byte reads |

The initcheck control's output still compared EXACT, because fresh cudaMalloc memory reads as zero. That is why the
poisoning in the suite is needed.

## 4. CUDA graphs: OK
`python E/scripts/check_graph.py E/cand.py {first_in_capture,warm}` -> logs/graph_first_in_capture.log and
logs/graph_warm.log

The graph, captured on a side stream:
1. producer copy_ writes in_a, then cand(in_a), then a consumer copies out;
2. producer writes in_b, then cand(in_b), then a consumer;
3. cand(in_a) again, then a consumer.

That is 3 candidate calls in one graph, with 3 distinct output buffers.
- In first_in_capture mode the very first candidate call of the process (including the CuTe compile) happens
  inside the capture. Capture succeeded.
- 20 replays per mode, all EXACT. Before each replay the sources were rewritten in place (position^replay,
  random bits, all -0, all 16-bit patterns, real input) and all 3 graph-owned outputs and 3 consumer buffers were
  poisoned (0xA5/0xFF).

Graph negative controls (negctl/graph_*.log) all MISMATCH, as they should:
- nc05 (zero tail not written);
- nc07 (last zero CTA missing);
- nc13 (kernel launched on the default stream).

## 5. Adversarial values (layout probes): OK
There are no reductions, so the order probes become layout probes. Input kinds:
- word32_index: 4-byte word i holds i.
- word32_index_hi: word i holds ~(i x 2654435761).
- hash64: 8-byte word i holds splitmix64(i). Every position is unique, so any word-, vector- or CTA-granular
  misplacement shows up.
- all16: every 16-bit pattern, i.e. all quiet and signalling NaN payloads of both signs, +-inf, +-0, all subnormals.
- specials17: 21 specials cycled with an odd period.
- negzero: all -0.
- poszero: all +0.
- snan_all: every element 0x7F81.
- onehot_bounds: distinct NaN payloads at the shard boundaries 593/594 and 151469/151470, the last row, and CTA
  boundaries.
- randn11.
- real: the captured gradient.

The side-stream test runs a slow producer of in0 (`_sleep` then copy_), then the candidate, then a consumer, all on
a non-default stream. It is EXACT.

Outside the spec:
- A contiguous view with a 32 B storage offset is EXACT.
- A 2 B offset, non-contiguous input, float32 input and CPU input each raise ValueError.

**Negative controls** (scripts/make_negctl.py; negctl/summary.log): all 13 FAIL the suite.

| Control | Broken behaviour | Suite result |
| --- | --- | --- |
| nc01 | src off by one vector | 5 FAIL, then the kernel hit an illegal address |
| nc02 | last copy CTA writes zeros | 52 FAIL |
| nc03 | 32 B lane-pair swap | 42 FAIL |
| nc04 | bf16 halfword swap | 42 FAIL |
| nc05 | zero tail not written | 56 FAIL (poison) |
| nc06 | last unrolled store dropped | 56 FAIL |
| nc07 | last zero CTA missing | 56 FAIL |
| nc08 | tail written as -0 | 57 FAIL |
| nc09 | body via fp32 round trip (quiets sNaN) | 37 FAIL |
| nc10 | body + 0.0 (-0 -> +0) | 42 FAIL |
| nc11 | output with storage offset | 77 FAIL (metadata) |
| nc12 | kernel writes into in0 | 45 FAIL |
| nc13 | launch on the default stream | 1 FAIL (side-stream test) |

## 6. Special cases: OK
- No GEMM and no PDL.
- SASS (sass/g111_cand.sass, from cubin kept with CUTE_DSL_KEEP=ptx,cubin):
  - each thread issues 4 x `LDG.E.NA.ENL2.256.CONSTANT` and 4 x `STG.E.ENL2.256`;
  - CTAs > 37983 (0x945f) only `STG ... RZ, RZ` (zeros);
  - no barriers, no griddepcontrol.
- The `.nc` loads read in0, which the previous kernel in the chain writes. Without PDL the kernel boundary makes
  those writes visible. This was exercised by the graph producer node that writes in0 right before the call, and by
  the side-stream producer. All EXACT.
- The compiler interleaves loads and stores. That is harmless because out is freshly allocated and never overlaps
  in0 (checked).
- Device assumptions: the compile cache is keyed by `in0.device.index`, the launch runs under
  `torch.cuda.device(dev)`, and the stream comes from the TVM-FFI environment (torch's current stream; verified on
  a non-default stream and under capture). Only cuda:0 was tested.

## 7. Timing (CUDA graph, rotating inputs, live outputs, interleaved): OK
`flock W/autokernel/run1/gpu2.lock env CUDA_VISIBLE_DEVICES=2 python E/scripts/bench.py E/cand.py 4 60`

Each graph holds 4 calls on 4 rotating inputs, with all 4 outputs live. Reference and candidate replays alternate.
Graph outputs were bytewise equal.

| Run | Reference us/call, median (p10-p90) | Candidate us/call, median (p10-p90) | Note |
| --- | --- | --- | --- |
| logs/bench_run3.log (used) | 1100.8 (1099.6-1101.7) | 354.7 (354.1-355.3) | quiet GPU |
| logs/bench_run2.log | 1101.7 (1100.1-1105.5) | 354.7 (353.9-356.4) | |
| logs/bench_run1.log | 1236.3 (1229.5-1482.7) | 353.8 (352.7-354.7) | GPU 2 at 100% from other jobs; discarded |

- Speedup 3.10x. The candidate moves 2.490 GB at 354.7 us, which is 7.02 TB/s, the HBM rate.
- **Saving: (1100.8 - 354.7) x 8 calls/step = 5.97 ms/step.**
- With the harness's 1-call graph instead: (1199.8 - 357.8) x 8 = 6.74 ms/step.

## Findings for the writer
None blocking. Design limits:
- Shape, dtype, contiguity and 32 B alignment are fixed by the spec. Any other input raises ValueError instead of
  falling back.
- The handoff's 6.7 ms/step uses the 1-call graph reference (1199.7 us). With rotating inputs and live outputs the
  reference is 1100.8 us, which gives 5.97 ms/step.
