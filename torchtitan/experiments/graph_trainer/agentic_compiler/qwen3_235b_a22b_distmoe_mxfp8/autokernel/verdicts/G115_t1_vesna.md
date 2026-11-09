# G115 eve t1 -- independent verification by vesna

**Verdict: PASS**

- Problem: pat_pqleqc2y6eqf6b7uiuqloik6hkhxfzjthcync7fg4zhq3rxtv6x / specialization_ajkjsaxdkdclitogl37szwx2akkbqi5sphvqklfrfyoaxlquplrs
  (FSDP all-gather unshard permutation of w13 + dist_moe::prepare_mxfp8_weight; flat bf16 [50331648] ->
  e4m3 [4,3072,4096], FPROP e8m0 [12288,128], DGRAD e8m0 [16384,96]); 94 calls/step.
- Candidate: candidates/G115/eve/t1/kernel.py, sha256 0eb2bdcde73583b0acef18f66db731d0a17004186b4662ab0174514fc1bd90f4
  (copied to cand.py here; hash rechecked at the end, still identical). No current kernel_name file exists, so the
  comparison is against reference.py (sha256 73c376ed90dacff9..., matches the board).
- Timing (CUDA graph, GPU 2): reference 68.78 us/call (p10-p90 68.73-69.01), candidate 23.70 us/call (p10-p90
  23.48-23.78): 2.90x faster. Saving (68.78 - 23.70) x 94 = **4.24 ms/step**.
- Environment: GB300, torch 2.16.0a0+git3c53c6f, nvidia-cutlass-dsl 4.8.0, CUDA 13.0. The contract checks, the
  harness, the permutation test, the adversarial suite and controls N01-N07 ran on GPU 0 (under gpu0.lock, before
  the 16:58 rule change). Controls N08-N17, CUDA graphs, sanitizers and timing ran on GPU 2 (timing under
  gpu2.lock).

Scripts are in `scripts/`, logs in `logs/`, broken copies in `negctl/`. Python commands were run from /tmp with
`python -I`. Nothing was run from, or written to, cache_dir.

## 1. Contract

| Check | Command | Result |
| --- | --- | --- |
| sha256 | `sha256sum cand.py` | 0eb2bdcd...90f4 == handoff (logs/sha256.log) |
| signature, standalone import | `python -I scripts/contract.py <E>` (fresh process) | `def run(in0)` == spec; imports only torch + cutlass; import + both JIT compiles 2.3 s (logs/contract.log) |
| output metadata | `scripts/t_meta_perm.py` | shape/stride/dtype/device equal to the spec and reference; storage_offset 0; storage nbytes 50331648 / 1572864 / 1572864 (same as the reference); alias groups [0,1,2] for both; no output aliases the input (logs/meta_perm.log) |
| input not mutated | same | bitwise-unchanged after 5 calls, for both alignment paths |
| repeated calls | same | 5 calls identical, for both paths |

## 2. Official gate

`python W/autokernel/harness.py <spec> cand.py --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G115.pt --time`
gives seed0-7 EXACT and real EXACT, PASS (logs/harness.log). The harness checks with poisoned output memory
(0xFF and 0x5A bait plus NaN fill) and reruns. Its 1-call graph timing is 73.8 us for the reference and 28.4 us for
the candidate.

## 3. Unshard permutation (position-encoded inputs)

Input row i, column c has |x| = 2^E(tile) * (1 + m/8), where m = (i >> 3*(c%5)) & 7, and the sign mixes bits of
c and i. E(tile) = (tile*7919) % 241 - 120 varies per input 32x32 tile. Every input row is distinct, every tile
has a different scale from its neighbours, and q = +-(1+m/8)*2^7 exactly, so the qdata bytes encode the source row.

- Candidate == reference bytewise at ptr%32 == 0 (32 B load path) and ptr%32 == 16 (16 B load path).
- I decoded the candidate's qdata independently. All 12288 logical rows decode to the derived source row
  ((s*4+e)*2+h)*192+r for logical row j = h*1536+s*192+r of expert e, so 12288 distinct rows map exactly.
  The reference decodes to the same map.
- There are 241 distinct scale bytes in both FPROP and DGRAD, so a misplaced scale atom or pattern shows up
  (random-normal inputs have nearly constant scale bytes and would hide scale-layout bugs).

## 4. Adversarial suite (`scripts/t_adv.py cand.py`, logs/adv_cand.log): ALL EXACT

Every case ran at both offsets (ptr%32 = 0 and 16).
- **A** Every one of the 65536 bf16 bit patterns (both signs, subnormals, +-inf, all NaN payloads) is the
  unique tile amax at least once. Its position is swept over all 1024 tile positions, and the other elements
  are random with |v| <= |amax|. I wrote an independent scale-byte formula (s = A *rn fl32(1/448), e8m0 round-up,
  NaN -> 255). It agrees with the reference's FPROP tile bytes on 2 x 49152 tiles, with 0 disagreements.
- **B** Exhaustive (scale byte, x) pairs. For each of the 248 finite scale bytes and for both A_max(b) and
  A_min(b), the tiles contain every bf16 x with |x| <= A, both signs. That is 16555 tiles and 17.0M elements,
  and it covers every e4m3 rounding tie, saturation and subnormal output.
- **C** Specials with full 1024-position sweeps (50 types): NaN payloads 0x7FC0/0xFFC0/0x7F81/0xFF81/0x7FFF/
  0xFFFF/0x7FA5, +-inf, unique max of 10 magnitudes (fp32-max bf16, 448, 1, smallest subnormal, -x), all +0,
  all -0, +-0 with one subnormal or normal, all-NaN, all-inf, NaN+inf, inf+max, 10 all-equal tiles, and
  448*2^k +-1 ulp as the unique max. There are also one-hot-row tiles with the row index swept.
- **D** Random bits with per-tile exponent windows (width 0/1/2/8/30/254, so extreme exponents and
  subnormal-only tiles are included), the same with NaN/inf in 1/16 of tiles, and fully random bits.
- **R** Real input and randn*0.02 at both offsets.
- **F** Misaligned inputs (ptr%32 = 2, 4, 8, 24, which take the clone path) are EXACT. A stride-2 view (outside
  the spec) is EXACT: both the reference and the candidate run on it.
- Coverage: 249/249 reachable e8m0 bytes in both FPROP and DGRAD (0..247 and 255), and 255 e4m3 qdata bytes.
- Reduction order: the only reduction is the 32x32 max, which is exact. NaN propagation and the masking of the
  xorsign sign bit are probed by A and C: negative amax patterns, and NaN/inf at every position. The
  cross-warp smem partial and the lane shuffles are probed by the unique-max position sweeps. N06 and N07 below
  show that these sweeps catch a missing stage.

### Negative controls (all caught)

`scripts/make_negctl.py` writes broken copies. Run with `t_adv.py <copy> --stop`; results are in
logs/negctl_adv.log.

| Control | Caught by |
| --- | --- |
| N01 permutation without the cat (as if no unshard) | A, off0 |
| N02 h/s roles swapped in the permutation | A, off0 |
| N03 off-by-one row inside a chunk | A, off0 |
| N04 cvt.rz scale instead of cvt.rp | A, off0 |
| N05 amax max without .NaN | A, off0 |
| N06 one lane-shuffle stage missing | A, off0 |
| N07 one warp's smem partial skipped | A, off0 |
| N08 DGRAD 4x4 pattern transposed | A, off0 (out2 only) |
| N09 1/448 constant +1 ulp | A, off0 |
| N10 reciprocal off by 2x | A, off0 |
| N11 FPROP 4x4 pattern transposed | A, off0 (out1 only) |
| N12 qdata stored at the input row (unpermuted) | A, off0 |
| N13 duplicated row only in the 16 B (ptr%32 == 16) path | A, **off8 only**, which proves both paths are exercised |
| N14 wrong DGRAD atom/group stride | A, off0 |
| N15 last CTA skips its qdata stores | harness poisoning: out0 16384 bytes differ; initcheck flags the read |
| N16 / N17 one DGRAD / FPROP 16 B scale chunk unwritten | harness poisoning: 49152 bytes differ (logs/negctl_poison.log) |

## 5. Memory (`scripts/run_san.sh`, `scripts/san_driver.py`; logs/sanitizer.log)

The driver calls the candidate on the real input at ptr%32 = 0 and 16, then reads every output byte. Both
compiled variants (align32 and align16) are exercised. PYTORCH_NO_CUDA_MEMORY_CACHING=1 is set, so each tensor is
its own allocation.

| Tool | Candidate | Planted-bug control |
| --- | --- | --- |
| memcheck (unfiltered, 4 calls) | 0 errors | S_memcheck_oob_dgrad: "Invalid __global__ write of size 16 ... 1 bytes after the nearest allocation" |
| initcheck (unfiltered, 4 calls) | 0 errors | N15: "Uninitialized __global__ memory read" of the unwritten qdata |
| racecheck (filtered `--kernel-name regex=PrepWeight`, 2 calls) | 0 hazards | S_racecheck_no_barrier: "Potential RAW hazard ... __shared__" |
| synccheck (filtered, 2 calls) | 0 errors | S_synccheck_divergent_bar: "Barrier error detected. Divergent thread(s) in warp" |

The first racecheck/synccheck attempt used the wrong filter syntax (`regex:`) and printed usage. It was rerun
with `regex=`; both runs are in the log. The harness's allocator poisoning (section 2) also shows every output
byte is written, and N15-N17 show it detects gaps.

## 6. CUDA graphs (`scripts/t_graph.py cand.py`, logs/graph_cand.log): ALL EXACT

- The very first run() call of the process happens inside capture.
- One graph contains:
  - neighbour copy kernels writing both inputs;
  - two run() calls (ptr%32 = 0, and ptr%32 = 16 through an offset view, so both variants are in one graph);
  - neighbour kernels reading all 6 outputs;
  - WAR writers that overwrite both inputs after the calls.
- 18 replays with the source buffers changed in place (real input plus 5 random sets with exponent spread and
  NaN/inf) are all EXACT against the eager reference, both in the copied buffers and in the graph-owned outputs.
- Back-to-back replays without host sync are EXACT.
- Eager on a side stream behind a slow producer (20 GEMMs, then an input overwrite) is EXACT, so the kernel
  launches on the current torch stream.

## 7. Special cases

- Not GEMM-adjacent.
- No PDL: no griddepcontrol or launch_dependents. The `ld.global.nc` loads read only in0, which earlier kernels
  write before this kernel starts.
- Multi-device: no device-index assumptions in the code. Outputs are allocated on `in0.device`, and the launch
  uses the TVM-FFI env stream. Tested only on visible device 0 (physical GPU 0, then GPU 2).

## 8. Timing (`scripts/bench.py cand.py 100` under gpu2.lock; logs/bench_run1.log, bench_run2.log)

Each graph holds 8 calls over 4 rotating 96 MiB inputs (real input plus 3 randn sets, more than L2 in total),
with all outputs kept alive. The reference and candidate graphs replay interleaved for 100 rounds, alternating
order. Outputs inside the timing graphs were checked EXACT for all 8 calls.

| Run | Reference us/call median (p10-p90) | Candidate us/call median (p10-p90) | Speedup (median of per-round ratios) |
| --- | ---: | ---: | ---: |
| 1 | 68.78 (68.73-69.01) | 23.70 (23.48-23.78) | 2.904x |
| 2 | 68.71 (68.64-68.94) | 23.64 (23.56-23.81) | 2.908x |

Saving: 4.24 ms/step at 94 calls. The writer's figures (69.6 -> 24.1 us) are confirmed. Other verifiers'
correctness processes shared GPU 2 without a lock; this explains the occasional max outliers in run 2, but the
p10-p90 bands are tight.

## Findings for the writer

None blocking. Behaviour outside the spec: inputs that are misaligned to 16 B or non-contiguous are cloned
first (slower but exact).
