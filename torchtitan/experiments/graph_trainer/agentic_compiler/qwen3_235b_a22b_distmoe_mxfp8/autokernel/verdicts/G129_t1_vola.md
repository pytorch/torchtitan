# G129 eve t1: verdict (verifier vola)

**PASS**

- Candidate: `candidates/G129/eve/t1/kernel.py`, sha256 `0c4a21bf77c3b3690233bacf9ed8967e729206b1b3117ba598a186e9cd77675c`
  (same as the handoff; copied here as `cand.py`).
- Problem: `pat_gbfxwxnjitoyy63k66uc7frtq4v4hk43wjok6bn7ky7juowgjkn` / `a6h3iapqonscl6fyy4oe2opznsovu5dvi5x3nmavvyre75vd3um4`.
  reference.py sha256 `3e6b0b7ea570bd05...` (matches the board). No current `<kernel_name>`, so the baseline is reference.py.
- Op: flat bf16 [25165824] all-gather output -> view [32, 512, 1536], split 8 ways, cat on dim 1 -> W [4, 4096, 1536] ->
  `dist_moe::prepare_mxfp8_weight` -> qdata e4m3 [4, 4096, 1536], FPROP e8m0 [16384, 48], DGRAD e8m0 [6144, 128].
- Diff vs the G115 t1 source: only `H = 1`, `CH = 512`, `C = 1536`, and docstring lines. The checks below ran on this
  file's own constants.

Timeline note: contract, harness and adv ran on GPU 0 under gpu0.lock, before the 16:58 brief revision. Everything
after that ran on GPU 2 with no lock, except timing, which held gpu2.lock.
Run commands from /tmp, e.g. `CUDA_VISIBLE_DEVICES=2 python scripts/<x>.py`. Logs are in `logs/` and `sanitizer/`.

## 1. Contract (`scripts/contract.py`, logs/contract.log): PASS
- The sha256 matches the handoff, and `run(in0)` matches the spec signature.
- A fresh isolated process (`python -I`, cwd /tmp) imports the file standalone. It loads no dist_moe, torchtitan,
  torchao, quack or triton.
- Output metadata equals the reference for all three outputs: shape, stride, dtype, device, storage_offset 0,
  contiguous, storage nbytes, requires_grad. It also matches the spec.
- Aliasing: no output aliases another output or the input (same as the reference), and every call returns fresh outputs.
- The input is unmutated after 3 calls; the 3 repeated calls are bitwise identical; the outputs are bitwise equal to the reference.

## 2. Official gate (logs/harness.log): PASS
`python W/autokernel/harness.py <spec> cand.py --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G129.pt --time`
gives seed0-7 EXACT and real EXACT. This includes the harness's poisoned-allocator reruns (0xFF / 0x5A bait).
Harness 1-call graph timing: reference 43.3 us, candidate 11.9 us.

## Unshard permutation (`scripts/perm.py`, logs/perm.log): PASS
- (a) The reference's own aten view/split/cat, replayed on int32 position codes, equals the formula
  `W[e, s*512 + r] = in0.view(32, 512, 1536)[s*4 + e, r]` on all 25,165,824 elements. This confirms the writer's mapping.
- (b) The inverse used by adv.py is exact.
- (c) A position-encoded bf16 input (every value is e4m3-exact at scale 1; 14 bits of input row and 6 bits of col block
  in each 32-col block):
  - decoding reference.py's qdata recovers the formula's input row for every logical row and every col block;
  - the candidate matches the reference bytewise on both load paths (ptr%32 = 0 and 16).
- (d) Two tile-hash scale patterns (61 and 53 distinct scale bytes, one hash value per tile) are EXACT on both paths.

## 5. Adversarial suite (`scripts/adv.py`, logs/adv.log): ALL EXACT, 58 candidate comparisons
Setup:
- Each case builds W tile by tile (24576 tiles of 32x32) and places it with the verified inverse.
- In0 sits between NaN guards. Each case runs on both load paths: 32-byte v8 loads at ptr%32 = 0, and 16-byte v4 loads
  at ptr%32 = 16.
- Every case checks all 3 outputs bytewise and checks that the input is unmutated.

Scale oracle:
- Running dist_moe on [N, 128, 128] constant tiles gives byte(amax) for all 65536 bf16 patterns. This is layout-free:
  each group's 512 FPROP and DGRAD bytes are uniform.
- An independent formula (fp32 `a * 0x3B124925`, round up to a power of two) agrees on all 65536 patterns with 0 disagreements.
- Reachable bytes are 0..247 plus 255 (249 total); byte(0) = 0.
- A torch qdata oracle (amax -> byte -> recip -> clamp(x * recip) -> e4m3, NaN -> 0x7F) equals the reference on every
  A, B and C case.

Cases:
- A0-A2: every bf16 pattern 0x0000-0xFFFF (with sign) as the tile's max element, at varying positions. Other elements
  are uniform below it; a third of the tiles also carry -|amax|.
- B: every bf16 x with |x| <= A_max(b), both signs, for all 248 finite scale bytes (8309 tiles). The same at A_min(b)
  (8248 tiles). Because recip depends only on b, this exhausts every (x, recip) pair the op can produce: all e4m3 ties,
  subnormals and satfinite boundaries.
- C1: 16 specials at each of the 1024 tile positions:
  - NaN payloads 7FC0, 7F81, 7FFF, FFC0, FF81, FFFF, 7FA5, FFA0;
  - +-inf;
  - +-bf16 max;
  - +-min subnormal;
  - min normal;
  - -0.
- C2: 25 special tile types:
  - uniform specials;
  - random NaN payloads;
  - inf/NaN mix;
  - +-0 mix;
  - subnormal-only tiles;
  - single nonzero among +-0 at every position, for 7 values;
  - 448 * 2^k +- 1 ulp as amax over the whole range (782 values).
- D1-D4:
  - uniform random 16-bit patterns;
  - random finite patterns;
  - per-tile 4-binade exponent windows over the full range, with 10% +-0;
  - randn * 2^k per tile, k in [-140, 120).
- E (reduction coverage): one dominant element per tile at every one of the 1024 positions, 24 times each.
  - Neighbouring tiles differ by 2^7, so a cross-tile leak changes a byte.
  - The background stays below dominant / 4, so dropping any element changes a byte.
  - The same layout with NaN as the dominant element tests NaN propagation through the per-thread tree, both shuffle
    stages and the cross-warp shared-memory combine.
  - Reduction order: the only reduction is the per-tile amax. `max.NaN.xorsign.abs` is exact, commutative and
    associative, with NaN dominant and the sign masked afterwards, so the output cannot depend on order. Coverage and
    NaN propagation are what can break, and E probes both directly.
- F (offsets and views, outside the spec):
  - ptr%32 = 0, 2, ..., 30, between NaN guards (non-16-byte-aligned inputs go through the clone path);
  - a non-contiguous stride-2 view;
  - a view at element offset 50331648 of a bucket buffer.
  All EXACT.
- G: the real input (absmax 0.0076, no NaN), and randn * {1e-30, 1, 1e30}.
- Coverage: the candidate was exercised on 249/249 reachable scale bytes.

## Negative controls (`negctl/make.py`, `negctl/run_negctl.sh`, logs/negctl_*.log)
Value mutants. Every one makes adv.py FAIL:

| Mutant | First failure |
| --- | --- |
| N01 expert/shard order swapped in the row map | A0, all outputs |
| N02 row off by one | A0 |
| N03 scale cvt.rz instead of cvt.rp | A0 |
| N04 1/448 constant 1 ulp high | A0, 54193 qdata bytes |
| N05 DGRAD 16-byte pattern transposed | A0, DGRAD only |
| N06 one lane-shuffle stage missing | A0 |
| N07 one row-shuffle stage missing | A0 |
| N08 cross-warp combine reads warp 0 only | A0 |
| N09 named barrier removed | A0 (also racecheck) |
| N10 block barrier removed | A0 (also racecheck) |
| N11 inf tile gets a non-NaN recip | A1 |
| N12 last row of each thread not stored | A0 (also direct poisoning) |
| N13 32-byte loads on a 16-byte-aligned input | ValueError from the TVM-FFI alignment guard |
| N14 amax max without .NaN | A1 |
| N15 DGRAD atoms in row-major order | A0, DGRAD only |
| N16 adjacent rows swapped inside a tile | A0, qdata only |

Output-equivalent mutants. These pass, as expected, which shows the suite does not over-claim:
- Q01 quant `mul.rz` instead of `mul.rn`: bf16 x times a power of two is exact. The only inexact products are fp32
  subnormals below 2^-126, and those round to e4m3 zero either way.
- Q02 NaN clamp removed: a NaN amax gives s = NaN, so `cvt.rp.ue8m0` returns 0xFF and `setp.ne` (ordered) gives the NaN recip.
- Q03 `s = (jj + 1) // CH`: unreachable. Each thread evaluates the row map only at its base row, and base rows satisfy
  `jj % RPW < RPI`, so they never sit at a chunk end. Rows loaded at `+i * RPI` stay inside the same chunk because
  RPW divides 32 and 32 divides CH.
- Q04 1/448 constant 1 ulp low (0x3B124924): `amax / 448` comes near a power of two only for bf16 mantissa 224. There
  `1 + 4.4e-8` rounds to 1.0 with the original constant, and `1 - 6e-8` rounds to `1 - 2^-24` with the mutant; both
  round up to the same power of two. The constant 1 ulp high is caught (N04).
- The first GPU 0 run had N02 = Q03 and N04 = Q04 under the old names. They were reclassified as above and replaced by
  effective mutants (logs `*.gpu0_first_run.log`).

## 3. Memory: PASS
- Harness allocator bait (0xFF / 0x5A): EXACT.
- Direct poisoning (`scripts/poison.py`, logs/poison.log): the candidate's torch.empty is prefilled with 0x00, 0x7F,
  0xA5 or 0xFF. All four values on both paths are EXACT, with 3 shim allocations per call. Each poison byte occurs in
  the correct outputs. Control: N12 FAILS for every poison value (logs/poison_neg_N12.log).
- compute-sanitizer 13.0 (`sanitizer/run_sanitizer.sh`). Settings: unfiltered, `PYTORCH_NO_CUDA_MEMORY_CACHING=1`
  (exact per-tensor bounds), both load paths in one driver; the 16-byte path's view ends exactly at its allocation end.
  - Candidate: memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors, initcheck 0 errors.
  - Tool controls, all flagged:

    | Control | Tool | Result |
    | --- | --- | --- |
    | S1 16-byte write past the DGRAD scale tensor | memcheck | 2 errors, "out of bounds ... after the nearest allocation of size 786,432" |
    | N10 no block barrier | racecheck | 4 hazards |
    | N09 no named barrier | racecheck | 2 hazards |
    | S3 named barrier under `if l < 16` | synccheck | 49456 errors, "Divergent thread(s) in warp" |
    | S4 4096 uninitialized input elements | initcheck | 768 errors, "Uninitialized __global__ memory read" in the PrepWeight kernel |

  - The flagged kernel names (`kernel_cutlass_kernel_cand_mod_*PrepWeight*`) show that the candidate kernels are instrumented.

## 4. CUDA graphs (`scripts/graphs.py`, logs/graphs_off{0,16,2}.log): PASS
Each run uses a fresh process per path (ptr%32 = 0 and 16, plus 2 for the clone path, which is outside the spec).
- Graph A: the very first candidate call of the process happens inside capture (import only, never run before). The
  graph contains:
  - a producer writing in0, then cand;
  - consumers (copies of the outputs and an int64 checksum kernel);
  - a WAR writer overwriting in0, then a second cand call and its consumers.

  24/24 replays were EXACT against eager reference.py, with both sources rewritten in place between replays (randn
  scales, random finite bits, exponent windows, NaN/inf, real input x k, +-0). An eager call after the graph equals the graph output.
- Graph B: captured on a user side stream, with 4 calls over 3 inputs and one in-graph producer. 12/12 replays were
  EXACT with in-place input changes.
- Graph controls, caught (logs/graphs_neg_*.log):
  - GN1 static output buffers: 0/24 and 0/12;
  - GN3 launch on an unjoined side stream: 0/24, eager mismatch, 0/12.

## 6. Special cases
- GEMM-adjacent: not applicable (no `_scaled_mm`).
- PDL: none. The PTX has no griddepcontrol, and the SASS has no ACQBULK or griddep instructions (`sass/v{32,16}/`).
  Input loads are `ld.global.nc`:
  - 32-byte path: 4x `LDG.E.NA.ENL2.256.CONSTANT`, 6x `STG.E.128`;
  - 16-byte path: 8x `LDG.E.NA.128.CONSTANT`, 2x `STG.E.128` and 8x `STG.E.64`.

  Without PDL, the input is written only by earlier kernels, so `.nc` is safe.
  The launch uses 256 threads (`.reqntid`), `.minnctapersm 1`, and 1536 CTAs.
- Multi-device: tested on one device only (the GPU 2 lease, visible as cuda:0).
  - The code hard-codes no device index. Outputs are allocated on `in0.device`, the kernel launches on the TVM-FFI env
    stream (the current torch stream), and compilation at import targets the current device's arch.
  - Calling it with in0 on a non-current device was not tested here. The 30B sibling, with the same launch mechanism,
    passed the coordinator's multidev_check.
- Outside the spec, by design: non-16-byte-aligned or non-contiguous inputs are cloned first, which adds a 48 MiB copy.
  The result is correct (F cases, graph ptr%32 = 2).

## 7. Timing (`scripts/bench.py`, logs/bench_seed{0,1}.log)
GPU 2 under gpu2.lock. Each graph holds 8 calls over 5 rotating 48 MiB inputs with live outputs. 100 rounds per run
alternate the reference and candidate graphs in shuffled order. GPU 2 was shared with other verifiers' unlocked
correctness runs, which caused a few outlier rounds; the medians are tight.

| Run | reference us/call: median (p10-p90) | candidate us/call: median (p10-p90) | cand/ref per round | cand faster |
| --- | --- | --- | --- | --- |
| seed 0 | 39.04 (38.98-39.10) | 13.45 (13.39-13.63) | 0.3445 (0.3427-0.3489) | 100/100 rounds |
| seed 1 | 39.05 (38.98-39.30) | 13.46 (13.38-13.64) | 0.3443 (0.3422-0.3474) | 99/100 rounds |

Saving: (39.04 - 13.45) us x 94 calls/step = **2.41 ms/step**, a 2.90x speedup. This is consistent with the writer's
39.5 -> 13.6 us.

## Findings for the writer
None blocking. Two notes, neither needs a change:
- The row map is evaluated only at each thread's base row (Q03). It stays correct only while a warp's RPW rows sit inside
  one CH chunk; the `CH % 32 == 0` assert plus RPW dividing 32 guarantee that.
- The 1/448 constant has 1-ulp-low slack (Q04).
