# G17 router fwd per-expert token counts -- cyd t1 -- verifier vince

VERDICT: **PASS**

- Problem: pat_bp6byfcwgcgl5kpglbew5h36egiu37aschnhpebujwy5nmu6gvx / specialization_aek4creykzwjkg3hb3reuewjqeu5pw5kwilodcleax2kguxrbox2
  (in0 [8192,128] f32 (shape only), in1 top-k idx [8192,8] i64, in2 mask [8192] bool -> counts [128] i64, [128] f32)
- Candidate: W/autokernel/run1/candidates/G17/cyd/t1/router_counts.py, sha256
  595abaac4b162077b16f20bb83d970b9a84e01102c0464536cb27202138e69f7 (== handoff); copy: cand.py
- Reference: reference.py sha256 fb7264f141f0348a... (== board); copy: reference_copy.py (all my scripts load the copy;
  nothing runs from cache_dir except the official harness command, which loads reference.py itself)
- No `<kernel_name>` file exists for this specialization: timing is against reference.py only.
- GPU 0 (GB300, sm_103), torch 2.16.0a0+git3c53c6f, nvidia-cutlass-dsl 4.8.0, compute-sanitizer 13.0.
- Scripts: scripts/ (vcommon.py helpers; every script takes the candidate from VCAND, default cand.py). Logs: logs/.

Run from /tmp: `CUDA_VISIBLE_DEVICES=0 python $V/scripts/<script>` (V = this directory).

## 1. Contract -- PASS

- sha256 of the candidate equals the handoff's (595abaac...).
- `python -I scripts/standalone.py $V real.pt` (fresh isolated process, no cwd/PYTHONPATH): imports standalone
  (only torch + cutlass), real input EXACT. `logs/contract.log`.
- `scripts/contract.py`: signature `(in0, in1, in2)` == spec; the only mutable module state is the compile cache
  `_CACHE` (keyed by device index); outputs ((128,), (1,), int64/float32, cuda:0, storage_offset 0) == reference,
  no aliasing (same alias structure as the reference); inputs byte-identical after the call; 20 repeats identical;
  every case also equals an independent CPU oracle (scatter + bincount). real + seeds 0-3 EXACT.
- Poisoning: blocks of the exact sizes the candidate allocates (32 KiB scratch, 1 KiB cnt, 512 B cntf) filled with
  0xFF / 0x5A / 0x00 / 0x80 and freed; torch.empty was wrapped to record pointers and 3/3 poisoned blocks were reused
  for the scratch and both outputs in every case: 12/12 EXACT. NaN fill of every torch.empty
  (fill_uninitialized_memory) 5/5 EXACT.

## 2. Official gate -- PASS

`python W/autokernel/harness.py <spec> cand.py --seeds 0 1 2 3 4 5 6 7 --real W/real_inputs/fusion_G17.pt --time`
(`logs/harness.log`): seed0-7 EXACT, real EXACT, PASS. Harness timing: reference cuda-graph 61.8 us, candidate 9.7 us
(1 call per graph, launch-bound); eager 253.3 vs 46.1 us.

## 3. Memory -- PASS

`scripts/run_san.sh`, `scripts/run_san2.sh` (`logs/sanitizers*.log`, full logs in `logs/san/`). Driver
`scripts/san_driver.py`: eager = real, random, in-row duplicates, all masked, 8192-count rows, plus two calls right
after the early-triggering PDL writer; graph = 2-call graph on one buffer, 3 replays with the input rewritten in place.
Every sanitizer run also checks its results against the reference (0 mismatches in all candidate runs).
Log map: `sanitizers.log` holds the final memcheck results; its racecheck/synccheck/initcheck lines (rc=1) failed on
a bad `--kernel-name regex:` syntax and were rerun: racecheck/synccheck in `sanitizers_2.log`, initcheck graph and
the filtered race/sync/init controls in `sanitizers_3.log`, unfiltered initcheck in `sanitizers_4.log`, and the nc10
(unfiltered initcheck) and nc17 (synccheck) controls in `sanitizers_5.log`.

| Tool | Candidate | Planted bug (negative control) |
| --- | --- | --- |
| memcheck (eager, no caching allocator, --padding 512) | 0 errors | nc14 (one 4 B write past the scratch): 1 error, "Invalid __global__ write ... out of bounds" |
| memcheck (graph) | 0 errors | -- |
| memcheck, out-of-range expert indices (oob.py, 11 bad values) | 0 errors | nc16 (range guard removed): 38 errors, "Invalid __shared__ atomic" |
| racecheck (eager, graph; kernel filter hist/reduce) | 0 hazards | nc12 (reduce barrier removed): 7 errors (smem write/read race, 512 hazards each) |
| synccheck (eager, graph) | 0 errors | nc17 (barrier executed by even lanes only): 1799 errors (1792 "Barrier error detected. Divergent thread(s) in warp" in hist) |
| initcheck (eager, unfiltered, no caching allocator, --check-api-memory-access no) | 0 errors | nc10 (lane 127 of each partial never written): 448 errors, all "Uninitialized __global__ memory read" in reduce (= 7 calls x 64 CTAs) |
| initcheck (graph, kernel filter hist/reduce; inputs written by memcpy nodes, which are tracked) | 0 errors | -- |

Tool notes (they do not affect the verdict, but say what a clean run means):
- initcheck with a `--kernel-name` filter reports ~237k false "uninitialized" reads of the inputs in eager mode,
  because the torch kernels that write the inputs are then not instrumented (graph mode, where the inputs are
  written by memcpy nodes, is clean even with the filter). The unfiltered run is the meaningful one (0 errors). With
  the filter, nc10 adds exactly 448 = 7 calls x 64 CTAs errors on top of the false positives.
- racecheck does not report the store-vs-shared-atomic race of nc08 (barrier between zeroing and the atomics
  removed): 0 hazards, although the values under the sanitizer were wrong in 6 of 7 calls. Normal-speed value tests
  also pass nc08 (the global-load latency hides the window). The candidate has the barrier: SASS of hist shows
  `STS [R2], RZ` -> `BAR.SYNC` -> `ATOMS.POPC.INC` -> `BAR.SYNC` -> `LDS`/`STG`.
- synccheck does not flag nc15 (barrier inside `if tidx < 128` in the 1024-thread reduce): the threads that skip it
  belong to whole warps that have already exited, which the hardware treats as arrived; nc17 is the meaningful
  divergent-barrier control.

## 4. CUDA graphs and statelessness -- PASS

`scripts/state_stress.py <scenario>`, one fresh process each (`logs/state_stress_1.log`, `_2.log`, `_occupier.log`).
Six input sets with pairwise different expected counts (half with in-row duplicates, mask densities 0-0.9); every
call's outputs are copied by a neighbour kernel into a history buffer and compared byte for byte with reference.py
for the set that call saw.

| Scenario | Calls checked | Wrong |
| --- | ---: | ---: |
| b2b eager, distinct input tensors, outputs kept alive, no sync (5000) + one buffer rewritten in place by copy_ before each call (5000) | 10000 | 0 |
| 94-call graph (= calls/step), [copy_ into input, run, copy_ outputs to log] per call, shared input buffer and separate buffers, 200 replays each with all inputs restaged between replays | 37600 | 0 |
| 2 calls in one graph on one buffer (input rewritten in between), 2000 replays; 1-call graph 2000 replays | 6000 | 0 |
| first call inside capture (fresh process, `_CACHE` empty, so cute.compile runs inside capture), 2-call graph x 500 replays, then 500 eager | 1500 | 0 |
| 3 concurrent streams with eager loops, x3 | 3600 | 0 |
| graph replayed on another stream while eager calls run on its capture stream (the 30B t5 hazard) | 9792 | 0 |
| two graphs captured on one stream replayed concurrently on two streams (the other 30B t5 hazard) | 24000 | 0 |
| side stream: `_sleep` -> rewrite input -> run -> log (would fail if run() launched on another stream) | 200 | 0 |
| early-triggering PDL writer (below): 3000 eager, 8-call graph x 600, separate buffers x 300 | 10200 | 0 |
| thread-slot occupier on another stream leaving k = 1/2/4/16 SMs' worth of slots, 94-call graph x 4 replays each, + 60 eager with k = 1 | 1564 | 0 |

Occupier details (`logs/state_stress_occupier.log`): `pdl_tools.occupier_threads` launches 2 x (152 - k) CTAs of
1024 threads + 100 KiB smem on another stream (two per SM, every thread slot taken); the current stream waits on a
residency counter until all of them have started, then an SM-id probe with the hist kernel's shape (64 x 128 threads)
runs before and after the timed replay. k = 1: probe confined to 1 SM, 214.5 us/call (multi-wave hist; the
PDL-launched 1024-thread reduce has to find the only free half-SM slot); k = 2: 3 SMs, 89.9 us/call; k = 4: 7 SMs,
74.1 us/call; k = 16: 31 SMs, 13.0 us/call; unoccupied 7.5 us/call (94-call graph incl. 4 neighbour memcpy nodes per
call). 4 checked replays per k + 60 eager calls with k = 1: all EXACT, no hang. (A first version with one
1024-thread CTA per SM did not confine anything -- the probe still found 51 SMs -- so the probe is what makes this
test meaningful.)

## 5. Adversarial values -- PASS

`scripts/adv.py` (`logs/adv.log`): 67/67 byte-exact vs reference.py and vs the CPU oracle, inputs unmutated:
- random distinct top-k at mask densities 0, 0.01, 0.3, 0.5, 0.9, 0.99, 1.0; all masked; none masked;
- masks on the new 128-row CTA split: even CTAs, odd CTAs, rows 127/128 of every CTA, row 0 only, row 8191 only,
  alternate rows, a single unmasked row (4097);
- in-row duplicates at every one of the 28 slot pairs, duplicate chains, all 8 slots equal (random expert per row);
- extreme counts: expert 0 (and 127) in all 8 slots of every row (count 8192), experts 0..7 in every row
  (8 x 8192), and odd counts above 2048/4096 that would expose any fp16/bf16-like rounding of the fp32 output
  (8191, 6145, 4097); only experts 0 and 127; Zipf hot experts with replacement (counts up to 6639);
  sorted / reverse-sorted rows; structured patterns;
- in0 filled with NaN, +inf, -inf, -0, subnormal (only its shape is used; nothing changes);
- real input, real idx with random and CTA-edge masks.
- Informational (outside the bool contract): mask bytes 0x02 / 0x80 / 0xFF also match the reference.

Reduction order: both outputs are counts of 0/1 values. The int64 sum is exact in any order and every partial
fp32 sum is an integer in [0, 8192] < 2^24, so every fp32 addition is exact and the result is order-independent;
zero counts give +0.0 in both (all-masked case byte-exact). There is no cancelling term to probe; the order-sensitive
failure modes that do exist (dropped/double-counted partials, rows, duplicates, rounding of large odd counts) are
covered by the cases and controls above.

Out-of-range expert indices (`scripts/oob.py`, `logs/oob.log`): the reference device-asserts (checked for 128, -1,
2^32, 255). The candidate skips every row that holds an index outside [0, 128) (11 values incl. 128, 255, -1, -128,
2^31, 2^32, 2^32+5, +-2^63 range ends), counts all other rows exactly, memcheck clean, and a valid call afterwards is
EXACT. Wrong shapes ([4096,8], [16384,8]) raise ValueError from the TVM-FFI shape check; a misaligned idx view and a
strided mask are copied by run() and EXACT (`logs/shape_misuse.log`).

Negative controls (`scripts/make_negctl.py` -> `negctl/*.py`, runner `scripts/run_negctl.sh`, logs
`logs/negctl_1.log`, `logs/negctl_2.log`, `negctl/*.log`): adv = adv.py, ctr = contract.py (incl. poisoning),
then the b2b / graph2 / pdl_writer stress scenarios (wrong calls / checked).

| Control | adv | ctr | b2b | graph2 | pdl_writer | Caught |
| --- | --- | --- | --- | --- | --- | --- |
| nc01 duplicates not skipped | 32/67 | PASS | 4966/10000 | 3019/6000 | 5011/10200 | yes |
| nc02 mask inverted | 0/67 | FAIL | 10000 | 6000 | 10200 | yes |
| nc03 reduce drops one partial | 4/67 | FAIL | 10000 | 6000 | 10200 | yes |
| nc04 row index off by one | crash (illegal address) | FAIL | 9178 | 1553 | 7655 | yes |
| nc05 no griddepcontrol.wait in reduce | 31/67 | PASS | 1121 | 1164 | 24 | yes |
| nc06 wait after the partial loads | 64/67 | PASS | 0 | 952 | 11 | yes |
| nc07 hist also launched with PDL | 67/67 | PASS | 0 | 0 | 8482/10200 | yes (only the early-trigger writer) |
| nc08 no barrier after smem zeroing | 67/67 | PASS | 0 | 0 | 0 | no (see sec. 3; candidate's barrier checked in SASS) |
| nc09 fp16-like rounding of counts > 2048 | 61/67 | PASS | 0 | 0 | 0 | yes (adv) |
| nc10 partial lane 127 never written | 4/67 | FAIL | 10000 | 6000 | 10200 | yes (+ initcheck) |
| nc11 range guard at 64 | 6/67 | FAIL | 10000 | 6000 | 10200 | yes |
| nc12 no barrier in reduce | 8/67 | FAIL | 8396 | 4693 | 8307 | yes (+ racecheck) |
| nc13 partial loads via ld.global.nc | 67/67 | PASS | 0 | 0 | 0 | no (informational; guarded by the SASS check) |
| nc14 4 B write past the scratch | 67/67 | FAIL (neighbour corrupted) | 1 | 0 | 0 | yes (+ memcheck) |

## 6. PDL -- PASS

- Graph edges (`scripts/edges.py`, `logs/edges.log`, cuGraphGetEdges on the captured
  [memcpy producer -> run -> memcpy consumer]): producer -> hist type 0 (full), hist -> reduce type 1
  (programmatic, from_port 1), reduce -> consumer type 0. Only hist -> reduce is programmatic.
- SASS (`dump/cand.sass`, from CUTE_DSL_KEEP_CUBIN): reduce begins `S2R / LDC / ACQBULK` (griddepcontrol.wait) before
  any global access; all 8 partial loads are `LDG.E.STRONG.GPU` (PTX `ld.relaxed.gpu`); hist has `PREEXIT`
  (launch_dependents) before its loads, which are plain `LDG.E.128` / `LDG.E.U8`; no `.CONSTANT` / `.nc` load anywhere
  in either kernel (grep count 0). hist is not a PDL secondary, so its plain input loads cannot see a predecessor's
  in-flight writes.
- Early-triggering predecessor (`scripts/pdl_tools.py` early_writer: one CTA, griddepcontrol.launch_dependents first,
  2 us spin, then ~80 us of writes of idx and mask, last elements first) right before every call: 3000 eager +
  7200 in-graph calls EXACT; the control nc07 (hist launched with PDL) fails 8482/10200, so the test has teeth.
- Multi-device: only cuda:0 tested (GPU 0 lease). Device-index assumptions: `_CACHE` is keyed by `in1.device.index`,
  scratch/outputs are allocated on `in1.device`, the launch stream is the TVM-FFI env stream (the current torch
  stream; the side-stream test proves it follows the current stream on cuda:0).

## 7. Timing

`VWINDOWS=<n> python scripts/timing.py <calls/graph> <rounds> <replays>` (`logs/timing_run2.log`,
`logs/timing_94calls_8sets.log`). One graph per implementation; call i of a graph reads its own input set (rotating
inputs; set 0 = real input) and its outputs stay live. Compile, warm-up and capture happen before gpu0.lock is taken;
the lock is held only for each timed window. Each sample = median of individually timed replays; reference and
candidate alternate within every round. Because correctness runs of other verifiers share GPU 0 without the lock
(time-slicing between contexts inflates long measurements), the script runs several windows 20 s apart and reports the
window with the fewest other GPU-0 processes, then the lowest reference median (contention only inflates); all
windows are in the log. No `<kernel_name>` exists, so the comparison is against reference.py only.

| Graph | Reference us/call median (p10-p90) | Candidate us/call median (p10-p90) | Speedup (per round) | Saving at 94 calls/step |
| --- | --- | --- | --- | --- |
| 8 calls, 8 rotating sets, 20 rounds x 10 replays (8 windows) | 53.30 (53.10-53.46) | 2.90 (2.87-2.92) | 18.4x (18.2-18.8), 20/20 rounds | **4.74 ms/step** |
| 94 calls (= calls/step), 94 rotating sets, 10 rounds x 5 replays (8 windows) | 105.31 (104.96-105.94) | 2.51 (2.51-2.53) | 41.9x (41.5-42.3), 10/10 rounds | 9.66 ms/step |
| 94 calls, 8 distinct sets reused (3 windows) | 103.17 (102.91-103.73) | 2.46 (2.45-2.47) | 41.9x | 9.47 ms/step |
| 1 call (launch-bound), 20 rounds x 20 replays (4 windows) | 56.05 (55.97-56.14) | 6.80 (6.56-6.98) | 8.2x | 4.63 ms/step |

- Reported saving: **4.74 ms/step** (8-call graph, the writer's methodology and the conservative number; writer
  claimed 53.81 -> 2.97 us/call, ~5.1 ms/step: consistent). In a 94-call graph the reference costs ~2x more per call
  (105 us) whether the inputs rotate over 94 or 8 sets, so it is not the input working set; most likely its ~8 MB of
  per-call intermediates (bool [8192,128] scatter target, the AND result, the fp32 copy) no longer stay L2-resident
  in a long graph. The candidate is unaffected (2.5 us/call), so the in-training saving is probably between
  4.7 and 9.7 ms/step; the step profile of the next training run will tell.
- Excluded: a first run (`logs/timing_run1.log`, back-to-back replays per sample) taken while ~7 sanitizer runs of
  other verifiers were active on GPU 0: reference 168-178 us/call, candidate 2.88 (8-call) / 7.48 (94-call), i.e.
  long measurements were stretched by context time-slicing. Not used.
- Every timing run also checked the last replay's outputs: 8/8, 94/94, 94/94, 1/1 EXACT.

## Notes for the writer / coordinator (no blocker)

- As documented: a row holding an expert index outside [0, 128) is skipped where the reference device-asserts.
- `cute.recast_ptr(...).align(16)` compiles to an add-15-and-mask of the row address (SASS `IADD3 +0xf; LOP3 &~0xf`),
  i.e. it rounds the pointer up rather than asserting alignment; run() guarantees 16 B alignment (clone otherwise), so
  it is a no-op here, but the guard in run() must stay.
