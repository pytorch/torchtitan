# Verifier brief (Qwen3 235B-A22B autokernel run1)

You are a newly spawned, independent verifier for ONE candidate kernel. You did not write it; do not trust the
writer's claims or tests: rerun and extend them yourself. Read the autokernel skill's "Exact correctness gate":
/home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/.claude/skills/autokernel/SKILL.md

## Paths

- W = /home/bahuang/local/torchtitan/outputs/agentic_compiler/qwen3_235b_a22b_distmoe_mxfp8
- Board: W/autokernel/run1/board.md (lock: W/autokernel/run1/board.lock; write board.md.tmp then mv)
- cache_dir (READ ONLY): W/kernel_cache; the problem's reference.py and spec JSON are the contract
- Harness: W/autokernel/harness.py; real inputs: W/real_inputs/fusion_G<id>.pt
- Evidence: W/autokernel/run1/verify/G<id>_<trial>_<your name>/ (your scripts, logs, verdict.md)
- 30B prior art: Q = /home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/agentic_compiler/qwen3_30b_a3b_distmoe_mxfp8
  (README.md sec. 7 "What every verifier checked" and Q/autokernel/board.md list what the 30B verifiers of the
  sibling kernel tested: use them as a checklist, adapted to the new shapes).

## GPU (revised 16:58: the single GPU 0 lock serialized ~20 verifiers on CuTe JIT compiles)

- Correctness, adversarial, sanitizer, CUDA-graph and compile runs: NO lock. Use your assigned GPU (below).
- Timing runs only: hold `flock W/autokernel/run1/gpu<N>.lock` for your assigned GPU, and keep the locked command
  short (warm up / compile inside the same process before the timed loop; never hold the lock while idle).
- Assigned GPU: verifiers of G06, G105, G111, G102, G115, G129, G125, G91, G07, G01 -> GPU 2;
  verifiers of G17, G90, G19, G133, G47, G15, G28, G45, G41 -> GPU 0.
- The coordinator runs ~210 GiB training jobs on GPU 0 under gpu0.lock. Never touch other users' processes.

## Checks (all required)

1. Contract: sha256 of the candidate equals the handoff's; `run` matches the spec's signature; a fresh process
   imports the file standalone (copy it to your evidence dir; never run Python from cache_dir).
2. Official gate: `python W/autokernel/harness.py <spec> <candidate> --seeds 0 1 2 3 4 5 6 7 --real <real input>`
   all EXACT; output structure, shape, stride, dtype, device, storage offsets and alias structure match the
   reference; inputs are not mutated; repeated calls are identical.
3. Memory: allocator poisoning shows every output byte is written; compute-sanitizer memcheck, racecheck,
   synccheck (and initcheck where meaningful) report 0 errors; demonstrate each tool flags a planted bug in a copy
   (negative control) so a clean result means something.
4. CUDA graphs: capture the candidate with neighbours that write its inputs and read its outputs; replay many
   times with inputs changed in place; two or more calls in one graph; the very first call inside capture.
5. Adversarial values: your own suite for this op (specials NaN/inf/-0/subnormals, extreme exponents, ties at
   rounding boundaries, all-equal and one-hot rows, edge indices where the op has indices, the exact amax/scale
   boundaries for MXFP8 kernels), plus direct probes of every reduction order the outputs depend on (use exactly
   representable cancelling terms where an output hides the order). Include negative controls: deliberately
   broken copies (wrong order, off-by-one, missing barrier, wrong rounding) that your suite must catch.
6. Special cases: GEMM-adjacent kernels: prove the `_scaled_mm_v2` call is identical and its output bitwise equal.
   PDL kernels: test with an early-triggering predecessor that writes the inputs, and check SASS for `.nc` loads of
   data written during the call chain. Multi-device: run on cuda:0 inputs only (GPU 0 lease); note any
   device-index assumptions in the code.
7. Timing: CUDA-graph replays with rotating inputs and live outputs, interleaved with the reference (and with the
   current `<kernel_name>` in the pattern dir if one exists); report medians and p10-p90 for both. Compute the
   saving in ms/step = (reference - candidate) x calls per step (from the board).

## Verdict

Write verdict.md in your evidence dir (PASS or FAIL, every check with its command and result, timings). Under the
board lock add a row to the "Verifications" table and update the problem row's "Exactness coverage". Do NOT touch
cache_dir: the coordinator promotes on PASS. End with a short report: verdict, sha256 prefix, reference us vs
candidate us (CUDA-graph), saving ms/step, and any finding the writer must fix (for FAIL, the minimal failing
input and how to reproduce it).
