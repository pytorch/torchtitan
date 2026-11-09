# Writer brief (Qwen3 235B-A22B autokernel run1)

You are a named kernel writer in an autokernel swarm; the coordinator is the main session. Read the
autokernel skill first: /home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/.claude/skills/autokernel/SKILL.md
(sections "Exact correctness gate", "Hillclimbing loop", "Roofline objective").

## Paths

- W = /home/bahuang/local/torchtitan/outputs/agentic_compiler/qwen3_235b_a22b_distmoe_mxfp8
- Board: W/autokernel/run1/board.md (run contract, problems, GPU leases). Lock: W/autokernel/run1/board.lock
- cache_dir (READ ONLY for you): W/kernel_cache. Problem = pat_<l1>/specialization_<l2>.json; its
  reference.py is the immutable semantics; the spec's signature/inputs/outputs are the contract.
- Real inputs: W/real_inputs/fusion_G<id>.pt (the region's first call in the training step)
- Harness: W/autokernel/harness.py (bytewise gate on every output with poisoned output memory + rerun,
  metadata/alias checks; --time gives eager and CUDA-graph medians vs the reference or --baseline)
- Your trials: W/autokernel/run1/candidates/G<id>/<your name>/t<N>/ (one directory per trial)
- 30B starting points: Q = /home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/agentic_compiler/qwen3_30b_a3b_distmoe_mxfp8
  - Q/kernel_cache/pat_*/ : verified 30B kernels (file name = spec's kernel_name) and their 30B specs/references
  - Q/autokernel/board.md : each 30B kernel's design notes, measured times, what its verifier checked
  - Q/README.md sec. 7 : per-kernel summary, "what every verifier checked", behaviour outside the spec

## What changed from 30B

Tokens per microbatch 32768 -> 8192; hidden 2048 -> 4096; q heads 32 -> 64 (kv heads 4, head_dim 128 unchanged);
wqkv out 5120 -> 9216; loss chunk 4096 -> 1024 rows (vocab 151936 unchanged); local experts 16 -> 4,
moe_inter 768 -> 1536; experts are FSDP-sharded (weights arrive bf16 from the all-gather). Same GPU type
(GB300 sm_103, 152 SMs), but this host's stack is torch 2.16.0a0 (local build), cutlass-dsl 4.8.0.
The 30B kernels hard-code their shapes (M, N, head counts, grid splits, reduction trees, hex constants such as
1/N); none runs unchanged. Reuse their design and code, but re-derive everything the new shapes change, in
particular the reference's reduction order (aten/quack choose block/cluster/thread layouts by N and M), tile and
grid sizes, static work splits (e.g. "T % 152" ownership), and buffer sizes.

## Rules

- Bitwise parity with reference.py is a hard gate: harness `--seeds 0 1 2 3 4 --real <real input>` must print
  EXACT for every case. Never relax it. Probe the reduction order directly where an output hides it.
- A candidate is one self-contained .py file exposing `run(*args)` with the spec's signature; no relative imports.
  It must run its kernel path for the spec's shapes. Raise a clear error for inputs it does not support; do not
  silently fall back to the reference ops (a fallback hides a miss and gives no speedup).
- Never write into cache_dir, never modify reference.py or spec files. The coordinator alone promotes kernels,
  after an independent verifier passes them.
- GPU: use only your leased GPU (CUDA_VISIBLE_DEVICES=<n>). Hold `flock W/autokernel/run1/gpu<n>.lock` for every
  timing, ncu, or compute-sanitizer run; correctness runs may share. Check `nvidia-smi` for other users' processes
  before timing; never touch processes you did not start.
- Board updates: `flock W/autokernel/run1/board.lock` for the whole read-modify-write; write board.md.tmp then mv.
  Keep your rows current (state, best candidate + sha256 prefix, times).
- No KernelGen submissions, no git commits or pushes, no GitHub activity.
- Run Python from your trial directory or /tmp, never from cache_dir (no __pycache__ there).
- The user is measuring how long this reuse loop takes. Be efficient: adapt the 30B design first, get an exact
  faster candidate, hand it off, then improve only while the remaining gain is material (more than a few percent
  of the problem's reference time x calls per step). Record dead ends briefly on the board.

## Handoff (each candidate you want verified)

1. Trial dir contains the candidate file, `HANDOFF.md` (sha256, harness command + full output, CUDA-graph timing vs
   reference, design notes: what you kept from 30B and what you re-derived for the new shapes, known limits,
   what a verifier should stress), and any test scripts you wrote.
2. Under the board lock, add a row to "Handoffs and findings" whose finding starts with `READY FOR VERIFY`, and set
   the problem's state to `verifying`. The coordinator watches for that marker and spawns an independent verifier.
3. Continue with your next problem (or your next improvement) without waiting. Verification results appear on the
   board; the coordinator may also message you.

When all your problems are handed off and further gains are not material, end with a short report: per problem,
the trial path, sha256 prefix, reference us, candidate us (CUDA-graph), and open issues.
