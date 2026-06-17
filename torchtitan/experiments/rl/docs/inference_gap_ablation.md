# Inference gap ablation (TorchTitan unified model in vLLM vs vLLM native)

Goal: make the TorchTitan unified Qwen3-32B model (run inside vLLM, TP=8) match
vLLM's native model throughput. Running log of method + results; append-only.

## Setup

- Model: Qwen3-32B, TP=8, `external_launcher` (`torchrun --nproc_per_node=8`).
- Harness: `torchtitan/experiments/rl/generate.py --benchmark` (excludes engine
  startup, runs `--warmup-runs` before timing, feeds synthetic token-id prompts,
  reports total tok/s = batch_size * generated_tokens / wall_time).
- Compile ownership: **TorchTitan's compile, never vLLM's.** All TorchTitan-path
  rungs use `--compile aot_eager` (per-layer torch.compile) + vLLM
  `cudagraph_mode = FULL_DECODE_ONLY` (vLLM compile mode = NONE; eager prefill,
  decode cudagraph). Capture sizes = powers of 2 up to `max_num_seqs`.
- Target: `--native --compile aot_eager --cudagraph on` (vLLM-native, same
  FULL_DECODE_ONLY mode). This is native(eager), the fair target -- not
  native(inductor), which would add comms/norm fusion the aot_eager path lacks.
- Kernel rungs are cumulative monkeypatches in `models/kernel_ablation.py`, each
  wrapped as a `torch.library` custom op so it survives compile + cudagraph.
- Numerics: every rung validated bitwise-identical greedy text on qwen3-1.7B
  TP=2 before benchmarking (most-patched chain validated at the top of each run).

## W1 -- bs=8, input_len=1024, generate=128 tokens

Ladder (cumulative, full_decode_only). Ratio vs native = 883.9 tok/s.

| # | rung | tok/s | ratio |
|---|------|-------|-------|
| 1 | baseline (eager DTensor, SP off) | 25.2 | 0.029 |
| 2 | + compile (aot_eager, per-layer) | 64.0 | 0.072 |
| 3 | + cudagraph (FULL_DECODE_ONLY) | 427.6 | 0.484 |
| 4 | + tree all-reduce (NCCL Ring,Tree) | 423.8 | 0.479 |
| 5 | + Helion RoPE | 469.5 | 0.531 |
| 6 | + FusedQKV + fused gate_up (proper) | 492.5 | 0.557 |
| 7 | + SiluAndMul (vLLM kernel) | 495.3 | 0.560 |
| 8 | + FlashAttention (FA3, vs CUSTOM varlen) | 499.9 | 0.566 |
| 9 | + fused add+RMSNorm (vLLM kernel) | 500.6 | 0.566 |
| 10 | + vLLM all-reduce | 637.5 | 0.721 |
| 11 | + vLLM RMSNorm | 656.2 | 0.742 |
| -- | (alt) vLLM RoPE instead of Helion | 623.5 | 0.705 |
| 12 | pure-local 2D model (localfused) | 801.1 | 0.906 |
| T | vLLM native (eager) | 883.9 | 1.000 |

Notes:
- Dominant DTensor-path levers: cudagraph (0.03 -> 0.48), Helion RoPE (+10%),
  FusedQKV+gate_up (+5%), and **vLLM all-reduce (+27%)** -- the single biggest
  kernel win. vLLM RMSNorm adds a further +3% to reach the DTensor ceiling 656.2
  (0.742x).
- SiluAndMul (+0.6%), FA3 (+0.9%), fused add+RMSNorm (+0.1%) are near-neutral on
  this path -- TorchTitan already has fast paths for these.
- vLLM RoPE is a regression vs Helion (623.5 < 656.2): Helion RoPE is the better
  choice; confirms vLLM-RoPE is neutral-to-negative.
- The DTensor ceiling is 0.742x. The remaining gap to native is the DTensor
  dispatch / cross-rank arrival-spin in the all-reduce (comm itself is already
  equal to native; proven by per-rank trace analysis: pure comm identical, the
  spread is arrival spin). Switching to the **pure-local 2D model** (plain local
  tensors + manual `tensor_model_parallel_all_reduce`, Megatron scheme) removes
  DTensor dispatch so ranks arrive together and the spin collapses -> 0.906x,
  closing most of the residual. The last ~9% is vLLM-native-only fusion not
  ported.

## W2 -- bs=32, input_len=4096, generate=1024 tokens

Ladder (cumulative, full_decode_only). Ratio vs native = 2366.5 tok/s.

| # | rung | tok/s | ratio |
|---|------|-------|-------|
| 1 | baseline (eager DTensor, SP off) | 98.0 | 0.041 |
| 2 | + compile (aot_eager, per-layer) | 249.5 | 0.105 |
| 3 | + cudagraph (FULL_DECODE_ONLY) | 1334.3 | 0.564 |
| 4 | + tree all-reduce (NCCL Ring,Tree) | 1335.6 | 0.564 |
| 5 | + Helion RoPE | 1476.6 | 0.624 |
| 6 | + FusedQKV + fused gate_up (proper) | 1528.5 | 0.646 |
| 7 | + SiluAndMul (vLLM kernel) | 1518.0 | 0.642 |
| 8 | + FlashAttention (FA3, vs CUSTOM varlen) | 1542.6 | 0.652 |
| 9 | + fused add+RMSNorm (vLLM kernel) | 1557.5 | 0.658 |
| 10 | + vLLM all-reduce | 1742.4 | 0.736 |
| 11 | + vLLM RMSNorm | 1797.1 | 0.759 |
| -- | (alt) vLLM RoPE instead of Helion | 1704.3 | 0.720 |
| 12 | pure-local 2D model (localfused) | 2246.7 | 0.949 |
| T | vLLM native (eager) | 2366.5 | 1.000 |

Notes:
- Same shape as W1, but every ratio is HIGHER: the larger workload amortizes the
  DTensor overhead. cudagraph floor 0.564 (vs 0.484 W1); DTensor ceiling 0.759
  (vs 0.742 W1); **pure-local 0.949 (vs 0.906 W1)** -- nearly closes the gap.
- vLLM all-reduce is still the biggest kernel lever but **smaller than W1
  (+11.9% vs +27%)**: at bs=32/in=4096 the AR messages are larger and less
  latency-bound, so swapping NCCL ring -> vLLM one-shot/multimem helps less.
  This is the expected decode-latency signature -- the AR win shrinks as the
  workload grows.
- SiluAndMul is a slight regression here (-0.7%), FA3 +0.9%, fused add+RMSNorm
  +0.1%, vLLM RMSNorm +3.1%: all near-neutral, same as W1. vLLM RoPE again
  regresses vs Helion (1704.3 < 1797.1).
- Residual above the DTensor ceiling (0.759) is again DTensor dispatch /
  cross-rank arrival-spin, not comm; pure-local removes it to reach 0.949. The
  last ~5% is native-only fusion not ported.

## spmd_types execution as a vLLM runtime (W1, 32B TP=8)

`--model-2d spmd`: same plain-local forward as `localfused` (vLLM paged
attn/rope/silu, threaded residual + fused add+RMSNorm) but the Partial ->
Replicate TP all-reduce (after wo/w2 + the vocab-parallel embedding) goes
through `spmd_types.redistribute(P->R)` with typecheck OFF, instead of vLLM's
custom all-reduce. spmd_types is otherwise a typechecking tool, not a runtime;
this measures it AS a runtime.

| path | tok/s | ratio | TP all-reduce |
|------|-------|-------|---------------|
| DTensor ceiling (vLLM-AR + vLLM-RMSNorm) | 656.2 | 0.742 | vLLM custom |
| spmd_types, NCCL AR (--model-2d spmd, default dist) | 617.3 | 0.698 | spmd.redistribute -> NCCL |
| spmd_types, vLLM AR (--model-2d spmd + set_dist shim) | 747.8 | 0.846 | spmd.redistribute -> vLLM custom |
| pure-local (--model-2d localfused) | 787.5 | 0.891 | vLLM custom (direct) |
| native | 883.9 | 1.000 | -- |

Finding: the TP all-reduce backend dominates. With spmd's default dist (NCCL),
spmd_types execution (0.698x) is SLOWER than pure-local (0.891x) and even below
the DTensor ceiling, despite being plain-local (no DTensor dispatch). Routing
`spmd.redistribute(P->R)` through vLLM's custom one-shot AR (a `spmd.set_dist`
shim, `_VLLMAllReduceDist`) lifts it to 747.8 (0.846x), +21% -- closing most of
the gap. The residual ~5% vs pure-local is `spmd.redistribute`'s out-of-place
path (`result = x.clone()`) plus the shim's `copy_` back (vLLM AR is
out-of-place), i.e. extra elementwise kernels per AR, not the collective. So
spmd_types ~= plain tensor as a runtime once its redistribute uses the same
collective; an in-place redistribute would recover the last few percent.

## Cross-workload summary (vs native eager)

| stage | W1 (bs8/in1024/gen128) | W2 (bs32/in4096/gen1024) |
|---|---|---|
| cudagraph floor | 0.484 | 0.564 |
| + all kernel rungs (DTensor ceiling) | 0.742 | 0.759 |
| pure-local 2D model | 0.906 | 0.949 |

The DTensor path plateaus at ~0.74-0.76x; closing the rest requires leaving
DTensor in the hot path (pure-local), which reaches 0.91-0.95x. The remaining
~5-9% is vLLM-native-only fusion.
