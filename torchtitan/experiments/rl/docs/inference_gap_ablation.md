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

(running; appended when complete)
