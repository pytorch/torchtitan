---
name: inference_perf_hillclimb
description: Profiler-driven hill-climbing to close the inference throughput gap between TorchTitan's unified model (running inside vLLM) and vLLM's native model. Benchmark with generate.py --benchmark, climb optimization rungs (compile / cudagraph / fused kernels), profile torchtitan vs the native target, then patch the single biggest gap at a time and re-measure. Use when the user wants to benchmark or optimize RL inference generation speed, reproduce the [3/3] "bridge the gap" study, or invokes /inference_perf_hillclimb.
---

# Inference Performance Hill-Climbing

Close the gap between the TorchTitan unified model (in vLLM) and vLLM's native
model on inference throughput, one profiler-found optimization at a time. Harness
is `torchtitan/experiments/rl/generate.py --benchmark`. Running log of method +
results lives in `torchtitan/experiments/rl/docs/inference_gap_ablation.md`
(append-only; keep adding findings).
Goal: match the vllm native perf 100%.

## Always first: Confirm the model and the parallelisms using

## Always first: confirm the vllm version, torchtitan version and PyTorch nightly version

## Always first: confirm the benchmarking data size: input sequence length, batch size, number of tokens generated

## ALWAYS FIRST: confirm compile + cudagraph config with the user

Before launching ANY benchmark run, confirm the two settings that dominate
inference perf with the user (use AskUserQuestion). Do not assume defaults.

1. **compile**: `off` | `aot_eager` | `inductor`
   - torchtitan path: `aot_eager` is the validated default (per-layer
     torch.compile). `inductor` per-layer is possible but heavier.
   - native target: vLLM accepts only `eager`/`inductor` for VLLM_COMPILE; the
     harness maps `aot_eager -> vLLM "eager"`. **The fair target is
     native(eager)**, NOT native(inductor) (inductor adds comms/norm fusion the
     aot_eager path does not -- worth only ~5%, see Pitfalls).
2. **cudagraph**: `on` | `off`, and **mode**: `full` | `full_and_piecewise`
   - `full` (default) is best for the torchtitan path: per-layer aot_eager +
     vLLM FULL cudagraph (vLLM compile mode = NONE).
   - `full_and_piecewise` forces vLLM to compile the whole model (so per-layer
     aot_eager is turned off, vLLM "eager" backend used). Measured slightly
     WORSE for torchtitan; provided for parity with native.

State the chosen config in the run command and in the results log.

## Making generate.py a benchmark script (changes to apply)

The stock `generate.py` on main is an EXAMPLE script (single chat prompt). On a
fresh branch it must be re-extended into a benchmark harness. The full restored
version lives on branch `ablation-inference` -- bring it over with
`git checkout ablation-inference -- torchtitan/experiments/rl/generate.py` plus
the patch modules (`models/{kernel_ablation,qwen3_vllm_2d,helion_rope,vllm_fused_ops}.py`)
and `.claude/skills/inference_perf_hillclimb/SKILL.md`, then fix the
`config_registry` import to the new path
(`from torchtitan.experiments.rl.examples.alphabet_sort import config_registry`).
The benchmark mode is these changes on top of the example script:

1. `--benchmark` mode: build the engine OUTSIDE the timed region (exclude
   startup/compile/capture); run `--warmup-runs` (default 2) full generation
   passes; then time `--num-runs` passes, report
   tok/s = batch_size * generated_tokens / wall_time. (Per-run times are flat
   run-to-run -> confirms compile/capture excluded.)
2. Skip the tokenizer: feed synthetic token-id prompts via
   `engine.renderer.render_cmpl([{"prompt_token_ids": ids}])` from
   `_make_token_id_lists(vocab_size, input_len, batch_size)`.
3. Workload flags: `--batch-size --input-len --max-tokens --ignore-eos`
   (exactly max_tokens/req) `--temperature 0` (greedy).
4. Override flags: `--model-path` (ABS hf dir), `--tp`, `--max-seq-len`
   (override per-layer rope.max_seq_len for long prompts), `--compile
   {off,aot_eager,inductor}`, `--cudagraph {on,off}`, `--cudagraph-mode`,
   `--nccl-algo`, `--native` (target), `--profile` (per-rank chrome trace).
5. `max_num_seqs = max(args.max_num_seqs, batch_size)` so cudagraph capture
   sizes cover the decode batch (cap = batch_size; see below).
6. Build the cudagraph `CompilationConfig` EXPLICITLY (the stock
   `VLLMCudagraphConfig.get_vllm_compilation_config` only emits `cudagraph_mode=
   "full"`); the harness needs all three modes -- see next section.
7. `enable_prefix_caching=False` in benchmark mode (so prefill is measured, not
   deduped across runs).
8. `rl_grpo_qwen3_32b` is NOT in the new config_registry -- add it (mirror
   `rl_grpo_qwen3_14b`, `model_registry("32B")`, TP=8, abs hf path).

## cudagraph mode: pick ONE, never mix (FULL vs FAP differ in COMPILE owner)

`--cudagraph-mode {full_decode_only, full, full_and_piecewise}`. These are NOT
just capture differences -- they change WHO owns torch.compile:
- **full_decode_only / full**: `CompilationConfig(mode=NONE, cudagraph_mode=...)`.
  vLLM does NOT compile; torchtitan's per-layer `aot_eager` is the only compile.
  full_decode_only = full cudagraph for decode, eager prefill (guaranteed);
  full = also tries prefill capture (attn-backend-dependent). **Keeps torchtitan
  compile.** Needs a decode cap (capture sizes = powers of 2 up to max_num_seqs).
- **full_and_piecewise**: `CompilationConfig(mode=VLLM_COMPILE, backend="eager",
  cudagraph_mode="FULL_AND_PIECEWISE")` AND set `config.compile=off`. Piecewise
  REQUIRES vLLM whole-model compile to split the graph around collectives, which
  conflicts with per-layer compile -- so per-layer must be turned off. **Drops
  torchtitan compile for vLLM's.**
Mixing FULL and FAP across rungs = mixing two compile strategies (the bug we hit:
re-run the WHOLE ladder on ONE mode). For "torchtitan compile, never vLLM
compile", use **full_decode_only** (default). native FULL ~= native FAP on W1
(878.9 vs 879.1), so the target is ~unchanged either way.

## Harness usage

Always launch via `torchrun --nproc_per_node=<TP>`. Kernel rungs (torchtitan
model, cumulative): `--rope-kernel {helion,vllm} --silu-vllm --rmsnorm-vllm
--allreduce-vllm --merged-gemm --fused-addnorm --attn-backend {custom,flash}`
(monkeypatches in `models/kernel_ablation.py` / `models/vllm_fused_ops.py`, each
a torch.library custom_op so it survives compile+cudagraph). FusedQKV+gate_up are
better done via the PROPER mechanisms (config `fuse_qkv=True` +
`overrides/fused_swiglu.py` `@override`), not `--merged-gemm`. Whole-model
native-style paths: `--model-2d {local,localfused,local3d,dtensor}` (mutually
exclusive with kernel rungs); `localfused` (pure-local + fused add+RMSNorm) is
the best, 0.91x.

Two workloads (Qwen3-32B, TP=8): W1 bs=8/in=1024/out=1024 (priority);
W2 bs=32/in=4096/out=1024 (needs `--max-seq-len 8192`).

## The hill-climbing loop

1. Confirm compile/cudagraph config (above).
2. Establish the ladder once: baseline (eager DTensor, SP off) -> +compile ->
   +cudagraph(FULL) -> +tree-AR -> kernel rungs. Record each tok/s.
3. Capture the **target**: `--native --compile aot_eager --cudagraph on`.
4. If torchtitan < target, PROFILE both: `--profile` on torchtitan-best and on
   native (same workload). Analyze rank-0 chrome traces (see
   `/tmp/ablation_logs/analyze_trace.py` pattern): total kernel launches, total
   CUDA us, top kernels by time, kernels present in one but not the other, CPU
   op count deltas.
5. Pick the SINGLE biggest gap. Implement a patch on top of the current best as
   a new flag in `kernel_ablation.py`.
6. **Validate numerics**: run example mode (`generate.py` no --benchmark) with
   and without the patch at TP>=2 and diff the greedy generated text -- it must
   be identical (bitwise). Kernel rungs also apply in example mode for this.
7. **Commit the rung** as its own git commit (one commit per table line).
8. Re-benchmark (W1, then W2). Append the result to the doc.
9. Repeat from 4 until torchtitan matches the target (within ~3%) or the
   remaining gap is structural (document it).

## An example hill climbing ladder
Target (vllm native Qwen3, Cudagraph(“FULL_AND_PIECEWISE”) + compile(“aot_eager”))
Baseline (vllm + torchtitan, eager DTensor, disabled SP, fixed some code changes)
Remove DTensor overhead by Compile(aot_eager)
Cudagraph(FULL_AND_PIECEWISE)
Use tree based TP all-reduce
RoPE (patch to Helion Kernel)
FusedQKV + Fused gate_up (not enabled by default)
SiluAndMul (patch to vllm’s kernel)
Fused add_rmsnorm (using vllm’s kernel instead)



## Pitfalls / hard-won lessons (read before optimizing)

- **Target = native(eager), not native(inductor).** Comparing against
  native-inductor invents a fake "structural comms-fusion gap": inductor fuses
  the TP all-reduce into gemm/RMSNorm (`fuse_allreduce_rms`), eager does NOT.
  native(eager) runs the SAME standalone all-reduce kernels as torchtitan, so
  there is no fusion to chase; inductor is only ~+5%.
- **Measure every patch; trace time can mislead.** The all-reduce had a huge
  trace duration (multimem 4.75x), but flattening 3D->2D moved throughput 0% --
  most of that time is cross-rank sync wait, not reducible compute. Confirm a
  fix with a throughput delta, not the trace alone. (GEMM kernel times match
  native across traces, so traces ARE comparable for counts/types.)
- **AR "slowness" is arrival SPIN, not kernel cost.** Decompose any collective
  with `analyze_ar_spread.py`: across the 8 per-rank traces, collectives end
  together so spread(dur)=arrival spin and min(dur)=pure comm. If comm matches
  native but spin doesn't, the fix is upstream (make per-rank work between
  collectives uniform), not the collective itself.
- **The structural lever is removing DTensor from the forward, NOT "2D".**
  compile+cudagraph kill DTensor's CPU cost but its boundaries
  (from_local/subclass-dispatch/views) fragment the graph -> launch jitter ->
  spin. `--model-2d local` (pure local tensors + manual collectives) is the win
  (0.70x->0.86x); `localfused` adds residual fusion (0.91x). Tensor rank is a
  red herring (3D==2D); register_sharding REGRESSES (-26%, more DTensor dispatch).
- **GEMM is already at native parity** in every best path (manual/merged fused
  QKV+gate_up). Don't chase fused_qkv/fused_swiglu for throughput -- they only
  matter for code quality / upstreaming the fusion.
- **Check GPU contention first** (`nvidia-smi`): a co-tenant job spikes variance
  (clean runs are +/-1-7 tok/s; contended showed +/-150). Re-run clean.
- **No Python-side logging/prints inside a patched forward** -- it forces a
  torch.compile graph break.
- **Long runs:** launch detached (`setsid nohup ... &`) so they survive session
  interruptions; clean GPU stragglers between runs.

## What's already known (32B TP=8, W1 FAP, vs native eager 879)

Ladder (full numbers + per-round findings in `docs/inference_gap_ablation.md`,
see its "FINAL SUMMARY"):

| stage | tok/s | ratio |
|---|---|---|
| cudagraph (DTensor 3D) | 424 | 0.48x |
| + Helion RoPE + vLLM all-reduce + merged GEMMs + fused-addnorm + FA3 | 612 | 0.70x |
| 2D-DTensor model (register_sharding, qwen3_vllm.py) | 637 | 0.72x |
| **pure-local model (no DTensor in forward)** | 754 | 0.86x |
| **pure-local + fused add+RMSNorm** | **801** | **0.91x** |
| native | 879 | 1.00x |

**ROOT CAUSE (proven): the residual all-reduce gap is cross-rank ARRIVAL SPIN,
not a slower kernel.** Per-rank trace method `/tmp/ablation_logs/analyze_ar_spread.py`
(all ranks' AR kernels end together, so across-rank spread(dur) = arrival spin,
min(dur) = pure comm): pure comm is IDENTICAL on every path (~146k us); only spin
differs (DTensor-best 597k -> pure-local 278k -> native 109k). compile+cudagraph
remove DTensor's CPU cost, but DTensor boundaries (from_local / subclass dispatch
/ placement views, +32% CPU ops) fragment the captured graph into eager glue ->
per-rank launch jitter (host-load sensitive) -> ranks arrive spread out -> the
collective spins on-device (GPU time cudagraph can't remove).

**THE LEVER: remove DTensor from the forward (pure-local), not the kernels.**
`--model-2d local` runs the whole forward on plain local tensors with manual vLLM
all-reduce after wo/w2 and the vocab-parallel embedding (Megatron scheme; norms/
attention need no collective); `localfused` adds native-style residual fusion.
Flag: `--model-2d {local,localfused,local3d,dtensor}` (whole-model, mutually
exclusive with kernel rungs), in `models/qwen3_vllm_2d.py`.

What does NOT help: **register_sharding / DTensor-native rope -26%** (adds DTensor
dispatch to the hot path); **tensor rank** (pure-local 3D == 2D, 755==754);
**embed-AR alone** (neutral); **fused_qkv / fused_swiglu** (GEMM already at native
parity, nvjet 459k both). SiluAndMul / vLLM-RMSNorm / vLLM-RoPE neutral-to-negative.
Remaining 0.91x->1.0x = Python masked-lookup embedding + per-layer head split/view
that native fuses (residual spin), and is host-load sensitive.

## Environment

conda env `titan-rl` (python at
`/data/users/jianiw/miniconda3/envs/titan-rl/bin`); detached shells need
`export PATH=.../titan-rl/bin:$PATH` (conda `activate` does not apply). HF
weights under `/data/users/jianiw/model/Qwen3-{1.7B,8B,32B}`. Versions are
pinned in the doc.
