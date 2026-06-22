---
name: inference_perf_hillclimb
description: Profiler-driven hill-climbing to close the inference throughput gap between TorchTitan's unified model (running inside vLLM) and vLLM's native model. Benchmark with generate.py --benchmark, climb optimization rungs (compile / cudagraph / fused kernels), profile torchtitan vs the native target, then patch the single biggest gap at a time and re-measure. Use when the user wants to benchmark or optimize RL inference generation speed, reproduce previous hill climbing study, or invokes /inference_perf_hillclimb.
---

# Inference Performance Hill-Climbing

Close the gap between the TorchTitan unified model (in vLLM) and vLLM's native
model on inference throughput, one profiler-found optimization at a time. Harness:
`torchtitan/experiments/rl/generate.py --benchmark`. Append-only running log +
full per-rung numbers live in `torchtitan/experiments/rl/docs/inference_gap_ablation.md`.
Goal: match vLLM native 100%.

## Scope: close the IMPLEMENTATION gap, not the serving knobs

This skill makes torchtitan's model run as fast as vLLM's NATIVE model at a FIXED
model, workload, precision, and topology -- i.e. it closes the framework/overhead
gap (target = native at the *same* config; ratio -> 1.0). It is explicitly NOT
about the orthogonal levers that change the setup itself:
- increasing batch size / seq_len / amount of input data
- changing the model
- lower precision (fp8 / quantization)
- speculative decoding
- disaggregated prefill
- changing topology / number of GPUs

Those move ABSOLUTE throughput but apply equally to native -- they don't change
the torchtitan-vs-native ratio this skill chases. Hold them FIXED and IDENTICAL
between torchtitan and native when measuring; otherwise you're no longer measuring
the implementation gap.

## Before any run: confirm with the user (don't assume defaults)

- **Model + parallelism**: which model, TP degree, SP/DP/PP/EP (e.g. Qwen3-32B,
  TP=8, SP off). Sets the sharding, all-reduce sizes, cudagraph capture sizes, and
  what "native" even is -- the dominant gap and right knobs change with it.
- **Versions**: vLLM / torchtitan / PyTorch-nightly. They move fast and change
  kernels, cudagraph behavior, custom AR, and APIs. Record all three -- numbers
  across versions are not comparable.
- **Workload**: input_len, batch_size, gen_tokens. Standard: W1 bs8/in1024/out128
  (priority); W2 bs32/in4096/out1024 (needs `--max-seq-len 8192`). Sets prefill-vs-
  decode weighting, AR/capture sizes, memory -- the dominant gap depends on it
  (e.g. prefill capture helps short-gen W1 more than long-gen W2).
- **compile + cudagraph** (the two settings that dominate perf; use AskUserQuestion):
  - compile: `aot_eager` (validated torchtitan default, per-layer torch.compile) |
    `inductor` (heavier) | `off`. **The fair target is native(eager)**, NOT
    native(inductor): inductor fuses the TP all-reduce into gemm/RMSNorm
    (`fuse_allreduce_rms`), eager does not -- it's ~+5% and not in the aot_eager
    path, so comparing to it invents a fake gap.
  - cudagraph mode: pick ONE for baseline AND target; never mix across rungs (see
    "cudagraph modes" under Harness).

## The hill-climbing loop (the core method -- PROFILE-DRIVEN, measure don't guess)

**Every rung is justified by a PROFILE, never intuition**, and KEPT only if a
re-benchmark moves throughput. The flag list is the residue of past profiles; the
method is: profile -> single biggest gap -> patch exactly that -> re-measure ->
re-profile. Two rules: (a) **trace time misleads** -- a huge all-reduce duration is
usually cross-rank SPIN (sync wait), not reducible compute; confirm with tok/s.
(b) **the biggest gap is often NOT a kernel** -- prefill cudagraph capture and
arrival-spin were the top levers; profile broadly.

1. Confirm model / parallelism / versions / workload / compile+cudagraph (above).
2. Establish the ladder once, capture the **target**:
   `--native --compile aot_eager --cudagraph on` (same workload).
3. **PROFILE both** torchtitan-best and native (`--profile`). Decompose for the gap:
   - kernels: `/tmp/ablation_logs/analyze_trace.py` (single-rank diff: kernel us,
     extra kernels present in one but not the other, CPU-op deltas).
   - collectives: `/tmp/ablation_logs/analyze_ar_spread.py` over the 8 per-rank
     traces -- min(dur)=pure comm, spread(dur)=ARRIVAL SPIN. Single-rank kernel
     sums mislead for collectives.
   - host-bound: per-rank GPU busy% + idle-gap location (eager prefill vs captured
     decode). Low busy% on the driver rank => host-launch-bound.
4. Pick the SINGLE biggest gap (kernel / spin / host lag / capture coverage).
5. Implement exactly that as a new knob (custom_op patch in `kernel_ablation.py`, a
   `--model-2d` variant, or a cudagraph/config flag); often "make torchtitan do
   what native does here".
6. **Validate numerics**: example mode (no --benchmark) at TP>=2, greedy text
   bitwise-identical with vs without the patch.
7. Commit the rung (one git commit per table line).
8. Re-benchmark (W1, then W2); append to the doc. KEEP only if throughput moved
   (trace promise != throughput).
9. Repeat from step 3 (RE-PROFILE -- the biggest gap shifts after each patch) until
   within ~3% of target, or the remainder is structural (document it).

## Gap buckets: how to spot, what we found, the lever

- **CPU / host overhead** (DTensor dispatch, ATen dispatch, per-launch Python). In
  an EAGER region the host can't feed the GPU -> it starves. Spot: per-rank GPU
  busy% < ~90% with idle gaps before compute kernels. Levers: cudagraph capture
  takes the CPU out of the per-kernel path (capturing PREFILL too was the single
  biggest lever, ~+20%); OR remove DTensor from the forward (`--model-2d`
  pure-local); OR cut per-launch overhead. Hidden under a captured graph -- only
  bites in eager prefill. NOTE: the structural lever is removing DTensor, not
  "2D" -- compile+cudagraph kill DTensor's CPU cost but its boundaries
  (from_local / subclass dispatch / placement views) fragment the captured graph
  -> launch jitter. Tensor rank is a red herring (3D==2D); register_sharding
  REGRESSES (-26%, more DTensor dispatch).
- **Single kernel slowness** (same shape, torchtitan > native). Spot:
  `analyze_trace.py` per-kernel us. Found: NCCL all-reduce 23us vs vLLM custom
  one-shot AR 6.3us for small decode messages (a real algorithm difference);
  faster RoPE/attn variants. Lever: `--allreduce-vllm`, `--rope-kernel helion`,
  `--attn-backend flash`. Caveat: GEMM and RMSNorm are already at native parity
  (torch/Quack `rms_norm` even BEATS vLLM's) -- don't chase them.
- **Fused vs separate kernels** (native fuses what torchtitan runs as N kernels:
  fused add+RMSNorm, fused QKV / gate-up GEMM, SiluAndMul). Spot: "extra kernels"
  in the torchtitan trace + higher CPU-op count. Lever: `fuse_qkv=True` +
  `fused_swiglu`, `--fused-addnorm`, `--silu-vllm`. Caveat: mostly a launch-count /
  host win -> matters in eager regions, shrinks under cudagraph.
- **Communication straggler** (one rank reaches the collective late, the other 7
  SPIN). Spot: `analyze_ar_spread.py` (min(dur)=comm==native, spread=arrival spin).
  Find WHAT BOUNDS the straggler: *compute?* per-rank compute time is usually
  UNIFORM, so no; *host-launch?* the DRIVER rank (rank 0) runs scheduler/sample/
  output on its Python thread and falls behind in the eager prefill -> arrives
  last (busy 54% vs 94%, spin ~600us) -- the usual cause; fix = capture prefill
  (spin ~600us -> ~3us, the +20% lever) or lighten the per-step host path.
  *hardware/topology?* if the SAME rank straggles in NATIVE too, it's a
  GPU/NVLink-position effect, not fixable in the model. Rule: the AR kernel is
  innocent -- fix uneven ARRIVAL, not the collective.

Operational gotchas: check GPU contention first (`nvidia-smi`; a co-tenant job
spikes variance to +/-150 vs clean +/-1-7 -- re-run clean); NO Python-side
logging/prints inside a patched forward (forces a torch.compile graph break);
launch long runs detached (`setsid nohup ... &`) and clean GPU stragglers between.

## Harness

**Build/restore**: stock `generate.py` on main is an EXAMPLE (single prompt). Bring
the benchmark harness over from branch `ablation-inference`:
`git checkout ablation-inference -- torchtitan/experiments/rl/generate.py` + the
patch modules (`models/{kernel_ablation,qwen3_vllm_2d,helion_rope,vllm_fused_ops}.py`),
fix the import (`from torchtitan.experiments.rl.examples.alphabet_sort import
config_registry`), and add `rl_grpo_qwen3_32b` (mirror `rl_grpo_qwen3_14b`,
`model_registry("32B")`, TP=8). `--benchmark` builds the engine OUTSIDE the timed
region (excludes startup/compile/capture), runs `--warmup-runs` then times
`--num-runs`, reports tok/s = batch*gen/wall; feeds synthetic token-ids (skips the
tokenizer); sets `enable_prefix_caching=False`. Build the cudagraph
`CompilationConfig` EXPLICITLY (the stock helper only emits `cudagraph_mode="full"`).

**Launch**: `torchrun --nproc_per_node=<TP> generate.py --benchmark ...`. Env:
conda `titan-rl` (see `torchtitan/experiments/rl/README.md` to build it).

**Knobs** are the residue of past profiles -- NOT a required checklist, and NOT
guaranteed to exist on a fresh checkout (the harness is rebuilt each time; re-add
what you need, add NEW knobs for new gaps):
- **Kernel patches** (torchtitan model; stack as the profile dictates):
  `--rope-kernel {helion,vllm} --silu-vllm --rmsnorm-vllm --allreduce-vllm
  --fused-addnorm --attn-backend {custom,flash}` -- custom_op monkeypatches in
  `models/kernel_ablation.py` / `vllm_fused_ops.py` (survive compile+cudagraph).
  FusedQKV+gate_up: use config `fuse_qkv=True` + `overrides/fused_swiglu.py`
  `@override`, not the superseded `--merged-gemm`.
- **Whole-model paths** (mutually exclusive with kernel patches):
  `--model-2d {local,localfused,spmd,local3d,dtensor}`. `localfused` = pure-local
  (no DTensor) + fused add+RMSNorm (best); `spmd` = localfused but the TP AR routes
  through `spmd_types.redistribute(P->R)`. spmd_types is a pre-run CHECK (validates
  SPMD sharding via typechecking) -- keep it OFF for perf runs.
- **compile / cudagraph**: `--compile {off,aot_eager,inductor}` `--cudagraph
  {on,off}` `--cudagraph-mode {full_decode_only,full,full_and_piecewise}`. Capture
  PREFILL with `--cudagraph-mode full --max-num-batched-tokens <P> --max-capture-size
  <P>`, where P >= the prefill CHUNK size (= max_num_batched_tokens), NOT input_len.
  Decode capture sizes default to powers of 2 up to max_num_seqs (= batch).
- **misc**: `--nccl-algo`, `--native` (target), `--profile` (per-rank chrome trace),
  `--max-seq-len` (long prompts).

### cudagraph modes (pick ONE for baseline AND target; never mix across rungs)
These change WHO owns torch.compile:
- **full_decode_only / full**: `CompilationConfig(mode=NONE)` -- vLLM does NOT
  compile; torchtitan's per-layer `aot_eager` is the only compile. full_decode_only
  = decode captured, eager prefill; **full** = also captures prefill (needs
  `--max-capture-size >= chunk`). KEEPS torchtitan compile.
- **full_and_piecewise**: `mode=VLLM_COMPILE, backend="eager"` AND `config.compile
  =off` -- vLLM compiles the whole model (to split the graph around collectives),
  so per-layer aot_eager is turned off. DROPS torchtitan compile for vLLM's.
Mixing FULL and FAP across rungs mixes two compile strategies (a bug we hit --
re-run the WHOLE ladder on ONE mode). native FULL ~= native FAP (878.9 vs 879.1 W1).

## Example ladder + results

ONE example trajectory, NOT a recipe -- each step was the biggest profiler-found
gap at that point: baseline (eager DTensor, SP off) -> +compile(aot_eager) ->
+cudagraph -> +tree all-reduce -> +Helion RoPE -> +FusedQKV/gate_up -> +SiluAndMul
-> +FA3 -> +fused add+RMSNorm -> +vLLM custom AR -> +vLLM RMSNorm -> pure-local.
Big jumps: cudagraph (0.03 -> 0.48x), vLLM custom AR (biggest KERNEL lever),
dropping DTensor (pure-local). SiluAndMul / vLLM-RMSNorm / vLLM-RoPE were
neutral-to-negative (torchtitan already fast). The biggest lever overall -- PREFILL
cudagraph capture (~+20%) -- isn't a kernel; it was found later by profiling the
residual arrival-spin straggler.

Ratios vs native eager (32B TP=8, full_decode_only; full per-rung numbers and the
FULL+prefill ladder are in the doc):

| path | W1 (bs8/in1024/gen128) | W2 (bs32/in4096/gen1024) |
|---|---|---|
| DTensor ceiling (vLLM-AR + vLLM-RMSNorm) | 0.742 | 0.759 |
| spmd_types, NCCL AR (`TT_SPMD_NCCL_AR=1`) | 0.698 | 0.796 |
| spmd_types, vLLM AR (set_dist shim) | 0.846 | 0.877 |
| pure-local (localfused) | 0.891 | 0.926 |
| native | 1.000 | 1.000 |

Capturing PREFILL too (`--cudagraph-mode full` + the max-batched-tokens/capture
knobs) lifts every row ~+0.07-0.18x: DTensor ceiling -> ~0.90x, pure-local ->
~0.975x, spmd+vLLM-AR -> ~0.93x, at BOTH W1 and W2; native barely moves (its
prefill is already lean).
