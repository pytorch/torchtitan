# Bitwise Numerics and Batch Invariance

This document describes the sources of numerical non-determinism in the RL
training loop and the mechanisms used to eliminate them, with pointers to where
each mechanism lives in the code. For user-facing setup and run instructions,
see the "Batch-invariant mode" and "Verifying generator/trainer logprob parity"
sections of `torchtitan/experiments/rl/README.md`.

## Problem

In RL for LLMs, the generator computes log-probs for completions in one batch
composition (e.g. 8 completions), while the trainer recomputes them in a
different batch composition (e.g. 2 completions after DP sharding). Without
special care, the same input can produce different log-probs in different batch
contexts due to floating-point accumulation order differences in GEMM, softmax,
reductions, and collective ops. Even tiny drift can flip top-k expert routing in
MoE models, so the two paths must match op-for-op.

## Where it is wired

Batch-invariant mode is gated by `DebugConfig(batch_invariant=True,
deterministic=True)` on both the trainer and generator configs. The shared
toggle is `set_batch_invariance()` in `torchtitan/distributed/utils.py`, called
from:

- `torchtitan/experiments/rl/actors/trainer.py` (trainer)
- `torchtitan/experiments/rl/actors/generator.py` (generator)
- `torchtitan/experiments/rl/generate.py` (standalone generation)

## Sources of non-batch-invariance

For the same input, the trainer and generator must run the same ops with the
same accumulation order. Three groups of fixes make that hold.

**Shared op overrides** -- applied by `set_batch_invariance()`
(`torchtitan/distributed/utils.py`), which both actors call:

- **`mm` / `addmm` / `_log_softmax` / `mean.dim`:** cuBLAS and the default
  reduction kernels pick tile/block schedules from the input shape, so the
  accumulation order (and rounding) changes with batch size. Overridden with the
  upstream
  [`batch_invariant_ops`](https://github.com/thinking-machines-lab/batch_invariant_ops)
  Triton kernels, which use a fixed tile iteration order.
- **Flash attention split-k:** FA3 picks `num_splits` from the sequence length,
  changing the partial-sum reduction tree. Forced to `num_splits=1` when
  `is_in_batch_invariant_mode()` is True
  (`torchtitan/models/common/attention.py`, `VarlenAttention.forward`); FA2 is
  already batch-invariant.
- **NCCL collectives:** all-reduce may use ring/tree algorithms with varying
  channel counts, reordering the cross-rank reduction. Forced to single-channel
  tree (`NCCL_ALGO=allreduce:tree`, `NCCL_MIN/MAX_NCHANNELS=1`,
  `NCCL_PROTO=Simple`, etc.) to match vLLM; must be set before
  `dist.init_process_group`.
- **Reduced-precision reductions and TF32:** disabled
  (`allow_bf16/fp16_reduced_precision_reduction = False`,
  `torch.backends.cuda.matmul.allow_tf32 = False`,
  `torch.backends.cudnn.allow_tf32 = False`).

**Generator-only patches** (`torchtitan/experiments/rl/batch_invariance.py`) --
the generator runs the same model inside vLLM, whose fused kernels must be routed
back to the trainer's ops:

- **Attention (Varlen / Flex / GPT-OSS):** the generator must run the same
  attention kernel as the trainer for whichever backend the config uses.
  - *Varlen* (Qwen3, the default): the generator runs torchtitan's own
    `VarlenAttention` through vLLM's `CUSTOM` backend -- the same FA3 varlen
    kernel as the trainer. The shared `num_splits=1` fix above removes the only
    batch-dependent split-k, so the two match bitwise.
  - *Flex* (Qwen3 flex): the generator runs vLLM's `FLEX_ATTENTION` backend.
    `BatchInvariantFlexConverter` (a `ModelConfigConverter` wired in via
    `converters=[BatchInvariantFlexConverter.Config()]`) pins
    `BLOCK_M = BLOCK_N = 16` and `BACKEND = "TRITON"` on every `FlexAttention`
    layer, so both sides use the same Triton tile size and avoid the
    `flex_decode` kernel.
  - *GPT-OSS* (varlen + per-layer sliding window + attention sinks): the same
    `CUSTOM` varlen path and `num_splits=1` as Varlen, with the sliding-window
    size baked into each layer's `VarlenAttention` config (identical on both
    sides). The attention sinks are a post-softmax rescale
    (`apply_attention_sink_rescale`, `sigmoid(lse - sinks)`); the vLLM wrapper's
    `_inject_attention_sinks` installs that same rescale as the generator
    attention's `out_transform`, so the sink math matches the trainer.
- **`patch_bmm_for_batch_invariance`:** `batch_invariant_ops` overrides
  `mm`/`addmm` but not `bmm`. The MoE router gate (3-D activation @ 2-D weight)
  lowers to `aten::bmm` in the generator but `aten::mm` in the trainer, so the
  gate scores drift and flip top-k expert routing. Installs vLLM's
  batch-invariant `bmm`.
- **`force_logprobs_fn_for_batch_invariance`:** vLLM's v2 GPU sampler computes
  per-token logprobs with a fused Triton kernel that inlines
  `log(softmax(logits))` and never calls PyTorch ops. Replaced with the trainer's
  `compute_logprobs`, so both share one logprob code path.

**RoPE cache dtype (`freqs_cis`):** `freqs_cis` is stored in fp32 and passed as a
forward input to each `TransformerBlock`. FSDP2's
`MixedPrecisionPolicy(cast_forward_inputs=True)` would cast it to bf16 at each
FSDP boundary, so the bf16 `cos`/`sin` in `apply_rotary_emb_cos_sin` lose
precision (the multiply promotes back to fp32, but the bf16 rounding already
happened). torchtitan sets `cast_forward_inputs=False` in the canonical
`apply_fsdp` (`torchtitan/distributed/fsdp.py`) so `freqs_cis` stays fp32 in all
paths -- required for bitwise-identical numerics across parallelism configs (e.g.
TP-only vs FSDP+TP). Safe for LLMs whose float inputs are already `param_dtype`;
image models (Flux, VLM) with external fp32 inputs (e.g. pixel values) may still
need `cast_forward_inputs=True`.

## Precision requirements

Batch-invariant mode requires bfloat16 computation in both trainer and
generator so they operate at the same precision. The generator always runs in
bfloat16 (`generator.model_dtype="bfloat16"`). The trainer reaches the same
precision with FSDP mixed precision (`training.mixed_precision_param="bfloat16"`,
the default): it keeps fp32 master weights and FSDP's `MixedPrecisionPolicy`
casts them to bf16 before the forward, so the forward is numerically identical to
the generator's bf16 path. The trainer always wraps the model in FSDP
(`parallelize_qwen3` applies `apply_fsdp_to_decoder` unconditionally for the
trainer), so this cast happens even at `data_parallel_shard_degree=1`, where FSDP
acts purely as a mixed-precision boundary (the degree-1 all-gather is a no-op but
still casts to bf16). No extra GPUs are required relative to TP-only. With
`data_parallel_shard_degree > 1` the same cast happens during the sharded
all-gather.

The batch-invariant configs keep fp32 master weights (`TrainingConfig()` with
default `dtype="float32"`) and rely on FSDP mixed precision for the bf16 forward,
so the optimizer updates fp32 weights while the forward stays bitwise identical
to the generator.

## Current limitations

- Trainer and generator must use the same tensor parallelism degree (logprob
  bitwise parity is only supported under matched parallelism).
- Sequence parallelism is not supported in batch-invariant mode because
  reduce-scatter lacks a tree-based implementation, and the ring-based
  implementation is non-deterministic.

## Usage

Run a batch-invariant config via the RL entrypoint, e.g.:

```bash
python -m torchtitan.experiments.rl.train \
    --module alphabet_sort \
    --config rl_grpo_qwen3_0_6b_varlen_batch_invariant
```

Batch-invariant configs in
`torchtitan/experiments/rl/examples/alphabet_sort/config_registry.py` all use
FSDP mixed precision (fp32 master weights, bf16-cast forward) on the trainer:
