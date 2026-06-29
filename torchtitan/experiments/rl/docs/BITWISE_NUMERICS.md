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

Generator-only patches live in `torchtitan/experiments/rl/batch_invariance.py`
(see "Generator-only mechanisms" below).

## Sources of non-determinism

### 1. GEMM / linear layers (`mm`, `addmm`)

cuBLAS selects tile iteration orders that depend on the M dimension (batch x
seq). Different batch sizes -> different tile schedules -> different
accumulation order -> different rounding.

**Fix:** `set_batch_invariance()` delegates to the upstream
[`batch_invariant_ops`](https://github.com/thinking-machines-lab/batch_invariant_ops)
package, which overrides these ATen ops with Triton kernels that use a fixed
tile iteration order.

### 2. Log-softmax and reductions (`log_softmax`, `mean.dim`)

Default kernels choose block sizes based on input shape.

**Fix:** Same as above -- `batch_invariant_ops` also overrides `_log_softmax`
and `mean.dim`.

### 3. Flash attention split-k (`num_splits`)

FA3 may choose different split-k factors depending on sequence length, leading
to different partial-sum reduction trees.

**Fix:** Force `num_splits=1` when `is_in_batch_invariant_mode()` is True
(`torchtitan/models/common/attention.py`, `VarlenAttention.forward`). FA2 is
already batch-invariant, so this only matters for FA3.

### 4. NCCL collectives

NCCL all-reduce can use ring or tree algorithms with varying channel counts,
changing the reduction order across ranks.

**Fix:** `set_batch_invariance()` forces single-channel tree all-reduce and a
deterministic protocol via environment variables (`NCCL_ALGO=allreduce:tree`,
`NCCL_MIN/MAX_NCHANNELS=1`, `NCCL_PROTO=Simple`, etc.). These match vLLM's
settings and must be set before `dist.init_process_group`.

### 5. Reduced-precision reductions and TF32

Reduced-precision accumulation and TF32 tensor cores introduce
hardware-dependent rounding that can vary with operand shapes.

**Fix:** `set_batch_invariance()` disables them:
`allow_bf16/fp16_reduced_precision_reduction = False`,
`torch.backends.cuda.matmul.allow_tf32 = False`, and
`torch.backends.cudnn.allow_tf32 = False`.

## Generator-only mechanisms

The trainer runs torchtitan ops directly, but the generator runs the same model
definition inside vLLM, which has its own fused kernels. These extra patches
(`torchtitan/experiments/rl/batch_invariance.py`) route the generator back
through the same ops as the trainer:

- **FlexAttention kernel pinning** (`BatchInvariantFlexConverter`): a
  `ModelConfigConverter` that pins `BLOCK_M = BLOCK_N = 16` and
  `BACKEND = "TRITON"` on every `FlexAttention` layer, matching vLLM's default
  tile size and avoiding the `flex_decode` kernel. Wired into a config via
  `converters=[BatchInvariantFlexConverter.Config()]`.
- **bmm override** (`patch_bmm_for_batch_invariance`): `batch_invariant_ops`
  overrides `mm`/`addmm` but not `bmm`. The MoE router gate (3-D activation @
  2-D weight) lowers to `aten::bmm` in the generator but `aten::mm` in the
  trainer, so without this the gate scores drift and flip top-k expert routing.
  This installs vLLM's batch-invariant `bmm` kernel.
- **logprob kernel override** (`force_logprobs_fn_for_batch_invariance`): vLLM's
  v2 GPU sampler computes per-token logprobs with a fused Triton kernel that
  inlines `log(softmax(logits))` and never calls PyTorch ops. This replaces it
  with the trainer's `compute_logprobs`, so both paths share one logprob code
  path.

## `freqs_cis` dtype and `cast_forward_inputs`

The `freqs_cis` buffer is computed and stored in fp32. It is passed as a
forward argument from `Decoder.forward` to each `TransformerBlock.forward`.
FSDP2's `MixedPrecisionPolicy` has a `cast_forward_inputs` flag that, when
`True`, casts **all** floating-point forward inputs to `param_dtype` (typically
bf16) at each FSDP boundary.

This creates a subtle numerics divergence for the cos/sin RoPE backend:

- **With `cast_forward_inputs=True`:** `freqs_cis` is cast from fp32 -> bf16
  at each transformer block. Inside `apply_rotary_emb_cos_sin`, `xq` and `xk`
  are upcast to fp32, then multiplied with the bf16 `cos`/`sin` values. PyTorch
  promotes the bf16 operand to fp32 for the multiply, but the bf16 rounding
  has already lost precision.
- **With `cast_forward_inputs=False`:** `freqs_cis` stays fp32 throughout.
  The fp32 x fp32 multiply preserves full precision.

torchtitan sets `cast_forward_inputs=False` in the canonical `apply_fsdp`
(`torchtitan/distributed/fsdp.py`) to avoid this precision loss. This is safe
for LLMs where all float inputs to FSDP-wrapped modules are already in
`param_dtype`. Image models (Flux, VLM) that receive external fp32 inputs (e.g.
pixel values) may still need `cast_forward_inputs=True`.

For bitwise-identical numerics between different parallelism configurations
(e.g. TP-only vs FSDP+TP), `freqs_cis` must remain fp32 in all code paths.

## Precision requirements

Batch-invariant mode requires bfloat16 computation in both trainer and
generator so they operate at the same precision. The generator always runs in
bfloat16 (`generator.model_dtype="bfloat16"`). The trainer reaches the same
precision in one of two ways:

1. **Full bf16 training** (`training.dtype="bfloat16"`): weights are bf16, used
   directly in the forward.
2. **FSDP mixed precision** (`training.mixed_precision_param="bfloat16"`, the
   default): with `data_parallel_shard_degree > 1`, FSDP all-gathers the full
   params in bf16 before the forward, so the dense forward is numerically
   identical to the generator's replicated bf16 dense path -- even when the
   master weights are fp32. This is what lets an FSDP2+TP trainer match a
   generator that uses pure DP+TP.

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
`torchtitan/experiments/rl/examples/alphabet_sort/config_registry.py` include:

- `rl_grpo_qwen3_0_6b_varlen_batch_invariant` -- dense, varlen attention, TP-only.
- `rl_grpo_qwen3_0_6b_flex_batch_invariant` -- dense, FlexAttention (uses
  `BatchInvariantFlexConverter`).
- `rl_grpo_gpt_oss_debug_varlen_batch_invariant` -- GPT-OSS dense debug model.
- `rl_grpo_qwen3_moe_debug_varlen_batch_invariant` -- MoE; trainer FSDP2+TP2+EP4
  matches generator DP2+TP2+EP4 bitwise (verified
  `bit_wise/logprob_diff/max == 0`).

Parity is verified by `torchtitan/experiments/rl/tests/test_bitwise_parity.py`
and the loss-comparison script in `torchtitan/experiments/rl/scripts/loss_compare.py`.
