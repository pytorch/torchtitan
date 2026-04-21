# Bitwise Numerics and Batch Invariance

This document describes the sources of numerical non-determinism in the RL
training loop and the mechanisms used to eliminate them.

## Problem

In RL for LLMs, the generator computes log-probs for completions in one batch
composition (e.g. 8 completions), while the trainer recomputes them in a
different batch composition (e.g. 2 completions after DP sharding). Without
special care, the same input can produce different log-probs in different batch
contexts due to floating-point accumulation order differences in GEMM, softmax,
reductions, and collective ops.

## Sources of non-determinism

### 1. GEMM / linear layers (`mm`, `addmm`)

cuBLAS selects tile iteration orders that depend on the M dimension (batch ×
seq). Different batch sizes → different tile schedules → different
accumulation order → different rounding.

**Fix:** Replace with Triton kernels that use a fixed tile iteration order
(via `batch_invariant_ops`).

### 2. Log-softmax and reductions (`log_softmax`, `mean.dim`)

Default kernels choose block sizes based on input shape.

**Fix:** Replace with fixed-schedule Triton kernels from `batch_invariant_ops`.

### 3. Flash Attention split-k (`num_splits`)

FA3 may choose different split-k factors depending on sequence length,
leading to different partial-sum reduction trees.

**Fix:** Force `num_splits=1` when `is_in_batch_invariant_mode()` is True
(`torchtitan/models/common/attention.py`, `VarlenAttention.forward`).

### 4. NCCL collectives

NCCL all-reduce can use ring or tree algorithms with varying channel counts,
changing the reduction order across ranks.

**Fix:** Force single-channel tree all-reduce via environment variables
(`torchtitan/distributed/utils.py`, `set_batch_invariance`).

### 5. Reduced-precision reductions and TF32

Reduced-precision accumulation and TF32 tensor cores introduce
hardware-dependent rounding that can vary with operand shapes.

**Fix:** Disable via `torch.backends.cuda.matmul.allow_tf32 = False` and
`torch.backends.cudnn.allow_tf32 = False` when batch-invariant mode is
enabled.

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
  The fp32 × fp32 multiply preserves full precision.

torchtitan sets `cast_forward_inputs=False` in `apply_fsdp`
(`torchtitan/models/llama4/parallelize.py`) to avoid this precision loss.
This is safe for LLMs where all float inputs to FSDP-wrapped modules are
already in `param_dtype`. Image models (Flux, VLM) that receive external
fp32 inputs (e.g. pixel values) may still need `cast_forward_inputs=True`.

For bitwise-identical numerics between different parallelism configurations
(e.g. TP-only vs FSDP+TP), `freqs_cis` must remain fp32 in all code paths.

## Precision requirements

Batch-invariant mode requires bfloat16 computation in both trainer and
generator so they operate at the same precision. The trainer achieves this
via either:

1. `training.dtype="bfloat16"` (full bf16 training), or
2. FSDP mixed precision with `training.mixed_precision_param="bfloat16"`

The generator always runs in bfloat16 (`generator.model_dtype="bfloat16"`).

## Current limitations

- Trainer and generator must use the same tensor parallelism degree.
- Sequence parallelism is not supported in batch-invariant mode because
  reduce-scatter lacks a tree-based implementation, and the ring-based
  implementation is non-deterministic.

## Usage

```bash
python torchtitan/experiments/rl/simple_grpo_sum_digits.py \
    --module rl --config rl_grpo_qwen3_0_6b_batch_invariant
```

This config sets `DebugConfig(batch_invariant=True, deterministic=True)`.
