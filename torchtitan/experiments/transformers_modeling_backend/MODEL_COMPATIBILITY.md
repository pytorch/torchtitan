# Model Compatibility: HF MoE Models with Titan Native MoE

This document tracks the numerical equivalence and architectural
compatibility of HF MoE models integrated into the transformers
modeling backend with titan's native MoE replacement.

## Numerical Equivalence Summary

All models are tested with production-scale hidden dimensions, reduced
expert count (E=8), and single-layer configs. HF is the gold standard —
titan's native MoE output is compared against unmodified HF forward.

| Model | model_type | Status | KL div | max_diff | cos_sim | Round-trip |
|-------|-----------|--------|--------|----------|---------|------------|
| Qwen3-30B-A3B | qwen3_moe | PASS | 5.31e-07 | 1.95e-03 | 0.999996 | PASS |
| Mixtral-8x7B | mixtral | WARN | 4.62e-05 | 3.12e-02 | 0.999996 | PASS |
| DeepSeek-V3 | deepseek_v3 | WARN | 3.66e-04 | 1.25e-01 | 0.999996 | PASS |
| OLMoE-1B-7B | olmoe | PASS | 4.04e-07 | 3.91e-03 | 0.999996 | PASS |
| DeepSeek-V2-Lite | deepseek_v2 | PASS | 2.64e-07 | 7.81e-03 | 0.999999 | PASS |
| GLM-4.7 | glm4_moe | WARN | 5.06e-05 | 6.25e-02 | 0.999996 | PASS |
| GLM-5 (DSA) | glm_moe_dsa | WARN | 2.67e-04 | 1.25e-01 | 0.999996 | PASS |
| Llama-4-Scout | llama4_text | WARN | 2.29e-04 | 1.25e-01 | 0.999991 | PASS |
| Gemma-4-26B | gemma4_text | FAIL | 4.19e-03 | 7.46e-01 | 0.946 | PASS |

**Thresholds:** PASS (KL < 1e-6), WARN (KL < 1e-3), FAIL (KL >= 1e-3).

## Known Sources of Numerical Difference

### Dispatcher precision (affects all WARN models)

Titan's `LocalTokenDispatcher.combine()` casts score-weighted expert
output to bf16 before `scatter_add`, causing bf16 accumulation.
HF accumulates in f32 via `reshape+sum`. Additionally, titan uses
`topk(sorted=False)` while HF defaults to `sorted=True`, causing
different f32 rounding in the normalization sum. With all three
dispatcher fixes applied, all PASS/WARN models produce max_diff=0.00.

### Expert kernel (Llama4)

Llama4 does not support `grouped_mm` expert implementation. HF uses
`torch.bmm` while titan uses `torch._grouped_mm`. These kernels
produce different floating-point results. Verified experimentally:
running titan weights through bmm gives max_diff=0.00.

### Activation function (Gemma4)

Titan's `GroupedExperts._experts_forward` hardcodes `F.silu` (SwiGLU).
Gemma4 uses `F.gelu(approximate="tanh")` (GeGLU). With matched
activation and identical routing, expert output is exactly 0.00.

### Router architecture (Gemma4)

Gemma4's router includes input RMSNorm, a learned `scale` vector, and
`per_expert_scale` per-expert scaling — none of which titan's
`TokenChoiceTopKRouter` supports. This causes completely different
routing decisions, which is the dominant source of Gemma4's FAIL-level
diff.

## Architectural Differences per Model

### Qwen3-30B-A3B (qwen3_moe)
- **Router:** softmax, norm_topk_prob=True — fully supported
- **Experts:** fused gate_up_proj (E, 2*I, H), SwiGLU — fully supported
- **Shared experts:** none
- **Differences from titan native:** dispatcher precision only

### Mixtral-8x7B (mixtral)
- **Router:** softmax, always normalizes (no config flag) — supported
- **Experts:** fused gate_up_proj, SwiGLU — fully supported
- **Shared experts:** none
- **Differences from titan native:** dispatcher precision only

### DeepSeek-V3 (deepseek_v3)
- **Router:** sigmoid with e_score_correction_bias, group routing — supported
- **Experts:** fused gate_up_proj, SwiGLU — fully supported
- **Shared experts:** additive (FeedForward) — supported
- **Attention:** MLA (q_lora_rank, kv_lora_rank) — supported via ShardingConfig TP plan
- **Differences from titan native:** dispatcher precision only

### OLMoE-1B-7B (olmoe)
- **Router:** softmax, norm_topk_prob=False — supported
- **Experts:** fused gate_up_proj, SwiGLU — fully supported
- **Shared experts:** none
- **Differences from titan native:** dispatcher precision only

### DeepSeek-V2-Lite (deepseek_v2)
- **Router:** softmax, no group routing — supported
- **Experts:** fused gate_up_proj, SwiGLU — fully supported
- **Shared experts:** additive (2 shared experts) — supported
- **Attention:** MLA — supported
- **Differences from titan native:** dispatcher precision only

### GLM-4.7 (glm4_moe)
- **Router:** sigmoid with e_score_correction_bias, group routing — supported
- **Experts:** fused gate_up_proj, SwiGLU — fully supported
- **Shared experts:** additive — supported
- **Differences from titan native:** dispatcher precision only

### GLM-5 DSA (glm_moe_dsa)
- **Router:** sigmoid with e_score_correction_bias, group routing — supported
- **Experts:** fused gate_up_proj, SwiGLU — fully supported
- **Shared experts:** additive — supported
- **Attention:** MLA + Dynamic Sparse Attention (DSA) indexer — MLA supported via
  ShardingConfig; the DSA indexer is **not supported under TP** (its no_grad forward
  uses scatter_/index ops needing local tensors; supporting it requires local_map
  execution of the indexer). Runs under FSDP/EP without TP.
- **Differences from titan native:** dispatcher precision only

### Llama-4-Scout (llama4_text)
- **Router:** sigmoid without e_score_correction_bias — supported (detected via source inspection)
- **Experts:** transposed layout (E, H, 2*I) — supported (transpose in state dict adapter)
- **Shared experts:** additive, singular form (`shared_expert`) — supported
- **MoE attribute:** `feed_forward` instead of `mlp` — supported
- **Expert kernel:** no settable experts implementation — requires `experts_implementation="native"` (uses the model's built-in `bmm`); any other value raises. Causes kernel-level numerical diff vs grouped_mm
- **score_before_experts:** True (top_k=1) — supported
- **Differences from titan native:** dispatcher precision + expert kernel (bmm vs grouped_mm)

### Gemma-4-26B (gemma4_text)
- **Router:** softmax with input RMSNorm, learned scale, per_expert_scale — **NOT FULLY SUPPORTED** (titan router lacks these components, causing different routing decisions)
- **Experts:** standard layout, but uses `gelu_pytorch_tanh` activation — **NOT SUPPORTED** (titan hardcodes `F.silu`)
- **Shared experts:** dense MLP as shared expert (layer-level MoE) — supported (mapped as shared_experts in native MoE)
- **MoE architecture:** layer-level (router/experts are siblings of dense MLP) — supported
- **Attention:** GQA with q_norm, k_norm, v_norm — supported
- **Differences from titan native:** activation function mismatch + router architecture mismatch (FAIL-level diff)

## Parallelism Support

| Model | FSDP | FSDP+EP |
|-------|------|---------|
| Qwen3 MoE | PASS | PASS |
| Mixtral | PASS | PASS |
| DeepSeek-V3 | PASS | PASS |
| OLMoE | PASS | PASS |
| DeepSeek-V2 | PASS | PASS |
| GLM-4.7 | PASS | PASS |
| GLM-5 DSA | PASS | PASS |
| Llama4 | PASS | PASS |
| Gemma4 | PASS | PASS |

## Core Changes Needed for Full Support

### Configurable expert activation (for Gemma4)
`GroupedExperts._experts_forward` hardcodes `F.silu`. Making the
activation configurable would support Gemma4 (`gelu_pytorch_tanh`).

### Extended router features (for Gemma4)
`TokenChoiceTopKRouter` needs support for input normalization, learned
scale vectors, and per-expert scaling to match Gemma4's router.
