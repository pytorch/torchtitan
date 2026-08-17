# Kimi K2.7

Kimi K2.5, K2.6, and K2.7-Code share an architecture that pairs a
**DeepSeek-V3-style** decoder (Multi-head Latent Attention + Mixture-of-Experts)
with a **MoonViT3d** vision encoder.

## Prerequisites

Install the additional dependencies:

```bash
pip install av torchvision
```

## Architecture

- **Decoder** — DeepSeek-V3 (MLA + MoE).
- **Vision encoder** — MoonViT3d: linear patch embedding, learnable 2D spatial
  position embeddings (plus a sinusoidal temporal term for video), 2D RoPE,
  pre-norm transformer blocks, temporal mean-pool + 2x2 spatial merge, then a
  2-layer MLP projector to the decoder hidden size.
- **Multimodal forward** — projected vision embeddings are scattered into the
  text embedding sequence at runs of the shared media placeholder token.

## Model variants

| Variant | LLM dim | Layers | Heads | Experts (top-k) | ViT dim | ViT layers | ViT heads |
|---------|---------|--------|-------|-----------------|---------|------------|-----------|
| debugmodel | 256 | 6 | 16 | 8 (top-3) | 256 | 4 | 4 |
| moonlight-16B-A3B | 2048 | 27 | 16 | 64 (top-6) | — | — | — |
| Kimi-VL-A3B | 2048 | 27 | 16 | 64 (top-6) | 1152 | 27 | 16 |
| Kimi-K2.5 | 7168 | 61 | 64 | 384 (top-8) | 1152 | 27 | 16 |

## Supported Parallelisms

| Feature | Notes |
|---------|-------|
| FSDP / HSDP | Decoder sharded per-layer. Without PP, the vision encoder is a separate FSDP unit; with PP, it belongs to the first-stage root FSDP unit |
| Tensor Parallelism (TP) | Model support exists, but DistMuon recipes currently reject TP-produced `_StridedShard` layouts ([#3353](https://github.com/pytorch/torchtitan/issues/3353)) |
| Expert Parallelism (EP) | DeepSeek-V3 routed + shared experts. DistMuon initially requires routed-expert storage `Shard(0)` on both `efsdp` and `ep`; it rejects the `efsdp` `Shard(1)` layout selected when `efsdp_size * ep_size > num_experts` ([#4122](https://github.com/pytorch/torchtitan/pull/4122)) |
| Pipeline Parallel (PP) | Model support exists; DistMuon PP support follows in [#4102](https://github.com/pytorch/torchtitan/pull/4102) |

## Numerical Checks

The HuggingFace comparison covers the Kimi-VL compatibility flavor, not the 1T
K2.x checkpoints. For full text+image float32 execution, it measures:

- vision-feature cosine similarity: `0.999977`
- normal end-to-end last-token logits: KL `4.3e-2`, top-1 match, top-5 4/5
- with expert routing pinned to HuggingFace's selections: KL `5.3e-4`, top-1
  match, top-5 5/5

The routing-pinned result is a diagnostic that isolates non-routing math; it is
not a normal end-to-end parity result.

- **Parallelism correctness**: bit-identical logits (max diff `0.0`) for
  no-parallel / FSDP / FSDP+EP; within bf16 tolerance for FSDP+EP+TP (with SP).

Test scripts:
- `scripts/checkpoint_conversion/numerical_tests_kimi.py`

## TODO

- Add a video dataset training pipeline.
- Add INT4 (compressed-tensors) checkpoint loading. The released K2.5, K2.6,
  and K2.7-Code 1T checkpoints are INT4 group-quantized; the inherited
  DeepSeek-V3 adapter only handles FP8 block-scale, so the 1T config trains from
  scratch but cannot load them yet.
- Add Context Parallel (CP) support.
