# Kimi K2.5

Kimi K2.5 pairs the **DeepSeek-V3** decoder (Multi-head Latent Attention +
Mixture-of-Experts) with a **MoonViT3d** vision encoder.

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
- **Multimodal forward** — vision features are scattered into the text embeddings
  at a single media placeholder.

## Model variants

| Variant | LLM dim | Layers | Heads | Experts (top-k) | ViT dim | ViT layers | ViT heads |
|---------|---------|--------|-------|-----------------|---------|------------|-----------|
| debugmodel | 256 | 6 | 16 | 8 (top-3) | 256 | 4 | 4 |
| moonlight-16B-A3B | 2048 | 27 | 16 | 64 (top-6) | — | — | — |
| Kimi-VL-A3B | 2048 | 27 | 16 | 64 (top-6) | 1152 | 27 | 16 |
| 1T-A32B | 7168 | 61 | 64 | 384 (top-8) | 1152 | 27 | 16 |

## Supported Parallelisms

| Feature | Notes |
|---------|-------|
| FSDP / HSDP | Decoder sharded per-layer; vision encoder sharded as a single unit (one AllGather) |
| Tensor Parallelism (TP) | With Sequence Parallel. The token embedding stays Replicate for the vision scatter and SP resumes at decoder layer 0; the vision encoder runs Replicate (no SP) |
| Expert Parallelism (EP) | DeepSeek-V3 routed + shared experts |
| Pipeline Parallel (PP) | Vision encoder folded into the first stage; 1F1B and Interleaved1F1B schedules |

## Numerical Parity

- **vs HuggingFace Kimi-VL** (full text+image, float32; last-token logits):
  vision features cosine `0.999977`.
  - end-to-end: KL `4.3e-2`, top-1 match, top-5 4/5
  - with routing pinned to HF's expert selections: KL `5.3e-4`, top-1 match,
    top-5 5/5
- **Parallelism correctness**: bit-identical logits (max diff `0.0`) for
  no-parallel / FSDP / FSDP+EP; within bf16 tolerance for FSDP+EP+TP (with SP).

Test scripts:
- `scripts/checkpoint_conversion/numerical_tests_kimi.py`

## TODO

- Add a video dataset training pipeline.
- Add int4 (compressed-tensors) checkpoint loading for the K2.6 release (the
  inherited DeepSeek-V3 adapter only dequantizes the fp8 block-scale format).
- Add Context Parallel (CP) support.
