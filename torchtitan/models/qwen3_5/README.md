# Qwen3.5: Multimodal Model with Hybrid Attention

## Overview

Qwen3.5 combines:
- **Hybrid Decoder**: 75% GatedDeltaNet (linear attention) + 25% full attention with output gating and partial RoPE.
- **Vision Encoder**: A Vision Transformer (ViT) with 2D RoPE and bilinear-interpolated learned position embeddings.
- **Patch Merger**: Reduces vision sequence length by merging spatial patches (e.g., 2x2 patches -> 1 token).
- **MRoPE**: Interleaves RoPE from temporal, height, and width position IDs in decoder layers.
- **MoE variant**: Routed experts + shared expert with sigmoid gate.

## Vision Scatter

- `tok_embeddings` produces text token embeddings of shape `B×S`.
- `vision_encoder` produces visual token embeddings of shape `N×L`.
- Valid visual tokens (excluding padding) are scattered into the placeholder positions in the text sequence, as illustrated below (credit: [@lkhphuc](https://github.com/lkhphuc)).

<img width="1398" height="840" alt="VLM Architecture" src="https://github.com/user-attachments/assets/63fcbbc1-c587-4a63-8246-411cb72f5789" />

Note: the diagram shows each patch mapping to one vision token. In practice, the Patch Merger groups `merge_size²` patches into one token (e.g., `merge_size=2` → 4 patches per token), reducing the vision sequence length by `merge_size²`.

## Prerequisites

Install the additional dependencies:

```bash
pip install av torchvision
```

For GatedDeltaNet GPU efficiency (optional, pure-torch fallback available):

```bash
pip install flash-linear-attention
```

## Model Variants

### Dense

| Variant | LLM dim | Layers | Heads | KV Heads | Head Dim | ViT dim | ViT layers |
|---------|---------|--------|-------|----------|----------|---------|------------|
| debugmodel | 256 | 8 | 4 | 2 | 64 | 256 | 4 |
| 0.8B | 1024 | 24 | 8 | 2 | 256 | 768 | 12 |
| 2B | 2048 | 24 | 8 | 2 | 256 | 1024 | 24 |
| 4B | 2560 | 32 | 16 | 4 | 256 | 1024 | 24 |
| 9B | 4096 | 32 | 16 | 4 | 256 | 1152 | 27 |
| 27B | 5120 | 64 | 24 | 4 | 256 | 1152 | 27 |

### MoE

| Variant | LLM dim | Layers | Experts | Top-k | Shared Expert |
|---------|---------|--------|---------|-------|---------------|
| debugmodel_moe | 256 | 4 | 8 | 2 | Yes |
| 35B-A3B | 2048 | 40 | 256 | 8 | Yes |
| 122B-A10B | 3072 | 48 | 256 | 8 | Yes |
| 397B-A17B | 4096 | 60 | 512 | 10 | Yes |

## Datasets

| Dataset | Type | Format |
|---------|------|--------|
| `cc12m` | Image-text pairs | WebDataset (streaming) |
| `cc12m-test` | Image-text pairs | Local WebDataset (for testing) |

## Supported Parallelisms

| Feature | Notes |
|---------|-------|
| FSDP / HSDP | Decoder sharded per-layer; vision encoder sharded as a single unit (one AllGather) |
| Tensor Parallelism (TP) | With Sequence Parallel; head-sharded TP on GatedDeltaNet projections |
| Expert Parallelism (EP) | For MoE variants |
| Pipeline Parallel (PP) | Vision encoder assigned to first stage; 1F1B and Interleaved1F1B schedules |
| Sample Packing | Configurable via `packing_buffer_size` in dataloader config |

## Numerical Parity

End-to-end KL divergence against HuggingFace Transformers (4B, multimodal inputs): **~3e-7** average, with **100% top-1 and top-5 match**.

Parallelism correctness: bit-identical logits (max diff `0.0`) across no-parallel, FSDP, FSDP+EP, and FSDP+EP+TP configs.

Test scripts:
- `scripts/checkpoint_conversion/numerical_tests_qwen3_5.py` — HF vs TT comparison
- `scripts/checkpoint_conversion/numerical_tests_qwen3_5_shard.py` — parallelism correctness

## TODO

- Add video dataset training configs
- Add Context Parallel (CP) support
