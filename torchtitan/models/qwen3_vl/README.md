# Qwen3-VL: Vision-Language Model

## Overview

Qwen3-VL combines:
- **Qwen3 LLM**: The base language model with QK-norm and RoPE.
- **Vision Encoder**: A Vision Transformer (ViT) that supports native resolution images (no fixed square crops) with 2D RoPE and bilinear-interpolated position embeddings.
- **Patch Merger**: Reduces vision sequence length by merging spatial patches (e.g., 2×2 patches → 1 token).
- **DeepStack**: Adds intermediate ViT features to vision positions in early decoder layers.
- **MRoPE**: Interleaves RoPE from temporal, height, and width position IDs in decoder layers.

## Vision Scatter

- `tok_embeddings` produces text token embeddings of shape `B×S`.
- `vision_encoder` produces visual token embeddings of shape `N×L`.
- Valid visual tokens (excluding padding) are scattered into the placeholder positions in the text sequence, as illustrated below (credit: [@lkhphuc](https://github.com/lkhphuc)).

<img width="1398" height="840" alt="VLM Architecture" src="https://github.com/user-attachments/assets/63fcbbc1-c587-4a63-8246-411cb72f5789" />

Note: the diagram shows each patch mapping to one vision token. In practice, the Patch Merger groups `merge_size²` patches into one token (e.g., `merge_size=2` → 4 patches per token), reducing the vision sequence length by `merge_size²`.

## Prerequisites

Install the additional dependencies required by Qwen3-VL:

```bash
pip install av torchvision
```

## Model Variants

| Variant | LLM dim | Layers | ViT dim | ViT layers | Patch size | MoE |
|---------|---------|--------|---------|------------|------------|-----|
| debugmodel | 256 | 4 | 256 | 4 | 16 | No |
| debugmodel_moe | 256 | 1 | 256 | 4 | 16 | Yes (8 experts) |
| 2B | 2048 | 28 | 1024 | 24 | 16 | No |
| 8B | 4096 | 36 | 1152 | 27 | 16 | No |
| 30B-A3B | 2048 | 48 | 1152 | 27 | 16 | Yes (128 experts) |
| 235B-A22B | 4096 | 94 | 1152 | 27 | 16 | Yes (128 experts) |

## Datasets

| Dataset | Type | Format |
|---------|------|--------|
| `cc12m` | Image-text pairs | WebDataset (streaming) |
| `cc12m-test` | Image-text pairs | Local WebDataset (for testing) |
| `obelics` | Interleaved image-text | HuggingFace (streaming) |

## Supported Features

| Feature | Notes |
|---------|-------|
| FSDP / HSDP | Both vision encoder and decoder are individually sharded |
| Tensor Parallelism (TP) | Applied to both vision encoder and decoder (without SequenceParallel due to vision scatter and DeepStack) |
| Expert Parallelism (EP) | For MoE variants (e.g., 30B-A3B) |
| Sample Packing | Configurable via `packing_buffer_size` in dataloader config |

## Numerical Parity

End-to-end KL divergence against HuggingFace Transformers (2B, 10 random samples): **~5e-8 to ~5e-5** per sample, with **100% top-1 and top-5 match**. Test scripts are in `scripts/checkpoint_conversion/numerical_tests_qwen3_vl_*.py`.

## TODO

- Add Pipeline Parallelism support
- Add Context Parallel support
- Add video dataset training configs
