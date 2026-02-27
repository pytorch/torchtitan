# Qwen3-VL: Vision-Language Model

This folder contains the implementation of Qwen3-VL, a multimodal vision-language model based on the Qwen3 architecture.

## Overview

Qwen3-VL combines:
- **Qwen3 LLM**: The base language model with QK-norm and RoPE
- **Vision Encoder**: A Vision Transformer (ViT) with 2D RoPE and bilinear-interpolated position embeddings
- **Patch Merger**: Reduces vision sequence length by merging spatial patches
- **DeepStack**: Visual features from intermediate ViT layers are added to early LLM hidden states
- **MRoPE**: Multi-dimensional Rotary Position Embedding with separate temporal, height, and width position IDs for vision-language alignment

## Prerequisites

Install the additional dependencies required by Qwen3-VL:

```bash
pip install einops pillow
```

## Supported Features

| Feature | Notes |
|---------|-------|
| FSDP / HSDP | Both vision encoder and decoder are individually sharded |
| Tensor Parallelism (TP) | Applied to both vision encoder and decoder (without SequenceParallel due to vision scatter and DeepStack) |
| Expert Parallelism (EP) | For MoE variants (e.g., 30B-A3B) |
| torch.compile | Applied to both vision encoder and decoder transformer blocks |
| Activation Checkpointing | Full and selective modes, applied to both vision encoder and decoder |
| Sample Packing | Configurable via `packing_buffer_size` in dataloader config |

## TODO

- Add Pipeline Parallelism support
- Add Context Parallelism support for long-sequence training
- Add SequenceParallel to the decoder (requires DTensor-aware vision scatter and DeepStack)
- Add video data loading and processing (model architecture already supports video)
