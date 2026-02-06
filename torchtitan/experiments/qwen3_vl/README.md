# Qwen3-VL: Vision-Language Model

This folder contains the implementation of Qwen3-VL, a multimodal vision-language model based on the Qwen3 architecture. The implementation follows the HuggingFace transformers reference implementation.

## Overview

Qwen3-VL combines:
- **Qwen3 LLM**: The base language model with QK-norm and RoPE
- **Vision Encoder**: A Vision Transformer (ViT) with 2D RoPE and bilinear-interpolated position embeddings
- **Patch Merger**: Reduces vision sequence length by merging spatial patches
- **DeepStack**: Visual features from intermediate ViT layers are added to early LLM hidden states
