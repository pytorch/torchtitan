# Qwen3-VL: Vision-Language Model

This folder contains the implementation of Qwen3-VL, a multimodal vision-language model based on the Qwen3 architecture.

## Overview

Qwen3-VL combines:
- **Qwen3 LLM**: The base language model with QK-norm and RoPE
- **Vision Encoder**: A Vision Transformer (ViT) that supports native resolution images (no fixed square crops) with 2D RoPE and bilinear-interpolated position embeddings
- **Patch Merger**: Reduces vision sequence length by merging spatial patches (e.g., 2×2 patches → 1 token)
- **DeepStack**: Visual features from intermediate ViT layers are added to early LLM hidden states
- **MRoPE**: Multi-dimensional Rotary Position Embedding with separate temporal, height, and width position IDs for vision-language alignment

## Design

Distributed training usually does not play nice with input of varying shapes. To handle a varying number of images and image sizes, we require two hyperparameters: image batch size `N` and image length `L` (in patches), and pad the actual image patches to this fixed size. Then we scatter the patch embeddings to their actual positions in the LLM input tokens.

<img width="1398" height="840" alt="VLM Architecture" src="https://github.com/user-attachments/assets/63fcbbc1-c587-4a63-8246-411cb72f5789" />

Note: the diagram shows the case where each patch maps to one vision token. In practice, Qwen3-VL uses a Patch Merger that groups `merge_size²` patches into one token (e.g., `merge_size=2` means 4 patches → 1 token), reducing the vision sequence length by `merge_size²`.

- After `tok_embeddings`, we obtain text tokens of shape `B×S`.
- After `vision_encoder`, we obtain visual tokens of shape `N×L`.
- We extract the valid visual tokens only (remove padding).
- Then scatter those tokens to their actual positions in the LLM input tokens.
- DeepStack adds intermediate ViT features to early decoder layers.
- MRoPE assigns separate (T, H, W) position IDs to vision tokens for spatial awareness.

## Prerequisites

Install the additional dependencies required by Qwen3-VL:

```bash
pip install av torchvision
```

## Model Variants

| Variant | LLM dim | Layers | ViT dim | ViT layers | Patch size | MoE |
|---------|---------|--------|---------|------------|------------|-----|
| debugmodel | 256 | 4 | 256 | 4 | 14 | No |
| debugmodel_moe | 256 | 1 | 256 | 4 | 14 | Yes (64 experts) |
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
| `nemotron-video` | Video QA | HuggingFace + local video files |

### Video dataset setup (nemotron-video)

The Nemotron dataset (`nvidia/Nemotron-VLM-Dataset-V2`) streams text/metadata from HuggingFace but references video files by path. You need to download the video files separately:

```bash
# Download NExT-QA videos (~150 GB)
huggingface-cli download rhymes-ai/NeXTVideo --repo-type dataset --local-dir ./assets/videos/
cd ./assets/videos && unzip NExTVideo.zip
```

Set `video_dir="./assets/videos"` in the dataloader config so paths resolve correctly.

## Supported Features

| Feature | Notes |
|---------|-------|
| FSDP / HSDP | Both vision encoder and decoder are individually sharded |
| Tensor Parallelism (TP) | Applied to both vision encoder and decoder (without SequenceParallel due to vision scatter and DeepStack) |
| Expert Parallelism (EP) | For MoE variants (e.g., 30B-A3B) |
| Sample Packing | Configurable via `packing_buffer_size` in dataloader config |

## TODO

- Add Pipeline Parallelism support
- Add default video dataset training configs
- Multi-worker data loading for video
- GPU-accelerated video decoding
