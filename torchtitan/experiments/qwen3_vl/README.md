# Qwen3-VL: Vision-Language Model

This folder contains the implementation of Qwen3-VL, a multimodal vision-language model based on the Qwen3 architecture. The implementation follows the HuggingFace transformers reference implementation.

## Overview

Qwen3-VL combines:
- **Qwen3 LLM**: The base language model with QK-norm and RoPE
- **Vision Encoder**: A Vision Transformer (ViT) with 2D RoPE and bilinear-interpolated position embeddings
- **Patch Merger**: Reduces vision sequence length by merging spatial patches
- **DeepStack**: Visual features from intermediate ViT layers are added to early LLM hidden states

## Key Features

### Vision Encoder
- **2D RoPE**: Separate rotary embeddings for height and width dimensions
- **Bilinear Position Interpolation**: Learnable position embeddings with bilinear interpolation for variable resolution
- **3D Patch Embedding**: Conv3d for handling both images (T=1) and videos (T>1)
- **Patch Merging**: Reduces sequence length by `spatial_merge_size^2` (default 4x)

### DeepStack
- Extracts visual features from intermediate ViT layers (e.g., layers 7, 15, 23)
- These features are added to the LLM hidden states at early layers
- Enables richer multimodal understanding by injecting visual information at multiple depths

### MRoPE (Multi-dimensional RoPE)
- For vision tokens: Position IDs encode temporal, height, and width dimensions
- Interleaved frequency layout for better position encoding
- Uses `mrope_section = [24, 20, 20]` for T/H/W dimension allocation

## Model Variants

| Variant | LLM Dim | Vision Dim | LLM Layers | Vision Layers | Total Params |
|---------|---------|------------|------------|---------------|--------------|
| debugmodel | 256 | 256 | 4 | 4 | ~50M |
| 2B | 1536 | 1280 | 28 | 32 | ~2B |
| 8B | 4096 | 1280 | 32 | 32 | ~8B |
| 72B | 8192 | 1280 | 80 | 32 | ~72B |

## Directory Structure

```
qwen3_vl/
├── __init__.py           # Module init with TrainSpec and model configs
├── job_config.py         # Job configuration extensions
├── README.md             # This file
├── datasets/             # Dataset loading and processing
│   ├── __init__.py
│   ├── mm_datasets.py    # Main dataset implementation
│   ├── mm_collator_nld.py # Batch collation
│   └── utils/            # Image/text processing utilities
│       ├── __init__.py
│       ├── image.py
│       ├── packing.py
│       └── text.py
├── infra/
│   ├── __init__.py
│   └── parallelize.py    # FSDP/parallelization
├── model/
│   ├── __init__.py
│   ├── args.py           # Model arguments (Qwen3VLModelArgs, etc.)
│   ├── model.py          # Main Qwen3-VL model
│   └── vision_encoder.py # Vision Transformer encoder with DeepStack
├── tests/
│   └── __init__.py
└── train_configs/
    └── debug_model.toml  # Debug config for testing
```

## Usage

requires `einops`

### Training

```bash
# Debug model training
torchrun --nproc_per_node=4 train.py \
    --config torchtitan/experiments/qwen3_vl/train_configs/debug_model.toml
```

### Model Configuration

The model is configured via `Qwen3VLModelArgs`:

```python
from torchtitan.experiments.qwen3_vl import Qwen3VLModelArgs, Qwen3VLVisionEncoderArgs

args = Qwen3VLModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    encoder=Qwen3VLVisionEncoderArgs(
        dim=1280,
        n_layers=32,
        n_heads=16,
        patch_size=14,
        spatial_merge_size=2,
        out_hidden_size=4096,  # Must match LLM dim
        deepstack_visual_indexes=[7, 15, 23],
    ),
)
```

## Architecture Details

### Forward Pass Flow

1. **Token Embedding**: Input tokens are embedded via `tok_embeddings`
2. **Vision Encoding** (if images/videos present):
   - Patches are embedded via 3D Conv (`patch_embed`)
   - Position embeddings added (bilinear interpolated)
   - Transformer blocks applied with 2D RoPE
   - DeepStack features extracted from intermediate layers
   - Final patch merging reduces sequence length
3. **Vision-Text Fusion**: Vision embeddings scattered into text at placeholder positions
4. **LLM Forward**: Transformer layers applied with DeepStack injection at early layers
5. **Output**: Final norm and linear projection to vocab

### Special Tokens

- `<|image_pad|>` (151655): Placeholder for image tokens
- `<|video_pad|>` (151656): Placeholder for video tokens
- `<|vision_start|>` (151652): Start of vision sequence
- `<|vision_end|>` (151653): End of vision sequence

## Datasets

Supported datasets:
- `obelics`: HuggingFaceM4/OBELICS (interleaved image-text)
- `cc12m`: pixparse/cc12m-wds (image-caption pairs)

## References

- [Qwen2-VL Paper](https://arxiv.org/abs/2409.12191)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [DeepStack Paper](https://arxiv.org/abs/2406.04334)
- [HuggingFace Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
