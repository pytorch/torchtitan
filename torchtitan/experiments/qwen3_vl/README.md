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
- **FlexAttention**: Efficient batched attention with per-image masking
- **Padded Batch Format (NLD)**: Images are processed as `(num_images, max_seq_len, dim)` for efficient GPU utilization
- **2D RoPE**: Separate rotary embeddings for height and width dimensions in block order
- **Bilinear Position Interpolation**: Learnable position embeddings with bilinear interpolation for variable resolution
- **3D Patch Embedding**: Conv3d for handling both images (T=1) and videos (T>1)
- **Patch Merging**: Reduces sequence length by `spatial_merge_size²` (default 4x)

### DeepStack
- Extracts visual features from intermediate ViT layers (e.g., layers 7, 15, 23)
- These features are added to the LLM hidden states at early layers
- Enables richer multimodal understanding by injecting visual information at multiple depths

### MRoPE (Multi-dimensional RoPE)
- For vision tokens: Position IDs encode temporal, height, and width dimensions
- Interleaved frequency layout for better position encoding
- Uses `mrope_section = [24, 20, 20]` for T/H/W dimension allocation

### Data Processing
- **Block Order Patches**: Patches are arranged to match position embedding expectations (2x2 merge blocks are contiguous)
- **Dynamic Image Batching**: Collator handles variable-sized images with padding
- **Token Alignment**: Number of `<|image_pad|>` tokens matches merged visual tokens: `num_tokens = (T × H × W) / spatial_merge_unit`

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
├── __init__.py              # Module init with TrainSpec and model configs
├── job_config.py            # DataConfig for vision-specific parameters
├── README.md                # This file
├── DOCUMENTATION.md         # Detailed implementation documentation
├── datasets/
│   ├── __init__.py
│   ├── mm_datasets.py       # Dataset implementations (CC12M, OBELICS)
│   ├── mm_collator_nld.py   # Multimodal collator for NLD format
│   └── utils/
│       ├── __init__.py
│       ├── image.py         # Image processing utilities
│       ├── packing.py       # Sequence packing utilities
│       └── text.py          # Text processing and padding utilities
├── infra/
│   ├── __init__.py
│   └── parallelize.py       # FSDP parallelization for VLM
├── model/
│   ├── __init__.py
│   ├── args.py              # Model arguments (Qwen3VLModelArgs, etc.)
│   ├── model.py             # Main Qwen3-VL model with vision-text fusion
│   └── vision_encoder.py    # Vision Transformer with FlexAttention
└── train_configs/
    └── debug_model.toml     # Debug config for testing
```

## Usage

### Requirements

```bash
pip install einops
```

Requires a Qwen3-VL tokenizer with special tokens:
- Download from HuggingFace: `Qwen/Qwen3-VL-8B-Instruct`

### Training

```bash
# Debug model training (single node, 4 GPUs)
torchrun --nproc_per_node=4 train.py \
    --job.config_file torchtitan/experiments/qwen3_vl/train_configs/debug_model.toml

# With custom config overrides
torchrun --nproc_per_node=4 train.py \
    --job.config_file torchtitan/experiments/qwen3_vl/train_configs/debug_model.toml \
    --training.steps 100 \
    --data.max_images_per_batch 8
```

### Configuration

Key configuration options in TOML:

```toml
[job]
custom_config_module = "torchtitan.experiments.qwen3_vl.job_config"

[model]
name = "qwen3_vl"
flavor = "debugmodel"  # or "2B", "8B", "72B"
hf_assets_path = "path/to/Qwen3-VL-8B-Instruct"  # Tokenizer path

[data]
patch_size = 14
temporal_patch_size = 2
spatial_merge_size = 2
max_patches_per_image = 320    # Merged patches per image
max_images_per_batch = 16      # Maximum images in a batch
packing_buffer_size = 100

[training]
dataset = "cc12m"
local_batch_size = 8
seq_len = 4096
```

### Model Configuration

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
   - Patches extracted and arranged in block order by collator
   - Patches embedded via 3D Conv (`patch_embed`)
   - Bilinear-interpolated position embeddings added
   - Transformer blocks applied with 2D RoPE and FlexAttention
   - DeepStack features extracted from intermediate layers
   - Final patch merging reduces sequence length
3. **Vision-Text Fusion**: Vision embeddings scattered into text at `<|image_pad|>` positions
4. **LLM Forward**: Transformer layers applied with DeepStack injection at early layers
5. **Output**: Final norm and linear projection to vocab

### Vision Encoder Data Flow

```
Input: pixel_values (N, L, patch_dim), grid_thw (N, 3)
  ↓
PatchEmbed: Conv3d projection → (N, L, dim)
  ↓
Add bilinear-interpolated position embeddings
  ↓
FlexAttention Transformer Blocks (with 2D RoPE)
  ├── Extract DeepStack features at layers [7, 15, 23]
  ↓
PatchMerger: Merge 2×2 patches → (N, L/4, out_hidden_size)
  ↓
Output: merged_embeds, deepstack_features
```

### Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<\|image_pad\|>` | 151655 | Placeholder for image tokens |
| `<\|video_pad\|>` | 151656 | Placeholder for video tokens |
| `<\|vision_start\|>` | 151652 | Start of vision sequence |
| `<\|vision_end\|>` | 151653 | End of vision sequence |
| `<\|endoftext\|>` | - | Padding token |

## Datasets

Supported datasets:
- **cc12m**: `pixparse/cc12m-wds` (image-caption pairs from Conceptual 12M)
- **obelics**: `HuggingFaceM4/OBELICS` (interleaved image-text)

## Implementation Notes

### Block Order for Patches

Patches are arranged in block order to match the position embedding expectations of the PatchMerger:
```
For a 4×4 image with merge_size=2:
Raster order:  0  1  2  3     Block order:  0  1  4  5
               4  5  6  7                   2  3  6  7
               8  9 10 11                   8  9 12 13
              12 13 14 15                  10 11 14 15
```

### Memory Considerations

- Adjust `max_images_per_batch` to control vision encoder memory usage
- Vision encoder can be memory-intensive with many high-resolution images
- Consider reducing `max_patches_per_image` for memory-constrained setups

## References

- [Qwen2-VL Paper](https://arxiv.org/abs/2409.12191)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [DeepStack Paper](https://arxiv.org/abs/2406.04334)
- [HuggingFace Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
