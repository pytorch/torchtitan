# Nemotron3 (NemotronH) Hybrid Model

NVIDIA's **Nemotron3** (also known as **NemotronH**) is a hybrid architecture that combines Mamba2 state-space models with traditional Transformer attention layers and Mixture of Experts (MoE) for efficient large-scale language modeling.

## Architecture Overview

The model uses a configurable layer pattern defined by `hybrid_override_pattern`:

| Symbol | Layer Type | Description |
|--------|------------|-------------|
| `M` | Mamba2 | State-space model layer for efficient long-range sequence modeling |
| `*` | Attention | Multi-head attention with Grouped Query Attention (GQA) |
| `E` | MLP | Standard feed-forward layer |
| `O` | MoE | Mixture of Experts (128 experts, 6 active per token) |

### nano-30B Configuration

- **Total Parameters**: ~31B
- **Active Parameters**: ~3B per token (due to MoE sparse activation)
- **Layers**: 52
- **Hidden Size**: 2688
- **Max Sequence Length**: 262,144 tokens

## Quick Start

### 1. Download Model Assets

```bash
# Download tokenizer, config, and index only (fast, small files)
python scripts/download_hf_assets.py \
    --repo_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --local_dir ./assets/hf \
    --assets tokenizer config index

# Or download EVERYTHING including weights (~60GB)
python scripts/download_hf_assets.py \
    --repo_id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --local_dir ./assets/hf \
    --all
```

### 2. Run Training

**Multi-GPU Training (8 GPUs):**
```bash
NGPU=8 CONFIG_FILE="./torchtitan/experiments/nemotron3/train_configs/nemotron3-nano-30B.toml" ./run_train.sh
```

**Single GPU (debug model):**
```bash
NGPU=1 CONFIG_FILE="./torchtitan/experiments/nemotron3/train_configs/debug_model.toml" ./run_train.sh
```

## Available Configurations

| Config | Description | Use Case |
|--------|-------------|----------|
| `nemotron3-nano-30B.toml` | Full Nemotron3 Nano-30B model with bf16 |  |
| `debug_model.toml` | Small 16-layer model | testing & debugging |

## Key Training Options

Edit the `.toml` config file to customize:

```toml
[training]
local_batch_size = 1      # Per-GPU batch size
seq_len = 4096            # Sequence length
dtype = "bfloat16"        # Training precision

[parallelism]
data_parallel_shard_degree = -1  # FSDP sharding (-1 = auto)
tensor_parallel_degree = 1       # Tensor parallelism

[checkpoint]
initial_load_path = "./assets/hf/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
initial_load_in_hf = true        # Load from HuggingFace format
```

## Hardware Requirements

- **nano-30B**: Requires multi-GPU setup with sufficient VRAM (tested on 8x B200)
- **debug_model**: Can run on a single GPU for development

## Roadmap

Currently using **FSDP** (Fully Sharded Data Parallel) to get training up and running. The following parallelism strategies are planned for future support:

| Feature | Status | Notes |
|---------|--------|-------|
| FSDP | ✅ Supported | Currently used for distributed training |
| Tensor Parallelism | 🚧 To be added | Likely needed for larger model variants |
| Context Parallelism | 🚧 To be added | Needed for very long sequence lengths |
| Expert Parallelism | 🚧 To be added | Essential for scaling MoE layers |
| MFU Optimizations | 🚧 To be added | Kernel fusion, better memory layout, etc. |

These advanced parallelism strategies will become necessary once NVIDIA releases larger models in the Nemotron series.

## Implementation Notes

The model implementation in `model/model.py` is adapted from NVIDIA's official HuggingFace implementation:
[`modeling_nemotron_h.py`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/modeling_nemotron_h.py)

## References

- [NVIDIA Nemotron-3-Nano-30B on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
