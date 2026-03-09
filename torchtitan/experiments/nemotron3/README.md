# Nemotron3 (NemotronH) Hybrid Model

NVIDIA's **Nemotron3** (also known as **NemotronH**) is a hybrid architecture that combines Mamba2 state-space models with traditional Transformer attention layers and Mixture of Experts (MoE) for efficient large-scale language modeling.

## Architecture Overview

The model uses a configurable layer pattern defined by `hybrid_override_pattern`:

| Symbol | Layer Type | Description |
|--------|------------|-------------|
| `M` | Mamba2 | State-space model layer for efficient long-range sequence modeling |
| `*` | Attention | Multi-head attention with Grouped Query Attention (GQA) |
| `-` | MLP | Standard feed-forward layer |
| `E` | MoE | Mixture of Experts (any char other than M, *, - maps to MoE) |

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
NGPU=8 MODULE=nemotron3 CONFIG=nemotron3_nano_30b ./run_train.sh
```

**Single GPU (debug model):**
```bash
NGPU=1 MODULE=nemotron3 CONFIG=nemotron3_debugmodel ./run_train.sh
```

## Available Configurations

| Config | Description | Use Case |
|--------|-------------|----------|
| `nemotron3_nano_30b` | Full Nemotron3 Nano-30B model with bf16 | production-scale training |
| `nemotron3_debugmodel` | Small debug configuration | testing & debugging |

## Key Training Options

Customize with CLI overrides:

```bash
NGPU=1 MODULE=nemotron3 CONFIG=nemotron3_debugmodel ./run_train.sh \
  --training.local_batch_size=1 \
  --training.seq_len=4096 \
  --training.dtype=bfloat16 \
  --parallelism.data_parallel_shard_degree=-1 \
  --parallelism.tensor_parallel_degree=1 \
  --checkpoint.initial_load_path=./assets/hf/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --checkpoint.initial_load_in_hf=true
```

## Legacy TOML Presets

The files under `train_configs/` are legacy presets. The active path is Python config registry:
- `torchtitan.experiments.nemotron3.config_registry.nemotron3_debugmodel`
- `torchtitan.experiments.nemotron3.config_registry.nemotron3_nano_30b`

## Hardware Requirements

- **nano-30B**: Requires multi-GPU setup with sufficient VRAM (tested on 8x B200)
- **debug_model**: Can run on a single GPU for development

## Roadmap

Currently using **FSDP** (Fully Sharded Data Parallel) to get training up and running. The next priorities are Expert Parallelism (EP), then Context Parallelism (CP), followed by MFU optimizations.

| Feature | Status | Notes |
|---------|--------|-------|
| FSDP | ✅ Supported | Currently used for distributed training |
| Expert Parallelism | 🚧 Next | Priority for scaling larger MoE models |
| Context Parallelism | 🚧 To be added | Needed for very long sequence lengths |
| MFU Optimizations | 🚧 To be added | Kernel fusion, better memory layout, etc. |

These additions are expected to become increasingly important as larger Nemotron variants are released.

## Implementation Notes

The model implementation in `model/model.py` is adapted from NVIDIA's official HuggingFace implementation:
[`modeling_nemotron_h.py`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/modeling_nemotron_h.py)

## References

- [NVIDIA Nemotron-3-Nano-30B on HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
