# DeepSeek-V3 in TorchTitan

DeepSeek-V3 is a Mixture-of-Experts (MoE) transformer model with Multi-head Latent Attention (MLA) architecture.

## Setup

### Download Tokenizer

```bash
# DeepSeek tokenizer (automatically downloads tokenizer.json and tokenizer_config.json)
python scripts/download_tokenizer.py --repo_id deepseek-ai/DeepSeek-V3
```

## Training

### Debug Training

```bash
# Quick debug run with small model
CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/debug_model.toml" ./run_train.sh
```

### Full Model Training

```bash
# 16B parameter model
CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh
```


## Supported Features
- FSDP, HSDP
- Activation checkpointing
- Tensor Parallel (TP)
- Expert Parallel (EP)
