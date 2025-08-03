# DeepSeek-V3 in TorchTitan

DeepSeek-V3 is a Mixture-of-Experts (MoE) transformer model with Multi-head Latent Attention (MLA) architecture.

## Setup

### Download Tokenizer

```bash
# DeepSeek 671B tokenizer (automatically downloads tokenizer.json and tokenizer_config.json)
python scripts/download_tokenizer.py --repo_id deepseek-ai/DeepSeek-V3
```

```bash
# For 16B model support:
python scripts/download_tokenizer.py --repo_id deepseek-ai/deepseek-moe-16b-chat
```

## Training

### Debug Training

```bash
# Quick debug run with small model
CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/debug_model.toml" ./run_train.sh
```

### Full Model Training

```bash
# 16B parameter model: adapted from older 16B parameter model from https://huggingface.co/deepseek-ai/deepseek-moe-16b-base
CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh
```

```bash
# 671B parameter model
CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b.toml" ./run_train.sh
```


## Supported Features
- FSDP, HSDP
- Activation checkpointing
- Tensor Parallel (TP)
- Expert Parallel (EP)
- Pipeline Parallel (PP)


## To be added
- Modeling
    - Merge DeepSeek-V3 and Llama4 MoE common components
    - Attention Layer: need to pass softmax_scale to sdpa() to support scaling
- Parallelism
    - Context Parallel support for DeepSeek-V3
- torch.compile
- Quantization
- Testing
    - perfomance and loss converging tests
    - CI integration
