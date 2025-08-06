# DeepSeek-V3 in `torchtitan`

DeepSeek-V3 is a Mixture-of-Experts (MoE) transformer model with Multi-head Latent Attention (MLA) architecture.

## Setup

### Download Tokenizer

```bash
# DeepSeek 671B tokenizer (automatically downloads tokenizer.json and tokenizer_config.json)
python scripts/download_tokenizer.py --repo_id deepseek-ai/DeepSeek-V3
```

```bash
# DeepSeek 16B tokenizer:
python scripts/download_tokenizer.py --repo_id deepseek-ai/deepseek-moe-16b-base
```

> **Note:** We are reusing the tokenizer from deepseek-ai/deepseek-moe-16b-base to help users test and run the 16B model. This is not the official tokenizer for the DeepSeek-V3-16B model. The DeepSeek-V3 model has a different architecture from the deepseek-moe models (different attention implementation, MoE router implementation, etc.), making it not feasible to load deepseek-moe-16b model weights into DeepSeek-V3-16B.


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
- Parallelism
    - Context Parallel support for DeepSeek V3
- torch.compile
- Quantization
- Testing
    - perfomance and loss converging tests
    - CI integration
