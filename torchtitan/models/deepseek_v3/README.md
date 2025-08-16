# DeepSeek-V3 in `torchtitan`

DeepSeek-V3 is a Mixture-of-Experts (MoE) transformer model with Multi-head Latent Attention (MLA) architecture.

## Setup

### Download Tokenizer

```bash
# DeepSeek 671B tokenizer (automatically downloads tokenizer.json and tokenizer_config.json)
python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3 --assets tokenizer
```

```bash
# DeepSeek 16B tokenizer:
python scripts/download_hf_assets.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer
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


## HuggingFace -> DCP Checkpoint Conversion

We implemented StateDictAdapter to preform HuggingFace safetensor to DCP format conversion. Currently, we only support conversion from HF checkpoints to DCP checkpoints offline (using CPU plain tensor).

Run the offine conversion script:
```bash
python scripts/checkpoint_conversion/convert_from_hf.py <hf_checkpoints_dir> <dcp_output_dir> --model_name deepseek_v3 --model_flavor 671B
```

Some limitations:
1. It can't be used to convert HF checkpoint on the fly using GPU DTensor, because of sharding and quantized blocks may not be aligned well and causing silent numerfical incorrectness.
2. It can't be used for weight sync to generate a state dict of bf16 because fake quantization to fp8 is applied.
3. When converting GroupedExperts weights from HF separate expert weights on-the-fly, `torch.split()` will cause huge GPU memory usage. This is because torchtitan GroupedExperts' weight has shape `(num_experts, dim1, dim2)`, and by default shard FSDP on dim-0. When we call `torch.split()` in `to_hf()` function on dim-0, this will incur and all-gather and get replicated expert memory.

## To be added
- Parallelism
    - Context Parallel support for DeepSeek V3
- torch.compile
- Quantization
- Testing
    - perfomance and loss converging tests
    - CI integration
