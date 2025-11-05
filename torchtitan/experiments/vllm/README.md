# TorchTitan Qwen3 Model with vLLM Inference

This directory contains code to run vLLM inference on models trained with TorchTitan.

## Overview

The integration consists of two main components:

1. **Model Adapter** (`model/qwen3.py`): A custom model class that extends vLLM's `Qwen3ForCausalLM` to handle TorchTitan checkpoint naming conventions
2. **Inference Script** (`infer.py`): A simple script to register the model and run inference


## Quick Start

### Prerequisites

1. Install vLLM:
```bash
pip install vllm
```

### Running Inference

#### Single Prompt

```bash
python torchtitan/experiments/vllm/infer.py \
    --model-path /path/to/torchtitan/checkpoint \
    --prompt "Explain quantum computing in simple terms"
```

#### Multiple Prompts from File

```bash
# Create a file with prompts (one per line)
cat > prompts.txt << EOF
What is the meaning of life?
Explain how transformers work
Write a poem about AI
EOF

# Run inference
python torchtitan/experiments/vllm/infer.py \
    --model-path /path/to/torchtitan/checkpoint \
    --prompts-file prompts.txt
```

#### With Tensor Parallelism

```bash
python torchtitan/experiments/vllm/infer.py \
    --model-path /path/to/torchtitan/checkpoint \
    --prompt "Explain deep learning" \
    --tensor-parallel-size 4 \
    --max-tokens 200
```

## Model Configuration

Your checkpoint directory should contain:

1. **`config.json`**: HuggingFace-style model configuration
2. **Model weights**: Either PyTorch checkpoint files or safetensors

Example `config.json` for a Qwen3-7B model:

```json
{
  "architectures": ["TorchTitanQwen3ForCausalLM"],
  "model_type": "qwen3",
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "vocab_size": 151936,
  "max_position_embeddings": 32768,
  "rope_theta": 1000000.0,
  "rms_norm_eps": 1e-06,
  "tie_word_embeddings": false,
  "head_dim": 128
}
```

## Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-path` | str | Required | Path to TorchTitan checkpoint directory |
| `--prompt` | str | "Hello, how are you?" | Single prompt to generate from |
| `--prompts-file` | str | None | Path to file with prompts (one per line) |
| `--max-tokens` | int | 100 | Maximum tokens to generate |
| `--temperature` | float | 0.8 | Sampling temperature |
| `--top-p` | float | 0.95 | Nucleus sampling parameter |
| `--tensor-parallel-size` | int | 1 | Number of GPUs for tensor parallelism |

## Implementation Details

### Model Registration

The inference script registers the custom model with vLLM's model registry:

```python
from vllm.model_executor.models import ModelRegistry
from torchtitan.experiments.vllm.model.qwen3 import TorchTitanQwen3ForCausalLM

ModelRegistry.register_model(
    "TorchTitanQwen3ForCausalLM",
    TorchTitanQwen3ForCausalLM,
)
```

### Weight Mapping

The `WeightsMapper` class handles automatic name translation:

```python
mapper = WeightsMapper(
    orig_to_new_substr={
        ".attention.wq": ".self_attn.q_proj",
        # ... other mappings
    },
    orig_to_new_prefix={
        "tok_embeddings.weight": "model.embed_tokens.weight",
        # ... other mappings
    },
)
```

### vLLM Engine Initialization

The script uses vLLM's high-level `LLM` class:

```python
llm = LLM(
    model=model_path,
    tensor_parallel_size=tensor_parallel_size,
    trust_remote_code=True,
)
```

## Troubleshooting

### "Model not found" Error

Ensure `config.json` exists in your checkpoint directory and specifies the correct architecture:
```json
{
  "architectures": ["TorchTitanQwen3ForCausalLM"],
  "model_type": "qwen3"
}
```

### Weight Loading Errors

Check that your checkpoint contains weights with TorchTitan naming conventions. You can inspect checkpoint keys:

```python
import torch
checkpoint = torch.load("path/to/checkpoint.pt")
print(checkpoint.keys())
```

### Memory Issues

- Reduce `--tensor-parallel-size` if you have limited GPU memory
- Use quantization (see vLLM documentation for quantization options)

## Performance Notes

- **Batch Processing**: The script processes multiple prompts in a single batch for efficiency
- **KV Caching**: vLLM automatically uses KV caching for fast autoregressive generation
- **Tensor Parallelism**: Use `--tensor-parallel-size` to distribute the model across multiple GPUs

## Next Steps

- See vLLM documentation for advanced features: https://docs.vllm.ai/
- Explore vLLM's serving capabilities for production deployments
- Configure quantization for reduced memory usage

## References

- [TorchTitan Qwen3 Model](../../../models/qwen3/model/model.py)
- [vLLM Qwen3 Model](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py)
- [vLLM Documentation](https://docs.vllm.ai/)
