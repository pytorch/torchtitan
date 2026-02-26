# Deterministic RL Training with vLLM

This experiment combines vLLM's deterministic kernels with PyTorch autograd to enable reinforcement learning training where forward passes produce bitwise-identical results across runs.

## Overview

RL training requires both fast inference for generating rollouts and gradient computation for policy updates. vLLM provides deterministic forward passes but does not support gradients. This experiment adds backward passes to vLLM's operations.

The implementation:
1. Uses vLLM's batch-invariant kernels for forward passes
2. Implements custom backward passes for gradient computation
3. Provides weight conversion utilities between TorchTitan and vLLM formats

### Features

- Bitwise determinism: Same inputs produce identical outputs across runs
- Gradient support: Backward passes through vLLM operations
- Weight conversion: Utilities to convert between model formats

Note: Currently supports single-device training only.

## Architecture

### Components

1. `models/attention.py`: VLLMCompatibleFlashAttention
   - Uses vLLM's Flash Attention for forward pass
   - Implements custom backward pass for gradient computation
   - Uses `num_splits=1` for deterministic behavior

2. `models/qwen3/model_vllm_compat.py`: Qwen3VLLMCompatModel
   - Qwen3 model with merged gate/up projections matching vLLM format
   - Uses VLLMRMSNorm with gradient support

3. `batch_invariant_backward.py`: Backward passes for vLLM operations
   - Registers gradients for vLLM's batch-invariant operations
   - Supports matmul, linear, and RMSNorm
   - Patches Flash Attention for autograd

4. `weights_vllm_compat.py`: Weight conversion utilities
   - Converts between TorchTitan format (separate w1, w2, w3) and vLLM format (merged gate_up_proj)
   - Provides bidirectional conversion functions

5. `simple_rl.py`: RL training loop
   - Generates rollouts using vLLM engine
   - Computes advantages using GRPO-style ranking
   - Updates policy using PPO

## Installation

### Prerequisites

```bash
# Install vLLM with deterministic support (from source)
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build.txt
uv pip install --no-build-isolation -e .

# Install TorchTitan (from the repository root)
pip install -e .

# Install additional dependencies
pip install transformers safetensors huggingface_hub tensorboard
```

### Enable Batch Invariance

Initialize vLLM's batch-invariant mode before training:

```python
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from vllm.v1.attention.backends.registry import AttentionBackendEnum
init_batch_invariance(AttentionBackendEnum.FLASH_ATTN)
```

## Usage

### Quick Start

```python
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from vllm.v1.attention.backends.registry import AttentionBackendEnum

import torch
from torchtitan.experiments.rl.vllm_compat import (
    enable_batch_invariant_backward_mode,
    Qwen3VLLMCompatModel,
)

# 1. Enable deterministic mode
init_batch_invariance(AttentionBackendEnum.FLASH_ATTN)
enable_batch_invariant_backward_mode()

# 2. Load model
from torchtitan.models.qwen3.model.model import Qwen3Model
model_args = Qwen3Model.Config(
    dim=2048,
    n_layers=24,
    n_heads=16,
    n_kv_heads=2,
    vocab_size=151936,
)
model = Qwen3VLLMCompatModel(model_args).to('cuda').to(torch.bfloat16)

# 3. Forward pass (deterministic)
input_ids = torch.randint(0, 151936, (2, 128), device='cuda')
logits = model(input_ids)

# 4. Backward pass
loss = logits.sum()
loss.backward()

print("Done running simple model")

```

### Full RL Training

Run the RL training loop:

```bash
VLLM_BATCH_INVARIANT=1 VLLM_FLASH_ATTN_VERSION=3 python -m torchtitan.experiments.rl.vllm_compat.simple_rl
```

This will:
1. Download Qwen3-1.7B from HuggingFace
2. Initialize vLLM engine for rollouts
3. Generate samples for training prompts
4. Compute rewards and advantages
5. Update the policy using PPO
6. Log metrics to TensorBoard

View training progress:
```bash
tensorboard --logdir=./outputs/rl_training
```

## How It Works

### Deterministic Forward Pass

vLLM's batch-invariant mode makes operations deterministic:

```python
# These operations are deterministic when batch_invariance is enabled
y = torch.matmul(a, b)  # Uses vLLM's deterministic matmul
output = flash_attn_varlen_func(q, k, v, num_splits=1)  # Deterministic FA
```

### Backward Pass with Gradients

Custom backward passes:
1. Re-compute attention weights deterministically
2. Use standard chain rule for gradients
3. Apply gradients through vLLM's deterministic operations

```python
class FlashAttnWithBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ...):
        # Use vLLM's forward implementation
        return flash_attn_varlen_func(q, k, v, num_splits=1, ...)

    @staticmethod
    def backward(ctx, grad_output):
        # Compute gradients deterministically
        # (re-compute attention weights and apply chain rule)
        return grad_q, grad_k, grad_v, ...
```

### Bitwise Determinism Verification

The training loop compares logprobs from vLLM and TorchTitan:

```python
# During training, compare logprobs
vllm_logprobs = [from vLLM rollout]
titan_logprobs = [from TorchTitan forward pass]

assert torch.equal(vllm_logprobs, titan_logprobs)
```

## Testing

Run the test suite:

```bash
cd torchtitan/experiments/rl/vllm_compat/tests

# Test backward passes
python test_batch_invariant_backward.py

# Test determinism
python test_exact_determinism.py
```

## Technical Details

### Why Determinism Matters for RL

RL training steps:
1. Generate rollouts by sampling from the policy
2. Compute rewards based on the samples
3. Update the policy using gradients

If the forward pass during training differs from the forward pass during rollout, policy gradients may be incorrect. This matters for algorithms like PPO that compare old and new policy probabilities.

This implementation uses the same kernels for both rollouts (vLLM) and training (TorchTitan) to ensure `logprobs_rollout == logprobs_training` bitwise.

### Performance

- Rollout speed: Uses vLLM's optimized kernels
- Training speed: Similar to standard TorchTitan
- Memory: Saves activations for custom backward passes

### Limitations

1. Custom backward requires uniform sequence lengths
2. Only causal attention is supported
3. Requires NVIDIA GPUs with Flash Attention support

## Project Structure

```
rl/vllm_compat/
├── README.md                          # Documentation
├── __init__.py                        # Package initialization
├── batch_invariant_backward.py        # Backward passes for vLLM ops
├── weights_vllm_compat.py             # Weight conversion utilities
├── simple_rl.py                       # RL training loop
├── models/
│   ├── __init__.py
│   ├── attention.py                   # VLLMCompatibleFlashAttention
│   └── qwen3/
│       ├── __init__.py
│       └── model_vllm_compat.py       # vLLM-compatible Qwen3 model
├── weights/
│   ├── __init__.py
│   ├── converter.py                   # Weight conversion script
│   └── README.md                      # Weight conversion documentation
└── tests/
    ├── __init__.py
    ├── test_batch_invariant_backward.py  # Test backward passes
    └── test_exact_determinism.py         # Test determinism
```

## TODO

- `FlashAttnWithBackward` will need to become more composable and should not live exclusively within this directory.
- vLLM integration will need to become more generic with a provided Attention operator that is KV-cache compatible.
- vLLM parallelism will need to add generic parallelism initialization to support Monarch managed TP/DP.

## Contributing

This experiment is part of TorchTitan. To contribute:

1. Test your changes with `pytest tests/`
2. Verify bitwise determinism is maintained
3. Update this README if adding new features

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)

## License

This code is licensed under the BSD-style license found in the LICENSE file in the TorchTitan repository root directory.
