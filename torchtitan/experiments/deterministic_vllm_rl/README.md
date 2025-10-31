# Deterministic RL Training with vLLM

This experiment provides a complete framework for **bitwise-deterministic reinforcement learning training** that combines:
- **vLLM** for fast, deterministic rollouts
- **TorchTitan** for efficient training with gradients
- **Custom backward passes** to maintain determinism through the entire training loop

## Overview

Traditional RL training faces a challenge: you want fast inference for generating rollouts, but you also need gradients for training. vLLM is extremely fast but doesn't support gradients. Standard PyTorch supports gradients but can be non-deterministic.

This experiment solves both problems by:
1. Using vLLM's deterministic kernels for forward passes (both rollouts and training)
2. Adding custom backward passes that are also deterministic
3. Achieving **bitwise-identical results** across runs for the entire training loop

### Key Features

- **Bitwise Determinism**: Same inputs always produce identical outputs (bit-for-bit)
- **vLLM Speed**: Fast rollouts using vLLM's optimized kernels
- **Gradient Support**: Full backward pass support for training
- **Model Compatibility**: Drop-in replacement for standard Qwen3 models in TorchTitan

## Architecture

### Components

1. **`models/attention.py`**: VLLMCompatibleFlashAttention
   - Uses vLLM's Flash Attention for forward pass
   - Implements custom backward pass for gradients
   - Maintains determinism with `num_splits=1`

2. **`models/qwen3/model_vllm_compat.py`**: Qwen3VLLMCompatModel
   - vLLM-compatible Qwen3 implementation
   - Merged gate/up projections (like vLLM)
   - Uses VLLMRMSNorm with gradient support

3. **`batch_invariant_backward.py`**: Gradient support for vLLM operations
   - Registers backward passes for vLLM's batch-invariant operations
   - Supports matmul, linear, and RMSNorm
   - Patches Flash Attention to work with autograd

4. **`simple_rl.py`**: End-to-end RL training loop
   - Generates rollouts using vLLM
   - Computes advantages using GRPO-style ranking
   - Updates policy using PPO with bitwise-deterministic gradients

## Installation

### Prerequisites

```bash
# Install vLLM with deterministic support
pip install vllm

# Install TorchTitan (from the repository root)
pip install -e .

# Install additional dependencies
pip install transformers safetensors huggingface_hub tensorboard
```

### Enable Batch Invariance

Before running any training, you must initialize vLLM's batch-invariant mode:

```python
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
init_batch_invariance()
```

## Usage

### Quick Start

```python
import torch
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from torchtitan.experiments.deterministic_vllm_rl import (
    patch_batch_invariant_with_gradients,
    Qwen3VLLMCompatModel,
)

# 1. Enable deterministic mode
init_batch_invariance()
patch_batch_invariant_with_gradients()

# 2. Load model
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
model_args = Qwen3ModelArgs(
    dim=2048,
    n_layers=24,
    n_heads=16,
    n_kv_heads=2,
    vocab_size=151936,
)
model = Qwen3VLLMCompatModel(model_args)

# 3. Forward pass (deterministic)
input_ids = torch.randint(0, 151936, (2, 128), device='cuda')
logits = model(input_ids)

# 4. Backward pass (also deterministic!)
loss = logits.sum()
loss.backward()
```

### Full RL Training

Run the complete RL training loop:

```bash
cd torchtitan/experiments/deterministic_vllm_rl
python simple_rl.py
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

vLLM's batch-invariant mode ensures that all operations are deterministic:

```python
# These operations are deterministic when batch_invariance is enabled
y = torch.matmul(a, b)  # Uses vLLM's deterministic matmul
output = flash_attn_varlen_func(q, k, v, num_splits=1)  # Deterministic FA
```

### Backward Pass with Gradients

We add custom backward passes that:
1. Re-compute attention weights (deterministic)
2. Use standard chain rule for gradients
3. Apply gradients through vLLM's deterministic operations

```python
class FlashAttnWithBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ...):
        # Use vLLM's fast forward
        return flash_attn_varlen_func(q, k, v, num_splits=1, ...)

    @staticmethod
    def backward(ctx, grad_output):
        # Compute gradients deterministically
        # (re-compute attention weights and apply chain rule)
        return grad_q, grad_k, grad_v, ...
```

### Bitwise Determinism Verification

The training loop verifies that vLLM and TorchTitan produce identical logprobs:

```python
# During training, compare logprobs
vllm_logprobs = [from vLLM rollout]
titan_logprobs = [from TorchTitan forward pass]

assert torch.equal(vllm_logprobs, titan_logprobs)  # Should be true!
```

## Testing

Run the test suite to verify determinism:

```bash
cd torchtitan/experiments/deterministic_vllm_rl/tests

# Test backward passes work correctly
python test_batch_invariant_backward.py

# Test exact determinism (bit-for-bit identical)
python test_exact_determinism.py
```

Expected output:
```
✓ All operations are exactly deterministic!
✓ vLLM-TorchTitan bitwise determinism verified: N tokens match exactly
```

## Technical Details

### Why Determinism Matters for RL

In RL training, we need to:
1. Generate rollouts (sampling from the policy)
2. Compute rewards based on the samples
3. Update the policy using gradients

**The problem**: If the forward pass during training differs from the forward pass during rollout, the gradients will be wrong! This is especially important for PPO, which compares old and new policy probabilities.

**The solution**: Use the same deterministic kernels for both rollouts (vLLM) and training (TorchTitan). This ensures that `logprobs_rollout == logprobs_training` (bitwise).

### Performance

- **Rollout speed**: ~100x faster than standard PyTorch (thanks to vLLM)
- **Training speed**: Same as standard TorchTitan
- **Memory**: Slightly higher (saves activations for custom backward)

### Limitations

1. **Sequence length**: Custom backward requires uniform sequence lengths
2. **Attention**: Only causal attention is supported
3. **Hardware**: Requires NVIDIA GPUs with Flash Attention support

## Project Structure

```
deterministic_vllm_rl/
├── README.md                          # This file
├── __init__.py                        # Package initialization
├── batch_invariant_backward.py        # Gradient support for vLLM ops
├── simple_rl.py                       # End-to-end RL training loop
├── models/
│   ├── __init__.py
│   ├── attention.py                   # VLLMCompatibleFlashAttention
│   └── qwen3/
│       ├── __init__.py
│       └── model_vllm_compat.py      # vLLM-compatible Qwen3 model
└── tests/
    ├── __init__.py
    ├── test_batch_invariant_backward.py  # Test gradients work
    └── test_exact_determinism.py         # Test bitwise determinism
```

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
