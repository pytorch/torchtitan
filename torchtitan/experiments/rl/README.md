# Deterministic RL Training with vLLM

This package provides two approaches for integrating TorchTitan models with vLLM:

1. vllm_compat/ - vLLM-Compatible approach
   - Separate model definition matching vLLM's weight format
   - Support batch-invariant and bit-wise identity between train and inference
   - Custom backward passes for attention gradient computation

2. unified/ - Unified approach
   - Uses canonical TorchTitan model definition for inference directly
   - Replaces attention with vLLM Compatible attention for inference
