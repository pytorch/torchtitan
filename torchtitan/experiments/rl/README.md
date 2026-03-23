# RL Training with TorchTitan and vLLM

This directory contains code for RL training using TorchTitan model definitions with vLLM inference engine for fast rollout generation.

## Overview
The integration consists of the following components:

1. **vLLM Model Wrapper** (`models/vllm_wrapper.py`): Adapts TorchTitan models for vLLM's inference engine
2. **RL Training Loop** (`simple_grpo_sum_digits.py`): GRPO-based RL training with Monarch actors
3. **Inference Script** (`inference_example.py`): Standalone inference using the vLLM engine


## Quick Start
### Prerequisites

0. Create and activate environment with uv:
```bash
pip install uv
uv venv --python 3.12 titan-rl
source titan-rl/bin/activate
```

1. Install Monarch and TorchStore from main:
```bash
uv pip install torchmonarch==0.3.0
uv pip install --no-deps "git+https://github.com/meta-pytorch/torchstore.git@main"
uv pip install pygtrie portpicker
```

2. Install Flash Attention 3 kernels:
```bash
# Flash Attention v3 (recommended for H100/H200 and newer GPUs)
uv pip install flash-attn-3 --extra-index-url=https://download.pytorch.org/whl/test/cu128
```

**NOTE:** FA2 is bundled with PyTorch and will be used automatically on older GPUs (e.g. A100) that don't support FA3.

3. Install PyTorch nightly for torchtitan, and pre-built vllm wheels (based on PyTorch nightly version).
```bash
# Install vllm with nightly torch
uv pip install torch vllm xformers  --pre \
--extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
--index-strategy unsafe-best-match
```

**NOTE:** The pre-built vLLM wheels are only compatible with CUDA 12.8, though they should work with most older CUDA versions. Alternatively, you can install the corresponding vLLM pre-built wheels directly from https://download.pytorch.org/whl/nightly/cu128, for example: `uv pip install vllm-1.0.0.dev20260219+cu130-<suffix>.whl`. Ensure the build version number (e.g., `dev20260219`) matches your PyTorch nightly installation.


4. Install TorchTitan in editable mode:
```bash
uv pip install -e .
```

5. Download `Qwen/Qwen3-0.6B` (or `Qwen/Qwen3-1.7B`) checkpoint from HuggingFace to `torchtitan/experiments/rl/example_checkpoint` folder.
```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B --local_dir torchtitan/experiments/rl/example_checkpoint --all --hf_token=...

python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B --local_dir torchtitan/experiments/rl/example_checkpoint --all --hf_token=...
```

6. Run inference with torchtitan model definition:
```bash
torchrun --nproc_per_node=2 torchtitan/experiments/rl/inference_example.py
```

**NOTE:**: Set `--nproc_per_node` to the world size, which should match the `tensor_parallel_degree` in the `VLLMGenerator` config.

7. Run simple GRPO RL loop to learn sum digits task
```bash
python torchtitan/experiments/rl/simple_grpo_sum_digits.py --module rl --config rl_grpo_qwen3_0_6b
```

**NOTE:** If you downloaded your HF model to a different path than the one in step 4, specify it in your command with `--hf_assets_path=<path_to_model_checkpoint>`.

We use a unified model definition from torchtitan for the trainer and generator, ensuring bitwise-identical models to address a class of subtle correctness bugs in RL for LLMs.



**Current status:** Batch invariance is only supported for single-GPU configurations (TP=1) for both the trainer and generator. When tensor parallelism is enabled (TP > 1), batch-invariant mode is not yet supported.
