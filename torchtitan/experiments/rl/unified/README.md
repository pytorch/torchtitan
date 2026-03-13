# Run vLLM inference with TorchTitan Qwen3 Model

This directory contains code to run a single canonical model definition (TorchTitan model definition) with vLLM inference engine (not batch-invariant yet, working in progress). This work is actively developing and only supports inference for now.

This work is inspired by https://github.com/vllm-project/vllm/pull/28685.

## Overview
The integration consists of two main components:

1. **Model Adapter** (`model/qwen3.py`): A custom model class that extends vLLM's `Qwen3ForCausalLM` to handle TorchTitan checkpoint naming conventions
2. **Inference Script** (`inference_example.py`): A simple script to register the model and run inference


## Quick Start
### Prerequisites

0. Create and activate environment with uv:
```bash
uv venv --python 3.12 titan-rl
source titan-rl/bin/activate
```

1. Install Monarch:
```bash
uv pip install torchmonarch
```


2. Install PyTorch nightly for torchtitan, and pre-built vllm wheels (based on PyTorch nightly version).
```bash
# Install vllm with nightly torch
uv pip install torch vllm xformers  --pre \
--extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
--index-strategy unsafe-best-match
```

**NOTE:** The pre-built vLLM wheels are only compatible with CUDA 12.8, though they should work with most older CUDA versions. Alternatively, you can install the corresponding vLLM pre-built wheels directly from https://download.pytorch.org/whl/nightly/cu128, for example: `uv pip install vllm-1.0.0.dev20260219+cu130-<suffix>.whl`. Ensure the build version number (e.g., `dev20260219`) matches your PyTorch nightly installation.


3. Install TorchTitan in editable mode:
```bash
uv pip install -e .
```

4. Download `Qwen/Qwen3-0.6B` (or `Qwen/Qwen3-1.7B`) checkpoint from HuggingFace to `torchtitan/experiments/rl/example_checkpoint` folder.
```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B --local_dir torchtitan/experiments/rl/example_checkpoint --all --hf_token=...

python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B --local_dir torchtitan/experiments/rl/example_checkpoint --all --hf_token=...
```

5. Run inference with unified model definition:
```bash
torchrun --nproc_per_node=2 torchtitan/experiments/rl/unified/inference_example.py
```

**NOTE:**: Set `--nproc_per_node` to the world size, which should match the `tensor_parallel_degree` in the `VLLMGenerator` config.

6. Run simple GRPO RL loop for model to learn sum digits task
```bash
python torchtitan/experiments/rl/unified/simple_grpo_sum_digits.py --module rl.unified --config rl_grpo_qwen3_0_6b
```

**NOTE:** If you downloaded your HF model to a different path than the one in step 4, specify it in your command with `--hf_assets_path=<path_to_model_checkpoint>`.

We use a unified model definition for the trainer and generator, ensuring bitwise-identical models to address a class of subtle correctness bugs in RL for LLMs.



**Current status:** Batch invariance is only supported for single-GPU configurations (TP=1) for both the trainer and generator. When tensor parallelism is enabled (TP > 1), batch-invariant mode is not yet supported.
