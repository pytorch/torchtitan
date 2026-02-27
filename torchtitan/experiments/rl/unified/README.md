# Run vLLM inference with TorchTitan Qwen3 Model

This directory contains code to run a single canonical model definition (TorchTitan model definition) with vLLM inference engine (not batch-invariant yet, working in progress). This work is actively developing and only supports inference for now.

This work is inspired by https://github.com/vllm-project/vllm/pull/28685.

## Overview
The integration consists of two main components:

1. **Model Adapter** (`model/qwen3.py`): A custom model class that extends vLLM's `Qwen3ForCausalLM` to handle TorchTitan checkpoint naming conventions
2. **Inference Script** (`infer.py`): A simple script to register the model and run inference


## Quick Start
### Prerequisites

1. Install Monarch:
```bash
uv pip install torchmonarch
```


2. Install PyTorch nightly for torchtitan, and pre-built vllm wheels (based on PyTorch nightly version).
```
# Install vllm with nightly torch
uv pip install torch vllm xformers  --pre \
--extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
--index-strategy unsafe-best-match
```

**NOTE:** The pre-built vLLM wheels are only compatible with CUDA 12.8, though they should work with most older CUDA versions. Alternatively, you can install the corresponding vLLM pre-built wheels directly from https://download.pytorch.org/whl/nightly/cu128, for example: `uv pip install vllm-1.0.0.dev20260219+cu130-<suffix>.whl`. Ensure the build version number (e.g., `dev20260219`) matches your PyTorch nightly installation.


3. Download `Qwen/Qwen3-0.6B` checkpoint from HuggingFace to `torchtitan/experiments/rl/example_checkpoint` folder.
```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B --local_dir torchtitan/experiments/rl/example_checkpoint --all --hf_token=...
```

4. Run inference with unified model definition:
```bash
torchrun --nproc_per_node=<world_size> \
      torchtitan/experiments/rl/unified/inference_example.py
```

5. Run simple GRPO rl loop
```
python3 torchtitan/experiments/rl/unified/simple_grpo.py --module rl.unified --config rl_grpo_qwen3_0_6b --hf_assets_path=<path_to_model_checkpoint>
```
We use a unified model definition for the trainer and generator, ensuring bitwise-identical models to address a class of subtle correctness bugs in RL for LLMs.

## TODO
Work on batch invariance:
1. Integrate with simple_rl_multiprocess.py to run end-to-end RL with one canonical model definition(UNIFIED mode).
2. Rewrite attention part to use vllm.Attention() with backward as the only attention path.
3. Leverage batch-invariant kernels into model definition.

Work on the RL loop:
1. Design trainer API and integrate with [train.py](https://github.com/pytorch/torchtitan/blob/main/torchtitan/train.py#L475)
2. Remove hardcoded configs and dependency on Qwen3 Model. Use torchtitan's config/TrainSpec instead, to work with any model.
3. Need to load the gsm8k dataset using TorchTitan dataset.
4. Need to properly implement weight saving and loading using TorchTitan's checkpoint mechanism, or use TorchStore. Also need to
    replace `vllm_to_torchtitan` and `torchtitan_to_vllm` calls to TorchTitan [state dict adaptor](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/qwen3/model/state_dict_adapter.py).
5. Right now we only support trainer run on multiple processes using DDP, and generator using TP, need to onboard more parallelism.
6. Right now we only support VLLM_COMPAT mode to achieve batch invariance and bitwise determinism, need to support UNIFIED mode.
7. In the longer term, need to add episode queue to achieve async, right now trainer and generator are running synchronously.
