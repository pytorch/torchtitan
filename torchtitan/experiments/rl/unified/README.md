# Run vLLM inference with TorchTitan Qwen3 Model

This directory contains code to run a single canonical model definition (TorchTitan model definition) with vLLM inference engine (not batch-invariant yet, working in progress). This work is actively developing and only supports inference for now.

This work is inspired by https://github.com/vllm-project/vllm/pull/28685.

## Overview
The integration consists of two main components:

1. **Model Adapter** (`model/qwen3.py`): A custom model class that extends vLLM's `Qwen3ForCausalLM` to handle TorchTitan checkpoint naming conventions
2. **Inference Script** (`infer.py`): A simple script to register the model and run inference


## Quick Start
### Prerequisites

1. Install PyTorch nightly & Monarch for torchtitan:
```bash
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
uv pip install torchmonarch
```

Install Flash Attention v3 kernels:
```
# CUDA 12
pip install flash-attn-3 --extra-index=https://download.pytorch.org/whl/test/cu126
# CUDA 13
pip install flash-attn-3 --extra-index=https://download.pytorch.org/whl/test/cu130
```

2. Install vLLM from source [vllm-use-an-existing-pytorch-installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#use-an-existing-pytorch-installation):
```bash
# install PyTorch first, either from PyPI or from source
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build.txt
uv pip install --no-build-isolation -e .
```


NOTE: If `flash_attn_varlen_func` hits error "torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain" during forward path, this is due to GPU driver version is not compatible with vLLM/PyTorch compiled version. Use the following command to recompile vLLM.

```bash
# Set CUDA version environment variable
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Clean previous build
rm -rf build dist *.egg-info
uv pip uninstall -y vllm

# Rebuild vLLM from source with CUDA 12.4
uv pip install -e .

```

3. Download Qwen/Qwen3-0.6B checkpoint from HuggingFace and put into `torchtitan/experiments/rl/example_checkpoint` folder.
```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B --local_dir torchtitan/experiments/rl/example_checkpoint --all --hf_token=...
```

4. Run inference:
```bash
python torchtitan/experiments/rl/unified/infer.py --model-ckpt-path <path_to_model_checkpoint>
```

Run with TP: (work in progress)
```bash
python torchtitan/experiments/rl/unified/infer.py --model-ckpt-path <path_to_model_checkpoint> --tensor-parallel-size 2

```

5. Run simple rl loop
```bash
VLLM_BATCH_INVARIANT=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN python3 torchtitan/experiments/rl/unified/simple_rl_multiprocess.py
```
Right now we only support VLLM_COMPAT mode, which could achieve trainer and generator bitwise identical. We are working on support UNIFIED mode,
which uses a unified model definition for trainer and generator.

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
7. In the longer term, need to add trajectory queue to achieve async, right now trainer and generator are running synchronously.
