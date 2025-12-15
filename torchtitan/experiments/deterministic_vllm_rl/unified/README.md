# Run vLLM inference with TorchTitan Qwen3 Model

This directory contains code to run a single canonical model definition (TorchTitan model definition) with vLLM inference engine (not batch-invariant yet, working in progress).
This work is inspired by https://github.com/vllm-project/vllm/pull/28685.

## Overview
The integration consists of two main components:

1. **Model Adapter** (`model/qwen3.py`): A custom model class that extends vLLM's `Qwen3ForCausalLM` to handle TorchTitan checkpoint naming conventions
2. **Inference Script** (`infer.py`): A simple script to register the model and run inference


## Quick Start
### Prerequisites

1. Install PyTorch nightly for torchtitan:
```
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
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

```
# Set CUDA version environment variable
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Clean previous build
rm -rf build dist *.egg-info
uv pip uninstall -y vllm

# Rebuild vLLM from source with CUDA 12.4
pip install -e .

```

3. Download Qwen3/Qwen3-0.6b checkpoint from HuggingFace and put into `example_checkpoint` folder.


4. Run inference:
```
python torchtitan/experiments/deterministic_vllm_rl/infer.py --model torchtitan/experiments/deterministic_vllm_rl/example_checkpoint/qwen3-0.6B
```

Run with TP: (work in progress)
```
python torchtitan/experiments/deterministic_vllm_rl/infer.py --model torchtitan/experiments/deterministic_vllm_rl/example_checkpoint/qwen3-0.6B --tensor-parallel-size 2

```

## TODO
1. Rewrite attention part to use vllm.Attention() with backward as the only attention path.
2. Integrate with simple_rl.py to run end-to-end RL with one canonical model definition.
3.  Leverage batch-invariant kernels into model definition.
