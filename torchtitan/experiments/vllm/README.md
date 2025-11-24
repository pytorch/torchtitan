# TorchTitan Qwen3 Model with vLLM Inference

This directory contains code to run vLLM inference on models trained with TorchTitan.

## Overview

The integration consists of two main components:

1. **Model Adapter** (`model/qwen3.py`): A custom model class that extends vLLM's `Qwen3ForCausalLM` to handle TorchTitan checkpoint naming conventions
2. **Inference Script** (`infer.py`): A simple script to register the model and run inference


## Quick Start

### Prerequisites

1. Install vLLM from source:
```bash
# install PyTorch first, either from PyPI or from source
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build.txt
uv pip install --no-build-isolation -e .
```
Using following command
https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#use-an-existing-pytorch-installation
