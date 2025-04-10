# FLUX model in torchtitan

## Overview

## Install dependencies

```bash
pip install -U -r requirements.txt
```

## Usage
First, download the autoencoder model from HuggingFace with your own access token:
```bash
python torchtitan/experiments/flux/scripts/download_autoencoder.py --repo_id black-forest-labs/FLUX.1-dev --ae_path ae.safetensors --hf_token <your_access_token>
```
This step will download the autoencoder model from HuggingFace and save it to the `torchtitan/experiments/flux/assets/autoencoder/ae.safetensors` file.

Run the following command to train the model on a single GPU:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=1 torchtitan/experiments/flux/train.py --job.config_file torchtitan/experiments/flux/train_configs/debug_model.toml
```

## TODO
- [ ] Supporting for multiple GPUs is comming soon (FSDP, etc)
- [ ] Implement test cases in CI for FLUX model. Adding more unit tests for FLUX model (eg, unit test for preprocessor, etc)
- [ ] More parallesim support (Tensor Parallelism, Context Parallelism, etc)
- [ ] Support for distributed checkpointing and loading
- [ ] Implement init_weights() function to initialize the model weights
- [ ] Implement the num_flops_per_token calculation in get_nparams_and_flops() function
