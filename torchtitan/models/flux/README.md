# FLUX model in torchtitan

## Overview
This directory contains the implementation of the [FLUX](https://github.com/black-forest-labs/flux/tree/main) model in torchtitan. In torchtitan, we showcase the pre-training process of text-to-image part of the FLUX model.

## Prerequisites
Install the required dependencies:
```bash
pip install -r requirements-flux.txt
```

## Usage
First, download the autoencoder model from HuggingFace with your own access token:
```bash
python scripts/download_hf_assets.py --repo_id black-forest-labs/FLUX.1-dev --additional_patterns ae.safetensors --hf_token <your_access_token>
```

This step will download the autoencoder model from HuggingFace and save it to the `assets/hf/FLUX.1-dev/ae.safetensors` file.

Run the following command to train the debug model on a single GPU:
```bash
MODULE=flux CONFIG=flux_debugmodel .run_train.sh
```

If you want to train with other configs, run the following command:
```bash
MODULE=flux CONFIG=flux_schnell ./run_train.sh
```


## Supported Features
- Parallelism: The model supports FSDP, HSDP, CP for training on multiple GPUs.
- Activation checkpointing: The model uses activation checkpointing to reduce memory usage during training.
- Distributed checkpointing and loading.
    - Notes on the current checkpointing implementation: To keep the model weights are sharded the same way as checkpointing, we need to shard the model weights before saving the checkpoint. This is done by checking each module at the end of evaluation, and sharding the weights of the module if it is a FSDPModule.
- CI for FLUX model. Supported periodically running integration tests on 8 GPUs, and unittests.


## TODO
- [ ] More parallesim support (Tensor Parallelism, Pipeline Parallelism, etc)
- [ ] Implement the num_flops_per_token calculation in get_nparams_and_flops() function
- [ ] Add `torch.compile` support
