<div align="center">

# FLUX model in torchtitan

[![integration tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_flux.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_flux.yaml/badge.svg?branch=main)

</div>

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
python torchtitan/experiments/flux/scripts/download_autoencoder.py --repo_id black-forest-labs/FLUX.1-dev --ae_path ae.safetensors --hf_token <your_access_token>
```

This step will download the autoencoder model from HuggingFace and save it to the `torchtitan/experiments/flux/assets/autoencoder/ae.safetensors` file.

Run the following command to train the model on a single GPU:
```bash
./torchtitan/experiments/flux/run_train.sh

```

If you want to train with other model config, run the following command:
```bash
CONFIG_FILE="./torchtitan/experiments/flux/train_configs/flux_schnell_model.toml" ./torchtitan/experiments/flux/run_train.sh
```

## Running Tests

### Unit Tests
To run the unit tests for the FLUX model, use the following command:
```bash
pytest -s torchtitan/experiments/flux/tests/
```

### Integration Tests
To run the integration tests for the FLUX model, use the following command:
```bash
python -m torchtitan.experiments.flux.tests.integration_tests <output_dir>
```


## Supported Features
- Parallelism: The model supports FSDP, HSDP for training on multiple GPUs.
- Activation checkpointing: The model uses activation checkpointing to reduce memory usage during training.
- Distributed checkpointing and loading.
    - Notes on the current checkpointing implementation: To keep the model wieghts are sharded the same way as checkpointing, we need to shard the model weights before saving the checkpoint. This is done by checking each module at the end of envaluation, and sharding the weights of the module if it is a FSDPModule.
- CI for FLUX model. Supported periodically running integration tests on 8 GPUs, and unittests.



## TODO
- [ ] More parallesim support (Tensor Parallelism, Context Parallelism, etc)
- [ ] Implement the num_flops_per_token calculation in get_nparams_and_flops() function
- [ ] Add `torch.compile` support
