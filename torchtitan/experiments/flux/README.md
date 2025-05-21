# FLUX model in torchtitan

## Overview
This directory contains the implementation of the [FLUX](https://github.com/black-forest-labs/flux/tree/main) model in torchtitan. In torchtitan, we showcase the pre-training process of text-to-image part of the FLUX model.

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

## Supported Features
- Parallelism: The model supports FSDP, HSDP for training on multiple GPUs.
- Activation checkpointing: The model uses activation checkpointing to reduce memory usage during training.
- Distributed checkpointing and loading.
    - Notes on the current checkpointing implementation: Currently we need to enable `reshard_after_forward=True` before eval
    process, and set it back to `False` after eval process. The reason is that eval step only runs forward, but not backward,
    so FSDP reshard_after_forward plan would interfere with how parameters look like for the potential subsequent checkpointing step.



## TODO
- [ ] More parallesim support (Tensor Parallelism, Context Parallelism, etc)
- [ ] Implement the num_flops_per_token calculation in get_nparams_and_flops() function
- [ ] Implement test cases in CI for FLUX model. Adding more unit tests for FLUX model (eg, unit test for preprocessor, etc)
- [ ] Checkpointing followup: Merge resharding strategy in `flux/trainer.py` to `parallel_flux.py`
