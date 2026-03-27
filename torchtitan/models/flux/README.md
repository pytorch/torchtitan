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
MODULE=flux CONFIG=flux_debugmodel ./run_train.sh
```

If you want to train with other configs, run the following command:
```bash
MODULE=flux CONFIG=flux_schnell ./run_train.sh
```


## Supported Features
- Parallelism: The model supports FSDP, HSDP, CP for training on multiple GPUs.
- Activation checkpointing: The model uses activation checkpointing to reduce memory usage during training.
- `torch.compile`: Per-block compilation for the Flux transformer (DoubleStreamBlock, SingleStreamBlock). See [torch.compile](#torchcompile) below.
- MXFP8 quantization: Dynamic MXFP8 quantization for linear layers on SM100+ (Blackwell) hardware. See [MXFP8 Quantization](#mxfp8-quantization) below.
- Distributed checkpointing and loading.
    - Notes on the current checkpointing implementation: To keep the model weights are sharded the same way as checkpointing, we need to shard the model weights before saving the checkpoint. This is done by checking each module at the end of evaluation, and sharding the weights of the module if it is a FSDPModule.
- CI for FLUX model. Supported periodically running integration tests on 8 GPUs, and unittests.

## torch.compile

The Flux model supports `torch.compile` for accelerating training. Compilation is applied per-block to the repeated DoubleStreamBlock and SingleStreamBlock layers in the main transformer.

To enable compilation, add the following flags:
```bash
MODULE=flux CONFIG=flux_debugmodel ./run_train.sh --compile.enable
```

By default, both the model and the loss function are compiled. You can control which components are compiled via `--compile.components`:
```bash
# Compile only the model (not the loss)
MODULE=flux CONFIG=flux_debugmodel ./run_train.sh --compile.enable --compile.components '["model"]'

# Compile only the loss
MODULE=flux CONFIG=flux_debugmodel ./run_train.sh --compile.enable --compile.components '["loss"]'
```

**Notes:**
- The Flux model blocks are compiled with `fullgraph=True` for maximum optimization.
- The default backend is `inductor`. You can change it with `--compile.backend <backend>`.


## MXFP8 Quantization

The Flux model supports MXFP8 (Microscaling FP8) quantization for accelerating training on SM100+ hardware (B200, B100). This uses the existing `MXFP8Converter` from torchtitan, which dynamically quantizes linear layers to MXFP8 precision.

**Requirements:**
- SM100+ GPU (e.g., NVIDIA B200, B100)
- `torchao` nightly

### Using Config Presets

Pre-configured presets with MXFP8 and `torch.compile` enabled:

```bash
# Flux schnell with MXFP8
MODULE=flux CONFIG=flux_schnell_mxfp8 ./run_train.sh

# Flux dev with MXFP8
MODULE=flux CONFIG=flux_dev_mxfp8 ./run_train.sh
```

### Custom Configuration

To create a custom MXFP8 config, define a new function in `config_registry.py`:

```python
def my_custom_mxfp8() -> FluxTrainer.Config:
    config = flux_schnell()  # or flux_dev()
    config.compile = CompileConfig(enable=True)
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            MXFP8Converter.Config(
                fqns=[
                    "double_blocks",
                    "single_blocks",
                    "img_in",
                    "txt_in",
                    "time_in",
                    "vector_in",
                    "final_layer",
                ],
            ),
        ],
    )
    return config
```

The `fqns` parameter specifies which fully qualified module names to quantize. The listed names cover all the linear-layer-containing submodules in the Flux transformer. Dimensions must be aligned to 32 bytes for MXFP8; Flux's default hidden sizes satisfy this requirement.

## TODO
- [ ] More parallelism support (Tensor Parallelism, Pipeline Parallelism, etc)
- [ ] Implement the num_flops_per_token calculation in get_nparams_and_flops() function
