# Flux model in torchtitan

## Overview

## Usage

1. Download the autoencoder used in Flux model with following command:
```
    python scripts/download_from_hf.py --repo_id black-forest-labs/FLUX.1-dev --autoencoder --autoencoder_path "ae.safetensors" --hf_token=...

```
2. Run the following command to train the model:
```
    torchrun torchtitan/experiments/flux/train.py --job.config_file torchtitan/experiments/flux/train_configs/debug_model.toml

```

## Code Structure
