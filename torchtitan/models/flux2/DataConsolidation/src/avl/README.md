# Generic HDF5 Dataloader

This repository provides a PyTorch-compatible dataloader for reading images from HDF5 files in a folder, with support for configurable transforms and multi-frame sampling.

## Features

- Loads images from all HDF5 files in a specified folder (supports `.hdf5` and `.h5`)
- Configurable transforms (resize, to_tensor, normalization)
- Multi-frame sampling (for video or sequence data)
- Augmentation options: center crop, random resized center crop
- YAML-based configuration for all hyperparameters and paths
- PyTorch `Dataset` and `DataLoader` interface

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Python Environment Setup

It is recommended to use a virtual environment for this project. You can use `venv` or `conda`:

### Using venv

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Using conda

```bash
conda create -n hdf5loader python=3.10
conda activate hdf5loader
pip install -r requirements.txt
```

## Configuration

The dataloader is configured using a YAML file.  
**Example `config.yaml`:**

```yaml
input:
  h5_folder: temp/ # Path to folder containing HDF5 files
  dataset_names: # List of dataset (topic) names to use, or leave empty/null for all
    - camera_0
    - camera_1

dataloader:
  batch_size: 8
  num_workers: 0

transforms:
  resize: [256, 256]
  to_tensor: true
  normalize: true

augmentation: resize_center # or random_resize_center
size: 256
scale_min: 0.15
scale_max: 0.5
num_frames: 1
frame_rate: 1
```

## Example Usage

Below is an example of how to use the provided dataloader with the YAML config:

```python
import yaml
from torch.utils.data import DataLoader
from dataloader import HDF5ImageDataset, build_transform

# Load configuration from YAML
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Build transforms from config
tf_cfg = config.get("transforms", {})
transform = build_transform(tf_cfg)
normalize = tf_cfg.get("normalize", True)

# Create the dataset
h5_folder = config["input"]["h5_folder"]
dataset_names = config["input"].get("dataset_names", None)
batch_size = config["dataloader"].get("batch_size", 2)
num_workers = config["dataloader"].get("num_workers", 0)
aug = config.get("augmentation", None)
size = config.get("size", None)
scale_min = config.get("scale_min", 0.08)
scale_max = config.get("scale_max", 1.0)
num_frames = config.get("num_frames", 1)
frame_rate = config.get("frame_rate", 1)

dataset = HDF5ImageDataset(
    h5_folder=h5_folder,
    dataset_names=dataset_names,
    transform=transform,
    normalize=normalize,
    aug=aug,
    size=size,
    scale_min=scale_min,
    scale_max=scale_max,
    num_frames=num_frames,
    frame_rate=frame_rate
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

# Example: iterate over one batch
print("Batch:")
for batch_imgs in loader:
    print("Batch shape:", batch_imgs.shape)
    break

dataset.close()
```

## Notes

- The dataloader will automatically find all `.hdf5` and `.h5` files in the specified folder.
- You can specify a particular dataset (topic) within the HDF5 files using `dataset_name`, or leave as `null` to use all datasets.
- Multi-frame sampling is supported via the `num_frames` and `frame_rate` parameters.
- Augmentation options are controlled by the `augmentation` field in the config.
