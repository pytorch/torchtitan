# Optimized Valeo Dataset Dataloader

This repository contains an optimized dataloader (`ValeoDatasetPreprocessed`) and a preprocessing script designed to significantly accelerate data loading from the Valeo dataset.

The key performance gain comes from a one-time preprocessing step that merges multiple camera videos and pre-calculates frame synchronization, minimizing I/O and computation during training.

## Synchronization Strategies

We offer two specialized versions of the `ValeoDatasetPreprocessed` class depending on your training requirements:

### A. Full Multi-Modal Synchronization (`valeo_dataset_full_lidm.py`)

This script provides a synchronized loading pipeline that retrieves camera slices from merged videos alongside corresponding Velodyne LiDAR range images stored in HDF5 format.

### B. High-Efficiency Camera Loading (`valeo_dataset.py`)

This script implements a high-performance "Open-Read-Close" strategy that slices and processes individual camera views one-by-one directly from the merged grid.

## Quickstart Workflow

The process involves two main steps:

1.  **Preprocess (Run Once):** Convert the raw dataset into an optimized format.
2.  **Train/Evaluate:** Use the `ValeoDatasetPreprocessed` class that points to the optimized data.

---

### Step 1: Preprocess the Raw Dataset

Run the `preprocess_valeo_dataset.py` script to create the optimized dataset. This script will:
-   Merge all camera videos into a single `merged_cameras.mp4`.
-   Create a `metadata.parquet` file with all frame synchronization info.
-   Copy LiDAR and GNSS data into a clean directory structure.

**Command:**
```bash
python preprocess_valeo_dataset.py \
    --input_path /path/to/raw/valeo/dataset \
    --output_path /path/to/save/preprocessed_dataset
```

This will create the `preprocessed_dataset` directory containing all the necessary files.

---

### Step 2: Use the Optimized Dataloader

Update your code to use the `ValeoDatasetPreprocessed` class, pointing it to the newly created directory.

**Example Instantiation:**
```python
from valeo_dataset import ValeoDatasetPreprocessed
from torch.utils.data import DataLoader

# Path to the directory created in Step 1
preprocessed_path = "/path/to/save/preprocessed_dataset"

# Instantiate the fast dataset
dataset = ValeoDatasetPreprocessed(base_path=preprocessed_path)

# Use with a PyTorch DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
```

You can also use `test_dataloader.py` for testing

#### **Important Note on Data Types**

For maximum speed, the dataloader defers some data conversion. You must handle this in your training loop.

-   **Camera Images:** The dataloader returns `torch.uint8` tensors. Convert and normalize them on the GPU.

```python
# In your training loop
for batch in dataloader:
    images_uint8 = batch['camera'].to('cuda')
    
    # Convert to float and normalize to [-1, 1]
    images_float32 = (images_uint8.float() / 127.5) - 1.0
    
    # ... feed images_float32 to your model
```
