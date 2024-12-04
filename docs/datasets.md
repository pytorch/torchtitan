# Custom Datasets in TorchTitan

TorchTitan is designed to work seamlessly with most HuggingFace datasets. While we provide the C4 dataset for numerics and convergence testing, you can easily add support for your own datasets. Here's how to do it using Wikipedia as an example.

## Quick Start

1. Install TorchTitan from source:
```bash
pip install -e .
```

2. Locate the dataset configuration file:
```
torchtitan/datasets/hf_datasets/hf_datasets.py
```

## Adding Your Dataset

You'll need to add two main components:

1. A dataset loader function
2. A sample processor function

### 1. Define Dataset Loader

Add a function that specifies how to load your dataset:

```python
def load_wikipedia_dataset(dataset_path: str, **kwargs):
    """Load Wikipedia dataset with specific configuration."""
    logger.info("Loading Wikipedia dataset...")
    return load_dataset(
        dataset_path,
        name="20220301.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

# Register your loader in DATASET_LOADERS
DATASET_LOADERS = {
    # ... existing loaders ...
    "wikipedia": load_wikipedia_dataset,
}
```

### 2. Define Sample Processor

Add a function that processes individual samples from your dataset:

```python
def process_wikipedia_text(sample: Dict[str, Any]) -> str:
    """Process Wikipedia dataset sample text."""
    return f"{sample['title']}\n\n{sample['text']}"

# Register your processor in DATASET_TEXT_PROCESSORS
DATASET_TEXT_PROCESSORS = {
    # ... existing processors ...
    "wikipedia": process_wikipedia_text,
}
```

### 3. Configure Your Training

In your training configuration file (`.toml`), set your dataset:

```toml
dataset = "wikipedia"
```

That's it! Your custom dataset is now ready to use with TorchTitan.

## Key Points

- The loader function should return a HuggingFace dataset object
- The processor function should return a string that combines the relevant fields from your dataset
- Make sure your dataset name matches exactly in both the loader and processor registrations
- Use streaming=True for large datasets to manage memory efficiently

Now you can start training with your custom dataset!
