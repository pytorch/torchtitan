# Custom Datasets in torchtitan

`torchtitan` is designed to work seamlessly with most HuggingFace datasets. While we provide the C4 dataset for numerics and convergence testing, you can easily add support for your own datasets. Here's how to do it using Wikipedia as an example.

## Quick Start
Locate the dataset configuration file:
```
torchtitan/datasets/hf_datasets/hf_datasets.py
```

## Adding Your Dataset
You'll need to add three components:
1. A dataset loader function
2. A sample processor function
3. A dataset configuration entry

### 1. Define Dataset Loader
Create a function that specifies how to load your dataset:

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
```

### 2. Define Sample Processor
Create a function that processes individual samples from your dataset:

```python
def process_wikipedia_text(sample: Dict[str, Any]) -> str:
    """Process Wikipedia dataset sample text."""
    return f"{sample['title']}\n\n{sample['text']}"
```

### 3. Register Your Dataset
Add your dataset configuration to the DATASETS dictionary:

```python
DATASETS = {
    # ... existing datasets ...
    "wikipedia": DatasetConfig(
        path="wikipedia",  # default HuggingFace dataset path
        loader=load_wikipedia_dataset,
        text_processor=process_wikipedia_text,
    ),
}
```

### 4. Configure Your Training
In your training configuration file (`.toml`), set your dataset:

```toml
dataset = "wikipedia"
```

That's it! Your custom dataset is now ready to use with `torchtitan`.

## Key Points
- The DatasetConfig contains all necessary components for a dataset:
  - `path`: The default path to the dataset (can be overridden during training)
  - `loader`: Function to load the dataset
  - `text_processor`: Function to process individual samples
- The loader function should return a HuggingFace dataset object
- The processor function should return a string that combines the relevant fields from your dataset
- Use `streaming=True` for large datasets to manage memory efficiently

Now you can start training with your custom dataset!
