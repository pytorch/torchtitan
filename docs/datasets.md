# Custom Datasets in torchtitan

`torchtitan` is designed to work seamlessly with most HuggingFace datasets. It supports three training flavours — **pre-training** (plain text), **instruction-tuning / SFT** (chat), and **multimodal** (vision) — each with its own dataloader. Both text flavours support single-source and multi-source interleaved configurations.

## Dataset file locations

```
torchtitan/hf_datasets/text_datasets.py        # pre-training and SFT
torchtitan/hf_datasets/multimodal/mm_datasets.py  # vision
```

---

## Pre-training datasets

### Adding a custom text dataset

You need three components: a loader function, a sample processor, and a registry entry.

#### 1. Define a dataset loader

```python
def load_wikipedia_dataset(dataset_path: str, **kwargs):
    """Load Wikipedia dataset with specific configuration."""
    return load_dataset(
        dataset_path,
        name="20220301.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
```

#### 2. Define a sample processor

```python
def process_wikipedia_text(sample: dict[str, Any]) -> str:
    """Process Wikipedia dataset sample text."""
    return f"{sample['title']}\n\n{sample['text']}"
```

#### 3. Register your dataset

```python
DATASETS = {
    # ... existing datasets ...
    "wikipedia": DatasetConfig(
        path="wikipedia",
        loader=load_wikipedia_dataset,
        sample_processor=process_wikipedia_text,
    ),
}
```

#### 4. Configure training

```python
dataloader=HuggingFaceTextDataLoader.Config(
    dataset="wikipedia",
    infinite=True,
),
```

---

## Instruction-tuning / SFT datasets (chat)

The `ChatDataLoader` handles single-turn `[user, assistant]` message pairs. It tokenizes samples using the model's chat template, masks prompt tokens in labels so loss is computed on the assistant response only, and packs multiple short samples into each sequence.

### Configuring a chat dataloader

```python
from torchtitan.hf_datasets.text_datasets import ChatDataLoader

def process_gsm8k(sample: dict) -> list[dict]:
    return [
        {"role": "user",      "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]

dataloader=ChatDataLoader.Config(
    dataset_path="openai/gsm8k",
    load_dataset_kwargs={"name": "main", "split": "train"},
    sample_processor=process_gsm8k,
    infinite=True,
),
```

---

## Multi-source interleaved dataloaders

Both text flavours support interleaving multiple sources with configurable sampling weights. At each step a source is drawn proportionally to its weight. Iteration stops when the first source is exhausted, defining an epoch boundary — re-looping and shuffling are handled per source exactly as in the single-source case.

All sources must share the same `infinite` setting.

### Interleaved pre-training

```python
from torchtitan.hf_datasets.text_datasets import (
    HFDataSource,
    InterleavedHuggingFaceTextDataLoader,
)

dataloader=InterleavedHuggingFaceTextDataLoader.Config(
    sources=[
        HFDataSource(dataset="c4",         weight=7.0, infinite=True),
        HFDataSource(dataset="wikipedia",  weight=2.0, infinite=True),
        HFDataSource(dataset="my_dataset", weight=1.0, infinite=True),
    ],
    seed=42,
),
```

### Interleaved SFT

```python
from torchtitan.hf_datasets.text_datasets import (
    ChatDataSource,
    InterleavedChatDataLoader,
)

def process_gsm8k(sample):
    return [
        {"role": "user",      "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]

def process_alpaca(sample):
    return [
        {"role": "user",      "content": sample["instruction"]},
        {"role": "assistant", "content": sample["output"]},
    ]

dataloader=InterleavedChatDataLoader.Config(
    sources=[
        ChatDataSource(
            dataset_path="openai/gsm8k",
            load_dataset_kwargs={"name": "main", "split": "train"},
            sample_processor=process_gsm8k,
            weight=3.0,
            infinite=True,
        ),
        ChatDataSource(
            dataset_path="tatsu-lab/alpaca",
            load_dataset_kwargs={"split": "train"},
            sample_processor=process_alpaca,
            weight=1.0,
            infinite=True,
        ),
    ],
    seed=42,
),
```

### Weight semantics

Weights are **sampling probabilities**, normalised internally. A weight of `3.0` alongside `1.0` means the first source is drawn three times as often on average — it does not mean the source is iterated three times per epoch. The epoch boundary is defined by whichever source exhausts first.

This makes weights easy to reason about as a **token mixture ratio**: if source A has weight 3 and source B has weight 1, roughly 75 % of training tokens will come from A and 25 % from B, regardless of the absolute dataset sizes.

### Checkpointing

Interleaved dataloaders are fully stateful. The interleaver RNG and the state of every source are saved together, so resuming from a checkpoint produces byte-identical continuations.

---

## Summary

| Use case | Dataloader |
|---|---|
| Single pre-training source | `HuggingFaceTextDataLoader` |
| Multiple pre-training sources | `InterleavedHuggingFaceTextDataLoader` |
| Single SFT source | `ChatDataLoader` |
| Multiple SFT sources | `InterleavedChatDataLoader` |
| Multimodal (vision + text) | `MultiModalDataLoader` |
