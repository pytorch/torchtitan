# Datasets

TorchTitan uses one Grain-backed data pipeline for text pretraining, SFT, and image training:

```text
source -> process -> mix/concat -> pack -> batch -> collate -> trainer
```

Start with the [Grain data pipeline README](../torchtitan/components/data/README.md). It contains complete examples for:

- built-in C4 and local JSONL pretraining
- Hugging Face random-access and streaming sources
- weighted dataset mixes and concatenation
- SFT with assistant-only labels
- pretokenized and OLMo-style custom sources
- Flux and Qwen multimodal datasets
- distributed sharding and exact dataloader checkpoint resume

The shortest built-in recipe is:

```python
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
)
from torchtitan.hf_datasets.text_datasets import DATASETS

config.dataloader = GrainDataLoader.Config(
    dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
)
```
