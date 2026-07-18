# Grain data pipeline

TorchTitan uses one data path for pretraining, SFT, and image training:

```text
source -> process -> mix/concat -> pack -> batch -> collate -> trainer
```

`GrainDataLoader` owns batching, prefetch, distributed run policy, and checkpoint state. Dataset configs describe the graph below it.

## Built-in text pretraining

Use the built-in C4 recipes directly:

```python
from torchtitan.hf_datasets.text_datasets import c4_text_dataloader

config.dataloader = c4_text_dataloader("c4_test")        # local test asset
config.dataloader = c4_text_dataloader("c4")             # streamed train split
config.dataloader = c4_text_dataloader("c4_validation")  # streamed validation split
```

Pretraining uses concat-then-split packing:

```text
tokenized documents: [1800 tokens] [5000 tokens] [900 tokens] ...
concatenated stream: [----------------------------------------- ...]
seq_len=4096 rows:   [       4096       ] [       4096       ] ...
trainer batch:       [local_batch_size, 4096]
```

Each document is shifted before packing, so a label never predicts the first token of the next document. Positions reset at every document boundary.

## Local JSONL

This complete recipe reads JSON objects from local files:

```python
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    IndexedJsonlSource,
    SingleDatasetConfig,
    TextCollator,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextProcessor


def row_to_text(row):
    return row["text"]


config.dataloader = GrainDataLoader.Config(
    dataset=ConcatThenSplitPackingConfig(
        dataset=SingleDatasetConfig(
            source=IndexedJsonlSource.Config(
                patterns=(
                    "/datasets/books/*.jsonl",
                    "/datasets/code/*.jsonl",
                ),
            ),
            process=HuggingFaceTextProcessor.Config(
                text_processor=row_to_text,
            ),
        ),
    ),
    collator=TextCollator.Config(),
)
```

Each non-empty line is one JSON object:

```json
{"text": "The first document."}
{"text": "The second document."}
```

Patterns are expanded in sorted order. Missing patterns and duplicate resolved files are errors.

## Hugging Face sources

A source receives a callable that loads its path. Use random access for a materialized dataset:

```python
from functools import partial

from datasets import load_dataset
from torchtitan.components.data import HuggingFaceRandomAccessSource

source = HuggingFaceRandomAccessSource.Config(
    path="openai/gsm8k",
    loader=partial(
        load_dataset,
        name="main",
        split="train",
        revision="<immutable-revision>",
    ),
)
```

Use streaming when the corpus should not be materialized:

```python
from torchtitan.components.data import HuggingFaceStreamingSource

source = HuggingFaceStreamingSource.Config(
    path="allenai/c4",
    loader=partial(
        load_dataset,
        name="en",
        split="train",
        streaming=True,
        revision="<immutable-revision>",
    ),
)
```

Hugging Face owns Hub access, caching, decoding, and the streaming cursor. Grain owns processing, shuffle, repeat, mixing, packing, batching, and recursive iterator state.

## Mixing datasets

Weights stay next to their datasets:

```python
from torchtitan.components.data import DatasetMixConfig, WeightedDataset

documents = DatasetMixConfig(
    datasets=(
        WeightedDataset(dataset=books, weight=2.0),
        WeightedDataset(dataset=code, weight=1.0),
    ),
)
```

Where the mix sits determines what the weights count.

Mix documents, then pack:

```python
packed_rows = ConcatThenSplitPackingConfig(dataset=documents)
```

```text
2 book documents : 1 code document
longer documents contribute more tokens
```

Pack each dataset, then mix:

```python
packed_rows = DatasetMixConfig(
    datasets=(
        WeightedDataset(
            dataset=ConcatThenSplitPackingConfig(dataset=books),
            weight=2.0,
        ),
        WeightedDataset(
            dataset=ConcatThenSplitPackingConfig(dataset=code),
            weight=1.0,
        ),
    ),
)
```

```text
2 book rows : 1 code row
each row has seq_len tokens
token ratio = 2:1
```

Use concatenation when finite map-style datasets should form one index space before shuffling and DP sharding:

```python
from torchtitan.components.data import DatasetConcatConfig

documents = DatasetConcatConfig(datasets=(books, code, math))
```

## SFT

SFT uses the same source, loader, batching, distributed, and checkpoint path. It changes row processing and uses whole-example first-fit packing.

```python
from torchtitan.hf_datasets.text_datasets import chat_dataloader


def question_answer_to_messages(row):
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]


config.dataloader = chat_dataloader(
    dataset_path="openai/gsm8k",
    load_dataset_kwargs={
        "name": "main",
        "split": "train",
        "revision": "<immutable-revision>",
    },
    sample_processor=question_answer_to_messages,
)
```

```text
raw row
  -> [user, assistant]
  -> chat template
  -> token IDs + assistant-only loss mask
  -> whole examples packed into fixed rows
  -> [local_batch_size, seq_len]
```

`ChatProcessor` currently accepts one user message followed by one assistant message. Samples longer than `seq_len` are dropped rather than training on truncated responses.

## Pretokenized data

Pretokenized corpora use the same pipeline. A random-access source only needs `__len__` and `__getitem__`:

```python
from dataclasses import dataclass

import numpy as np

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    SingleDatasetConfig,
    TokenSequence,
)
from torchtitan.config import Configurable


class MemmapTokenSource(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        tokens_path: str
        document_offsets_path: str

    def __init__(self, config: Config, **_):
        self.tokens = np.memmap(config.tokens_path, dtype=np.uint32, mode="r")
        self.offsets = np.load(config.document_offsets_path)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        start, end = self.offsets[index : index + 2]
        return np.asarray(self.tokens[start:end], dtype=np.int64)


def tokens_to_sequence(token_ids):
    return TokenSequence(
        token_ids=token_ids,
        loss_mask=np.ones(token_ids.shape, dtype=np.bool_),
    )


packed_rows = ConcatThenSplitPackingConfig(
    dataset=SingleDatasetConfig(
        source=MemmapTokenSource.Config(
            tokens_path="tokens.bin",
            document_offsets_path="document_offsets.npy",
        ),
        process=tokens_to_sequence,
    ),
)
```

An OLMo-style token-budget mix can select a deterministic document prefix inside a custom source by using its token offsets. The selected source then composes with the same `DatasetMixConfig`, packing, loader, and checkpoint path.

If storage already contains fixed trainer rows, skip packing:

```python
config.dataloader = GrainDataLoader.Config(
    dataset=SingleDatasetConfig(
        source=MyFixedTrainingRows.Config(...),
    ),
    collator=TextCollator.Config(),
)
```

Each source item must already be `(inputs, labels)`, where `inputs` contains `input` and `positions` tensors of length `seq_len`.

## Flux and Qwen multimodal

Images use the same loader and source contracts. Their processors and collators produce modality-specific batches.

Flux:

```python
from torchtitan.models.flux.flux_datasets import flux_dataloader

config.dataloader = flux_dataloader(
    dataset="cc12m-wds",
    prompt_dropout_prob=0.447,
    img_size=256,
)
```

Qwen:

```python
from torchtitan.hf_datasets.multimodal.mm_datasets import multimodal_dataloader

config.dataloader = multimodal_dataloader(
    dataset="cc12m-wds",
    packing_buffer_size=128,
    max_images_per_batch=128,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    min_pixels=65_536,
    max_pixels=16_777_216,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
    build_mrope_positions=True,
)
```

Custom image augmentation belongs in a `SampleProcessor`. Grain passes a deterministic `numpy.random.Generator` to configured processors, so crop and dropout decisions resume exactly.

## Run policy

Set run-wide behavior once on the loader:

```python
import grain.python as grain

config.dataloader = GrainDataLoader.Config(
    dataset=packed_rows,
    collator=TextCollator.Config(),
    seed=42,
    shuffle=True,
    repeat=True,
    read_options=grain.ReadOptions(
        num_threads=16,
        prefetch_buffer_size=32,
    ),
    batch_prefetch_buffer_size=8,
)
```

`shuffle`, `repeat`, and prefetch do not need to be repeated on every leaf dataset.

## Distributed behavior

Only the effective data-parallel coordinate selects data:

```text
effective DP = data_parallel_replicate_degree * data_parallel_shard_degree

different effective-DP ranks -> disjoint source rows
TP/PP/CP peers              -> the same rows for their effective-DP coordinate
```

Random-access data is shuffled globally, then stride-sharded. Hugging Face streams are split by effective DP rank at the source. Packing currently happens after that split and is rank-local.

Finite filtered datasets with DP greater than one are rejected because ranks can produce different row counts and hang. Repeated training datasets are supported.

## Checkpointing

`GrainDataLoader.state_dict()` records:

```text
pipeline identity:
  dataset and collator config types + public fields
  seed, shuffle, repeat
  seq_len, local_batch_size
  tokenizer type

iterator state:
  source cursor/index
  repeat and shuffle state
  mix schedule and child cursors
  packing buffers
  batch and prefetch state
```

Resume is exact when code, config, source contents, tokenizer, and effective DP degree stay the same. Changing the composed pipeline or effective DP degree is rejected.

Packing is rank-local, so changing the effective DP topology is not supported. The packing implementations carry a TODO for a future global pack plan that could make topology-independent resume possible.

Use top-level functions or configured `SampleProcessor` classes in checkpointed recipes. Capturing closures do not have a stable pipeline identity and are rejected.

Custom dataset graph nodes are frozen dataclasses with an explicit `build()`:

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class MyDatasetConfig:
    dataset: DatasetConfig

    def build(self, *, runtime: DataRuntime, options: BuildOptions):
        dataset = self.dataset.build(runtime=runtime, options=options)
        return MyCheckpointableGrainDataset(dataset)
```

Sources, processors, collators, and loaders that own runtime behavior use TorchTitan `Configurable`. Dataset graph configs describe composition and return a Grain dataset.
