# Datasets

TorchTitan uses one Grain-based data pipeline for text pretraining, SFT, and image training. Mental model:

```text
1. Define a source (e.g. jsonl):
    class: SourceConfig
    output: RandomAccessDataSource | IterDataset

2. Define a dataset (filter/process applied to the source):
    class: SingleDatasetConfig
    does:   pre-filter -> process -> post-filter
    output: MapDataset | IterDataset

3. Compose datasets (optional):
    class: e.g. FirstFitPackingConfig(dataset=DatasetMixConfig(...))
    input:  one or more child DatasetConfig values
    does:   mix, concatenate, and/or pack
    output: MapDataset | IterDataset

4. Dataloader:
    runtime: GrainDataLoader
    config:  GrainDataLoader.Config
    input:  MapDataset | IterDataset
    does:   convert to iterable if needed -> batch -> collate -> prefetch
    output: TrainerBatch

5. Trainer:
    input: TrainerBatch
    does:  model forward and backward
```

# Text Pretraining

## Local JSONL

Each non-empty line must contain one JSON object:

```json
{"title": "First", "body": "The first document."}
{"title": "Second", "body": "The second document."}
```

```python
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    IndexedJsonlSource,
    SingleDatasetConfig,
)
from torchtitan.hf_datasets.text_datasets import TextProcessor


def article_text(row):
    return row["title"] + "\n\n" + row["body"]


books_ds = SingleDatasetConfig(
    source=IndexedJsonlSource.Config(
        patterns=(
            "/datasets/books/*.jsonl",
        ),
    ),
    processor=TextProcessor.Config(text_fn=article_text),
    post_filters=(lambda sample: sample is not None,),
)

books_packed_ds = ConcatThenSplitPackingConfig(dataset=books_ds)

config.dataloader = GrainDataLoader.Config(
    dataset=books_packed_ds,
)
```

## Hugging Face sources

Use random access for a materialized dataset:

```python
from torchtitan.components.data import HuggingFaceRandomAccessSource

source = HuggingFaceRandomAccessSource.Config(
    path="openai/gsm8k",
    name="main",
    split="train",
)
```

Use streaming when the corpus should not be materialized:

```python
from torchtitan.components.data import HuggingFaceStreamingSource

source = HuggingFaceStreamingSource.Config(
    path="allenai/c4",
    name="en",
    split="train",
)
```

## Adding your own source -- Example: Pretokenized data

This example uses token offsets to expose one pretokenized document per index, then reuses the normal processing and packing configs.

```python
from dataclasses import dataclass

import numpy as np

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    DatasetBuildContext,
    DatasetIterationPolicy,
    RandomAccessDataSource,
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
)
from torchtitan.config import Configurable


class PretokenizedMemmapSource(Configurable, RandomAccessDataSource):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        tokens_path: str
        document_offsets_path: str

    def __init__(
        self,
        config: Config,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ):
        del dataset_iteration_policy
        self.tokens = np.memmap(config.tokens_path, dtype=np.uint32, mode="r")
        self.offsets = np.load(config.document_offsets_path)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        start, end = self.offsets[index : index + 2]
        return np.asarray(self.tokens[start:end], dtype=np.int64)


class TokensToTextSequence(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del config, context

    def __call__(self, token_ids, rng):
        del rng
        if len(token_ids) < 2:
            return None
        return TextSequence(
            input_ids=token_ids[:-1],
            labels=token_ids[1:],
        )


token_documents_ds = SingleDatasetConfig(
    source=PretokenizedMemmapSource.Config(
        tokens_path="tokens.bin",
        document_offsets_path="document_offsets.npy",
    ),
    processor=TokensToTextSequence.Config(),
    post_filters=(lambda sample: sample is not None,),
)

packed_tokens_ds = ConcatThenSplitPackingConfig(
    dataset=token_documents_ds,
)
```

# SFT

SFT changes the processor and packing policy, not the loader:

```python
from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.hf_datasets.text_datasets import ChatProcessor


def gsm8k_messages(row):
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]


gsm8k_ds = SingleDatasetConfig(
    source=HuggingFaceRandomAccessSource.Config(
        path="openai/gsm8k",
        name="main",
        split="train",
    ),
    processor=ChatProcessor.Config(messages_fn=gsm8k_messages),
    post_filters=(lambda sample: sample is not None,),
)

gsm8k_packed_ds = FirstFitPackingConfig(dataset=gsm8k_ds)

config.dataloader = GrainDataLoader.Config(
    dataset=gsm8k_packed_ds,
)
```

`ChatProcessor` applies the tokenizer's chat template, creates next-token input
and label pairs, and sets prompt labels to `IGNORE_INDEX`.

# Mixing datasets

Keep each weight next to its dataset:

```python
from torchtitan.components.data import DatasetMixConfig, WeightedDataset

# `books_ds` and `code_ds` are SingleDatasetConfig values.
pretraining_mix_ds = DatasetMixConfig(
    datasets=(
        WeightedDataset(dataset=books_ds, weight=0.75),
        WeightedDataset(dataset=code_ds, weight=0.25),
    ),
)
```

Examples use probabilities that sum to `1.0`, but any positive relative weights are accepted and normalized internally. `weight=0.75` next to `weight=0.25` means the first dataset is drawn three times as often.

### Mix documents before packing when weights should count documents

```python
packed_pretraining_ds = ConcatThenSplitPackingConfig(
    dataset=pretraining_mix_ds,
)
```

### Pack each child before mixing when weights should count fixed-length rows

```python
books_packed_ds = ConcatThenSplitPackingConfig(dataset=books_ds)
code_packed_ds = ConcatThenSplitPackingConfig(dataset=code_ds)

token_ratio_mix_ds = DatasetMixConfig(
    datasets=(
        WeightedDataset(dataset=books_packed_ds, weight=0.67),
        WeightedDataset(dataset=code_packed_ds, weight=0.33),
    ),
)
```

Automatic adjustment from observed document or supervised-token counts is not implemented. A custom mix can maintain per-dataset moving averages and rebalance its weights.

### Concatenate datasets

Use concatenation to treat finite datasets as one corpus. Every row appears once; dataset sizes determine their proportions. Use mixing instead for explicit weights or streaming datasets.

```python
from torchtitan.components.data import DatasetConcatConfig

pretraining_corpus_ds = DatasetConcatConfig(
    datasets=(books_ds, code_ds, math_ds),
)
```

To include every row from one finite dataset multiple times per epoch, repeat that child in the concatenation:

```python
pretraining_corpus_ds = DatasetConcatConfig(
    datasets=(books_ds,) * 3 + (code_ds, math_ds),
)
```

Every `books_ds` row now appears three times in the combined finite index space. With `shuffle=True`, TorchTitan shuffles that combined index space before DP sharding.

Use mixing instead when the desired behavior is relative source-selection frequency rather than exact finite duplication:

```python
pretraining_mix_ds = DatasetMixConfig(
    datasets=(
        WeightedDataset(dataset=books_ds, weight=0.6),
        WeightedDataset(dataset=code_ds, weight=0.2),
        WeightedDataset(dataset=math_ds, weight=0.2),
    ),
)
```

`GrainDataLoader.Config(repeat=True)` repeats the complete concatenated or mixed dataset; it does not change one child's relative contribution.

# Images and multimodal data

Images use the same source, dataset, loader, sharding, and checkpoint contracts. Their processors preserve modality-specific sample dictionaries, and their collators create model-specific batches.

Example for Qwen multimodal:

```python
from torchtitan.components.data import (
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    MMSamplePackingConfig,
    MultiModalProcessor,
    _process_cc12_wd_sample,
)

mm_processor = MultiModalProcessor.Config(
    sample_processor=_process_cc12_wd_sample,
)

mm_ds = SingleDatasetConfig(
    source=HuggingFaceStreamingSource.Config(
        path="pixparse/cc12m-wds",
        split="train",
    ),
    processor=mm_processor,
    post_filters=(lambda sample: sample is not None,),
)

packed_mm_ds = MMSamplePackingConfig(
    dataset=mm_ds,
    num_packing_bins=8,
)

config.dataloader = GrainDataLoader.Config(
    dataset=packed_mm_ds,
    collator=MultiModalCollator.Config(
        build_mrope_positions=True,
        patch_size=mm_processor.patch_size,
        temporal_patch_size=mm_processor.temporal_patch_size,
        spatial_merge_size=mm_processor.spatial_merge_size,
    ),
    streaming_shuffle_buffer_size=128,
)
```

`num_packing_bins` is the number of candidate packed rows kept open, not an input sample buffer.

Custom image augmentation belongs in a `SampleProcessor`.

# Loader policy

Configure run-wide behavior once:

```python
import grain.python as grain

config.dataloader = GrainDataLoader.Config(
    dataset=packed_pretraining_ds,
    seed=42,
    shuffle=True,
    repeat=True,
    streaming_shuffle_buffer_size=1_000,
    read_options=grain.ReadOptions(
        num_threads=16,
        prefetch_buffer_size=500,
    ),
    num_prefetch_batches=2,
)
```

### Reading indexed datasets

A random-access source becomes a `MapDataset`, which supports `dataset[index]`. It becomes an `IterDataset` when a later stage needs to consume samples sequentially:

```text
all children are MapDataset: DatasetMixConfig remains a MapDataset
any child is IterDataset:    DatasetMixConfig converts each MapDataset child
packing:                     converts its child if needed and returns IterDataset
GrainDataLoader:             converts a MapDataset if no earlier stage did
```

`read_options` controls each `MapDataset`-to-`IterDataset` conversion:

```python
grain.ReadOptions(
    num_threads=16,            # indexed samples read concurrently
    prefetch_buffer_size=500,  # samples waiting for the consumer
)
```

Each conversion has its own threads and buffer. An all-map mix converts once; a mix containing a stream converts each map child separately.

### Streaming shuffle and ready batches

`streaming_shuffle_buffer_size` is the number of raw rows retained for approximate shuffling. A larger buffer improves mixing but uses more memory.

`num_prefetch_batches` is the number of complete, collated batches allowed to wait for the trainer:

```text
trainer computes batch 10
background thread prepares batches 11 and 12
```

# Distributed and checkpoint behavior

Only the effective data-parallel coordinate selects data:

```text
effective DP = data_parallel_replicate_degree * data_parallel_shard_degree

different effective-DP ranks -> disjoint source rows
TP/PP/CP peers               -> same rows for their effective-DP coordinate
```

Data ownership is decided before batching:

```text
random-access source -> global shuffle -> contiguous balanced DP shard
Hugging Face stream -> source-level DP shard
DatasetMixConfig    -> combines children already owned by this DP rank
DatasetConcatConfig -> concatenates, globally shuffles, then DP-shards
packing             -> packs samples locally on each DP rank
GrainDataLoader     -> batches and collates that rank's samples
```

For random-access training, each rank receives a contiguous slice of the globally shuffled index space, not a contiguous region of the original corpus.

With effective DP greater than one, `repeat=False` is rejected because ranks can exhaust at different steps and hang training collectives. Use `repeat=True` and let the trainer's step count stop training.

`GrainDataLoader.state_dict()` records:

- source cursors and shuffle/repeat progress
- mix-child and packing-buffer state
- batching and prefetch state
- effective DP degree

Resume requires unchanged code, config, source contents, tokenizer, and effective DP degree.
