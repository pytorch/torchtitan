# Datasets

TorchTitan uses one Grain-backed data pipeline for text pretraining, SFT, and image training:

```text
source -> filter -> process -> mix/concat -> pack -> batch -> collate -> trainer
```

Dataset configs describe the graph up to packing. `GrainDataLoader` applies run-wide shuffle, repeat, data-parallel sharding, batching, prefetch, collation, and checkpointing.

## Text pretraining

The built-in C4 recipes are ready to compose:

```python
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    TextCollator,
)
from torchtitan.hf_datasets.text_datasets import DATASETS

config.dataloader = GrainDataLoader.Config(
    dataset=ConcatThenSplitPackingConfig(
        dataset=DATASETS["c4_test"],
    ),
    collator=TextCollator.Config(),
)
```

Use `DATASETS["c4"]` for streamed C4 training and `DATASETS["c4_validation"]` for streamed validation.

Text processors produce variable-length `TextSequence` objects. Packing emits `seq_len` aligned tokens. `TextCollator` keeps that input width, shifts labels left, and pads the final label with `IGNORE_INDEX`:

```text
TextSequence([10, 11, 12, 13, 14])
    -> TextCollator(seq_len=5)
    -> input  [10, 11, 12, 13, 14]
       labels [11, 12, 13, 14, IGNORE_INDEX]
```

A full `seq_len` row therefore contributes `seq_len - 1` supervised next-token targets.

```text
TextSequence[NumPy] -> Grain packing[NumPy] -> TextCollator -> TrainerBatch[PyTorch]
```

Concat-then-split packing treats tokenized documents as one stream and splits it into fixed rows. Positions reset at each document boundary so the collator masks cross-document targets.

## Local JSONL

Each non-empty line must contain one JSON object:

```json
{"text": "The first document."}
{"text": "The second document."}
```

```python
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    IndexedJsonlSource,
    SingleDatasetConfig,
    TextCollator,
)
from torchtitan.hf_datasets.text_datasets import TextProcessor


def article_text(row):
    return row["title"] + "\n\n" + row["body"]


books_and_code = SingleDatasetConfig(
    source=IndexedJsonlSource.Config(
        patterns=(
            "/datasets/books/*.jsonl",
            "/datasets/code/*.jsonl",
        ),
    ),
    processor=TextProcessor.Config(text_fn=article_text),
    post_filters=(lambda sample: sample is not None,),
)

config.dataloader = GrainDataLoader.Config(
    dataset=ConcatThenSplitPackingConfig(dataset=books_and_code),
    collator=TextCollator.Config(),
)
```

Patterns are expanded in sorted order. A missing pattern or a file selected by more than one pattern is an error.

## Hugging Face sources

Use random access for a materialized dataset:

```python
from torchtitan.components.data import HuggingFaceRandomAccessSource

source = HuggingFaceRandomAccessSource.Config(
    path="openai/gsm8k",
    name="main",
    split="train",
    revision="<immutable-revision>",
)
```

Use streaming when the corpus should not be materialized:

```python
from torchtitan.components.data import HuggingFaceStreamingSource

source = HuggingFaceStreamingSource.Config(
    path="allenai/c4",
    name="en",
    split="train",
    revision="<immutable-revision>",
)
```

TorchTitan supplies `streaming=True` or `False` from the source class. Other `datasets.load_dataset` arguments belong in `load_dataset_kwargs`.

The streaming source checkpoints the Hugging Face cursor and shards it by effective data-parallel rank. Exact continuation requires unchanged source contents and the same effective data-parallel degree.

## SFT

SFT changes the processor and packing policy, not the loader:

```python
from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
    TextCollator,
)
from torchtitan.hf_datasets.text_datasets import ChatProcessor


def gsm8k_messages(row):
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]


gsm8k = SingleDatasetConfig(
    source=HuggingFaceRandomAccessSource.Config(
        path="openai/gsm8k",
        name="main",
        split="train",
        revision="<immutable-revision>",
    ),
    processor=ChatProcessor.Config(messages_fn=gsm8k_messages),
    post_filters=(lambda sample: sample is not None,),
)

config.dataloader = GrainDataLoader.Config(
    dataset=FirstFitPackingConfig(dataset=gsm8k),
    collator=TextCollator.Config(),
)
```

`ChatProcessor` applies the tokenizer's chat template and sets prompt labels to `IGNORE_INDEX`. It currently accepts one user message followed by one assistant message. Samples longer than `seq_len` are dropped instead of truncating the response.

First-fit packing keeps each example whole while combining short examples into fixed rows.

## Mixing datasets

Keep each weight next to its dataset:

```python
from torchtitan.components.data import DatasetMixConfig, WeightedDataset

pretraining_mix = DatasetMixConfig(
    datasets=(
        WeightedDataset(dataset=books, weight=2.0),
        WeightedDataset(dataset=code, weight=1.0),
    ),
)
```

The mixed element determines what the weight counts.

Mix documents before packing when weights should count documents:

```python
packed_pretraining = ConcatThenSplitPackingConfig(dataset=pretraining_mix)
```

```text
2 book documents : 1 code document
long documents contribute more tokens
```

Pack each child before mixing when weights should count fixed-length rows and therefore physical tokens:

```python
token_ratio_mix = DatasetMixConfig(
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
each row contains seq_len tokens
```

Use concatenation to make finite map-style datasets one index space before global shuffle and sharding:

```python
from torchtitan.components.data import DatasetConcatConfig

pretraining_corpus = DatasetConcatConfig(datasets=(books, code, math))
```

With `repeat=False`, a mix stops when its first child exhausts, while concatenation consumes every child in order.

## Pretokenized data

A custom random-access source needs `__len__` and `__getitem__`. Its constructor receives the loader's run policy:

```python
from dataclasses import dataclass

import numpy as np

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    DatasetBuildContext,
    DatasetIterationPolicy,
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
)
from torchtitan.config import Configurable


class MemmapTokenSource(Configurable):
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
        return TextSequence(
            input_ids=token_ids,
            labels=token_ids.copy(),
        )


packed_tokens = ConcatThenSplitPackingConfig(
    dataset=SingleDatasetConfig(
        source=MemmapTokenSource.Config(
            tokens_path="tokens.bin",
            document_offsets_path="document_offsets.npy",
        ),
        processor=TokensToTextSequence.Config(),
    ),
)
```

`RandomAccessDataSource` is a protocol, so the custom source does not need to inherit from it.

An OLMo-style source can use token offsets to select a deterministic document prefix before composing with the same mixing and packing configs. TorchTitan does not prescribe that storage format.

If storage already contains fixed `seq_len` rows, return them as `TextSequence` objects, skip packing, and use `TextCollator`.

## Images and multimodal data

Images use the same source, dataset, loader, sharding, and checkpoint contracts. Their processors preserve modality-specific sample dictionaries, and their collators create model-specific batches.

Flux:

```python
from torchtitan.components.data import GrainDataLoader
from torchtitan.models.flux.flux_datasets import DATASETS, FluxCollator

config.dataloader = GrainDataLoader.Config(
    dataset=DATASETS["cc12m-wds"],
    collator=FluxCollator.Config(),
    streaming_shuffle_buffer_size=128,
)
```

Qwen multimodal:

```python
from torchtitan.components.data import GrainDataLoader
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    MM_DATASETS,
    MMSamplePackingConfig,
)

config.dataloader = GrainDataLoader.Config(
    dataset=MMSamplePackingConfig(
        dataset=MM_DATASETS["cc12m"],
        buffer_size=128,
    ),
    collator=MultiModalCollator.Config(build_mrope_positions=True),
    streaming_shuffle_buffer_size=128,
)
```

Custom image augmentation belongs in a `SampleProcessor`. Grain supplies its deterministic `numpy.random.Generator`, so crop and dropout decisions participate in exact resume.

## Loader policy

Configure run-wide behavior once:

```python
config.dataloader = GrainDataLoader.Config(
    dataset=packed_pretraining,
    collator=TextCollator.Config(),
    seed=42,
    shuffle=True,
    repeat=True,
    streaming_shuffle_buffer_size=1_000,
    num_workers=0,
    num_prefetch_batches=8,
)
```

`streaming_shuffle_buffer_size` is the number of raw streaming rows available to window shuffle. `num_workers` runs an eligible map-root processing graph in one multiprocessing pool per rank; it is not multiplied by the number of child datasets. `num_prefetch_batches` queues completed collated batches in one background thread.

Multiprocessing prefetch is opt-in and requires a map-root dataset. Streaming, mixing, and packing roots remain in-process.

## Distributed and checkpoint behavior

Only the effective data-parallel coordinate selects data:

```text
effective DP = data_parallel_replicate_degree * data_parallel_shard_degree

different effective-DP ranks -> disjoint source rows
TP/PP/CP peers               -> same rows for their effective-DP coordinate
```

Random-access data is globally shuffled, then stride-sharded. Hugging Face streams are sharded at the source. Packing happens after sharding and is rank-local.

With effective DP greater than one, `repeat=False` is rejected because finite ranks can exhaust at different steps and hang collectives. Use `repeat=True` with a trainer-controlled step count.

`GrainDataLoader.state_dict()` recursively records source cursors, shuffle/repeat progress, mix child state, packing buffers, batching/prefetch state, effective DP degree, and process worker count. Resume requires unchanged code, config, source contents, tokenizer, effective DP degree, and process worker count.

Custom dataset graph nodes use frozen config dataclasses with an explicit `build()`:

```python
from dataclasses import dataclass

from torchtitan.components.data import DatasetConfig


@dataclass(frozen=True, kw_only=True, slots=True)
class MyDatasetConfig:
    dataset: DatasetConfig

    def build(self, *, context, dataset_iteration_policy):
        dataset = self.dataset.build(
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
        return MyCheckpointableGrainDataset(dataset)
```

Sources, processors, collators, and loaders own configured runtime behavior and use TorchTitan `Configurable`. Dataset graph configs describe composition and return Grain datasets.
