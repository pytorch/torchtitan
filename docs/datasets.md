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
    class: GrainDataLoader
    input:  MapDataset | IterDataset
    does:   convert to iterable if needed -> batch -> collate -> prefetch
    output: TrainerBatch
```

# Text Pretraining

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

# SFT

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


gsm8k_ds = SingleDatasetConfig(
    source=HuggingFaceRandomAccessSource.Config(
        path="openai/gsm8k",
        name="main",
        split="train",
        revision="<immutable-revision>",
    ),
    processor=ChatProcessor.Config(messages_fn=gsm8k_messages),
    post_filters=(lambda sample: sample is not None,),
)

gsm8k_packed_ds = FirstFitPackingConfig(dataset=gsm8k_ds)

config.dataloader = GrainDataLoader.Config(
    dataset=gsm8k_packed_ds,
    collator=TextCollator.Config(),
)
```

`ChatProcessor` applies the tokenizer's chat template and sets prompt labels to `IGNORE_INDEX`.

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

NOTE: When every child is random-access, Grain keeps one map root and weights select indexed draw attempts before sparse filters. If any child is streaming, random-access children are converted individually and all weights select emitted elements. This preserves one read-prefetch budget for the common all-random-access pretraining mix; a mixed graph has one budget per converted random-access child.

For an all-random-access mix with a sparse filter:

```text
configured draw attempts: sparse 50%, dense 50%
sparse accepts 1 in 10:   sparse 5%, dense 50%
accepted output share:    sparse 5 / (5 + 50) = 9.1%
```

The mixed element determines what each draw counts.

Mix documents before packing when weights should count documents:

```python
packed_pretraining_ds = ConcatThenSplitPackingConfig(
    dataset=pretraining_mix_ds,
)
```

```text
2 book documents : 1 code document
long documents contribute more tokens
```

Pack each child before mixing when weights should count fixed-length rows and therefore physical tokens:

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

```text
2 book rows : 1 code row
each row contains seq_len tokens
```

Automatic token-weighted mixing from source token counts is not implemented.

Use concatenation to make finite map-style datasets one index space before global shuffle and sharding:

```python
from torchtitan.components.data import DatasetConcatConfig

pretraining_corpus_ds = DatasetConcatConfig(
    datasets=(books_ds, code_ds, math_ds),
)
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


token_documents_ds = SingleDatasetConfig(
    source=MemmapTokenSource.Config(
        tokens_path="tokens.bin",
        document_offsets_path="document_offsets.npy",
    ),
    processor=TokensToTextSequence.Config(),
)

packed_tokens_ds = ConcatThenSplitPackingConfig(
    dataset=token_documents_ds,
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

flux_ds = DATASETS["cc12m-test"]

config.dataloader = GrainDataLoader.Config(
    dataset=flux_ds,
    collator=FluxCollator.Config(),
    streaming_shuffle_buffer_size=128,
)
```

Use `DATASETS["cc12m-wds"]` for the full streamed corpus.

Qwen multimodal:

```python
from torchtitan.components.data import GrainDataLoader
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    MM_DATASETS,
    MMSamplePackingConfig,
    MultiModalProcessor,
)

mm_ds = MM_DATASETS["cc12m"]
processor = mm_ds.processor
assert isinstance(processor, MultiModalProcessor.Config)

packed_mm_ds = MMSamplePackingConfig(
    dataset=mm_ds,
    buffer_size=128,
)

config.dataloader = GrainDataLoader.Config(
    dataset=packed_mm_ds,
    collator=MultiModalCollator.Config(
        build_mrope_positions=True,
        patch_size=processor.patch_size,
        temporal_patch_size=processor.temporal_patch_size,
        spatial_merge_size=processor.spatial_merge_size,
    ),
    streaming_shuffle_buffer_size=128,
)
```

Custom image augmentation belongs in a `SampleProcessor`. Grain supplies its deterministic `numpy.random.Generator`, so crop and dropout decisions participate in exact resume.

## Loader policy

Configure run-wide behavior once:

```python
import grain.python as grain

config.dataloader = GrainDataLoader.Config(
    dataset=packed_pretraining_ds,
    collator=TextCollator.Config(),
    seed=42,
    shuffle=True,
    repeat=True,
    streaming_shuffle_buffer_size=1_000,
    read_options=grain.ReadOptions(
        num_threads=16,
        prefetch_buffer_size=500,
    ),
    num_workers=0,
    num_prefetch_batches=2,
)
```

`streaming_shuffle_buffer_size` is the number of raw streaming rows available to window shuffle.

`read_options` controls concurrent indexed reads whenever a `MapDataset` becomes an `IterDataset`. An all-map mix has one conversion; a mixed map/stream graph converts each map child separately. Native streams do not use this option. Set `num_threads=0` for in-memory Python data.

`num_workers` adds one multiprocessing pool for an eligible map-root graph; it is not multiplied by child datasets.

`num_prefetch_batches` queues completed collated batches in one background thread.

Multiprocessing prefetch is opt-in and requires a map-root dataset. Streaming, mixing, and packing roots remain in-process.

Omitting `collator` uses `DefaultCollator`, which stacks rows that are already `(model_inputs, labels)` pairs.

## Distributed and checkpoint behavior

Only the effective data-parallel coordinate selects data:

```text
effective DP = data_parallel_replicate_degree * data_parallel_shard_degree

different effective-DP ranks -> disjoint source rows
TP/PP/CP peers               -> same rows for their effective-DP coordinate
```

Random-access data is globally shuffled, then stride-sharded. Hugging Face streams are sharded at the source. Packing happens after sharding and is rank-local.

Repeated map datasets receive a fresh Grain shuffle permutation each epoch. Shuffled Hugging Face streams advance their shard order on each repeat.

With effective DP greater than one, `repeat=False` is rejected because finite ranks can exhaust at different steps and hang collectives. Use `repeat=True` with a trainer-controlled step count.

`GrainDataLoader.state_dict()` records:

- source cursors and shuffle/repeat progress
- mix-child and packing-buffer state
- batching and prefetch state
- effective DP degree and process-worker count

Resume requires unchanged code, config, source contents, tokenizer, effective DP degree, and process-worker count.

Custom dataset graph nodes use frozen config dataclasses with an explicit `build()`:

```python
from dataclasses import dataclass

from torchtitan.components.data import DatasetConfig


@dataclass(frozen=True, kw_only=True, slots=True)
class MyDatasetConfig:
    dataset: DatasetConfig

    def build(self, *, context, dataset_iteration_policy):
        child_ds = self.dataset.build(
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
        return MyCheckpointableGrainDataset(child_ds)
```

Sources, processors, collators, and loaders own configured runtime behavior and use TorchTitan `Configurable`. Dataset graph configs describe composition and return Grain datasets.
