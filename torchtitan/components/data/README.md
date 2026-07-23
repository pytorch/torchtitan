# Grain data pipeline

TorchTitan uses one data path for pretraining, SFT, and image training:

```text
source -> pre-filter -> process -> post-filter -> mix/concat -> pack -> batch -> collate -> trainer
```

`GrainDataLoader` owns batching, prefetch, distributed run policy, and checkpoint state. Dataset configs describe the graph below it.

## Built-in text pretraining

Compose the built-in C4 dataset directly:

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

Use `DATASETS["c4"]` or `DATASETS["c4_validation"]` for the streamed splits.

Pretraining uses concat-then-split packing:

```text
tokenized documents: [1800 tokens] [5000 tokens] [900 tokens] ...
concatenated stream: [----------------------------------------- ...]
seq_len=4096 rows:   [       4096       ] [       4096       ] ...
trainer batch:       [local_batch_size, 4096]
```

Each document is shifted before packing, so a label never predicts the first token of the next document. Positions reset at every document boundary.

```text
token_ids=[0,1,2,3,4,5], seq_len=4

inputs  [0,1,2,3]  labels [1,2,3,4]
inputs  [4,0,0,0]  labels [5,-100,-100,-100]
```

## Local JSONL

This complete recipe reads JSON objects from local files:

```python
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    IndexedJsonlSource,
    SingleDatasetConfig,
)
from torchtitan.hf_datasets.text_datasets import TextProcessor


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
            processor=TextProcessor.Config(
                text_fn=row_to_text,
            ),
        ),
    ),
)
```

Each non-empty line is one JSON object:

```json
{"text": "The first document."}
{"text": "The second document."}
```

Patterns are expanded in sorted order. Missing patterns and duplicate resolved files are errors.

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

TorchTitan sets `streaming=True` or `False` from the source class. Hugging Face owns Hub access, caching, decoding, and the streaming cursor. Grain owns processing, shuffle, repeat, mixing, packing, batching, and recursive iterator state.

Exact streaming cursor continuation requires unchanged source contents and the raw Hugging Face cursor path. Hugging Face-side shuffle buffers and batched maps are outside this guarantee.

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

`repeat` controls whether either graph is finite:

```text
concat + repeat=False -> consume every child once, in order
concat + repeat=True  -> repeat the complete concatenation
mix + repeat=False    -> stop when the first child exhausts
mix + repeat=True     -> mix repeated children indefinitely
```

## SFT

SFT uses the same source, loader, batching, distributed, and checkpoint path. It changes row processing and uses whole-example first-fit packing.

```python
from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.hf_datasets.text_datasets import ChatProcessor


def question_answer_to_messages(row):
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]


config.dataloader = GrainDataLoader.Config(
    dataset=FirstFitPackingConfig(
        dataset=SingleDatasetConfig(
            source=HuggingFaceRandomAccessSource.Config(
                path="openai/gsm8k",
                name="main",
                split="train",
                revision="<immutable-revision>",
            ),
            processor=ChatProcessor.Config(
                messages_fn=question_answer_to_messages,
            ),
            post_filters=(lambda sample: sample is not None,),
        ),
    ),
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
    DatasetBuildContext,
    SampleProcessor,
    SingleDatasetConfig,
    TokenSequence,
)
from torchtitan.config import Configurable


class MemmapTokenSource(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        tokens_path: str
        document_offsets_path: str

    def __init__(self, config: Config, *, dp_rank: int, dp_world_size: int):
        del dp_rank, dp_world_size
        self.tokens = np.memmap(config.tokens_path, dtype=np.uint32, mode="r")
        self.offsets = np.load(config.document_offsets_path)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        start, end = self.offsets[index : index + 2]
        return np.asarray(self.tokens[start:end], dtype=np.int64)


class TokensToSequence(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del config, context

    def __call__(self, token_ids, rng):
        del rng
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
        processor=TokensToSequence.Config(),
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
)
```

Each source item must already be `(inputs, labels)`, where `inputs` contains `input` and `positions` tensors of length `seq_len`.

## Flux and Qwen multimodal

Images use the same loader and source contracts. Their processors and collators produce modality-specific batches.

Flux:

```python
from torchtitan.components.data import GrainDataLoader
from torchtitan.models.flux.flux_datasets import DATASETS

config.dataloader = GrainDataLoader.Config(
    dataset=DATASETS["cc12m-wds"],
    streaming_shuffle_window_size=128,
)
```

Qwen:

```python
from torchtitan.components.data import GrainDataLoader
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.hf_datasets.multimodal.mm_datasets import (
    MM_DATASETS,
    MMSamplePackingConfig,
)

config.dataloader = GrainDataLoader.Config(
    dataset=MM_DATASETS["cc12m"],
    collator=MultiModalCollator.Config(
        build_mrope_positions=True,
    ),
    streaming_shuffle_window_size=128,
)
```

Multimodal packing is opt-in:

```python
config.dataloader.dataset = MMSamplePackingConfig(
    dataset=MM_DATASETS["cc12m"],
    buffer_size=128,
)
```

Custom image augmentation belongs in a `SampleProcessor`. Grain passes a deterministic `numpy.random.Generator` to configured processors, so crop and dropout decisions resume exactly.

## Run policy

Set run-wide behavior once on the loader:

```python
import grain.python as grain

config.dataloader = GrainDataLoader.Config(
    dataset=packed_rows,
    seed=42,
    shuffle=True,
    repeat=True,
    streaming_shuffle_window_size=1_000,
    read_options=grain.ReadOptions(
        num_threads=16,
        prefetch_buffer_size=32,
    ),
    batch_prefetch_buffer_size=8,
    process_workers=0,
    process_prefetch_buffer_size=1,
)
```

The buffers have different jobs:

```text
streaming_shuffle_window_size -> raw streaming rows available to shuffle
read_options                  -> parallel map-source reads and their row buffer
batch_prefetch_buffer_size    -> completed trainer batches ready for training
process_prefetch_buffer_size  -> rows buffered by each process worker
```

`shuffle`, `repeat`, and these buffering controls are configured once on the loader, not on every leaf dataset.

Process prefetch is opt-in and currently requires a map-root dataset. Streaming, mixing, and packing remain single-process. Each worker owns its own `ReadOptions.num_threads` pool, so map-stage concurrency scales as `process_workers * max(num_threads, 1)` per rank. Custom map sources and processors must be picklable and reopen process-local resources. Worker count is checkpoint topology, so resume requires the same `process_workers` value.

## Distributed behavior

Only the effective data-parallel coordinate selects data:

```text
effective DP = data_parallel_replicate_degree * data_parallel_shard_degree

different effective-DP ranks -> disjoint source rows
TP/PP/CP peers              -> the same rows for their effective-DP coordinate
```

Random-access data is shuffled globally, then stride-sharded. Hugging Face streams are split by effective DP rank at the source. Packing currently happens after that split and is rank-local.

With effective DP greater than one, `repeat=False` is rejected because finite ranks can exhaust at different steps and hang collectives. Use `repeat=True` and a trainer-controlled step count.

## Checkpointing

`GrainDataLoader.state_dict()` records:

```text
version, effective DP degree, and process worker count
per-rank iterator state:
  source cursor/index
  repeat and shuffle state
  mix schedule and child cursors
  packing buffers
  batch and prefetch state
```

Resume is exact when code, config, source contents, tokenizer, effective DP degree, and process worker count stay the same. Changing either topology value is rejected. The loader does not store a duplicate fingerprint of the pipeline config.

Packing is rank-local, so changing the effective DP topology is not supported. The packing implementations carry a TODO for a future global pack plan that could make topology-independent resume possible.

Custom dataset graph nodes are frozen dataclasses with an explicit `build()`:

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class MyDatasetConfig:
    dataset: DatasetConfig

    def build(
        self,
        *,
        context: DatasetBuildContext,
        iteration: DatasetIterationPolicy,
    ):
        dataset = self.dataset.build(context=context, iteration=iteration)
        return MyCheckpointableGrainDataset(dataset)
```

Sources, processors, collators, and loaders that own runtime behavior use TorchTitan `Configurable`. Dataset graph configs describe composition and return a Grain dataset.
