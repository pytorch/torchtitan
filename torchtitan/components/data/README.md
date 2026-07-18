# Grain data pipeline

TorchTitan has one data path:

```text
source -> process rows -> mix/concat -> pack -> batch -> collate -> trainer
```

`GrainDataLoader` owns run policy, batching, prefetch, distributed sharding, and checkpoint state. Dataset recipes describe the graph below it.

## Built-in text pretraining

The built-in C4 recipes are the shortest starting point:

```python
from torchtitan.hf_datasets.text_datasets import c4_text_dataloader

config.dataloader = c4_text_dataloader("c4_test")        # local test asset
config.dataloader = c4_text_dataloader("c4")             # streamed train split
config.dataloader = c4_text_dataloader("c4_validation")  # streamed validation split
```

The full `c4` recipe is:

```python
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
    TextCollator,
    TextToTokenSequence,
)

config.dataloader = GrainDataLoader.Config(
    dataset=ConcatThenSplitPackingConfig(
        dataset=SingleDatasetConfig(
            source=HuggingFaceStreamingSource.Config(
                path="allenai/c4",
                load_dataset_kwargs={
                    "name": "en",
                    "split": "train",
                    "revision": "<immutable-revision>",
                },
            ),
            process=TextToTokenSequence.Config(text_field="text"),
        ),
    ),
    collator=TextCollator.Config(),
    seed=42,
    shuffle=True,
    repeat=True,
)
```

For `seq_len=4096`, `ConcatThenSplitPackingConfig` produces exactly one 4096-token training row at a time:

```text
tokenized documents: [1800] [5000] [900] ...
concatenated stream: [------------------------- ...]
training rows:       [4096] [4096] [4096] ...
batch:               [local_batch_size, 4096]
```

## Local JSONL

`IndexedJsonlSource` builds compact byte offsets and reads rows on demand:

```python
dataset = SingleDatasetConfig(
    source=IndexedJsonlSource.Config(
        patterns=(
            "/datasets/books/*.jsonl",
            "/datasets/code/*.jsonl",
        ),
    ),
    process=TextToTokenSequence.Config(text_field="text"),
)
```

Each non-empty line must be one JSON object:

```json
{"text": "The first document."}
{"text": "The second document."}
```

Patterns are expanded in sorted order. Missing patterns and duplicate resolved files are errors.

## Hugging Face sources

Use random access for materialized datasets:

```python
source = HuggingFaceRandomAccessSource.Config(
    path="openai/gsm8k",
    load_dataset_kwargs={
        "name": "main",
        "split": "train",
        "revision": "<immutable-revision>",
    },
)
```

Use streaming when the corpus should not be materialized:

```python
source = HuggingFaceStreamingSource.Config(
    path="my-org/large-corpus",
    load_dataset_kwargs={
        "split": "train",
        "revision": "<immutable-revision>",
    },
)
```

Hugging Face owns Hub access, caching, Arrow/WebDataset decoding, and its streaming cursor. Grain owns processing, shuffle, repeat, mixing, packing, batching, and recursive checkpoint state.

## Weighted mixes

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

Choose where mixing happens based on what the weight means.

Pack first when weights describe fixed token rows:

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
therefore the token ratio is 2:1
```

Mix first when weights describe documents:

```python
packed_rows = ConcatThenSplitPackingConfig(dataset=documents)
```

```text
2 book documents : 1 code document
document lengths may differ
therefore the token ratio need not be 2:1
```

`DatasetConcatConfig` is different: it appends finite map-style datasets into one global index space, then applies one shuffle and DP shard.

```python
from torchtitan.components.data import DatasetConcatConfig

documents = DatasetConcatConfig(datasets=(books, code, math))
```

## SFT

SFT uses the same source, loader, batching, and checkpoint path. It changes row processing and usually uses whole-example first-fit packing.

```python
from torchtitan.components.data import (
    ChatToTokenSequence,
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
    TextCollator,
)


def question_answer_to_messages(sample):
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


config.dataloader = GrainDataLoader.Config(
    dataset=FirstFitPackingConfig(
        dataset=SingleDatasetConfig(
            source=HuggingFaceRandomAccessSource.Config(
                path="json",
                load_dataset_kwargs={
                    "data_files": "data/sft.json",
                    "split": "train",
                },
            ),
            process=ChatToTokenSequence.Config(
                sample_to_messages=question_answer_to_messages,
                train_on_assistant_only=True,
            ),
        ),
    ),
    collator=TextCollator.Config(),
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

`ChatToTokenSequence` currently accepts one user message followed by one assistant message. Samples longer than `seq_len` are dropped by first-fit packing. Set `train_on_assistant_only=False` to train on every non-padding token.

Use a top-level function or a configured `SampleProcessor`; capturing closures cannot provide stable checkpoint identity.

## Pretokenized data

Pretokenized corpora use the same recipes. A source only needs integer random access:

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
        path: str
        document_offsets_path: str

    def __init__(self, config: Config, **_):
        self.tokens = np.memmap(config.path, dtype=np.uint32, mode="r")
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
            path="tokens.bin",
            document_offsets_path="document_offsets.npy",
        ),
        process=tokens_to_sequence,
    ),
)
```

An OLMo-style token-budget selector belongs inside a custom source config: select the deterministic document range once while building the source, then reuse the standard mix, packing, loader, and checkpoint layers.

Already-fixed trainer rows skip packing:

```python
config.dataloader = GrainDataLoader.Config(
    dataset=SingleDatasetConfig(
        source=MyFixedTokenRows.Config(...),
        process=fixed_tokens_to_training_row,
    ),
    collator=TextCollator.Config(),
)
```

## Flux and Qwen multimodal

Flux changes the processor and collator, not the loader:

```python
from torchtitan.components.data import GrainDataLoader
from torchtitan.models.flux.flux_datasets import (
    FluxCollator,
    flux_dataset_config,
)

config.dataloader = GrainDataLoader.Config(
    dataset=flux_dataset_config(
        "cc12m-wds",
        image_size=256,
        prompt_dropout_prob=0.447,
    ),
    collator=FluxCollator.Config(),
)
```

Qwen uses a modality-specific packing recipe because image and patch admission limits apply to a whole local batch:

```python
from torchtitan.components.data import (
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.hf_datasets.multimodal import (
    QwenCC12MProcessor,
    QwenMultimodalCollator,
    QwenMultimodalPackingConfig,
)

config.dataloader = GrainDataLoader.Config(
    dataset=QwenMultimodalPackingConfig(
        dataset=SingleDatasetConfig(
            source=HuggingFaceStreamingSource.Config(
                path="pixparse/cc12m-wds",
                load_dataset_kwargs={"split": "train"},
            ),
            process=QwenCC12MProcessor.Config(),
        ),
        max_images_per_batch=128,
        max_patches_per_batch=8_388_608,
    ),
    collator=QwenMultimodalCollator.Config(
        build_mrope_positions=True,
    ),
)
```

Custom image augmentation should be a `SampleProcessor`. Grain passes a deterministic `numpy.random.Generator` to each call, so crop and dropout decisions restore exactly.

## Run policy

Configure run-wide behavior once:

```python
config.dataloader = GrainDataLoader.Config(
    dataset=packed_rows,
    collator=TextCollator.Config(),
    seed=42,
    shuffle=True,
    repeat=True,
    batch_prefetch_buffer_size=8,
)
```

`shuffle`, `repeat`, and batch prefetch do not belong on leaf datasets. Grain process prefetch is intentionally not exposed until it works consistently across composed map, stream, mix, and packing graphs.

## Distributed behavior

Only the effective data-parallel coordinate selects data:

```text
effective DP = data_parallel_replicate_degree * data_parallel_shard_degree

different effective-DP ranks -> disjoint source rows
same effective-DP rank      -> identical rows on TP/PP/CP peers
```

Random-access data is shuffled globally, then stride-sharded. Hugging Face streams are split by effective DP rank at the source. Packing happens after that split and is rank-local.

Finite filtered streams with DP greater than one are rejected because ranks can exhaust at different steps. Repeated training streams are supported.

## Checkpointing

The loader checkpoint contains:

```text
pipeline identity:
  dataset and collator config types + public fields
  seed, shuffle, repeat
  seq_len, local_batch_size
  tokenizer type

mutable iterator state:
  source cursor/index
  repeat and shuffle state
  mix schedule and child cursors
  packing buffers and lookahead
  partial batch and terminal prefetch alignment
```

Resume is exact for the same code, config, immutable source revision, tokenizer, and effective DP degree. Changing the pipeline or effective DP degree is rejected.

Custom recipes implement one visible method:

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class MyDatasetConfig:
    dataset: DatasetConfig

    def build(self, *, runtime: DataRuntime, options: BuildOptions):
        dataset = self.dataset.build(runtime=runtime, options=options)
        return MyCheckpointableGrainDataset(dataset)
```

Runtime owners such as sources, processors, collators, and the loader use TorchTitan `Configurable`. Dataset graph recipes are frozen dataclasses with an explicit `build() -> Grain dataset`.
