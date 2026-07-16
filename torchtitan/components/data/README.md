# Datasets and data loading

The Grain data pipeline separates storage, sample processing, dataset composition, packing, and loading:

```text
source -> process -> filter -> combine -> shuffle -> DP shard -> repeat
       -> next-token shift -> pack -> batch -> prefetch -> Trainer
```

Most text pretraining jobs only define a source and a sample processor. Each deeper layer is replaceable when an experiment needs different behavior.

## Quick start: local JSONL pretraining

Given JSON Lines files such as:

```json
{"text": "The first document."}
{"text": "The second document."}
```

Define how one row becomes one tokenized document:

```python
import numpy as np

from torchtitan.components.data.dataset import DataRuntime, TokenSample


def text_row_to_token_sample(row: dict, runtime: DataRuntime) -> TokenSample:
    tokenizer = runtime.tokenizer
    assert tokenizer is not None

    token_ids = np.asarray(
        tokenizer.encode(row["text"], add_bos=True, add_eos=True),
        dtype=np.int64,
    )
    return TokenSample(
        token_ids=token_ids,
        loss_mask=np.ones(token_ids.shape, dtype=np.bool_),
    )
```

Use it in a config-registry function:

```python
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import PackedTokenDatasetConfig
from torchtitan.components.data.sources import JsonlSourceConfig


def llama3_my_corpus() -> Trainer.Config:
    config = llama3_debugmodel()
    config.dataloader = GrainDataLoader.Config(
        dataset_config=PackedTokenDatasetConfig(
            dataset=SingleDatasetConfig(
                source=JsonlSourceConfig(
                    patterns=("/datasets/corpus/shard-*.jsonl",),
                ),
                sample_processor=text_row_to_token_sample,
            ),
        ),
        seed=42,
        shuffle=True,
        infinite=True,
    )
    return config
```

TorchTitan supplies the tokenizer, sequence length, local batch size, and effective data-parallel rank when it builds the loader.

Put the function beside the model's other recipes, such as in `torchtitan/models/llama3/config_registry.py`, then launch it with `CONFIG=llama3_my_corpus`.

The repository includes a runnable example:

```bash
NGPU=1 MODULE=llama3 CONFIG=llama3_debugmodel_grain ./run_train.sh
```

## The reusable pieces

```python
GrainDataLoader.Config(
    dataset_config=PackedTokenDatasetConfig(
        dataset=SingleDatasetConfig(
            source=...,
            sample_processor=...,
            sample_filters=(...,),
        ),
        packing=...,
    ),
    seed=42,
    shuffle=True,
    infinite=True,
    prefetch_buffer_size=8,
)
```

`source`
: Provides deterministic integer indexing through `__len__` and `__getitem__`.

`sample_processor`
: Converts one source row to the value consumed downstream. Text packing expects `TokenSample`.

`sample_filters`
: Runs after processing. A filter may accept either `(sample)` or `(sample, runtime)`.

`dataset`
: A single source, a combination of sources, or a fully custom Grain recipe.

`packing`
: Converts tokenized documents to fixed-length sequences. The default concatenates then splits.

`GrainDataLoader`
: Builds the recipe, adds terminal prefetch, and saves/restores exact iterator state.

## Read files from another mounted path

Use regex path rewriting when manifests contain producer paths that differ from the training host:

```python
from torchtitan.components.data.sources import JsonlSourceConfig, PathRewrite


source = JsonlSourceConfig(
    patterns=("/producer/checkpoints/data/shard-*.jsonl",),
    path_rewrites=(
        PathRewrite(
            pattern=r"^/producer/checkpoints",
            replacement="/mnt/training-data",
        ),
    ),
)

# Opens /mnt/training-data/data/shard-*.jsonl
```

Patterns are expanded in their configured order, and matches within each pattern are sorted. Duplicate files and patterns that match nothing fail early.

`JsonlSourceConfig` loads all rows into memory. It is intended for small local corpora, prompt sets, tests, and examples. Large corpora should implement an indexed source.

## Filter documents

Filters run after the sample processor, so they can inspect token lengths or processor output:

```python
def has_enough_tokens(sample: TokenSample) -> bool:
    return len(sample.token_ids) >= 2


def fits_context(sample: TokenSample, runtime: DataRuntime) -> bool:
    # Shifting N tokens produces N - 1 input/label positions.
    return len(sample.token_ids) - 1 <= runtime.seq_len


dataset = SingleDatasetConfig(
    source=source,
    sample_processor=text_row_to_token_sample,
    sample_filters=(has_enough_tokens, fits_context),
)
```

Processors and filters should be deterministic functions of their inputs.

## Use pretokenized data

Pretokenized storage is a source implementation, not a special loader. This small fixed-row memmap example can be replaced by an OLMo index, an object-store reader, or another indexed format without changing mixing, packing, distributed sharding, or resume:

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MemmapTokenSourceConfig:
    path: str
    num_rows: int
    row_length: int

    def build(self) -> "MemmapTokenSource":
        return MemmapTokenSource(
            path=self.path,
            num_rows=self.num_rows,
            row_length=self.row_length,
        )

    def fingerprint(self) -> str:
        file = Path(self.path)
        return f"{file.name}:{file.stat().st_size}:{self.num_rows}:{self.row_length}"


class MemmapTokenSource:
    def __init__(self, *, path: str, num_rows: int, row_length: int) -> None:
        self._rows = np.memmap(
            path,
            mode="r",
            dtype=np.int64,
            shape=(num_rows, row_length),
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict:
        return {"token_ids": np.asarray(self._rows[index])}


def pretokenized_row_to_token_sample(row: dict) -> TokenSample:
    token_ids = row["token_ids"]
    return TokenSample(
        token_ids=token_ids,
        loss_mask=np.ones(token_ids.shape, dtype=np.bool_),
    )
```

Configure it exactly like raw text:

```python
dataset = SingleDatasetConfig(
    source=MemmapTokenSourceConfig(
        path="/datasets/tokens.bin",
        num_rows=1_000_000,
        row_length=4096,
    ),
    sample_processor=pretokenized_row_to_token_sample,
)
```

The source owns storage identity. Its `fingerprint()` must change when data selection or interpretation changes so an incompatible checkpoint is rejected.

## Combine datasets

### Deterministic weighted interleave

```python
from torchtitan.components.data.dataset import weighted_interleave


dataset = weighted_interleave(
    [
        (math_dataset, 2.0),
        (code_dataset, 1.0),
    ]
)
```

This produces a deterministic 2:1 document interleave before global shuffle. Weights describe document proportions, not token proportions. If document lengths differ substantially, use a token-budget combiner.

### Concatenate finite selections

```python
from torchtitan.components.data.dataset import concat


dataset = concat([phase_one_dataset, phase_two_dataset])
```

### Select by token budget

Token-budget selection is intentionally user-defined because each pretokenized format stores token counts and document indexes differently. A combiner receives leaf configs before global shuffle, sharding, and repetition:

```python
from dataclasses import dataclass, replace

import grain.python as grain

from torchtitan.components.data.dataset import (
    BuildOptions,
    DataRuntime,
    MultiDatasetConfig,
    SingleDatasetConfig,
)


@dataclass(frozen=True)
class TokenBudgetCombine:
    target_tokens: tuple[int, ...]

    def __call__(
        self,
        datasets: tuple[SingleDatasetConfig, ...],
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.MapDataset:
        selected = []
        for source_index, (dataset, target_tokens) in enumerate(
            zip(datasets, self.target_tokens)
        ):
            # `select` is an API on this experiment's source config. It returns
            # a new source config containing a deterministic document selection.
            selected_source = dataset.source.select(
                target_tokens=target_tokens,
                seed=options.seed + source_index,
            )
            selected_dataset = replace(dataset, source=selected_source)
            selected.append(
                selected_dataset.build_processed_dataset(runtime=runtime)
            )
        return grain.MapDataset.concatenate(selected)

    def fingerprint(self) -> str:
        return f"{type(self).__qualname__}:{self.target_tokens}"


dataset = MultiDatasetConfig(
    datasets=(books_dataset, code_dataset),
    combine_fn=TokenBudgetCombine(
        target_tokens=(800_000_000_000, 200_000_000_000),
    ),
)
```

Calling `build_processed_dataset()` preserves each leaf's processor and filters. `MultiDatasetConfig` applies global shuffle, DP sharding, and repetition once after the combination.

Configured callable objects such as `TokenBudgetCombine` must implement `fingerprint()`. Plain processors and filters should be top-level named functions so their module-qualified identity is stable and config logs remain readable.

## Choose a packing policy

### Concatenate then split

```python
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig


recipe = PackedTokenDatasetConfig(
    dataset=dataset,
    packing=ConcatThenSplitPackingConfig(),
)
```

This is the default for pretraining. It fills sequences efficiently and may split a document across packed rows. Positions restart at document boundaries. Labels are shifted before packing, so the last token in one document never predicts the first token in the next document.

### Keep examples whole

```python
from torchtitan.components.data.packing import FirstFitPackingConfig


recipe = PackedTokenDatasetConfig(
    dataset=dataset,
    packing=FirstFitPackingConfig(),
)
```

First-fit packing places whole examples into bins and pads unused space. Use it when examples should not fragment, such as SFT examples.

### Add an experiment-specific packer

A custom packing config receives a checkpointable Grain iterator of shifted features:

```python
from dataclasses import dataclass

import grain.python as grain

from torchtitan.components.data.dataset import BuildOptions, DataRuntime


@dataclass(frozen=True)
class MyPackingConfig:
    max_documents_per_sequence: int

    def build(
        self,
        parent: grain.IterDataset,
        *,
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.IterDataset:
        return MyCheckpointablePackingDataset(
            parent=parent,
            sequence_length=runtime.seq_len,
            max_documents=self.max_documents_per_sequence,
            seed=options.seed,
        )

    def fingerprint(self) -> str:
        return (
            f"{type(self).__qualname__}:"
            f"{self.max_documents_per_sequence}"
        )
```

The returned iterator must preserve exact Grain `get_state()` / `set_state()` behavior.

## SFT: what changes from pretraining?

The loader, distributed sharding, checkpointing, and trainer contract do not change. The sample processor changes two things:

1. It applies the model's chat template.
2. It sets `loss_mask=False` for prompt tokens and `True` for assistant tokens.

Use first-fit packing and filter examples that exceed the context length:

```python
import numpy as np

from torchtitan.components.data.dataset import DataRuntime, SingleDatasetConfig, TokenSample
from torchtitan.components.data.packing import (
    FirstFitPackingConfig,
    PackedTokenDatasetConfig,
)
from torchtitan.components.data.sources import JsonlSourceConfig


def chat_row_to_token_sample(row: dict, runtime: DataRuntime) -> TokenSample:
    tokenizer = runtime.tokenizer
    assert tokenizer is not None

    messages = [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]

    full_text = tokenizer.apply_chat_template(messages).rstrip("\n")
    token_ids = tokenizer.encode(full_text, add_bos=True, add_eos=False)
    if tokenizer.eos_id is not None and (
        not token_ids or token_ids[-1] != tokenizer.eos_id
    ):
        token_ids.append(tokenizer.eos_id)

    prompt_text = tokenizer.apply_chat_template(
        messages[:1],
        add_generation_prompt=True,
    )
    prompt_length = len(
        tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
    )

    loss_mask = np.ones(len(token_ids), dtype=np.bool_)
    loss_mask[:prompt_length] = False
    return TokenSample(
        token_ids=np.asarray(token_ids, dtype=np.int64),
        loss_mask=loss_mask,
    )


def sft_example_fits_context(
    sample: TokenSample,
    runtime: DataRuntime,
) -> bool:
    return len(sample.token_ids) - 1 <= runtime.seq_len


sft_recipe = PackedTokenDatasetConfig(
    dataset=SingleDatasetConfig(
        source=JsonlSourceConfig(patterns=("/datasets/sft/*.jsonl",)),
        sample_processor=chat_row_to_token_sample,
        sample_filters=(sft_example_fits_context,),
    ),
    packing=FirstFitPackingConfig(),
)
```

`PackedTokenDatasetConfig` converts every `False` label position to `IGNORE_INDEX`, which the existing cross-entropy loss already ignores.

For pretokenized SFT data, the processor is smaller:

```python
def pretokenized_sft_row_to_token_sample(row: dict) -> TokenSample:
    return TokenSample(
        token_ids=np.asarray(row["token_ids"], dtype=np.int64),
        loss_mask=np.asarray(row["loss_mask"], dtype=np.bool_),
    )
```

For a single-turn dataset, moving chat formatting and prompt-boundary detection into the processor plus choosing first-fit packing is the main implementation lift. Multi-turn SFT needs a processor that marks every assistant span, not only one prompt/response boundary. Boundary detection should tokenize chat-template prefixes rather than infer spans from character offsets, because tokenization can merge across text boundaries.

The existing [`ChatDataLoader`](../../../docs/datasets.md#instruction-tuning--sft-datasets-chat) remains available and already owns single-turn chat templating and boundary detection. A shared built-in Grain chat processor can be added when the desired single-turn and multi-turn policy is settled.

## Indexed remote data and sequential streams

An indexed object-store dataset follows the same source contract:

```python
@dataclass(frozen=True)
class S3TokenSourceConfig:
    manifest_uri: str
    manifest_version: str

    def build(self) -> "S3TokenSource":
        return S3TokenSource(self.manifest_uri)

    def fingerprint(self) -> str:
        return f"{self.manifest_uri}:{self.manifest_version}"


class S3TokenSource:
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> dict:
        ...
```

The standard `SingleDatasetConfig` then provides shuffle, DP sharding, repeat, processing, filtering, and packing.

A source that cannot support deterministic integer indexing should provide a custom `DatasetConfig` returning `grain.IterDataset`. It receives `BuildOptions`, so it must apply DP rank/world size, seed, shuffle, and repetition semantics itself:

```python
@dataclass(frozen=True)
class SequentialStreamConfig:
    uri: str
    version: str

    def build(
        self,
        *,
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.IterDataset:
        return MyCheckpointableStream(
            uri=self.uri,
            dp_rank=options.dp_rank,
            dp_world_size=options.dp_world_size,
            seed=options.seed,
            repeat=options.infinite,
        )

    def fingerprint(self) -> str:
        return f"{self.uri}:{self.version}"


recipe = PackedTokenDatasetConfig(
    dataset=SequentialStreamConfig(
        uri="s3://bucket/corpus",
        version="manifest-2026-07-16",
    )
)
```

The stream must expose exact iterator state through Grain. A Python generator with no checkpoint state is not sufficient for resumable training.

## Return trainer-ready custom batches

`GrainDataLoader` does not require text. A fully custom recipe may return the trainer's existing batch contract directly:

```python
(
    {
        "input": token_ids,            # [batch, sequence]
        "positions": positions,        # [batch, sequence]
        "pixel_values": pixel_values,  # model-specific fields are forwarded
        "grid_thw": grid_thw,
    },
    labels,
)
```

For example:

```python
from torchtitan.components.data.dataset import finish_map_dataset


@dataclass(frozen=True)
class ImageBatchDatasetConfig:
    source: ImageSourceConfig

    def build(
        self,
        *,
        runtime: DataRuntime,
        options: BuildOptions,
    ) -> grain.IterDataset:
        dataset = grain.MapDataset.source(self.source.build())
        dataset = finish_map_dataset(dataset, options=options)
        return dataset.map(image_row_to_training_batch).to_iter_dataset()

    def fingerprint(self) -> str:
        return f"{type(self).__qualname__}:{self.source.fingerprint()}"
```

Custom recipes own modality-specific collation and must return fields accepted by their model. The loader still provides prefetch and exact resume.

## Distributed data and checkpoint resume

For `SingleDatasetConfig` and `MultiDatasetConfig`, do not shard the source manually:

```text
global deterministic order
    -> shuffle(seed)
    -> rows[dp_rank::dp_world_size]
    -> repeat
```

TorchTitan derives `dp_rank` and `dp_world_size` from the batch mesh:

- Different DP ranks receive disjoint strided shards.
- TP, PP, and CP ranks that share an effective DP rank receive the same logical batch.
- Checkpoints store iterator state under `dp_rank_N`.
- Changing DP degree is rejected; dataloader resharding is not implemented.
- Changing source identity, configured processor/filter identity, combination, packing, seed, shuffle, repetition, sequence length, or local batch size is rejected by the pipeline fingerprint.

`JsonlSourceConfig` fingerprints file names and sizes, not full contents. Do not edit a corpus in place while expecting to resume from an old checkpoint. Custom large-data sources should fingerprint a versioned manifest or another stable corpus identifier. Plain functions are identified by module and qualified name, so start a new run when changing a processor's implementation without renaming it.

Fixed-step training normally uses `infinite=True`. Finite exhaustive DP runs can produce different packed batch counts on different ranks when document lengths differ, so they are not safe for a training loop with per-step collectives.

## Common configuration changes

Use a finite deterministic pass:

```python
GrainDataLoader.Config(
    dataset_config=recipe,
    shuffle=False,
    infinite=False,
)
```

Change the terminal prefetch buffer:

```python
GrainDataLoader.Config(
    dataset_config=recipe,
    prefetch_buffer_size=32,
)
```

Disable terminal prefetch while debugging:

```python
GrainDataLoader.Config(
    dataset_config=recipe,
    prefetch_buffer_size=0,
)
```

Use a top-level function when no values need configuration:

```python
def keep_english(sample: dict) -> bool:
    return sample["language"] == "en"
```

Use a frozen callable with `fingerprint()` when behavior has configured values:

```python
@dataclass(frozen=True)
class MinimumQuality:
    threshold: float

    def __call__(self, sample: dict) -> bool:
        return sample["quality"] >= self.threshold

    def fingerprint(self) -> str:
        return f"{type(self).__qualname__}:{self.threshold}"
```

Avoid closures over mutable state and random functions that do not derive randomness from the configured seed. Exact resume assumes deterministic recipe behavior.

## Existing Hugging Face loaders

The Grain path is additive. Existing configs using these loaders continue to work:

- `HuggingFaceTextDataLoader`
- `InterleavedHuggingFaceTextDataLoader`
- `ChatDataLoader`
- `InterleavedChatDataLoader`
- `MMDataLoader`

See [Hugging Face dataset loaders](../../../docs/datasets.md) for their configuration.
