# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the composed Grain data pipeline."""

import json
from dataclasses import dataclass, replace
from typing import Any
from unittest import mock

import datasets
import grain.python as grain
import numpy as np
import pytest
import torch

from torchtitan.components.data.collators import Collator, TextCollator, TrainerBatch
from torchtitan.components.data.dataset import (
    DatasetConcatConfig,
    DatasetMixConfig,
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
    WeightedDataset,
)
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
)
from torchtitan.components.data.sources import (
    _HuggingFaceCursorIterator,
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
    IndexedJsonlSource,
)
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets.text_datasets import ChatProcessor, TextProcessor


class FakeTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, add_bos=False, add_eos=False):
        tokens = [ord(char) % 250 + 10 for char in text]
        return [self.bos_id] * add_bos + tokens + [self.eos_id] * add_eos

    def apply_chat_template(self, messages, add_generation_prompt=False):
        text = " ".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        if add_generation_prompt:
            text += " assistant:"
        return text


CONTEXT = DatasetBuildContext(
    tokenizer=FakeTokenizer(),
    seq_len=9,
    local_batch_size=2,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)


def dataset_iteration_policy(**overrides):
    values = {
        "seed": 42,
        "shuffle": False,
        "repeat": False,
        "dp_rank": 0,
        "dp_world_size": 1,
        "streaming_shuffle_buffer_size": 4,
    }
    return DatasetIterationPolicy(**(values | overrides))


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows))


class RowToTokens(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del config, context

    def __call__(self, sample, rng):
        del rng
        tokens = np.asarray(sample["tokens"], dtype=np.int64)
        if len(tokens) < 2:
            return None
        return TextSequence(
            input_ids=tokens[:-1],
            labels=tokens[1:],
        )


class PairCollator(Collator):
    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del config, context

    def __call__(self, rows) -> TrainerBatch:
        inputs, labels = zip(*rows)
        return {
            key: torch.stack([row[key] for row in inputs]) for key in inputs[0]
        }, torch.stack(labels)


class VerifyFilterOrder(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del config, context

    def __call__(self, sample, rng):
        del rng
        assert sample["keep"]
        return {"value": sample["value"] * 2, "processed": True}


def test_indexed_jsonl_random_access(tmp_path):
    write_jsonl(tmp_path / "b.jsonl", [{"id": 2}, {"id": 3}])
    write_jsonl(tmp_path / "a.jsonl", [{"id": 0}, {"id": 1}])
    source = IndexedJsonlSource.Config(patterns=(str(tmp_path / "*.jsonl"),)).build(
        dataset_iteration_policy=dataset_iteration_policy(),
    )

    assert len(source) == 4
    assert [source[index]["id"] for index in range(4)] == [0, 1, 2, 3]
    assert source[-1]["id"] == 3


def test_indexed_jsonl_rejects_missing_and_duplicate_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        IndexedJsonlSource.Config(patterns=(str(tmp_path / "missing*.jsonl"),)).build(
            dataset_iteration_policy=dataset_iteration_policy(),
        )

    write_jsonl(tmp_path / "rows.jsonl", [{"id": 0}])
    with pytest.raises(ValueError, match="more than once"):
        IndexedJsonlSource.Config(
            patterns=(
                str(tmp_path / "rows.jsonl"),
                str(tmp_path / "*.jsonl"),
            )
        ).build(
            dataset_iteration_policy=dataset_iteration_policy(),
        )


def test_hugging_face_streaming_source_shards_and_restores(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"id": index} for index in range(10)])
    config = HuggingFaceStreamingSource.Config(
        path="json",
        split="train",
        load_dataset_kwargs={"data_files": str(path)},
    )
    rank_rows = []
    for rank in range(2):
        dataset = config.build(
            dataset_iteration_policy=dataset_iteration_policy(
                dp_rank=rank,
                dp_world_size=2,
            ),
        )
        rank_rows.append([row["id"] for row in dataset])

    assert set(rank_rows[0]).isdisjoint(rank_rows[1])
    assert set(rank_rows[0]) | set(rank_rows[1]) == set(range(10))

    iterator = iter(
        config.build(
            dataset_iteration_policy=dataset_iteration_policy(),
        )
    )
    next(iterator)
    state = iterator.get_state()
    expected = next(iterator)
    restored = iter(
        config.build(
            dataset_iteration_policy=dataset_iteration_policy(),
        )
    )
    restored.set_state(state)

    assert next(restored) == expected


def _hf_sharded_rows():
    return datasets.Dataset.from_dict({"id": list(range(16))}).to_iterable_dataset(
        num_shards=8
    )


def test_hf_shuffled_repeat_advances_epoch():
    iterator = _HuggingFaceCursorIterator(
        _hf_sharded_rows(),
        repeat=True,
        shuffle=True,
    )

    first_epoch = [next(iterator)["id"] for _ in range(16)]
    second_epoch = [next(iterator)["id"] for _ in range(16)]

    assert first_epoch != second_epoch
    assert set(first_epoch) == set(second_epoch) == set(range(16))


def test_hf_unshuffled_repeat_replays_order():
    iterator = _HuggingFaceCursorIterator(
        _hf_sharded_rows(),
        repeat=True,
        shuffle=False,
    )

    first_epoch = [next(iterator)["id"] for _ in range(16)]
    second_epoch = [next(iterator)["id"] for _ in range(16)]

    assert first_epoch == second_epoch


def test_hf_resume_mid_second_epoch():
    iterator = _HuggingFaceCursorIterator(
        _hf_sharded_rows(),
        repeat=True,
        shuffle=True,
    )
    for _ in range(20):
        next(iterator)
    state = iterator.get_state()
    expected = [next(iterator) for _ in range(16)]

    restored = _HuggingFaceCursorIterator(
        _hf_sharded_rows(),
        repeat=True,
        shuffle=True,
    )
    restored.set_state(state)

    assert [next(restored) for _ in range(16)] == expected


def test_hf_empty_repeat_stops():
    empty = datasets.IterableDataset.from_generator(lambda: iter(()))
    iterator = _HuggingFaceCursorIterator(
        empty,
        repeat=True,
        shuffle=True,
    )

    with pytest.raises(StopIteration):
        next(iterator)


def test_hf_cursor_restores_through_grain_wrappers(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"id": index} for index in range(10)])
    config = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={"data_files": str(path)},
        ),
        post_filters=(lambda row: row["id"] % 2 == 0,),
    )
    policy = dataset_iteration_policy(repeat=True, shuffle=True)
    iterator = iter(config.build(context=CONTEXT, dataset_iteration_policy=policy))
    for _ in range(3):
        next(iterator)
    state = iterator.get_state()
    expected = [next(iterator) for _ in range(5)]

    restored = iter(config.build(context=CONTEXT, dataset_iteration_policy=policy))
    restored.set_state(state)

    assert [next(restored) for _ in range(5)] == expected


@pytest.mark.parametrize(
    ("source_type", "streaming"),
    [
        (HuggingFaceRandomAccessSource, False),
        (HuggingFaceStreamingSource, True),
    ],
)
def test_hf_explicit_fields_are_passed_to_load_dataset(
    monkeypatch, source_type, streaming
):
    loaded = (
        datasets.IterableDataset.from_generator(lambda: iter(({"id": 0},)))
        if streaming
        else datasets.Dataset.from_dict({"id": [0]})
    )
    load_dataset = mock.Mock(return_value=loaded)
    monkeypatch.setattr(
        "torchtitan.components.data.sources.datasets.load_dataset",
        load_dataset,
    )

    source_type.Config(
        path="owner/dataset",
        name="configuration",
        split="validation",
        revision="abc123",
        load_dataset_kwargs={"token": "secret"},
    ).build(
        dataset_iteration_policy=dataset_iteration_policy(),
    )

    load_dataset.assert_called_once_with(
        "owner/dataset",
        name="configuration",
        split="validation",
        revision="abc123",
        streaming=streaming,
        token="secret",
    )


@pytest.mark.parametrize(
    "source_type", [HuggingFaceRandomAccessSource, HuggingFaceStreamingSource]
)
def test_hf_lineage_fields_are_serialized(source_type):
    config = GrainDataLoader.Config(
        dataset=SingleDatasetConfig(
            source=source_type.Config(
                path="owner/dataset",
                name="configuration",
                split="validation",
                revision="abc123",
            )
        )
    )

    source = config.to_dict()["dataset"]["source"]

    assert source["path"] == "owner/dataset"
    assert source["name"] == "configuration"
    assert source["split"] == "validation"
    assert source["revision"] == "abc123"


@pytest.mark.parametrize(
    "source_type", [HuggingFaceRandomAccessSource, HuggingFaceStreamingSource]
)
def test_hf_config_requires_split(source_type):
    with pytest.raises(TypeError, match="split"):
        source_type.Config(path="owner/dataset")


@pytest.mark.parametrize(
    "source_type", [HuggingFaceRandomAccessSource, HuggingFaceStreamingSource]
)
@pytest.mark.parametrize("field", ["split", "name", "revision", "streaming"])
def test_hf_config_rejects_duplicate_first_class_fields(source_type, field):
    with pytest.raises(ValueError, match="repeated in kwargs"):
        source_type.Config(
            path="owner/dataset",
            split="train",
            load_dataset_kwargs={field: "duplicate"},
        )


@pytest.mark.parametrize(
    ("source_type", "wrong_leaf", "message"),
    [
        (
            HuggingFaceRandomAccessSource,
            datasets.IterableDataset.from_generator(lambda: iter(({"id": 0},))),
            "requires one Dataset",
        ),
        (
            HuggingFaceRandomAccessSource,
            datasets.DatasetDict({"train": datasets.Dataset.from_dict({"id": [0]})}),
            "requires one Dataset",
        ),
        (
            HuggingFaceStreamingSource,
            datasets.Dataset.from_dict({"id": [0]}),
            "requires one IterableDataset",
        ),
        (
            HuggingFaceStreamingSource,
            datasets.IterableDatasetDict(
                {
                    "train": datasets.IterableDataset.from_generator(
                        lambda: iter(({"id": 0},))
                    )
                }
            ),
            "requires one IterableDataset",
        ),
    ],
)
def test_hf_source_rejects_wrong_leaf(monkeypatch, source_type, wrong_leaf, message):
    monkeypatch.setattr(
        "torchtitan.components.data.sources.datasets.load_dataset",
        lambda *args, **kwargs: wrong_leaf,
    )

    with pytest.raises(TypeError, match=message):
        source_type.Config(path="owner/dataset", split="train").build(
            dataset_iteration_policy=dataset_iteration_policy(),
        )


def test_loader_requires_repeat_with_data_parallelism(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"id": index} for index in range(4)])
    config = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={"data_files": str(path)},
        ),
        post_filters=(lambda row: row["id"] % 2 == 0,),
    )

    with pytest.raises(ValueError, match="repeat=False with data parallelism"):
        GrainDataLoader.Config(dataset=config, repeat=False).build(
            dp_world_size=2,
            dp_rank=0,
            tokenizer=FakeTokenizer(),
            seq_len=8,
            local_batch_size=2,
        )


@dataclass(frozen=True)
class RowsSourceConfig:
    rows: tuple[Any, ...]

    def build(
        self,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ):
        del dataset_iteration_policy
        return self.rows


@dataclass(frozen=True)
class StreamingRowsSourceConfig:
    rows: tuple[dict, ...]

    def build(
        self,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ):
        dp_rank = dataset_iteration_policy.dp_rank
        dp_world_size = dataset_iteration_policy.dp_world_size
        dataset = grain.MapDataset.source(self.rows)
        dataset = dataset[dp_rank::dp_world_size]
        if dataset_iteration_policy.repeat:
            dataset = dataset.repeat()
        return dataset.to_iter_dataset()


@pytest.mark.parametrize("source_type", [RowsSourceConfig, StreamingRowsSourceConfig])
def test_pre_and_post_filters_restore_exactly(source_type):
    rows = tuple({"value": index, "keep": index % 2 == 0} for index in range(12))
    config = SingleDatasetConfig(
        source=source_type(rows=rows),
        pre_filters=(lambda row: row["keep"],),
        processor=VerifyFilterOrder.Config(),
        post_filters=(lambda row: row["processed"] and row["value"] % 4 == 0,),
    )
    dataset = config.build(
        context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy()
    )
    if isinstance(dataset, grain.MapDataset):
        dataset = dataset.to_iter_dataset(read_options=CONTEXT.read_options)
    iterator = iter(dataset)

    assert next(iterator)["value"] == 0
    state = iterator.get_state()
    expected = list(iterator)

    restored_dataset = config.build(
        context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy()
    )
    if isinstance(restored_dataset, grain.MapDataset):
        restored_dataset = restored_dataset.to_iter_dataset(
            read_options=CONTEXT.read_options
        )
    restored = iter(restored_dataset)
    restored.set_state(state)

    assert (
        list(restored)
        == expected
        == [
            {"value": 4, "processed": True},
            {"value": 8, "processed": True},
            {"value": 12, "processed": True},
            {"value": 16, "processed": True},
            {"value": 20, "processed": True},
        ]
    )


def test_single_dataset_shuffle_shard_repeat_order():
    config = SingleDatasetConfig(
        source=RowsSourceConfig(rows=tuple({"value": index} for index in range(12)))
    )
    rank_0 = config.build(
        context=CONTEXT,
        dataset_iteration_policy=dataset_iteration_policy(
            shuffle=True, dp_world_size=2, dp_rank=0
        ),
    )
    rank_1 = config.build(
        context=CONTEXT,
        dataset_iteration_policy=dataset_iteration_policy(
            shuffle=True, dp_world_size=2, dp_rank=1
        ),
    )
    rank_0_peer = config.build(
        context=CONTEXT,
        dataset_iteration_policy=dataset_iteration_policy(
            shuffle=True, dp_world_size=2, dp_rank=0
        ),
    )
    values_0 = {row["value"] for row in rank_0}
    values_1 = {row["value"] for row in rank_1}

    assert values_0.isdisjoint(values_1)
    assert values_0 | values_1 == set(range(12))
    assert list(rank_0) == list(rank_0_peer)
    assert [row["value"] for row in rank_0] != [0, 2, 4, 6, 8, 10]


def test_map_dataset_unshuffled_repeat_replays_order():
    config = SingleDatasetConfig(
        source=RowsSourceConfig(
            rows=tuple({"value": index} for index in range(4)),
        ),
    )
    dataset = config.build(
        context=CONTEXT,
        dataset_iteration_policy=dataset_iteration_policy(
            repeat=True,
            shuffle=False,
        ),
    )

    assert [dataset[index]["value"] for index in range(8)] == [
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
    ]


def test_weighted_map_mix_keeps_weight_with_dataset():
    left = SingleDatasetConfig(
        source=RowsSourceConfig(
            rows=tuple({"source": "left", "index": index} for index in range(20))
        )
    )
    right = SingleDatasetConfig(
        source=RowsSourceConfig(
            rows=tuple({"source": "right", "index": index} for index in range(20))
        )
    )
    dataset = DatasetMixConfig(
        datasets=(
            WeightedDataset(dataset=left, weight=2.0),
            WeightedDataset(dataset=right, weight=1.0),
        )
    ).build(
        context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy(repeat=True)
    )

    assert isinstance(dataset, grain.MapDataset)

    iterator = iter(dataset)
    values = [next(iterator)["source"] for _ in range(12)]

    assert values.count("left") == 8
    assert values.count("right") == 4


def _mixed_child_rows(source_type):
    large = SingleDatasetConfig(
        source=source_type(
            rows=tuple({"source": "large", "index": index} for index in range(100))
        )
    )
    small = SingleDatasetConfig(
        source=source_type(
            rows=tuple({"source": "small", "index": index} for index in range(10))
        )
    )
    dataset = DatasetMixConfig(
        datasets=(
            WeightedDataset(dataset=large),
            WeightedDataset(dataset=small),
        )
    ).build(
        context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy(repeat=True)
    )
    if isinstance(dataset, grain.MapDataset):
        return [dataset[index] for index in range(2_000)]
    iterator = iter(dataset)
    return [next(iterator) for _ in range(2_000)]


def test_mix_reaches_all_rows_of_larger_map_child():
    rows = _mixed_child_rows(RowsSourceConfig)
    seen = {row["index"] for row in rows if row["source"] == "large"}

    assert seen == set(range(100))


def test_mix_reaches_all_rows_of_larger_iterable_child():
    rows = _mixed_child_rows(StreamingRowsSourceConfig)
    seen = {row["index"] for row in rows if row["source"] == "large"}

    assert seen == set(range(100))


def test_filtered_map_mix_weights_draw_attempts():
    sparse = SingleDatasetConfig(
        source=RowsSourceConfig(
            rows=tuple({"source": "sparse", "index": index} for index in range(100))
        ),
        pre_filters=(lambda row: row["index"] % 10 == 0,),
    )
    dense = SingleDatasetConfig(
        source=RowsSourceConfig(
            rows=tuple({"source": "dense", "index": index} for index in range(100))
        )
    )
    dataset = DatasetMixConfig(
        datasets=(
            WeightedDataset(dataset=sparse),
            WeightedDataset(dataset=dense),
        )
    ).build(
        context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy(repeat=True)
    )

    assert isinstance(dataset, grain.MapDataset)

    dataset = dataset.to_iter_dataset(read_options=CONTEXT.read_options)
    iterator = iter(dataset)
    sources = [next(iterator)["source"] for _ in range(2_000)]

    # Draw attempts are equal, but the sparse child accepts one row in ten.
    assert 150 <= sources.count("sparse") <= 220


def test_mixed_map_and_iterable_weights_emitted_rows():
    map_child = SingleDatasetConfig(
        source=RowsSourceConfig(
            rows=tuple({"source": "map", "index": index} for index in range(100))
        ),
        pre_filters=(lambda row: row["index"] % 10 == 0,),
    )
    stream_child = SingleDatasetConfig(
        source=StreamingRowsSourceConfig(
            rows=tuple({"source": "stream", "index": index} for index in range(100))
        )
    )
    dataset = DatasetMixConfig(
        datasets=(
            WeightedDataset(dataset=map_child),
            WeightedDataset(dataset=stream_child),
        )
    ).build(
        context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy(repeat=True)
    )

    assert isinstance(dataset, grain.IterDataset)

    iterator = iter(dataset)
    sources = [next(iterator)["source"] for _ in range(40)]

    assert sources.count("map") == 20
    assert sources.count("stream") == 20


@pytest.mark.parametrize("lengths", [(3, 3), (2, 10)])
def test_finite_mix_stops_when_first_child_exhausts(lengths):
    children = tuple(
        WeightedDataset(
            dataset=SingleDatasetConfig(
                source=RowsSourceConfig(
                    rows=tuple(
                        {"source": source, "index": index} for index in range(length)
                    )
                )
            )
        )
        for source, length in enumerate(lengths)
    )
    rows = list(
        DatasetMixConfig(datasets=children).build(
            context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy()
        )
    )
    counts = [sum(row["source"] == source for row in rows) for source in range(2)]

    assert any(count == length for count, length in zip(counts, lengths, strict=True))
    assert all(count <= length for count, length in zip(counts, lengths, strict=True))


def test_finite_mix_with_empty_iterable_child_is_empty():
    empty = SingleDatasetConfig(source=StreamingRowsSourceConfig(rows=()))
    nonempty = SingleDatasetConfig(
        source=StreamingRowsSourceConfig(
            rows=tuple({"value": index} for index in range(4))
        )
    )

    rows = list(
        DatasetMixConfig(
            datasets=(
                WeightedDataset(dataset=empty),
                WeightedDataset(dataset=nonempty),
            )
        ).build(context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy())
    )

    assert rows == []


def test_iterable_mix_restores_exactly():
    config = DatasetMixConfig(
        datasets=tuple(
            WeightedDataset(
                dataset=SingleDatasetConfig(
                    source=StreamingRowsSourceConfig(
                        rows=tuple(
                            {"source": source, "index": index} for index in range(20)
                        )
                    )
                ),
                weight=source + 1,
            )
            for source in range(2)
        )
    )
    policy = dataset_iteration_policy(repeat=True)
    dataset = config.build(context=CONTEXT, dataset_iteration_policy=policy)

    assert isinstance(dataset, grain.IterDataset)

    iterator = iter(dataset)
    for _ in range(17):
        next(iterator)
    state = iterator.get_state()
    expected = [next(iterator) for _ in range(20)]

    restored = iter(config.build(context=CONTEXT, dataset_iteration_policy=policy))
    restored.set_state(state)

    assert [next(restored) for _ in range(20)] == expected


def test_map_mix_restores_exactly_after_root_conversion():
    config = DatasetMixConfig(
        datasets=tuple(
            WeightedDataset(
                dataset=SingleDatasetConfig(
                    source=RowsSourceConfig(
                        rows=tuple(
                            {"source": source, "index": index} for index in range(20)
                        )
                    )
                ),
                weight=source + 1,
            )
            for source in range(2)
        )
    )
    policy = dataset_iteration_policy(repeat=True)
    dataset = config.build(context=CONTEXT, dataset_iteration_policy=policy)

    assert isinstance(dataset, grain.MapDataset)

    iterator = iter(dataset.to_iter_dataset(read_options=CONTEXT.read_options))
    for _ in range(17):
        next(iterator)
    state = iterator.get_state()
    expected = [next(iterator) for _ in range(20)]

    restored_dataset = config.build(
        context=CONTEXT,
        dataset_iteration_policy=policy,
    )
    assert isinstance(restored_dataset, grain.MapDataset)
    restored = iter(restored_dataset.to_iter_dataset(read_options=CONTEXT.read_options))
    restored.set_state(state)

    assert [next(restored) for _ in range(20)] == expected


def test_concat_shards_after_one_global_index_space():
    left = SingleDatasetConfig(
        source=RowsSourceConfig(rows=tuple({"value": index} for index in range(4)))
    )
    right = SingleDatasetConfig(
        source=RowsSourceConfig(rows=tuple({"value": index} for index in range(4, 8)))
    )
    config = DatasetConcatConfig(datasets=(left, right))
    rank_0 = list(
        config.build(
            context=CONTEXT,
            dataset_iteration_policy=dataset_iteration_policy(
                dp_world_size=2, dp_rank=0
            ),
        )
    )
    rank_1 = list(
        config.build(
            context=CONTEXT,
            dataset_iteration_policy=dataset_iteration_policy(
                dp_world_size=2, dp_rank=1
            ),
        )
    )

    assert [row["value"] for row in rank_0] == [0, 1, 2, 3]
    assert [row["value"] for row in rank_1] == [4, 5, 6, 7]


@pytest.mark.parametrize(
    "packing_type",
    [ConcatThenSplitPackingConfig, FirstFitPackingConfig],
)
def test_packing_yields_rows_and_loader_batches(packing_type):
    documents = SingleDatasetConfig(
        source=RowsSourceConfig(
            rows=(
                {"tokens": [1, 10, 11, 12, 2]},
                {"tokens": [1, 20, 21, 2]},
                {"tokens": [1, 30, 31, 32, 2]},
                {"tokens": [1, 40, 41, 2]},
            )
        ),
        processor=RowToTokens.Config(),
    )
    recipe = packing_type(dataset=documents)
    packed_sequences = recipe.build(
        context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy()
    )
    first_sequence = next(iter(packed_sequences))

    assert first_sequence.input_ids.shape == (9,)
    assert first_sequence.labels.shape == (9,)
    assert first_sequence.positions.shape == (9,)

    loader = GrainDataLoader.Config(
        dataset=recipe,
        collator=TextCollator.Config(),
        shuffle=False,
        repeat=True,
        num_prefetch_batches=1,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    inputs, labels = next(iter(loader))
    assert inputs["input"].shape == (2, 8)
    assert labels.shape == (2, 8)


def test_first_fit_num_packing_bins_is_independent_of_local_batch_size(monkeypatch):
    captured = {}

    def capture_options(dataset, **kwargs):
        captured.update(kwargs)
        return dataset

    monkeypatch.setattr(
        grain.experimental,
        "FirstFitPackIterDataset",
        capture_options,
    )
    dataset = SingleDatasetConfig(
        source=RowsSourceConfig(rows=({"tokens": [1, 2]},)),
        processor=RowToTokens.Config(),
    )

    FirstFitPackingConfig(dataset=dataset).build(
        context=replace(CONTEXT, local_batch_size=64),
        dataset_iteration_policy=dataset_iteration_policy(),
    )

    assert captured["num_packing_bins"] == 8


def test_first_fit_meta_features_are_packed_arrays():
    dataset = SingleDatasetConfig(
        source=RowsSourceConfig(
            rows=(
                {"tokens": [1, 2]},
                {"tokens": [3, 4, 5]},
            )
        ),
        processor=RowToTokens.Config(),
    )

    packed = next(
        iter(
            FirstFitPackingConfig(dataset=dataset).build(
                context=CONTEXT,
                dataset_iteration_policy=dataset_iteration_policy(),
            )
        )
    )

    assert isinstance(packed.input_ids, np.ndarray)
    assert isinstance(packed.labels, np.ndarray)
    assert isinstance(packed.positions, np.ndarray)
    assert packed.input_ids.shape == packed.labels.shape == packed.positions.shape


def test_first_fit_positions_reset_per_document():
    dataset = SingleDatasetConfig(
        source=RowsSourceConfig(
            rows=(
                {"tokens": [1, 2]},
                {"tokens": [3, 4, 5]},
            )
        ),
        processor=RowToTokens.Config(),
    )

    packed = next(
        iter(
            FirstFitPackingConfig(dataset=dataset).build(
                context=CONTEXT,
                dataset_iteration_policy=dataset_iteration_policy(),
            )
        )
    )

    assert packed.positions[:3].tolist() == [0, 0, 1]
    assert packed.labels[:3].tolist() == [2, 4, 5]
    assert (packed.labels[3:] == IGNORE_INDEX).all()


def test_nested_packing_preserves_inner_document_boundaries():
    documents = SingleDatasetConfig(
        source=RowsSourceConfig(
            rows=(
                {"tokens": [1, 2]},
                {"tokens": [3, 4, 5]},
            )
        ),
        processor=RowToTokens.Config(),
    )
    inner = FirstFitPackingConfig(dataset=documents)
    outer = ConcatThenSplitPackingConfig(dataset=inner)

    packed = next(
        iter(
            outer.build(
                context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy()
            )
        )
    )

    assert packed.positions[:3].tolist() == [0, 0, 1]
    assert packed.labels[:3].tolist() == [2, 4, 5]


def test_unpacked_text_collator_creates_range_positions():
    sequence = TextSequence(
        input_ids=np.asarray([1, 2, 3]),
        labels=np.asarray([2, 3, 4]),
    )

    inputs, labels = TextCollator.Config().build(context=CONTEXT)([sequence])

    assert inputs["input"][0, :3].tolist() == [1, 2, 3]
    assert inputs["positions"][0, :3].tolist() == [0, 1, 2]
    assert labels[0, :3].tolist() == [2, 3, 4]
    assert (labels[0, 3:] == IGNORE_INDEX).all()


def test_pack_then_pack_then_collate_preserves_aligned_pairs():
    documents = SingleDatasetConfig(
        source=RowsSourceConfig(
            rows=(
                {"tokens": [1, 2]},
                {"tokens": [3, 4, 5]},
            )
        ),
        processor=RowToTokens.Config(),
    )
    packed = next(
        iter(
            FirstFitPackingConfig(
                dataset=ConcatThenSplitPackingConfig(dataset=documents)
            ).build(
                context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy()
            )
        )
    )

    inputs, labels = TextCollator.Config().build(context=CONTEXT)([packed])

    assert inputs["input"][0, :3].tolist() == [1, 3, 4]
    assert labels[0, :3].tolist() == [2, 4, 5]
    assert inputs["positions"][0, :3].tolist() == [0, 0, 1]


class SftTokens(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del config, context

    def __call__(self, sample, rng):
        del sample, rng
        return TextSequence(
            input_ids=np.asarray([1, 10, 11, 20, 21]),
            labels=np.asarray([IGNORE_INDEX, IGNORE_INDEX, 20, 21, 2]),
        )


def test_sft_labels_survive_packing_and_collation():
    recipe = FirstFitPackingConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(rows=({"id": 0}, {"id": 1})),
            processor=SftTokens.Config(),
        )
    )
    packed = next(
        iter(
            recipe.build(
                context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy()
            )
        )
    )
    _, labels = TextCollator.Config().build(context=CONTEXT)([packed])

    assert labels[0, 0].item() == IGNORE_INDEX
    assert labels[0, 1].item() == IGNORE_INDEX
    assert labels[0, 2].item() == 20


def test_text_processor_leaves_positions_unmaterialized():
    processor = TextProcessor.Config().build(context=CONTEXT)

    sequence = processor({"text": "hello"}, np.random.default_rng(0))

    assert sequence is not None
    assert sequence.positions is None
    assert sequence.input_ids.tolist() == [1, 114, 111, 118, 118, 121]
    assert sequence.labels.tolist() == [114, 111, 118, 118, 121, 2]


def test_text_processor_returns_none_for_short_row():
    tokenizer = mock.Mock()
    tokenizer.encode.return_value = [FakeTokenizer.bos_id]
    processor = TextProcessor.Config().build(
        context=replace(CONTEXT, tokenizer=tokenizer)
    )

    assert processor({"text": ""}, np.random.default_rng(0)) is None


def test_single_dataset_post_filter_removes_none():
    tokenizer = mock.Mock()
    tokenizer.encode.side_effect = ([FakeTokenizer.bos_id], [1, 10, 2])
    dataset = SingleDatasetConfig(
        source=RowsSourceConfig(rows=({"text": ""}, {"text": "kept"})),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    ).build(
        context=replace(CONTEXT, tokenizer=tokenizer),
        dataset_iteration_policy=dataset_iteration_policy(),
    )

    sequences = list(dataset)
    assert [sequence.input_ids.tolist() for sequence in sequences] == [[1, 10]]
    assert [sequence.labels.tolist() for sequence in sequences] == [[10, 2]]


def test_chat_processor_masks_prompt_and_trains_assistant():
    def question_answer_to_messages(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    processor = ChatProcessor.Config(
        messages_fn=question_answer_to_messages,
    ).build(context=replace(CONTEXT, seq_len=65))
    token_sequence = processor(
        {"question": "2+2?", "answer": "4"},
        np.random.default_rng(0),
    )
    prompt = FakeTokenizer().apply_chat_template(
        [{"role": "user", "content": "2+2?"}],
        add_generation_prompt=True,
    )
    prompt_length = len(FakeTokenizer().encode(prompt, add_bos=True, add_eos=False))

    assert (token_sequence.labels[: prompt_length - 1] == IGNORE_INDEX).all()
    assert (token_sequence.labels[prompt_length - 1 :] != IGNORE_INDEX).all()
    assert token_sequence.labels[-1] == FakeTokenizer.eos_id


def test_chat_processor_rejects_non_single_turn_messages():
    processor = ChatProcessor.Config(
        messages_fn=lambda sample: sample["messages"],
    ).build(context=CONTEXT)

    with pytest.raises(ValueError, match="Expected single-turn"):
        processor(
            {"messages": [{"role": "user", "content": "hello"}]},
            np.random.default_rng(0),
        )


def test_first_fit_drops_rows_longer_than_sequence_length():
    recipe = FirstFitPackingConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(
                rows=(
                    {"tokens": list(range(20))},
                    {"tokens": [1, 10, 11, 2]},
                )
            ),
            processor=RowToTokens.Config(),
        )
    )

    sequence = next(
        iter(
            recipe.build(
                context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy()
            )
        )
    )
    assert sequence.input_ids[0].item() == 1


class AddRandomOffset(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        maximum: int

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del context
        self.maximum = config.maximum

    def __call__(self, sample, rng):
        return sample | {"offset": int(rng.integers(self.maximum))}


def test_configured_random_processor_is_deterministic():
    config = SingleDatasetConfig(
        source=RowsSourceConfig(rows=tuple({"id": index} for index in range(10))),
        processor=AddRandomOffset.Config(maximum=1000),
    )
    first = list(
        config.build(
            context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy(seed=7)
        )
    )
    second = list(
        config.build(
            context=CONTEXT, dataset_iteration_policy=dataset_iteration_policy(seed=7)
        )
    )

    assert first == second


class RandomTextSequence(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del config, context

    def __call__(self, sample, rng):
        random_token = int(rng.integers(10, 200))
        input_ids = np.asarray([1, sample["id"] + 10, random_token, 2])
        return TextSequence(
            input_ids=input_ids,
            labels=input_ids.copy(),
        )


def test_loader_restores_configured_random_map():
    config = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=SingleDatasetConfig(
                source=RowsSourceConfig(
                    rows=tuple({"id": index} for index in range(20))
                ),
                processor=RandomTextSequence.Config(),
            )
        ),
        collator=TextCollator.Config(),
        repeat=True,
        num_prefetch_batches=1,
    )
    loader = config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    next(iter(loader))
    state = loader.state_dict()
    expected = next(iter(loader))

    restored = config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    restored.load_state_dict(state)
    actual = next(iter(restored))

    assert torch.equal(expected[0]["input"], actual[0]["input"])
    assert torch.equal(expected[1], actual[1])


def test_loader_exact_restore_with_nonempty_packing_buffers():
    recipe = ConcatThenSplitPackingConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(
                rows=tuple(
                    {"tokens": [1, index + 10, index + 11, 2]} for index in range(20)
                )
            ),
            processor=RowToTokens.Config(),
        )
    )
    config = GrainDataLoader.Config(
        dataset=recipe,
        collator=TextCollator.Config(),
        shuffle=True,
        repeat=True,
        num_prefetch_batches=2,
    )
    loader = config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    expected = next(iterator)

    restored = config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    restored.load_state_dict(state)
    actual = next(iter(restored))

    assert torch.equal(expected[0]["input"], actual[0]["input"])
    assert torch.equal(expected[0]["positions"], actual[0]["positions"])
    assert torch.equal(expected[1], actual[1])


def test_loader_exact_restore_with_map_mix_before_first_fit():
    def tokenized_documents(offset):
        return SingleDatasetConfig(
            source=RowsSourceConfig(
                rows=tuple(
                    {
                        "tokens": [
                            1,
                            offset + index,
                            offset + index + 1,
                            2,
                        ]
                    }
                    for index in range(20)
                )
            ),
            processor=RowToTokens.Config(),
        )

    config = GrainDataLoader.Config(
        dataset=FirstFitPackingConfig(
            dataset=DatasetMixConfig(
                datasets=(
                    WeightedDataset(
                        dataset=tokenized_documents(10),
                        weight=0.67,
                    ),
                    WeightedDataset(
                        dataset=tokenized_documents(100),
                        weight=0.33,
                    ),
                ),
            )
        ),
        collator=TextCollator.Config(),
        repeat=True,
        num_prefetch_batches=1,
    )
    loader = config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    iterator = iter(loader)
    for _ in range(5):
        next(iterator)
    state = loader.state_dict()
    expected = [next(iterator) for _ in range(8)]

    restored = config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    restored.load_state_dict(state)
    restored_iterator = iter(restored)
    actual = [next(restored_iterator) for _ in range(8)]

    for expected_batch, actual_batch in zip(expected, actual, strict=True):
        assert torch.equal(expected_batch[0]["input"], actual_batch[0]["input"])
        assert torch.equal(expected_batch[0]["positions"], actual_batch[0]["positions"])
        assert torch.equal(expected_batch[1], actual_batch[1])


def test_loader_exact_restore_with_nested_weighted_mix():
    def packed_rows(offset):
        return ConcatThenSplitPackingConfig(
            dataset=SingleDatasetConfig(
                source=RowsSourceConfig(
                    rows=tuple(
                        {
                            "tokens": [
                                1,
                                offset + index,
                                offset + index + 1,
                                2,
                            ]
                        }
                        for index in range(20)
                    )
                ),
                processor=RowToTokens.Config(),
            )
        )

    config = GrainDataLoader.Config(
        dataset=DatasetMixConfig(
            datasets=(
                WeightedDataset(dataset=packed_rows(10), weight=2.0),
                WeightedDataset(dataset=packed_rows(100), weight=1.0),
            ),
        ),
        collator=TextCollator.Config(),
        repeat=True,
        num_prefetch_batches=1,
    )
    loader = config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    iterator = iter(loader)
    for _ in range(40):
        next(iterator)
    state = loader.state_dict()
    expected = [next(iterator) for _ in range(8)]

    restored = config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    restored.load_state_dict(state)
    restored_iterator = iter(restored)
    actual = [next(restored_iterator) for _ in range(8)]

    for expected_batch, actual_batch in zip(expected, actual):
        assert torch.equal(expected_batch[0]["input"], actual_batch[0]["input"])
        assert torch.equal(expected_batch[0]["positions"], actual_batch[0]["positions"])
        assert torch.equal(expected_batch[1], actual_batch[1])


def test_empty_shard_rejected():
    dataset = SingleDatasetConfig(source=RowsSourceConfig(rows=({"id": 0},)))

    with pytest.raises(ValueError, match="fewer than"):
        dataset.build(
            context=CONTEXT,
            dataset_iteration_policy=dataset_iteration_policy(
                dp_rank=1, dp_world_size=2
            ),
        )


def test_concat_rejects_iterable_children(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"id": 0}])
    stream = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={"data_files": str(path)},
        )
    )

    with pytest.raises(TypeError, match="map-style children"):
        DatasetConcatConfig(datasets=(stream,)).build(
            context=CONTEXT,
            dataset_iteration_policy=dataset_iteration_policy(),
        )


def test_loader_rejects_dp_change():
    recipe = ConcatThenSplitPackingConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(
                rows=tuple({"tokens": [1, index, index + 1, 2]} for index in range(8))
            ),
            processor=RowToTokens.Config(),
        )
    )
    config = GrainDataLoader.Config(
        dataset=recipe,
        collator=TextCollator.Config(),
        shuffle=False,
        repeat=True,
    )
    loader = config.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    state = loader.state_dict()

    different_dp = config.build(
        dp_world_size=2,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    with pytest.raises(ValueError, match="data-parallel"):
        different_dp.load_state_dict(state)


def test_concat_then_split_normalizes_split_continuation_positions():
    recipe = ConcatThenSplitPackingConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(rows=({"tokens": list(range(10))},)),
            processor=RowToTokens.Config(),
        )
    )
    rows = list(
        recipe.build(
            context=replace(CONTEXT, seq_len=5),
            dataset_iteration_policy=dataset_iteration_policy(
                shuffle=False, repeat=False
            ),
        )
    )

    assert rows[0].input_ids.tolist() == [0, 1, 2, 3, 4]
    assert rows[1].input_ids.tolist() == [5, 6, 7, 8, 0]
    assert rows[0].labels.tolist() == [1, 2, 3, 4, 5]
    assert rows[1].labels.tolist() == [6, 7, 8, 9, IGNORE_INDEX]
    assert rows[0].positions.tolist() == [0, 1, 2, 3, 4]
    assert rows[1].positions.tolist() == [0, 1, 2, 3, 0]

    collator = TextCollator.Config().build(context=replace(CONTEXT, seq_len=5))
    first_inputs, first_labels = collator([rows[0]])
    second_inputs, second_labels = collator([rows[1]])

    assert first_inputs["input"].tolist() == [[0, 1, 2, 3, 4]]
    assert first_labels.tolist() == [[1, 2, 3, 4, 5]]
    assert second_inputs["input"].tolist() == [[5, 6, 7, 8, 0]]
    assert second_labels.tolist() == [[6, 7, 8, 9, IGNORE_INDEX]]


def test_map_dataset_reshuffles_deterministically_across_repeats():
    config = SingleDatasetConfig(
        source=RowsSourceConfig(rows=tuple({"value": index} for index in range(8)))
    )
    policy = dataset_iteration_policy(shuffle=True, repeat=True)
    first = config.build(context=CONTEXT, dataset_iteration_policy=policy)
    peer = config.build(context=CONTEXT, dataset_iteration_policy=policy)
    first_two = [first[index]["value"] for index in range(16)]
    peer_two = [peer[index]["value"] for index in range(16)]

    assert first_two[:8] != first_two[8:]
    assert first_two == peer_two


def test_concat_then_split_resets_positions_between_documents():
    recipe = ConcatThenSplitPackingConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(
                rows=(
                    {"tokens": [1, 10, 2]},
                    {"tokens": [1, 20, 21, 2]},
                )
            ),
            processor=RowToTokens.Config(),
        )
    )
    sequence = next(
        iter(
            recipe.build(
                context=replace(CONTEXT, seq_len=9),
                dataset_iteration_policy=dataset_iteration_policy(),
            )
        )
    )

    assert sequence.input_ids.tolist() == [1, 10, 1, 20, 21, 0, 0, 0, 0]
    assert sequence.labels.tolist() == [
        10,
        2,
        20,
        21,
        2,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]
    assert sequence.positions.tolist() == [0, 1, 0, 1, 2, 0, 0, 0, 0]

    inputs, labels = TextCollator.Config().build(context=CONTEXT)([sequence])
    assert inputs["input"].tolist() == [[1, 10, 1, 20, 21, 0, 0, 0, 0]]
    assert labels.tolist() == [
        [
            10,
            2,
            20,
            21,
            2,
            IGNORE_INDEX,
            IGNORE_INDEX,
            IGNORE_INDEX,
            IGNORE_INDEX,
        ]
    ]


@pytest.fixture
def finite_rows_loader():
    # Direct trainer rows survive SingleDatasetConfig -> GrainDataLoader unchanged.
    rows = tuple(
        (
            {
                "input": torch.tensor([index]),
                "positions": torch.tensor([0]),
            },
            torch.tensor([index]),
        )
        for index in range(5)
    )
    loader = GrainDataLoader.Config(
        dataset=SingleDatasetConfig(source=RowsSourceConfig(rows=rows)),
        collator=PairCollator.Config(),
        shuffle=False,
        repeat=False,
        num_prefetch_batches=1,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=1,
        local_batch_size=2,
    )
    yield loader
    loader.close()


def test_mix_rejects_empty_datasets():
    with pytest.raises(ValueError, match="positive-weight"):
        DatasetMixConfig(datasets=()).build(
            context=CONTEXT,
            dataset_iteration_policy=dataset_iteration_policy(),
        )


@pytest.mark.parametrize(
    "weight", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_mix_rejects_non_finite_or_nonpositive_weight(weight):
    dataset = SingleDatasetConfig(source=RowsSourceConfig(rows=({"value": 0},)))

    with pytest.raises(ValueError, match="positive-weight"):
        DatasetMixConfig(
            datasets=(WeightedDataset(dataset=dataset, weight=weight),)
        ).build(
            context=CONTEXT,
            dataset_iteration_policy=dataset_iteration_policy(),
        )


def test_loader_rejects_unknown_checkpoint_version(finite_rows_loader):
    state = finite_rows_loader.state_dict()
    state["version"] = 2

    with pytest.raises(ValueError, match="version"):
        finite_rows_loader.load_state_dict(state)


def test_loader_rejects_missing_rank_state(finite_rows_loader):
    state = finite_rows_loader.state_dict()
    del state["dp_rank_0"]

    with pytest.raises(ValueError, match="missing dataloader state"):
        finite_rows_loader.load_state_dict(state)


def test_loader_batches_exact_rows_and_preserves_finite_tail(finite_rows_loader):
    batches = list(finite_rows_loader)

    assert len(batches) == 3
    assert batches[0][0]["input"].tolist() == [[0], [1]]
    assert batches[0][1].tolist() == [[0], [1]]
    assert batches[1][0]["input"].tolist() == [[2], [3]]
    assert batches[1][1].tolist() == [[2], [3]]
    assert batches[2][0]["input"].tolist() == [[4]]
    assert batches[2][1].tolist() == [[4]]


def mock_grain_loader():
    loader = object.__new__(GrainDataLoader)
    loader._dp_world_size = 1
    loader._rank_id = "dp_rank_0"
    loader._iterator = mock.Mock()
    return loader


def test_restore_failure_closes_loader():
    loader = mock_grain_loader()
    loader._iterator.set_state.side_effect = RuntimeError("invalid Grain state")
    state = loader.state_dict()

    with pytest.raises(RuntimeError, match="invalid Grain state"):
        loader.load_state_dict(state)

    loader._iterator.close.assert_called_once_with()


def test_indexed_jsonl_loader_restores_exactly_on_each_rank(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(
        path,
        [{"tokens": [1, index + 10, index + 11, 2]} for index in range(20)],
    )
    config = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=SingleDatasetConfig(
                source=IndexedJsonlSource.Config(patterns=(str(path),)),
                processor=RowToTokens.Config(),
            )
        ),
        collator=TextCollator.Config(),
        seed=42,
        shuffle=True,
        repeat=True,
        num_prefetch_batches=1,
    )

    for rank in range(2):
        loader = config.build(
            dp_world_size=2,
            dp_rank=rank,
            tokenizer=FakeTokenizer(),
            seq_len=8,
            local_batch_size=2,
        )
        iterator = iter(loader)
        for _ in range(10):
            next(iterator)
        state = loader.state_dict()
        expected = [next(iterator) for _ in range(4)]

        restored = config.build(
            dp_world_size=2,
            dp_rank=rank,
            tokenizer=FakeTokenizer(),
            seq_len=8,
            local_batch_size=2,
        )
        restored.load_state_dict(state)
        restored_iterator = iter(restored)
        actual = [next(restored_iterator) for _ in range(4)]

        for expected_batch, actual_batch in zip(expected, actual):
            assert torch.equal(expected_batch[0]["input"], actual_batch[0]["input"])
            assert torch.equal(
                expected_batch[0]["positions"],
                actual_batch[0]["positions"],
            )
            assert torch.equal(expected_batch[1], actual_batch[1])


def test_first_fit_oversized_row_does_not_discard_buffered_row():
    recipe = FirstFitPackingConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(
                rows=(
                    {"tokens": [1, 2, 3, 4, 5, 6]},
                    {"tokens": list(range(20, 40))},
                    {"tokens": [10, 11, 12]},
                )
            ),
            processor=RowToTokens.Config(),
        )
    )

    sequence = next(
        iter(
            recipe.build(
                context=CONTEXT,
                dataset_iteration_policy=dataset_iteration_policy(),
            )
        )
    )

    assert sequence.input_ids.tolist() == [1, 2, 3, 4, 5, 10, 11, 0, 0]
    assert sequence.labels.tolist() == [
        2,
        3,
        4,
        5,
        6,
        11,
        12,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]


def test_loader_passes_read_options_to_map_conversion(monkeypatch):
    captured = []
    to_iter_dataset = grain.MapDataset.to_iter_dataset

    def capture_read_options(dataset, read_options=None, **kwargs):
        captured.append(read_options)
        return to_iter_dataset(dataset, read_options=read_options, **kwargs)

    monkeypatch.setattr(
        grain.MapDataset,
        "to_iter_dataset",
        capture_read_options,
    )
    read_options = grain.ReadOptions(num_threads=3, prefetch_buffer_size=7)
    loader = GrainDataLoader.Config(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(rows=({"value": 0},)),
        ),
        shuffle=False,
        repeat=False,
        read_options=read_options,
        num_prefetch_batches=0,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=1,
        local_batch_size=1,
    )
    loader.close()

    assert captured == [read_options]
