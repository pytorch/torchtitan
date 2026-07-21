# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the composed Grain data pipeline."""

import json
from dataclasses import dataclass, replace
from typing import Any

import grain.python as grain
import numpy as np
import pytest
import torch

from torchtitan.components.data.dataset import (
    DatasetBuildContext,
    DatasetConcatConfig,
    DatasetIterationPolicy,
    DatasetMixConfig,
    SampleProcessor,
    SingleDatasetConfig,
    TokenSequence,
    WeightedDataset,
)
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
)
from torchtitan.components.data.sources import (
    HuggingFaceStreamingSource,
    IndexedJsonlSource,
)
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets.text_datasets import ChatProcessor


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
    seq_len=8,
    local_batch_size=2,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)


def iteration(**overrides):
    values = {
        "seed": 42,
        "shuffle": False,
        "repeat": False,
        "dp_rank": 0,
        "dp_world_size": 1,
        "streaming_shuffle_window_size": 4,
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
        token_ids = np.asarray(sample["tokens"], dtype=np.int64)
        return TokenSequence(
            token_ids=token_ids,
            loss_mask=np.ones(token_ids.shape, dtype=np.bool_),
        )


def test_indexed_jsonl_random_access(tmp_path):
    write_jsonl(tmp_path / "b.jsonl", [{"id": 2}, {"id": 3}])
    write_jsonl(tmp_path / "a.jsonl", [{"id": 0}, {"id": 1}])
    source = IndexedJsonlSource.Config(patterns=(str(tmp_path / "*.jsonl"),)).build(
        dp_rank=0, dp_world_size=1
    )

    assert len(source) == 4
    assert [source[index]["id"] for index in range(4)] == [0, 1, 2, 3]
    assert source[-1]["id"] == 3


def test_indexed_jsonl_rejects_missing_and_duplicate_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        IndexedJsonlSource.Config(patterns=(str(tmp_path / "missing*.jsonl"),)).build(
            dp_rank=0, dp_world_size=1
        )

    write_jsonl(tmp_path / "rows.jsonl", [{"id": 0}])
    with pytest.raises(ValueError, match="more than once"):
        IndexedJsonlSource.Config(
            patterns=(
                str(tmp_path / "rows.jsonl"),
                str(tmp_path / "*.jsonl"),
            )
        ).build(dp_rank=0, dp_world_size=1)


def test_hugging_face_streaming_source_shards_and_restores(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"id": index} for index in range(10)])
    config = HuggingFaceStreamingSource.Config(
        path="json",
        load_dataset_kwargs={"data_files": str(path), "split": "train"},
    )
    rank_rows = []
    for rank in range(2):
        dataset = config.build(
            dp_rank=rank,
            dp_world_size=2,
        )
        rank_rows.append([row["id"] for row in dataset])

    assert set(rank_rows[0]).isdisjoint(rank_rows[1])
    assert set(rank_rows[0]) | set(rank_rows[1]) == set(range(10))

    iterator = iter(config.build(dp_rank=0, dp_world_size=1))
    next(iterator)
    state = iterator.get_state()
    expected = next(iterator)
    restored = iter(config.build(dp_rank=0, dp_world_size=1))
    restored.set_state(state)

    assert next(restored) == expected


def test_hf_cursor_restores_through_grain_wrappers(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"id": index} for index in range(10)])
    config = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="json",
            load_dataset_kwargs={"data_files": str(path), "split": "train"},
        ),
        filters=(lambda row: row["id"] % 2 == 0,),
    )
    policy = iteration(repeat=True, shuffle=True)
    iterator = iter(config.build(context=CONTEXT, iteration=policy))
    for _ in range(3):
        next(iterator)
    state = iterator.get_state()
    expected = [next(iterator) for _ in range(5)]

    restored = iter(config.build(context=CONTEXT, iteration=policy))
    restored.set_state(state)

    assert [next(restored) for _ in range(5)] == expected


def test_loader_requires_repeat_with_data_parallelism(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"id": index} for index in range(4)])
    config = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="json",
            load_dataset_kwargs={"data_files": str(path), "split": "train"},
        ),
        filters=(lambda row: row["id"] % 2 == 0,),
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

    def build(self, *, dp_rank: int, dp_world_size: int):
        del dp_rank, dp_world_size
        return self.rows


@dataclass(frozen=True)
class StreamingRowsSourceConfig:
    rows: tuple[dict, ...]

    def build(self, *, dp_rank: int, dp_world_size: int):
        dataset = grain.MapDataset.source(self.rows)
        dataset = dataset[dp_rank::dp_world_size]
        return dataset.to_iter_dataset()


def test_single_dataset_shuffle_shard_repeat_order():
    config = SingleDatasetConfig(
        source=RowsSourceConfig(rows=tuple({"value": index} for index in range(12)))
    )
    rank_0 = config.build(
        context=CONTEXT,
        iteration=iteration(shuffle=True, dp_world_size=2, dp_rank=0),
    )
    rank_1 = config.build(
        context=CONTEXT,
        iteration=iteration(shuffle=True, dp_world_size=2, dp_rank=1),
    )
    rank_0_peer = config.build(
        context=CONTEXT,
        iteration=iteration(shuffle=True, dp_world_size=2, dp_rank=0),
    )
    values_0 = {row["value"] for row in rank_0}
    values_1 = {row["value"] for row in rank_1}

    assert values_0.isdisjoint(values_1)
    assert values_0 | values_1 == set(range(12))
    assert list(rank_0) == list(rank_0_peer)
    assert [row["value"] for row in rank_0] != [0, 2, 4, 6, 8, 10]


def test_weighted_mix_keeps_weight_with_dataset():
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
    ).build(context=CONTEXT, iteration=iteration(repeat=True))
    values = [dataset[index]["source"] for index in range(12)]

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
    ).build(context=CONTEXT, iteration=iteration(repeat=True))
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
            iteration=iteration(dp_world_size=2, dp_rank=0),
        )
    )
    rank_1 = list(
        config.build(
            context=CONTEXT,
            iteration=iteration(dp_world_size=2, dp_rank=1),
        )
    )

    assert [row["value"] for row in rank_0] == [0, 2, 4, 6]
    assert [row["value"] for row in rank_1] == [1, 3, 5, 7]


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
    rows = recipe.build(context=CONTEXT, iteration=iteration())
    first_inputs, first_labels = next(iter(rows))

    assert first_inputs["input"].shape == (8,)
    assert first_inputs["positions"].shape == (8,)
    assert first_labels.shape == (8,)

    loader = GrainDataLoader.Config(
        dataset=recipe,
        shuffle=False,
        repeat=True,
        batch_prefetch_buffer_size=1,
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


class SftTokens(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del config, context

    def __call__(self, sample, rng):
        del sample, rng
        return TokenSequence(
            token_ids=np.asarray([1, 10, 11, 20, 21, 2]),
            loss_mask=np.asarray([False, False, False, True, True, True]),
        )


def test_sft_loss_mask_survives_packing():
    recipe = FirstFitPackingConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(rows=({"id": 0}, {"id": 1})),
            processor=SftTokens.Config(),
        )
    )
    _, labels = next(iter(recipe.build(context=CONTEXT, iteration=iteration())))

    assert labels[0].item() == IGNORE_INDEX
    assert labels[1].item() == IGNORE_INDEX
    assert labels[2].item() == 20


def test_chat_processor_masks_prompt_and_trains_assistant():
    def question_answer_to_messages(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    processor = ChatProcessor.Config(
        messages_fn=question_answer_to_messages,
    ).build(context=replace(CONTEXT, seq_len=64))
    token_sequence = processor(
        {"question": "2+2?", "answer": "4"},
        np.random.default_rng(0),
    )
    prompt = FakeTokenizer().apply_chat_template(
        [{"role": "user", "content": "2+2?"}],
        add_generation_prompt=True,
    )
    prompt_length = len(FakeTokenizer().encode(prompt, add_bos=True, add_eos=False))

    assert not token_sequence.loss_mask[1:prompt_length].any()
    assert token_sequence.loss_mask[prompt_length:].all()
    assert token_sequence.token_ids[-1] == FakeTokenizer.eos_id


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

    inputs, _ = next(iter(recipe.build(context=CONTEXT, iteration=iteration())))
    assert inputs["input"][0].item() == 1


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
    first = list(config.build(context=CONTEXT, iteration=iteration(seed=7)))
    second = list(config.build(context=CONTEXT, iteration=iteration(seed=7)))

    assert first == second


class RandomTokenSequence(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del config, context

    def __call__(self, sample, rng):
        random_token = int(rng.integers(10, 200))
        token_ids = np.asarray([1, sample["id"] + 10, random_token, 2])
        return TokenSequence(
            token_ids=token_ids,
            loss_mask=np.ones(token_ids.shape, dtype=np.bool_),
        )


def test_loader_restores_configured_random_map():
    config = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=SingleDatasetConfig(
                source=RowsSourceConfig(
                    rows=tuple({"id": index} for index in range(20))
                ),
                processor=RandomTokenSequence.Config(),
            )
        ),
        repeat=True,
        batch_prefetch_buffer_size=1,
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
        shuffle=True,
        repeat=True,
        batch_prefetch_buffer_size=2,
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
        repeat=True,
        batch_prefetch_buffer_size=1,
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
            iteration=iteration(dp_rank=1, dp_world_size=2),
        )


def test_concat_rejects_iterable_children(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"id": 0}])
    stream = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="json",
            load_dataset_kwargs={"data_files": str(path), "split": "train"},
        )
    )

    with pytest.raises(TypeError, match="map-style children"):
        DatasetConcatConfig(datasets=(stream,)).build(
            context=CONTEXT,
            iteration=iteration(),
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


def test_concat_then_split_keeps_long_document_next_token_pairs():
    recipe = ConcatThenSplitPackingConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(rows=({"tokens": list(range(10))},)),
            processor=RowToTokens.Config(),
        )
    )
    rows = list(
        recipe.build(
            context=replace(CONTEXT, seq_len=4),
            iteration=iteration(shuffle=False, repeat=False),
        )
    )

    assert rows[0][0]["input"].tolist() == [0, 1, 2, 3]
    assert rows[0][1].tolist() == [1, 2, 3, 4]
    assert rows[1][0]["input"].tolist() == [4, 5, 6, 7]
    assert rows[1][1].tolist() == [5, 6, 7, 8]
    assert rows[2][0]["input"].tolist() == [8, 0, 0, 0]
    assert rows[2][1].tolist() == [
        9,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]
    assert rows[0][0]["positions"].tolist() == [0, 1, 2, 3]
    assert rows[1][0]["positions"].tolist() == [0, 1, 2, 3]
    assert rows[2][0]["positions"].tolist() == [0, 0, 0, 0]


def test_map_dataset_reshuffles_deterministically_across_repeats():
    config = SingleDatasetConfig(
        source=RowsSourceConfig(rows=tuple({"value": index} for index in range(8)))
    )
    policy = iteration(shuffle=True, repeat=True)
    first = config.build(context=CONTEXT, iteration=policy)
    peer = config.build(context=CONTEXT, iteration=policy)
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
    inputs, labels = next(
        iter(
            recipe.build(
                context=replace(CONTEXT, seq_len=8),
                iteration=iteration(),
            )
        )
    )

    assert inputs["input"].tolist() == [1, 10, 1, 20, 21, 0, 0, 0]
    assert labels.tolist() == [
        10,
        2,
        20,
        21,
        2,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]
    assert inputs["positions"].tolist() == [0, 1, 0, 1, 2, 0, 0, 0]


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
    return GrainDataLoader.Config(
        dataset=SingleDatasetConfig(source=RowsSourceConfig(rows=rows)),
        shuffle=False,
        repeat=False,
        batch_prefetch_buffer_size=1,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=1,
        local_batch_size=2,
    )


def test_mix_rejects_empty_datasets():
    with pytest.raises(ValueError, match="positive-weight"):
        DatasetMixConfig(datasets=()).build(
            context=CONTEXT,
            iteration=iteration(),
        )


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_mix_rejects_nonpositive_weight(weight):
    dataset = SingleDatasetConfig(source=RowsSourceConfig(rows=({"value": 0},)))

    with pytest.raises(ValueError, match="positive-weight"):
        DatasetMixConfig(
            datasets=(WeightedDataset(dataset=dataset, weight=weight),)
        ).build(
            context=CONTEXT,
            iteration=iteration(),
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


def test_loader_batches_exact_rows_and_drops_remainder(finite_rows_loader):
    batches = list(finite_rows_loader)

    assert len(batches) == 2
    assert batches[0][0]["input"].tolist() == [[0], [1]]
    assert batches[0][1].tolist() == [[0], [1]]
    assert batches[1][0]["input"].tolist() == [[2], [3]]
    assert batches[1][1].tolist() == [[2], [3]]


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
        seed=42,
        shuffle=True,
        repeat=True,
        batch_prefetch_buffer_size=1,
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
