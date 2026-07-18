# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the composed Grain data pipeline."""

import json
from dataclasses import dataclass

import grain.python as grain
import numpy as np
import pytest
import torch

from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.dataset import (
    BuildOptions,
    ChatToTokenSequence,
    DataRuntime,
    DatasetConcatConfig,
    DatasetMixConfig,
    SampleProcessor,
    SingleDatasetConfig,
    TokenSequence,
    WeightedDataset,
)
from torchtitan.components.data.loader import _normalize_config, GrainDataLoader
from torchtitan.components.data.packing import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
)
from torchtitan.components.data.sources import (
    HuggingFaceStreamingSource,
    IndexedJsonlSource,
)
from torchtitan.components.loss import IGNORE_INDEX


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


RUNTIME = DataRuntime(
    tokenizer=FakeTokenizer(),
    seq_len=8,
    local_batch_size=2,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)


def options(**overrides):
    values = {
        "seed": 42,
        "shuffle": False,
        "repeat": False,
        "dp_rank": 0,
        "dp_world_size": 1,
    }
    return BuildOptions(**(values | overrides))


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows))


def row_to_tokens(row):
    token_ids = np.asarray(row["tokens"], dtype=np.int64)
    return TokenSequence(
        token_ids=token_ids,
        loss_mask=np.ones(token_ids.shape, dtype=np.bool_),
    )


def test_indexed_jsonl_random_access(tmp_path):
    write_jsonl(tmp_path / "b.jsonl", [{"id": 2}, {"id": 3}])
    write_jsonl(tmp_path / "a.jsonl", [{"id": 0}, {"id": 1}])
    source = IndexedJsonlSource.Config(patterns=(str(tmp_path / "*.jsonl"),)).build(
        runtime=RUNTIME, options=options()
    )

    assert len(source) == 4
    assert [source[index]["id"] for index in range(4)] == [0, 1, 2, 3]
    assert source[-1]["id"] == 3


def test_indexed_jsonl_rejects_missing_and_duplicate_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        IndexedJsonlSource.Config(patterns=(str(tmp_path / "missing*.jsonl"),)).build(
            runtime=RUNTIME, options=options()
        )

    write_jsonl(tmp_path / "rows.jsonl", [{"id": 0}])
    with pytest.raises(ValueError, match="more than once"):
        IndexedJsonlSource.Config(
            patterns=(
                str(tmp_path / "rows.jsonl"),
                str(tmp_path / "*.jsonl"),
            )
        ).build(runtime=RUNTIME, options=options())


def test_hugging_face_streaming_source_shards_and_restores(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"id": index} for index in range(10)])
    config = HuggingFaceStreamingSource.Config(
        path="json",
        load_dataset_kwargs={
            "data_files": str(path),
            "split": "train",
        },
    )
    rank_rows = []
    for rank in range(2):
        dataset = config.build(
            options=options(dp_rank=rank, dp_world_size=2),
        )
        rank_rows.append([row["id"] for row in dataset])

    assert set(rank_rows[0]).isdisjoint(rank_rows[1])
    assert set(rank_rows[0]) | set(rank_rows[1]) == set(range(10))

    iterator = iter(config.build(options=options()))
    next(iterator)
    state = iterator.get_state()
    expected = next(iterator)
    restored = iter(config.build(options=options()))
    restored.set_state(state)

    assert next(restored) == expected


def test_finite_filtered_stream_rejects_data_parallelism(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"id": index} for index in range(4)])
    config = SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="json",
            load_dataset_kwargs={
                "data_files": str(path),
                "split": "train",
            },
        ),
        filters=(lambda row: row["id"] % 2 == 0,),
    )

    with pytest.raises(ValueError, match="finite filtered"):
        config.build(
            runtime=RUNTIME,
            options=options(dp_world_size=2),
        )


@dataclass(frozen=True)
class RowsSourceConfig:
    rows: tuple[dict, ...]

    def build(self, **_):
        return self.rows


def test_single_dataset_shuffle_shard_repeat_order():
    config = SingleDatasetConfig(
        source=RowsSourceConfig(rows=tuple({"value": index} for index in range(12)))
    )
    rank_0 = config.build(
        runtime=RUNTIME,
        options=options(shuffle=True, dp_world_size=2, dp_rank=0),
    )
    rank_1 = config.build(
        runtime=RUNTIME,
        options=options(shuffle=True, dp_world_size=2, dp_rank=1),
    )
    rank_0_peer = config.build(
        runtime=RUNTIME,
        options=options(shuffle=True, dp_world_size=2, dp_rank=0),
    )
    values_0 = {row["value"] for row in rank_0}
    values_1 = {row["value"] for row in rank_1}

    assert values_0.isdisjoint(values_1)
    assert values_0 | values_1 == set(range(12))
    assert list(rank_0) == list(rank_0_peer)


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
    ).build(runtime=RUNTIME, options=options(repeat=True))
    values = [dataset[index]["source"] for index in range(12)]

    assert values.count("left") == 8
    assert values.count("right") == 4


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
            runtime=RUNTIME,
            options=options(dp_world_size=2, dp_rank=0),
        )
    )
    rank_1 = list(
        config.build(
            runtime=RUNTIME,
            options=options(dp_world_size=2, dp_rank=1),
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
        process=row_to_tokens,
    )
    recipe = packing_type(dataset=documents)
    rows = recipe.build(runtime=RUNTIME, options=options())
    first_inputs, first_labels = next(iter(rows))

    assert first_inputs["input"].shape == (8,)
    assert first_inputs["positions"].shape == (8,)
    assert first_labels.shape == (8,)

    loader = GrainDataLoader.Config(
        dataset=recipe,
        collator=TextCollator.Config(),
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


def test_sft_loss_mask_survives_packing():
    def sft_tokens(_row):
        return TokenSequence(
            token_ids=np.asarray([1, 10, 11, 20, 21, 2]),
            loss_mask=np.asarray([False, False, False, True, True, True]),
        )

    recipe = FirstFitPackingConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(rows=({"id": 0}, {"id": 1})),
            process=sft_tokens,
        )
    )
    _, labels = next(iter(recipe.build(runtime=RUNTIME, options=options())))

    assert labels[0].item() == IGNORE_INDEX
    assert labels[1].item() == IGNORE_INDEX
    assert labels[2].item() == 20


def test_chat_processor_masks_prompt_and_trains_assistant():
    def question_answer_to_messages(sample):
        return [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]

    processor = ChatToTokenSequence.Config(
        sample_to_messages=question_answer_to_messages,
    ).build(runtime=RUNTIME)
    token_sequence = processor(
        {"question": "2+2?", "answer": "4"},
        np.random.default_rng(0),
    )
    prompt = FakeTokenizer().apply_chat_template(
        [{"role": "user", "content": "2+2?"}],
        add_generation_prompt=True,
    )
    prompt_length = len(FakeTokenizer().encode(prompt, add_bos=True, add_eos=False))

    assert not token_sequence.loss_mask[:prompt_length].any()
    assert token_sequence.loss_mask[prompt_length:].all()
    assert token_sequence.token_ids[-1] == FakeTokenizer.eos_id


def test_chat_processor_rejects_non_single_turn_messages():
    processor = ChatToTokenSequence.Config(
        sample_to_messages=lambda sample: sample["messages"],
    ).build(runtime=RUNTIME)

    with pytest.raises(ValueError, match="one user and one assistant"):
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
            process=row_to_tokens,
        )
    )

    inputs, _ = next(iter(recipe.build(runtime=RUNTIME, options=options())))
    assert inputs["input"][0].item() == 1


class AddRandomOffset(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        maximum: int

    def __init__(self, config: Config, *, runtime: DataRuntime):
        del runtime
        self.maximum = config.maximum

    def __call__(self, sample, rng):
        return sample | {"offset": int(rng.integers(self.maximum))}


def test_configured_random_processor_is_deterministic():
    config = SingleDatasetConfig(
        source=RowsSourceConfig(rows=tuple({"id": index} for index in range(10))),
        process=AddRandomOffset.Config(maximum=1000),
    )
    first = list(config.build(runtime=RUNTIME, options=options(seed=7)))
    second = list(config.build(runtime=RUNTIME, options=options(seed=7)))

    assert first == second


class RandomTokenSequence(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, runtime: DataRuntime):
        del config, runtime

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
                process=RandomTokenSequence.Config(),
            )
        ),
        collator=TextCollator.Config(),
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


@dataclass(frozen=True)
class RecipeA:
    value: int


@dataclass(frozen=True)
class RecipeB:
    value: int


def test_normalized_config_keeps_recipe_type_identity():
    assert _normalize_config(RecipeA(value=1)) != _normalize_config(RecipeB(value=1))


def test_loader_exact_restore_with_nonempty_packing_buffers():
    recipe = ConcatThenSplitPackingConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(
                rows=tuple(
                    {"tokens": [1, index + 10, index + 11, 2]} for index in range(20)
                )
            ),
            process=row_to_tokens,
        )
    )
    config = GrainDataLoader.Config(
        dataset=recipe,
        collator=TextCollator.Config(),
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
                process=row_to_tokens,
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


@pytest.mark.parametrize(
    "packing_type",
    [ConcatThenSplitPackingConfig, FirstFitPackingConfig],
)
def test_finite_packing_rejects_data_parallelism(packing_type):
    recipe = packing_type(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(rows=({"tokens": [1, 10, 11, 2]},)),
            process=row_to_tokens,
        )
    )

    with pytest.raises(ValueError, match="finite packed"):
        recipe.build(
            runtime=RUNTIME,
            options=options(dp_world_size=2),
        )


def test_loader_rejects_pipeline_and_dp_changes():
    recipe = ConcatThenSplitPackingConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(
                rows=tuple({"tokens": [1, index, index + 1, 2]} for index in range(8))
            ),
            process=row_to_tokens,
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

    changed = GrainDataLoader.Config(
        dataset=recipe,
        collator=TextCollator.Config(),
        seed=999,
    ).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    with pytest.raises(ValueError, match="pipeline"):
        changed.load_state_dict(state)

    different_dp = config.build(
        dp_world_size=2,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=8,
        local_batch_size=2,
    )
    with pytest.raises(ValueError, match="data-parallel"):
        different_dp.load_state_dict(state)
