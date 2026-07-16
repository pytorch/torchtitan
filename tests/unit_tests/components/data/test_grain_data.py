# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the grain data components.

Covers the CPU half of the locked release checks (the dp=2 run is a separate script):
built-in JSONL+rewrite; generic custom source / mixer / packing fixtures proving the
extension seams; CTS+FirstFit alignment and state restore; fingerprint sensitivity;
empty-shard and resume guards; the trainer-batch contract.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import grain.python as grain
import numpy as np
import pytest
import torch

from torchtitan.components.data.dataset import (
    BuildOptions,
    concat,
    DataRuntime,
    MultiDatasetConfig,
    SingleDatasetConfig,
    TokenSample,
    weighted_interleave,
)
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
    packed_features_to_training_batch,
    PackedTokenDatasetConfig,
)
from torchtitan.components.data.sources import JsonlSourceConfig, PathRewrite
from torchtitan.components.loss import IGNORE_INDEX


# --- shared test doubles -------------------------------------------------------------


class FakeTokenizer:
    """Deterministic 'tokenizer': char codes; bos=1, eos=2."""

    bos_id = 1
    eos_id = 2

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [ord(c) % 250 + 10 for c in text]
        return [self.bos_id] * add_bos + ids + [self.eos_id] * add_eos


def text_row_to_token_sample(row, runtime):
    ids = np.asarray(runtime.tokenizer.encode(row["text"], add_bos=True, add_eos=True))
    return TokenSample(token_ids=ids, loss_mask=np.ones(ids.shape, dtype=np.bool_))


RUNTIME = DataRuntime(tokenizer=FakeTokenizer(), seq_len=16, local_batch_size=2)


def options(**overrides):
    defaults = dict(seed=42, shuffle=True, infinite=False, dp_rank=0, dp_world_size=1)
    return BuildOptions(**{**defaults, **overrides})


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows))


# --- check 1: built-in JSONL glob + regex rewrite ------------------------------------


def test_jsonl_glob_and_rewrite(tmp_path):
    real = tmp_path / "downloaded"
    real.mkdir()
    write_jsonl(real / "data_0.jsonl", [{"text": "aa"}, {"text": "bb"}])
    write_jsonl(real / "data_1.jsonl", [{"text": "cc"}])

    # the config names a producer root that only exists after the rewrite
    source = JsonlSourceConfig(
        patterns=("/producer/root/data_*.jsonl",),
        path_rewrites=(PathRewrite(pattern="^/producer/root", replacement=str(real)),),
    ).build()
    assert len(source) == 3
    assert [source[i]["text"] for i in range(3)] == ["aa", "bb", "cc"]


def test_jsonl_missing_pattern_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        JsonlSourceConfig(patterns=(str(tmp_path / "nope_*.jsonl"),)).build()


def test_duplicate_paths_fail(tmp_path):
    write_jsonl(tmp_path / "a.jsonl", [{"text": "x"}])
    with pytest.raises(ValueError, match="more than once"):
        JsonlSourceConfig(
            patterns=(str(tmp_path / "a.jsonl"), str(tmp_path / "*.jsonl"))
        ).build()


# --- check 2: generic custom mmap source composes without core changes ---------------


@dataclass(frozen=True)
class ExternalMemmapSourceConfig:
    """Test fixture standing in for any user-owned binary source (e.g. pretokenized)."""

    path: str
    num_rows: int
    row_len: int

    def build(self):
        return ExternalMemmapSource(self.path, self.num_rows, self.row_len)

    def fingerprint(self):
        return f"external:{Path(self.path).name}:{Path(self.path).stat().st_size}"


class ExternalMemmapSource:
    def __init__(self, path, num_rows, row_len):
        self._values = np.memmap(
            path, mode="r", dtype=np.int64, shape=(num_rows, row_len)
        )

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return {"input_ids": np.asarray(self._values[index])}


def make_memmap(tmp_path, num_rows=6, row_len=5):
    path = tmp_path / "tokens.bin"
    values = (
        np.arange(num_rows * row_len, dtype=np.int64).reshape(num_rows, row_len) + 10
    )
    values.tofile(path)
    return ExternalMemmapSourceConfig(
        path=str(path), num_rows=num_rows, row_len=row_len
    )


def pretokenized_row_to_sample(row):
    ids = row["input_ids"]
    return TokenSample(token_ids=ids, loss_mask=np.ones(ids.shape, dtype=np.bool_))


def test_custom_source_composes(tmp_path):
    dataset_config = SingleDatasetConfig(
        source=make_memmap(tmp_path),
        sample_processor=pretokenized_row_to_sample,
    )
    dataset = dataset_config.build(runtime=RUNTIME, options=options(shuffle=False))
    samples = list(dataset)
    assert len(samples) == 6
    assert all(isinstance(sample, TokenSample) for sample in samples)


# --- arity dispatch: optionals never receive the runtime ------------------------------


def test_process_arity_dispatch(tmp_path):
    write_jsonl(tmp_path / "rows.jsonl", [{"text": "ab"}])
    source = JsonlSourceConfig(patterns=(str(tmp_path / "rows.jsonl"),))

    def process_with_optional(row, suffix="!"):
        return row["text"] + suffix

    dataset = SingleDatasetConfig(
        source=source,
        sample_processor=process_with_optional,
    ).build(runtime=RUNTIME, options=options(shuffle=False))
    assert list(dataset) == ["ab!"]  # optional param did NOT get the runtime

    def process_with_runtime(row, runtime):
        return len(runtime.tokenizer.encode(row["text"]))

    dataset = SingleDatasetConfig(
        source=source,
        sample_processor=process_with_runtime,
    ).build(runtime=RUNTIME, options=options(shuffle=False))
    assert list(dataset) == [2]

    def bad_signature(row, runtime, extra):
        return row

    with pytest.raises(TypeError, match="must take"):
        SingleDatasetConfig(
            source=source,
            sample_processor=bad_signature,
        ).build(runtime=RUNTIME, options=options(shuffle=False))


# --- TorchTitan config logging --------------------------------------------------------


def test_grain_config_to_dict_is_json_serializable(tmp_path):
    config = GrainDataLoader.Config(
        dataset_config=PackedTokenDatasetConfig(
            dataset=SingleDatasetConfig(
                source=make_memmap(tmp_path),
                sample_processor=pretokenized_row_to_sample,
            ),
        ),
    )
    serialized = json.dumps(config.to_dict())
    assert "pretokenized_row_to_sample" in serialized
    assert "tokens.bin" in serialized


# --- check 3: built-in weighted mix + concat -----------------------------------------


def two_leaf_configs(tmp_path):
    write_jsonl(tmp_path / "a.jsonl", [{"text": f"a{i}"} for i in range(4)])
    write_jsonl(tmp_path / "b.jsonl", [{"text": f"b{i}"} for i in range(2)])
    make = lambda name: SingleDatasetConfig(
        source=JsonlSourceConfig(patterns=(str(tmp_path / name),)),
        sample_processor=text_row_to_token_sample,
    )
    return make("a.jsonl"), make("b.jsonl")


def test_weighted_mix_is_deterministic_interleave(tmp_path):
    ds_a, ds_b = two_leaf_configs(tmp_path)
    dataset = weighted_interleave([(ds_a, 1.0), (ds_b, 1.0)]).build(
        runtime=RUNTIME, options=options(shuffle=False)
    )
    first_ids = [sample.token_ids[1] for sample in dataset]
    assert first_ids[0] != first_ids[1]  # equal weights: strict a/b alternation


def test_concat_preserves_order(tmp_path):
    ds_a, ds_b = two_leaf_configs(tmp_path)
    dataset = concat([ds_a, ds_b]).build(
        runtime=RUNTIME, options=options(shuffle=False)
    )
    assert len(list(dataset)) == 6


# --- check 4: custom config-level mixer composes without core changes ----------------


def select_first_n_mixer(datasets, runtime, opts):
    """Fixture for metadata-aware mixing: builds each leaf, takes a finite prefix view."""
    del opts
    selected = [
        dataset.build_processed_dataset(runtime=runtime)[:2] for dataset in datasets
    ]
    return grain.MapDataset.concatenate(selected)


def test_custom_mixer_composes(tmp_path):
    ds_a, ds_b = two_leaf_configs(tmp_path)
    dataset = MultiDatasetConfig(
        datasets=(ds_a, ds_b),
        combine_fn=select_first_n_mixer,
    ).build(runtime=RUNTIME, options=options(shuffle=False))
    assert len(list(dataset)) == 4  # 2 from each leaf: the mixer's selection stuck


# --- fingerprints: every semantic change registers ------------------------------------


def test_fingerprint_sensitivity(tmp_path):
    ds_a, ds_b = two_leaf_configs(tmp_path)

    # mix weights
    assert (
        weighted_interleave([(ds_a, 1.0), (ds_b, 1.0)]).fingerprint()
        != weighted_interleave([(ds_a, 1.0), (ds_b, 3.0)]).fingerprint()
    )

    # packing algorithm
    recipe_cts = PackedTokenDatasetConfig(dataset=ds_a)
    recipe_ff = PackedTokenDatasetConfig(dataset=ds_a, packing=FirstFitPackingConfig())
    assert recipe_cts.fingerprint() != recipe_ff.fingerprint()

    # configured custom packer values
    @dataclass(frozen=True)
    class ConfiguredPacking:
        max_documents: int

        def build(self, parent, *, runtime, options):
            return parent

        def fingerprint(self):
            return f"{type(self).__qualname__}:{self.max_documents}"

    assert (
        PackedTokenDatasetConfig(
            dataset=ds_a, packing=ConfiguredPacking(2)
        ).fingerprint()
        != PackedTokenDatasetConfig(
            dataset=ds_a, packing=ConfiguredPacking(3)
        ).fingerprint()
    )

    # distinct same-file lambdas (line number disambiguates)
    with_filter_a = SingleDatasetConfig(
        source=ds_a.source,
        sample_filters=(lambda sample: True,),
    )
    with_filter_b = SingleDatasetConfig(
        source=ds_a.source,
        sample_filters=(lambda sample: False,),
    )
    assert with_filter_a.fingerprint() != with_filter_b.fingerprint()

    # configured callable without fingerprint() is rejected
    @dataclass(frozen=True)
    class OpaqueMixer:
        limit: int

        def __call__(self, datasets, runtime, opts):
            return datasets[0].build_processed_dataset(runtime=runtime)

    with pytest.raises(TypeError, match="must implement fingerprint"):
        MultiDatasetConfig(
            datasets=(ds_a,),
            combine_fn=OpaqueMixer(1),
        ).fingerprint()


# --- check 5: CTS + FirstFit alignment and state restore ------------------------------


def packed_recipe(tmp_path, packing):
    return PackedTokenDatasetConfig(
        dataset=SingleDatasetConfig(
            source=make_memmap(tmp_path),
            sample_processor=pretokenized_row_to_sample,
        ),
        packing=packing,
    )


PACKERS = [ConcatThenSplitPackingConfig(), FirstFitPackingConfig()]


@pytest.mark.parametrize("packing", PACKERS)
def test_packing_alignment_and_trainer_contract(tmp_path, packing):
    runtime = DataRuntime(tokenizer=FakeTokenizer(), seq_len=8, local_batch_size=2)
    recipe = packed_recipe(tmp_path, packing)
    inputs, labels = next(
        iter(recipe.build(runtime=runtime, options=options(shuffle=False)))
    )

    # trainer contract: exactly the supported kwargs, long dtype
    assert set(inputs) == {"input", "positions"}
    assert all(t.dtype == torch.long for t in [*inputs.values(), labels])

    # alignment: every trainable label is the next input token (memmap docs are arange)
    trainable = labels != IGNORE_INDEX
    assert torch.equal(labels[trainable], inputs["input"][trainable] + 1)

    # a debugmodel-shaped forward accepts the extra kwargs
    class DebugModel:
        def __call__(self, tokens, positions=None, attention_masks=None):
            return tokens

    DebugModel()(inputs["input"], **{k: v for k, v in inputs.items() if k != "input"})


@pytest.mark.parametrize("packing", PACKERS)
def test_packing_state_restore(tmp_path, packing):
    runtime = DataRuntime(tokenizer=FakeTokenizer(), seq_len=8, local_batch_size=1)
    build = lambda: iter(
        packed_recipe(tmp_path, packing).build(
            runtime=runtime, options=options(shuffle=False)
        )
    )
    iterator = build()
    next(iterator)
    state = iterator.get_state()
    expected_inputs, expected_labels = next(iterator)

    restored = build()
    restored.set_state(state)
    actual_inputs, actual_labels = next(restored)
    assert torch.equal(expected_inputs["input"], actual_inputs["input"])
    assert torch.equal(expected_labels, actual_labels)


def test_cts_labels_must_not_be_meta_feature():
    """Regression pin: meta features never split, silently dropping documents."""
    from grain import experimental as grain_experimental

    docs = [
        {
            "input_ids": np.arange(5, dtype=np.int64),
            "labels": np.arange(5, dtype=np.int64),
        }
        for _ in range(2)
    ]
    parent = grain.MapDataset.source(docs).to_iter_dataset()
    packed = grain_experimental.ConcatThenSplitIterDataset(
        parent,
        length_struct={"input_ids": 8, "labels": 8},
        meta_features=("labels",),
    )
    rows = list(packed)
    # with meta labels, the second document does NOT share the first row: packing degrades
    assert np.count_nonzero(rows[0]["input_ids_segment_ids"]) < 8


# --- check 6: custom packing config composes without core changes --------------------


@dataclass(frozen=True)
class TruncatePackingConfig:
    """Fixture packer: one doc per row, truncated/padded to seq_len."""

    def build(self, parent, *, runtime, options):
        del options
        seq_len = runtime.seq_len

        def truncate(features):
            ids = np.zeros(seq_len, dtype=np.int64)
            labels = np.full(seq_len, IGNORE_INDEX, dtype=np.int64)
            n = min(len(features["input_ids"]), seq_len)
            ids[:n] = features["input_ids"][:n]
            labels[:n] = features["labels"][:n]
            return {
                "input_ids": ids,
                "labels": labels,
                "input_ids_positions": np.arange(seq_len),
                "input_ids_segment_ids": (ids != 0).astype(np.int32),
            }

        return parent.map(truncate)

    def fingerprint(self):
        return type(self).__qualname__


def test_custom_packing_composes(tmp_path):
    runtime = DataRuntime(tokenizer=FakeTokenizer(), seq_len=8, local_batch_size=1)
    inputs, _ = next(
        iter(
            packed_recipe(tmp_path, TruncatePackingConfig()).build(
                runtime=runtime, options=options(shuffle=False)
            )
        )
    )
    assert inputs["input"].shape[-1] == 8


# --- check 7: batch conversion contract ------------------------------------------------


def test_packed_features_to_training_batch_contract():
    features = {
        "input_ids": np.array([[5, 6, 9, 0]]),
        "labels": np.array([[6, 7, 10, 0]]),
        "input_ids_positions": np.array([[0, 1, 0, 0]]),
        "input_ids_segment_ids": np.array([[1, 1, 2, 0]]),
    }
    inputs, labels = packed_features_to_training_batch(features)
    assert set(inputs) == {"input", "positions"}  # nothing extra reaches model kwargs
    assert labels[0, 3] == IGNORE_INDEX  # padding masked via segment 0


# --- check 8 (CPU half): DP sharding disjoint + exhaustive, guards, resume ------------


def test_dp_shards_disjoint_and_exhaustive(tmp_path):
    write_jsonl(
        tmp_path / "rows.jsonl", [{"text": f"row{i}", "id": i} for i in range(21)]
    )
    config = SingleDatasetConfig(
        source=JsonlSourceConfig(patterns=(str(tmp_path / "rows.jsonl"),)),
        sample_processor=lambda row: row["id"],
    )
    rank_rows = [
        set(
            config.build(
                runtime=RUNTIME,
                options=options(shuffle=True, dp_rank=rank, dp_world_size=2),
            )
        )
        for rank in (0, 1)
    ]
    assert rank_rows[0].isdisjoint(rank_rows[1])
    assert rank_rows[0] | rank_rows[1] == set(range(21))


def test_empty_shard_guard(tmp_path):
    write_jsonl(tmp_path / "one.jsonl", [{"text": "only"}])
    config = SingleDatasetConfig(
        source=JsonlSourceConfig(patterns=(str(tmp_path / "one.jsonl"),))
    )
    with pytest.raises(ValueError, match="fewer than dp_world_size"):
        config.build(runtime=RUNTIME, options=options(dp_rank=1, dp_world_size=2))


def loader_for_rank(
    tmp_path,
    dp_rank,
    dp_world_size,
    *,
    seed=42,
    shuffle=True,
    seq_len=16,
    local_batch_size=2,
):
    if not (tmp_path / "docs.jsonl").exists():
        write_jsonl(
            tmp_path / "docs.jsonl",
            [{"text": f"document number {i}"} for i in range(50)],
        )
    config = GrainDataLoader.Config(
        dataset_config=PackedTokenDatasetConfig(
            dataset=SingleDatasetConfig(
                source=JsonlSourceConfig(patterns=(str(tmp_path / "docs.jsonl"),)),
                sample_processor=text_row_to_token_sample,
            ),
        ),
        seed=seed,
        shuffle=shuffle,
        prefetch_buffer_size=8,
    )
    return GrainDataLoader(
        config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=FakeTokenizer(),
        seq_len=seq_len,
        local_batch_size=local_batch_size,
    )


def test_loader_rank_streams_differ_and_are_deterministic(tmp_path):
    def first_tokens(rank):
        inputs, _ = next(iter(loader_for_rank(tmp_path, rank, dp_world_size=2)))
        return inputs["input"].flatten().tolist()

    rank0, rank1 = first_tokens(0), first_tokens(1)
    assert rank0 != rank1
    assert first_tokens(0) == rank0  # same seed, same stream


def test_resume_round_trip(tmp_path):
    loader = loader_for_rank(tmp_path, dp_rank=0, dp_world_size=1)
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    expected, _ = next(iterator)

    resumed = loader_for_rank(tmp_path, dp_rank=0, dp_world_size=1)
    resumed.load_state_dict(state)
    actual, _ = next(iter(resumed))
    assert torch.equal(expected["input"], actual["input"])


def test_resume_guards(tmp_path):
    loader = loader_for_rank(tmp_path, dp_rank=0, dp_world_size=1)
    state = loader.state_dict()

    fresh = loader_for_rank(tmp_path, dp_rank=0, dp_world_size=1)
    fresh.load_state_dict({})  # empty state is valid (BaseDataLoader contract)

    with pytest.raises(ValueError, match="dp_world_size"):
        fresh.load_state_dict({**state, "dp_world_size": 2})

    with pytest.raises(ValueError, match="fingerprint"):
        fresh.load_state_dict({**state, "pipeline_fingerprint": "not-the-same"})

    with pytest.raises(ValueError, match="version"):
        fresh.load_state_dict({**state, "version": 99})

    rank_state = state.pop("dp_rank_0")
    with pytest.raises(ValueError, match="missing dataloader state"):
        fresh.load_state_dict(state)
    state["dp_rank_0"] = rank_state

    changed_seed = loader_for_rank(tmp_path, dp_rank=0, dp_world_size=1, seed=43)
    with pytest.raises(ValueError, match="fingerprint"):
        changed_seed.load_state_dict(state)

    changed_shuffle = loader_for_rank(
        tmp_path, dp_rank=0, dp_world_size=1, shuffle=False
    )
    with pytest.raises(ValueError, match="fingerprint"):
        changed_shuffle.load_state_dict(state)

    changed_shape = loader_for_rank(
        tmp_path,
        dp_rank=0,
        dp_world_size=1,
        seq_len=8,
        local_batch_size=1,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        changed_shape.load_state_dict(state)

    # a real data change flips the fingerprint through the whole recipe
    with open(loader_docs_path(tmp_path), "a") as handle:
        handle.write('\n{"text": "appended row"}')
    changed = loader_for_rank(tmp_path, dp_rank=0, dp_world_size=1)
    with pytest.raises(ValueError, match="fingerprint"):
        changed.load_state_dict(state)


def loader_docs_path(tmp_path):
    return tmp_path / "docs.jsonl"


# --- SFT, iterable packing, and non-text extension seams -------------------------------


@dataclass(frozen=True)
class RowsSourceConfig:
    rows: tuple
    identity: str = "rows"

    def build(self):
        return self.rows

    def fingerprint(self):
        return f"{self.identity}:{len(self.rows)}"


def sft_row_to_token_sample(row):
    token_ids = np.asarray([*row["prompt_ids"], *row["response_ids"]])
    loss_mask = np.asarray(
        [False] * len(row["prompt_ids"]) + [True] * len(row["response_ids"]),
        dtype=np.bool_,
    )
    return TokenSample(token_ids=token_ids, loss_mask=loss_mask)


def fits_sequence(sample, runtime):
    return len(sample.token_ids) <= runtime.seq_len + 1


def test_sft_masks_and_whole_example_packing():
    rows = (
        {"prompt_ids": [10, 11], "response_ids": [20, 21]},
        {"prompt_ids": [30, 31], "response_ids": [40, 41]},
        {"prompt_ids": list(range(20)), "response_ids": [99]},
    )
    recipe = PackedTokenDatasetConfig(
        dataset=SingleDatasetConfig(
            source=RowsSourceConfig(rows, identity="sft"),
            sample_processor=sft_row_to_token_sample,
            sample_filters=(fits_sequence,),
        ),
        packing=FirstFitPackingConfig(),
    )
    runtime = DataRuntime(seq_len=8, local_batch_size=1)
    inputs, labels = next(
        iter(recipe.build(runtime=runtime, options=options(shuffle=False)))
    )

    assert inputs["input"].tolist() == [[10, 11, 20, 30, 31, 40, 0, 0]]
    assert inputs["positions"].tolist() == [[0, 1, 2, 0, 1, 2, 0, 0]]
    assert labels.tolist() == [
        [IGNORE_INDEX, 20, 21, IGNORE_INDEX, 40, 41, IGNORE_INDEX, IGNORE_INDEX]
    ]


@dataclass(frozen=True)
class StreamingTokenDatasetConfig:
    samples: tuple[TokenSample, ...]

    def build(self, *, runtime, options):
        del runtime, options
        return grain.MapDataset.source(self.samples).to_iter_dataset()

    def fingerprint(self):
        return f"stream:{len(self.samples)}"


def test_iter_dataset_reuses_standard_packing():
    samples = tuple(
        TokenSample(
            token_ids=np.arange(start, start + 5),
            loss_mask=np.ones(5, dtype=np.bool_),
        )
        for start in (10, 20, 30)
    )
    recipe = PackedTokenDatasetConfig(
        dataset=StreamingTokenDatasetConfig(samples=samples)
    )
    runtime = DataRuntime(seq_len=8, local_batch_size=1)
    iterator = iter(recipe.build(runtime=runtime, options=options(shuffle=False)))
    next(iterator)
    state = iterator.get_state()
    expected_inputs, expected_labels = next(iterator)

    restored = iter(recipe.build(runtime=runtime, options=options(shuffle=False)))
    restored.set_state(state)
    actual_inputs, actual_labels = next(restored)
    assert torch.equal(actual_inputs["input"], expected_inputs["input"])
    assert torch.equal(actual_labels, expected_labels)


@dataclass(frozen=True)
class ImageBatchDatasetConfig:
    def build(self, *, runtime, options):
        del runtime, options
        batches = tuple(
            (
                {
                    "input": torch.tensor([[value, value + 1, value + 2]]),
                    "positions": torch.tensor([[0, 1, 2]]),
                    "pixel_values": torch.full((1, 3, 8, 8), value),
                    "grid_thw": torch.tensor([[1, 2, 2]]),
                },
                torch.tensor([[value + 1, value + 2, IGNORE_INDEX]]),
            )
            for value in (1, 5)
        )
        return grain.MapDataset.source(batches).repeat().to_iter_dataset()

    def fingerprint(self):
        return type(self).__qualname__


def test_image_shaped_custom_recipe_through_loader_resume():
    config = GrainDataLoader.Config(
        dataset_config=ImageBatchDatasetConfig(),
        prefetch_buffer_size=0,
    )
    loader = GrainDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=3,
        local_batch_size=1,
    )
    inputs, labels = next(iter(loader))
    assert inputs["pixel_values"].shape == (1, 3, 8, 8)
    assert inputs["grid_thw"].tolist() == [[1, 2, 2]]

    state = loader.state_dict()
    expected_inputs, expected_labels = next(iter(loader))
    assert not torch.equal(expected_inputs["pixel_values"], inputs["pixel_values"])

    restored = GrainDataLoader(
        config,
        dp_world_size=1,
        dp_rank=0,
        tokenizer=FakeTokenizer(),
        seq_len=3,
        local_batch_size=1,
    )
    restored.load_state_dict(state)
    restored_inputs, restored_labels = next(iter(restored))
    assert torch.equal(restored_inputs["pixel_values"], expected_inputs["pixel_values"])
    assert torch.equal(restored_labels, expected_labels)


# --- OLMo-style source selection must preserve processing and filters -----------------


@dataclass(frozen=True)
class SelectablePretokenizedSourceConfig:
    rows: tuple[dict, ...]
    selected_documents: int | None = None

    def build(self):
        count = self.selected_documents or len(self.rows)
        return self.rows[:count]

    def select_config(self, *, target_tokens, seed):
        del seed
        selected = 0
        documents = 0
        for row in self.rows:
            if selected >= target_tokens:
                break
            selected += len(row["input_ids"])
            documents += 1
        return replace(self, selected_documents=documents)

    def fingerprint(self):
        return f"selectable:{len(self.rows)}:{self.selected_documents}"


def nonempty_token_sample(sample):
    return len(sample.token_ids) > 1


@dataclass(frozen=True)
class TokenBudgetCombine:
    targets: tuple[int, ...]

    def __call__(self, datasets, runtime, opts):
        selected = []
        for source_index, (dataset, target) in enumerate(zip(datasets, self.targets)):
            selected_source = dataset.source.select_config(
                target_tokens=target,
                seed=opts.seed + source_index,
            )
            selected_dataset = replace(dataset, source=selected_source)
            selected.append(selected_dataset.build_processed_dataset(runtime=runtime))
        return grain.MapDataset.concatenate(selected)

    def fingerprint(self):
        return f"{type(self).__qualname__}:{self.targets}"


def test_olmo_style_selection_runs_processing_filters_and_packing():
    source_a = SelectablePretokenizedSourceConfig(
        rows=(
            {"input_ids": np.asarray([10, 11, 12])},
            {"input_ids": np.asarray([13])},
            {"input_ids": np.asarray([14, 15, 16])},
        )
    )
    source_b = SelectablePretokenizedSourceConfig(
        rows=(
            {"input_ids": np.asarray([20, 21, 22])},
            {"input_ids": np.asarray([23, 24, 25])},
        )
    )
    leaves = tuple(
        SingleDatasetConfig(
            source=source,
            sample_processor=pretokenized_row_to_sample,
            sample_filters=(nonempty_token_sample,),
        )
        for source in (source_a, source_b)
    )
    recipe = PackedTokenDatasetConfig(
        dataset=MultiDatasetConfig(
            datasets=leaves,
            combine_fn=TokenBudgetCombine(targets=(4, 3)),
        )
    )
    runtime = DataRuntime(seq_len=4, local_batch_size=1)
    batches = list(recipe.build(runtime=runtime, options=options(shuffle=False)))

    assert batches
    assert all(isinstance(inputs["input"], torch.Tensor) for inputs, _ in batches)
    trainable_labels = torch.cat(
        [labels[labels != IGNORE_INDEX] for _, labels in batches]
    ).tolist()
    assert set(trainable_labels).issubset({11, 12, 15, 16, 21, 22})
