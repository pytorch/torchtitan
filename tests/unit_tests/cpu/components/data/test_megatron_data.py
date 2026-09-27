# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Megatron data port. The indexed-dataset, blend-parsing and builder tests are
ported from lm-engine's ``tests/megatron_data_test.py`` and
``tests/blended_megatron_dataset_builder_test.py``. No process group is
initialized here, so the code takes its single-process (rank 0) paths."""

import os
import pathlib
import pickle
import shutil
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.data.megatron import (
    build,
    compile_helpers,
    MegatronBatchSampler,
    MegatronDataLoader,
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
)
from torchtitan.components.data.megatron.bin import get_bin_path
from torchtitan.components.data.megatron.blended_dataset import BlendedDataset
from torchtitan.components.data.megatron.builder import (
    _get_blend_from_list,
    _get_num_samples_per_prefix,
    _get_sizes_for_blend,
    _resolve_idx_path,
)
from torchtitan.components.data.megatron.concatenated_dataset import ConcatenatedDataset
from torchtitan.components.data.megatron.gpt_dataset import GPTDataset
from torchtitan.components.data.megatron.indexed_dataset import get_idx_path
from torchtitan.components.data.megatron.utils import normalize

_DOC_STRIDE = 1000


@pytest.fixture(scope="module", autouse=True)
def _helpers():
    compile_helpers()


def _write(prefix: str, documents: list[np.ndarray]) -> None:
    builder = MMapIndexedDatasetBuilder(get_bin_path(prefix))
    for document in documents:
        builder.add_item(torch.tensor(document))
        builder.end_document()
    builder.finalize(get_idx_path(prefix))


def _repeated(num_documents: int, document: np.ndarray) -> list[np.ndarray]:
    return [document] * num_documents


def _numbered_documents(num_documents: int, length: int) -> list[np.ndarray]:
    """Document ``d`` holds ``d * _DOC_STRIDE + [0, length)``: every token names
    its document and position."""
    return [
        np.arange(length, dtype=np.int64) + d * _DOC_STRIDE
        for d in range(num_documents)
    ]


# ------------------------------------------------ indexed dataset (lm-engine)


def test_megatron_dataset_builder(tmp_path) -> None:
    prefix = str(tmp_path / "file")
    document = np.array([1, 2])
    _write(prefix, _repeated(1000, document))

    assert os.path.exists(get_bin_path(prefix))
    assert os.path.exists(get_idx_path(prefix))

    dataset = MMapIndexedDataset(prefix)
    assert len(dataset) == 1000
    for item in dataset:
        assert (item == document).all()


def test_megatron_dataset_merge(tmp_path) -> None:
    prefix1, prefix2 = str(tmp_path / "file1"), str(tmp_path / "file2")
    document1, document2 = np.array([1, 2]), np.array([3, 4, 5])
    _write(prefix1, _repeated(1000, document1))
    _write(prefix2, _repeated(2000, document2))

    prefix_merged = str(tmp_path / "merged")
    builder = MMapIndexedDatasetBuilder(get_bin_path(prefix_merged))
    builder.add_index(prefix1)
    builder.add_index(prefix2)
    builder.finalize(get_idx_path(prefix_merged))

    dataset = MMapIndexedDataset(prefix_merged)
    assert len(dataset) == 3000
    for i, item in enumerate(dataset):
        assert (item == (document1 if i < 1000 else document2)).all()


# ------------------------------------------------ builder (lm-engine)


@pytest.mark.parametrize(
    "blend, expected_prefixes, expected_weights",
    [
        (["path1"], ["path1"], None),
        (["path1", "path2"], ["path1", "path2"], None),
        (["path1", "path2", "path3"], ["path1", "path2", "path3"], None),
        (["  path1  ", "  path2  "], ["path1", "path2"], None),
        (["30", "path1", "70", "path2"], ["path1", "path2"], [30.0, 70.0]),
        (["30", " path1 ", "70", " path2 "], ["path1", "path2"], [30.0, 70.0]),
    ],
)
def test_get_blend_from_list(blend, expected_prefixes, expected_weights) -> None:
    prefixes, weights = _get_blend_from_list(blend)
    assert prefixes == expected_prefixes
    assert weights == expected_weights


def test_get_sizes_for_blend() -> None:
    sizes_per_dataset = _get_sizes_for_blend([0.5, 0.5], [100, 10, 0])
    expected = [int(np.ceil(100 * 0.5 * 1.005)), int(np.ceil(10 * 0.5 * 1.005)), 0]
    assert sizes_per_dataset == [expected, expected]


def test_resolve_idx_path_returns_none_for_local_prefix() -> None:
    assert (
        _resolve_idx_path(
            "/some/local/path", node_uses_local_storage=False, path_to_cache=None
        )
        is None
    )


def test_get_num_samples_per_prefix(tmp_path) -> None:
    prefix1, prefix2 = str(tmp_path / "dataset1"), str(tmp_path / "dataset2")
    _write(prefix1, _repeated(200, np.arange(8)))
    _write(prefix2, _repeated(400, np.arange(8)))

    assert _get_num_samples_per_prefix(
        [prefix1, prefix2], node_uses_local_storage=False, path_to_cache=None
    ) == [200, 400]


def _build_two(tmp_path, blend_of) -> tuple:
    prefix1, prefix2 = str(tmp_path / "dataset1"), str(tmp_path / "dataset2")
    _write(prefix1, _repeated(200, np.arange(8)))
    _write(prefix2, _repeated(400, np.arange(8)))
    return build(
        sizes=[50, 0, 0],
        sequence_length=4,
        blend=blend_of(prefix1, prefix2),
        split="100,0,0",
        path_to_cache=str(tmp_path / "cache"),
        tokenizer=None,
        node_uses_local_storage=False,
        random_seed=1234,
    )


def test_build_with_unweighted_blend_is_concatenated_by_num_samples(
    tmp_path,
) -> None:
    train, valid, test = _build_two(tmp_path, lambda p1, p2: [p1, p2])

    assert valid is None and test is None
    assert isinstance(train, ConcatenatedDataset)
    weights = normalize([200, 400])
    expected_sizes = [sizes[0] for sizes in _get_sizes_for_blend(weights, [50])]
    assert [len(dataset) for dataset in train.datasets] == expected_sizes
    assert len(train) == sum(expected_sizes)


def test_build_with_explicitly_weighted_blend_uses_blended_dataset(tmp_path) -> None:
    train, valid, test = _build_two(tmp_path, lambda p1, p2: ["30", p1, "70", p2])

    assert valid is None and test is None
    assert isinstance(train, BlendedDataset)
    assert train.weights == normalize([30.0, 70.0])


# ------------------------------------------------ GPTDataset


def _build_gpt(tmp_path, *, num_samples: int, sequence_length: int, seed: int = 1):
    prefix = str(tmp_path / "numbered")
    if not os.path.exists(get_idx_path(prefix)):
        _write(prefix, _numbered_documents(num_documents=40, length=7))
    (dataset, _, _) = build(
        sizes=[num_samples, 0, 0],
        sequence_length=sequence_length,
        blend=[prefix],
        split="100,0,0",
        path_to_cache=str(tmp_path / "cache"),
        tokenizer=None,
        node_uses_local_storage=False,
        random_seed=seed,
    )
    return dataset


def test_gpt_samples_are_contiguous_windows_over_documents(tmp_path) -> None:
    dataset = _build_gpt(tmp_path, num_samples=30, sequence_length=10)
    assert isinstance(dataset, GPTDataset)
    assert len(dataset) >= 30

    for i in range(len(dataset)):
        text = dataset[i]["text"]
        assert text.shape == (11,)
        # Within a document tokens step by 1; a new document starts at position 0.
        for prev, token in zip(text[:-1], text[1:]):
            assert token == prev + 1 or token % _DOC_STRIDE == 0


def test_gpt_index_cache_is_reused(tmp_path) -> None:
    first = _build_gpt(tmp_path, num_samples=30, sequence_length=10)
    cache_files = sorted(os.listdir(tmp_path / "cache"))
    assert any(name.endswith("-GPTDataset-shuffle_index.npy") for name in cache_files)

    second = _build_gpt(tmp_path, num_samples=30, sequence_length=10)
    assert sorted(os.listdir(tmp_path / "cache")) == cache_files
    np.testing.assert_array_equal(first.shuffle_index, second.shuffle_index)
    np.testing.assert_array_equal(first.sample_index, second.sample_index)


class _UnpicklableTokenizer:
    def __init__(self) -> None:
        self.hook = lambda: None


def test_gpt_dataset_without_fim_pickles_with_unpicklable_tokenizer(tmp_path) -> None:
    # DataLoader workers started with spawn/forkserver pickle the dataset.
    prefix = str(tmp_path / "numbered")
    _write(prefix, _numbered_documents(num_documents=40, length=7))
    (dataset, _, _) = build(
        sizes=[10, 0, 0],
        sequence_length=10,
        blend=[prefix],
        split="100,0,0",
        path_to_cache=str(tmp_path / "cache"),
        tokenizer=_UnpicklableTokenizer(),
        node_uses_local_storage=False,
        random_seed=1,
    )
    restored = pickle.loads(pickle.dumps(dataset))
    np.testing.assert_array_equal(restored[0]["text"], dataset[0]["text"])


# ------------------------------------------------ sampler


def test_sampler_shards_disjoint_global_batches() -> None:
    per_rank = [
        list(
            MegatronBatchSampler(
                total_samples=20,
                consumed_samples=0,
                micro_batch_size=3,
                num_replicas=2,
                rank=rank,
            )
        )
        for rank in range(2)
    ]
    # 20 samples, global batch 6 -> 3 full batches; the last 2 samples drop.
    assert per_rank[0] == [[0, 1, 2], [6, 7, 8], [12, 13, 14]]
    assert per_rank[1] == [[3, 4, 5], [9, 10, 11], [15, 16, 17]]


def test_sampler_resumes_from_consumed_samples() -> None:
    resumed = MegatronBatchSampler(
        total_samples=20, consumed_samples=6, micro_batch_size=3, num_replicas=2, rank=1
    )
    assert list(resumed) == [[9, 10, 11], [15, 16, 17]]


# ------------------------------------------------ MegatronDataLoader


def _loader(
    tmp_path,
    *,
    dp_world_size=1,
    dp_rank=0,
    max_context_length=8,
    num_tokens_per_batch=16,
    **overrides,
) -> MegatronDataLoader:
    prefix = str(tmp_path / "numbered")
    if not os.path.exists(get_idx_path(prefix)):
        _write(prefix, _numbered_documents(num_documents=40, length=7))
    fields = dict(
        data_paths=[prefix],
        sequence_length=8,
        num_samples=24,
        split="100,0,0",
        data_cache_path=str(tmp_path / "cache"),
        num_workers=0,
    )
    fields.update(overrides)
    return MegatronDataLoader.Config(**fields).build(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=None,
        max_context_length=max_context_length,
        num_tokens_per_batch=num_tokens_per_batch,
    )


def test_loader_yields_flat_trainer_batches(tmp_path) -> None:
    loader = _loader(tmp_path)
    batch = next(iter(loader))

    assert set(batch) == {
        "input",
        "labels",
        "positions",
        "padding_mask",
        "num_valid_tokens",
    }
    for key in ("input", "labels", "positions", "padding_mask"):
        assert batch[key].shape == (16,)
    # Labels are the inputs shifted by one within each sample.
    sample0 = loader._dataset[0]["text"]
    np.testing.assert_array_equal(batch["input"][:8].numpy(), sample0[:-1])
    np.testing.assert_array_equal(batch["labels"][:8].numpy(), sample0[1:])
    assert batch["positions"].tolist() == list(range(8)) * 2
    assert not batch["padding_mask"].any()
    assert batch["num_valid_tokens"] == 16


def test_loader_state_dict_resumes_at_next_batch(tmp_path) -> None:
    loader = _loader(tmp_path)
    iterator = iter(loader)
    for _ in range(3):
        next(iterator)
    state = loader.state_dict()
    expected = next(iterator)

    resumed = _loader(tmp_path)
    resumed.load_state_dict(state)
    actual = next(iter(resumed))
    torch.testing.assert_close(actual["input"], expected["input"])


def test_loader_without_repeat_stops(tmp_path) -> None:
    loader = _loader(tmp_path, repeat=False)
    batches = list(loader)
    # 2 samples per batch over the dataset's full global batches.
    assert len(batches) == len(loader._dataset) // 2
    assert list(loader) == []


def test_loader_ranks_read_disjoint_samples(tmp_path) -> None:
    rank0 = next(iter(_loader(tmp_path, dp_world_size=2, dp_rank=0)))
    rank1 = next(iter(_loader(tmp_path, dp_world_size=2, dp_rank=1)))
    assert not torch.equal(rank0["input"], rank1["input"])


def test_loader_rejects_indivisible_batch(tmp_path) -> None:
    with pytest.raises(ValueError):
        MegatronDataLoader.Config(
            data_paths=[str(tmp_path / "x")], sequence_length=8, num_samples=4
        ).build(
            dp_world_size=1,
            dp_rank=0,
            tokenizer=None,
            max_context_length=8,
            num_tokens_per_batch=12,
        )


def test_loader_config_validation() -> None:
    with pytest.raises(ValueError):
        MegatronDataLoader.Config(
            data_paths=["a", "b"], data_weights=[1.0], sequence_length=8, num_samples=4
        )
    with pytest.raises(ValueError):
        MegatronDataLoader.Config(
            data_paths=["a"], sequence_length=8, num_samples=4, fim_rate=1.5
        )


def _document_ids(tokens: torch.Tensor) -> set[int]:
    return set((tokens // _DOC_STRIDE).tolist())


def test_loader_weighted_blend_uses_blended_dataset(tmp_path) -> None:
    other = str(tmp_path / "other")
    _write(other, _numbered_documents(num_documents=40, length=7))
    loader = _loader(
        tmp_path,
        data_paths=[str(tmp_path / "numbered"), other],
        data_weights=[0.3, 0.7],
    )
    assert isinstance(loader._dataset, BlendedDataset)
    assert loader._dataset.weights == normalize([0.3, 0.7])
    assert next(iter(loader))["input"].shape == (16,)


def test_loader_valid_split_is_disjoint_from_train(tmp_path) -> None:
    def all_documents(split_name: str) -> set[int]:
        loader = _loader(
            tmp_path,
            split="50,50,0",
            split_name=split_name,
            num_samples=8,
            repeat=False,
        )
        return set().union(*(_document_ids(b["input"]) for b in loader))

    train, valid = all_documents("train"), all_documents("valid")
    assert train and valid
    # Splits partition the documents (split by sequence; one sequence per document).
    assert train.isdisjoint(valid)
    assert max(train) < min(valid)


def test_loader_empty_split_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="split no data"):
        _loader(tmp_path, split="100,0,0", split_name="valid")


def test_loader_repeat_wraps_to_first_batch(tmp_path) -> None:
    loader = _loader(tmp_path)
    num_batches = len(loader._dataset) // 2
    iterator = iter(loader)
    first = next(iterator)
    for _ in range(num_batches - 1):
        next(iterator)
    wrapped = next(iterator)

    torch.testing.assert_close(wrapped["input"], first["input"])
    assert loader.state_dict()["dp_rank_0"]["consumed_samples"] == 2


def test_loader_state_dict_is_per_rank(tmp_path) -> None:
    loader = _loader(tmp_path, dp_world_size=2, dp_rank=1)
    next(iter(loader))
    assert loader.state_dict() == {
        "version": 1,
        "dp_world_size": 2,
        # One global batch is 2 samples on each of 2 ranks.
        "dp_rank_1": {"consumed_samples": 4},
    }


def test_loader_load_empty_state_is_noop(tmp_path) -> None:
    loader = _loader(tmp_path)
    expected = next(iter(_loader(tmp_path)))
    loader.load_state_dict({})
    torch.testing.assert_close(next(iter(loader))["input"], expected["input"])


@pytest.mark.parametrize(
    "state",
    [
        {"version": 2, "dp_world_size": 1, "dp_rank_0": {"consumed_samples": 0}},
        {"version": 1, "dp_world_size": 4, "dp_rank_0": {"consumed_samples": 0}},
        {"version": 1, "dp_world_size": 1, "dp_rank_3": {"consumed_samples": 0}},
    ],
)
def test_loader_load_state_dict_rejects_mismatch(tmp_path, state) -> None:
    loader = _loader(tmp_path)
    with pytest.raises(ValueError):
        loader.load_state_dict(state)


def test_loader_rejects_sequence_longer_than_context(tmp_path) -> None:
    with pytest.raises(ValueError, match="exceeds max_context_length"):
        _loader(tmp_path, sequence_length=16, num_tokens_per_batch=32)


def test_loader_rejects_too_few_max_documents(tmp_path) -> None:
    # 16 tokens / sequence_length 8 = 2 samples (documents) per batch.
    with pytest.raises(ValueError, match="max_num_documents"):
        _loader(tmp_path, max_num_documents=1)
    assert _loader(tmp_path, max_num_documents=2).max_num_documents == 2


def test_loader_rejects_fewer_samples_than_one_global_batch(tmp_path) -> None:
    with pytest.raises(ValueError, match="fewer than one global batch"):
        # 40 documents * 7 tokens / 8 ~ 34 samples < one global batch of 64.
        _loader(tmp_path, num_tokens_per_batch=8 * 64, max_context_length=8)


def test_loader_close_keeps_position(tmp_path) -> None:
    loader = _loader(tmp_path)
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    loader.close()

    # close() drops the iterator but keeps consumed_samples, so iteration
    # resumes at the third batch.
    resumed = next(iter(loader))
    expected = list(_loader(tmp_path, repeat=False))[2]
    torch.testing.assert_close(resumed["input"], expected["input"])


# ------------------------------------------------ FIM
# ------------------------------------------------ FIM


class _CharTokenizer:
    """Maps characters to ``ord(c)``; FIM sentinels and EOS get small ids."""

    eos_id = 1
    _special = {"<fim_prefix>": 2, "<fim_middle>": 3, "<fim_suffix>": 4, "<fim_pad>": 5}

    def token_to_id(self, token: str) -> int | None:
        return self._special.get(token)

    def encode(self, text: str, *, add_bos: bool, add_eos: bool) -> list[int]:
        assert not add_bos and not add_eos
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return "".join(chr(t) for t in token_ids)


def _build_fim(tmp_path, *, fim_rate: float, fim_spm_rate: float):
    prefix = str(tmp_path / "text")
    documents = [
        np.array([ord(c) for c in f"document number {i:03d}"] + [1]) for i in range(30)
    ]
    _write(prefix, documents)
    (dataset, _, _) = build(
        sizes=[20, 0, 0],
        sequence_length=40,
        blend=[prefix],
        split="100,0,0",
        path_to_cache=str(tmp_path / f"cache-{fim_rate}-{fim_spm_rate}"),
        tokenizer=_CharTokenizer(),
        node_uses_local_storage=False,
        random_seed=3,
        fim_rate=fim_rate,
        fim_spm_rate=fim_spm_rate,
    )
    return dataset


def test_fim_psm_layout(tmp_path) -> None:
    dataset = _build_fim(tmp_path, fim_rate=1.0, fim_spm_rate=0.0)
    sample = dataset[0]["text"]
    assert sample.shape == (41,)
    # Every document segment after an EOS starts with <fim_prefix> (PSM).
    eos_positions = np.flatnonzero(sample == 1)
    for position in eos_positions:
        if position + 1 < sample.shape[0]:
            assert sample[position + 1] == 2
    assert {2, 3, 4} <= set(sample.tolist())


def test_fim_spm_layout(tmp_path) -> None:
    dataset = _build_fim(tmp_path, fim_rate=1.0, fim_spm_rate=1.0)
    sample = dataset[0]["text"]
    # SPM starts each full segment with <fim_prefix>, <fim_suffix>.
    eos_positions = np.flatnonzero(sample == 1)
    for position in eos_positions:
        if position + 2 < sample.shape[0]:
            assert sample[position + 1] == 2 and sample[position + 2] == 4


def test_fim_rate_zero_leaves_samples_unchanged(tmp_path) -> None:
    dataset = _build_fim(tmp_path, fim_rate=0.0, fim_spm_rate=0.5)
    sample = dataset[0]["text"]
    assert not {2, 3, 4, 5} & set(sample.tolist())


# ------------------------------------------------ multi-rank


class TestMegatronDataLoaderDistributed(DTensorTestBase):
    """Two gloo ranks: global rank 0 builds and caches the indices, then a
    barrier, then rank 1 loads them (lm-engine's build-then-load order)."""

    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_ranks_share_cache_and_split_global_batches(self):
        # Rank 0 picks a directory; everyone uses it.
        shared = [tempfile.mkdtemp() if self.rank == 0 else None]
        dist.broadcast_object_list(shared, src=0)
        tmp_path = pathlib.Path(shared[0])
        if self.rank == 0:
            _write(str(tmp_path / "numbered"), _numbered_documents(40, 7))
        dist.barrier()

        loader = _loader(tmp_path, dp_world_size=2, dp_rank=self.rank)
        batch = next(iter(loader))

        cache_files = os.listdir(tmp_path / "cache")
        # One set of index files, written once by rank 0.
        assert (
            sum(name.endswith("-GPTDataset-shuffle_index.npy") for name in cache_files)
            == 1
        )
        # The global batch is samples [0, 4): rank r gets [2r, 2r + 2).
        expected = torch.cat(
            [
                torch.from_numpy(loader._dataset[i]["text"][:-1])
                for i in range(2 * self.rank, 2 * self.rank + 2)
            ]
        )
        torch.testing.assert_close(batch["input"], expected)

        gathered = [None, None]
        dist.all_gather_object(gathered, batch["input"])
        assert not torch.equal(gathered[0], gathered[1])

        loader.close()
        dist.barrier()
        if self.rank == 0:
            shutil.rmtree(tmp_path)
