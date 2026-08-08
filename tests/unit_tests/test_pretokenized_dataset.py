# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from array import array

import pytest
import torch

import torchtitan.hf_datasets.pretokenized as pretokenized
from torchtitan.hf_datasets.pretokenized import PreTokenizedTextDataset


def _write_tokens(path, tokens: list[int]) -> None:
    token_array = array("I", tokens)
    with path.open("wb") as f:
        token_array.tofile(f)


def _write_dataset(path, tokens: list[int]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_tokens(path / "data.bin", tokens)
    with (path / "metadata.json").open("w") as f:
        json.dump(
            {
                "data_file": "data.bin",
                "dtype": "uint32",
                "num_tokens": len(tokens),
            },
            f,
        )


def _write_multi_bin_dataset(path, parts: list[list[int]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    data_files = []
    num_tokens = 0
    for idx, tokens in enumerate(parts):
        name = f"part-{idx:06d}.bin"
        _write_tokens(path / name, tokens)
        data_files.append({"data_file": name, "num_tokens": len(tokens)})
        num_tokens += len(tokens)
    with (path / "metadata.json").open("w") as f:
        json.dump(
            {
                "data_files": data_files,
                "dtype": "uint32",
                "num_tokens": num_tokens,
            },
            f,
        )


def _instances(dataset: PreTokenizedTextDataset) -> list[int]:
    """Instance index of every sample the dataset yields."""
    return [int(input_dict["input"][0]) // dataset.seq_len for input_dict, _ in dataset]


class _FakeWorkerInfo:
    def __init__(self, worker_id: int, num_workers: int):
        self.id = worker_id
        self.num_workers = num_workers


def _instances_for_worker(monkeypatch, worker_id, num_workers, **kwargs) -> list[int]:
    monkeypatch.setattr(
        pretokenized,
        "get_worker_info",
        lambda: _FakeWorkerInfo(worker_id, num_workers),
    )
    return _instances(PreTokenizedTextDataset(**kwargs))


def test_pretokenized_dataset_packs_contiguous_tokens(tmp_path):
    _write_dataset(tmp_path, list(range(10)))

    ds = PreTokenizedTextDataset(
        dataset_path=str(tmp_path),
        seq_len=4,
        local_batch_size=1,
        dp_rank=0,
        dp_world_size=1,
    )

    iterator = iter(ds)
    input_dict, labels = next(iterator)
    assert torch.equal(input_dict["input"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(labels, torch.tensor([1, 2, 3, 4]))
    assert torch.equal(input_dict["positions"], torch.tensor([0, 1, 2, 3]))

    input_dict, labels = next(iterator)
    assert torch.equal(input_dict["input"], torch.tensor([4, 5, 6, 7]))
    assert torch.equal(labels, torch.tensor([5, 6, 7, 8]))


def test_pretokenized_dataset_shards_instances_by_data_parallel_rank(tmp_path):
    _write_dataset(tmp_path, list(range(17)))

    ds = PreTokenizedTextDataset(
        dataset_path=str(tmp_path),
        seq_len=4,
        local_batch_size=1,
        dp_rank=1,
        dp_world_size=2,
    )

    input_dict, labels = next(iter(ds))
    assert torch.equal(input_dict["input"], torch.tensor([4, 5, 6, 7]))
    assert torch.equal(labels, torch.tensor([5, 6, 7, 8]))


def test_pretokenized_dataset_reads_across_bin_boundaries(tmp_path):
    _write_multi_bin_dataset(tmp_path, [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]])

    ds = PreTokenizedTextDataset(
        dataset_path=str(tmp_path),
        seq_len=4,
        local_batch_size=1,
        dp_rank=0,
        dp_world_size=1,
    )

    iterator = iter(ds)
    input_dict, labels = next(iterator)
    assert torch.equal(input_dict["input"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(labels, torch.tensor([1, 2, 3, 4]))

    input_dict, labels = next(iterator)
    assert torch.equal(input_dict["input"], torch.tensor([4, 5, 6, 7]))
    assert torch.equal(labels, torch.tensor([5, 6, 7, 8]))


def test_pretokenized_dataset_discovers_child_metadata(tmp_path):
    _write_dataset(tmp_path / "source-a", list(range(9)))
    _write_dataset(tmp_path / "source-b", list(range(9, 18)))

    ds = PreTokenizedTextDataset(
        dataset_path=str(tmp_path), seq_len=4, local_batch_size=1
    )

    assert ds.num_tokens == 18
    assert len(ds._token_files) == 2
    assert _instances(ds) == [0, 1, 2, 3]


def test_pretokenized_dataset_rejects_dataset_smaller_than_one_row(tmp_path):
    _write_dataset(tmp_path, list(range(33)))

    with pytest.raises(ValueError, match="At least 16 are required"):
        PreTokenizedTextDataset(
            dataset_path=str(tmp_path),
            seq_len=4,
            local_batch_size=4,
            dp_world_size=4,
        )


def test_pretokenized_dataset_matches_olmo_core_sharding(tmp_path):
    """The per-rank instance stream must equal OLMo-core's index math.

    OLMo-core reshapes the global order into rows of ``instances_per_batch``
    (one full global batch) and takes ``indices[:, dp_rank::dp_world_size]``.
    This dataset rows by ``dp_world_size * local_batch_size`` instead, because
    torchtitan's dataloader batches per worker at ``local_batch_size``; the two
    must produce the same per-rank sequence.
    """
    dp_world_size, local_batch_size, grad_accum = 4, 2, 3
    instances_per_batch = dp_world_size * local_batch_size * grad_accum
    _write_dataset(tmp_path, list(range(4 * 200 + 1)))

    kwargs = {
        "dataset_path": str(tmp_path),
        "seq_len": 4,
        "local_batch_size": local_batch_size,
        "dp_world_size": dp_world_size,
        "shuffle": True,
        "shuffle_seed": 7,
        "shuffle_strategy": "global",
    }

    for dp_rank in range(dp_world_size):
        ds = PreTokenizedTextDataset(dp_rank=dp_rank, **kwargs)

        # Rebuild OLMo-core's `_get_local_instance_indices` over the very same
        # permutation, so only the sharding math is under test.
        global_order = [ds._global_index_at(p, 0) for p in range(ds.num_sequences)]
        total_size = instances_per_batch * (len(global_order) // instances_per_batch)
        rows = [
            global_order[b : b + instances_per_batch]
            for b in range(0, total_size, instances_per_batch)
        ]
        olmo_core = [
            row[col]
            for row in rows
            for col in range(dp_rank, instances_per_batch, dp_world_size)
        ]

        ours = _instances(ds)
        assert ours[: len(olmo_core)] == olmo_core, dp_rank
        # Our tail rounds to a whole row rather than a whole global batch.
        assert 0 <= len(ours) - len(olmo_core) < instances_per_batch // local_batch_size


def test_pretokenized_dataset_shuffle_is_deterministic_and_exhaustive(tmp_path):
    _write_dataset(tmp_path, list(range(33)))
    kwargs = {
        "dataset_path": str(tmp_path),
        "seq_len": 4,
        "local_batch_size": 1,
        "shuffle": True,
        "shuffle_seed": 7,
        "shuffle_block_size": 2,
    }

    first_order = _instances(PreTokenizedTextDataset(**kwargs))
    second_order = _instances(PreTokenizedTextDataset(**kwargs))

    assert first_order == second_order
    assert first_order != list(range(8))
    assert sorted(first_order) == list(range(8))


def test_pretokenized_dataset_shuffle_has_no_dp_overlap(tmp_path):
    _write_dataset(tmp_path, list(range(33)))
    kwargs = {
        "dataset_path": str(tmp_path),
        "seq_len": 4,
        "local_batch_size": 2,
        "dp_world_size": 2,
        "shuffle": True,
        "shuffle_seed": 7,
        "shuffle_block_size": 2,
    }

    rank_0 = _instances(PreTokenizedTextDataset(dp_rank=0, **kwargs))
    rank_1 = _instances(PreTokenizedTextDataset(dp_rank=1, **kwargs))

    assert len(rank_0) == len(rank_1) == 4
    assert set(rank_0).isdisjoint(rank_1)
    assert sorted(rank_0 + rank_1) == list(range(8))


def test_pretokenized_dataset_drops_tail_that_does_not_fill_a_row(tmp_path):
    # 7 usable instances with rows of 2 x 2 = 4 leaves one whole row.
    _write_dataset(tmp_path, list(range(29)))
    kwargs = {
        "dataset_path": str(tmp_path),
        "seq_len": 4,
        "local_batch_size": 2,
        "dp_world_size": 2,
        "shuffle": True,
        "shuffle_seed": 7,
        "shuffle_strategy": "global",
    }

    rank_0 = _instances(PreTokenizedTextDataset(dp_rank=0, **kwargs))
    rank_1 = _instances(PreTokenizedTextDataset(dp_rank=1, **kwargs))

    combined = rank_0 + rank_1
    assert len(rank_0) == len(rank_1) == 2
    assert len(set(combined)) == 4
    assert set(combined).issubset(set(range(7)))


def test_pretokenized_dataset_shuffle_restores_position(tmp_path):
    _write_dataset(tmp_path, list(range(4 * 64 + 1)))
    kwargs = {
        "dataset_path": str(tmp_path),
        "seq_len": 4,
        "local_batch_size": 2,
        "dp_world_size": 2,
        "shuffle": True,
        "shuffle_seed": 7,
        "shuffle_block_size": 2,
    }
    # Stop mid-row and on a row boundary.
    for num_consumed in (3, 4):
        ds = PreTokenizedTextDataset(dp_rank=1, **kwargs)
        iterator = iter(ds)
        for _ in range(num_consumed):
            next(iterator)
        state = ds.state_dict()
        expected_remaining = [int(inputs["input"][0]) // 4 for inputs, _ in iterator]

        restored = PreTokenizedTextDataset(dp_rank=1, **kwargs)
        restored.load_state_dict(state)

        assert _instances(restored) == expected_remaining, num_consumed


def test_pretokenized_dataset_rejects_incompatible_checkpoint(tmp_path):
    _write_dataset(tmp_path, list(range(49)))
    ds = PreTokenizedTextDataset(
        dataset_path=str(tmp_path), seq_len=4, local_batch_size=1
    )

    for stale in ({"sequence_idx": 3, "epoch": 0}, {"stream_position": 3, "epoch": 0}):
        with pytest.raises(ValueError, match="earlier version"):
            ds.load_state_dict(stale)


def test_pretokenized_dataset_global_shuffle_is_deterministic_and_exhaustive(tmp_path):
    _write_dataset(tmp_path, list(range(65)))
    kwargs = {
        "dataset_path": str(tmp_path),
        "seq_len": 4,
        "local_batch_size": 1,
        "shuffle": True,
        "shuffle_seed": 7,
        "shuffle_strategy": "global",
    }

    first_order = _instances(PreTokenizedTextDataset(**kwargs))
    second_order = _instances(PreTokenizedTextDataset(**kwargs))

    assert first_order == second_order
    assert first_order != list(range(16))
    assert sorted(first_order) == list(range(16))


def test_pretokenized_dataset_global_shuffle_has_no_dp_overlap(tmp_path):
    _write_dataset(tmp_path, list(range(65)))
    kwargs = {
        "dataset_path": str(tmp_path),
        "seq_len": 4,
        "local_batch_size": 2,
        "dp_world_size": 2,
        "shuffle": True,
        "shuffle_seed": 7,
        "shuffle_strategy": "global",
    }

    rank_0 = _instances(PreTokenizedTextDataset(dp_rank=0, **kwargs))
    rank_1 = _instances(PreTokenizedTextDataset(dp_rank=1, **kwargs))

    assert set(rank_0).isdisjoint(rank_1)
    assert sorted(rank_0 + rank_1) == list(range(16))


def test_pretokenized_dataset_chunk_size_keeps_runs_together(tmp_path):
    """chunk_size > 1 must permute runs of instances, as OLMo-core does."""
    _write_dataset(tmp_path, list(range(4 * 256 + 1)))
    ds = PreTokenizedTextDataset(
        dataset_path=str(tmp_path),
        seq_len=4,
        local_batch_size=1,
        chunk_size=4,
        shuffle=True,
        shuffle_seed=7,
        shuffle_strategy="global",
    )

    order = [ds._global_index_at(p, 0) for p in range(ds.num_sequences // 4 * 4)]
    assert sorted(order) == list(range(len(order)))
    for start in range(0, len(order), 4):
        run = order[start : start + 4]
        assert run == list(range(run[0], run[0] + 4))
        assert run[0] % 4 == 0


def test_pretokenized_dataset_shards_rows_by_dataloader_worker(monkeypatch, tmp_path):
    _write_dataset(tmp_path, list(range(4 * 256 + 1)))
    kwargs = {
        "dataset_path": str(tmp_path),
        "seq_len": 4,
        "local_batch_size": 2,
        "dp_world_size": 2,
        "shuffle": True,
        "shuffle_seed": 7,
        "shuffle_strategy": "global",
    }

    single = {
        rank: _instances(PreTokenizedTextDataset(dp_rank=rank, **kwargs))
        for rank in range(2)
    }

    for num_workers in (2, 3, 4):
        combined = []
        for rank in range(2):
            for worker_id in range(num_workers):
                combined += _instances_for_worker(
                    monkeypatch, worker_id, num_workers, dp_rank=rank, **kwargs
                )
        assert len(combined) == len(set(combined)), num_workers
        assert set(combined) == set(single[0]) | set(single[1]), num_workers


def test_pretokenized_dataset_reader_threads_preserve_order(tmp_path):
    _write_dataset(tmp_path, list(range(4 * 256 + 1)))
    kwargs = {
        "dataset_path": str(tmp_path),
        "seq_len": 4,
        "local_batch_size": 2,
        "shuffle": True,
        "shuffle_seed": 7,
        "shuffle_strategy": "global",
    }

    serial = _instances(PreTokenizedTextDataset(num_threads=0, **kwargs))
    threaded = _instances(PreTokenizedTextDataset(num_threads=4, **kwargs))

    assert serial == threaded


def test_pretokenized_dataset_limits_open_files(tmp_path):
    _write_multi_bin_dataset(
        tmp_path,
        [list(range(3)), list(range(3, 7)), list(range(7, 10))],
    )
    ds = PreTokenizedTextDataset(
        dataset_path=str(tmp_path),
        seq_len=4,
        local_batch_size=1,
        num_threads=0,
        max_open_files=1,
    )

    list(ds)

    assert len(ds._open_fds) == 1


def test_pretokenized_dataset_order_is_independent_of_fd_cache_size(tmp_path):
    """`_borrow_fd` opens outside the lock; that must not perturb data order."""
    _write_multi_bin_dataset(
        tmp_path, [list(range(i * 64, (i + 1) * 64)) for i in range(16)]
    )
    kwargs = {
        "dataset_path": str(tmp_path),
        "seq_len": 8,
        "local_batch_size": 2,
        "dp_rank": 1,
        "dp_world_size": 4,
        "shuffle": True,
        "shuffle_seed": 7,
        "shuffle_strategy": "global",
    }

    reference = _instances(
        PreTokenizedTextDataset(num_threads=0, max_open_files=64, **kwargs)
    )

    assert reference
    # max_open_files=1 makes every read miss the cache and race on the open.
    assert (
        _instances(PreTokenizedTextDataset(num_threads=4, max_open_files=1, **kwargs))
        == reference
    )
    assert (
        _instances(PreTokenizedTextDataset(num_threads=4, max_open_files=4, **kwargs))
        == reference
    )


def test_pretokenized_dataset_racing_opens_read_correct_bytes(tmp_path):
    """Threads racing to open the same file must not cross descriptors."""
    num_files, tokens_per_file, seq_len = 16, 64, 8
    _write_multi_bin_dataset(
        tmp_path,
        [
            list(range(i * tokens_per_file, (i + 1) * tokens_per_file))
            for i in range(num_files)
        ],
    )
    ds = PreTokenizedTextDataset(
        dataset_path=str(tmp_path),
        seq_len=seq_len,
        local_batch_size=2,
        shuffle=True,
        shuffle_seed=7,
        shuffle_strategy="global",
        num_threads=4,
        max_open_files=1,
    )

    fd_dir = "/proc/self/fd"
    open_before = len(os.listdir(fd_dir)) if os.path.isdir(fd_dir) else None

    samples = list(ds)

    assert samples
    # Token value equals the token's absolute index, so a crossed descriptor
    # shows up as a discontinuity within a window.
    for input_dict, labels in samples:
        base = int(input_dict["input"][0])
        assert input_dict["input"].tolist() == list(range(base, base + seq_len))
        assert labels.tolist() == list(range(base + 1, base + 1 + seq_len))

    assert not any(ds._fd_inflight.values())
    if open_before is not None:
        # Every descriptor opened is either still cached or closed: the loser
        # of an open race has to close its duplicate. `_open_fds` itself can
        # sit above max_open_files here, because eviction skips descriptors a
        # concurrent read is still using and only runs when the cache grows.
        assert len(os.listdir(fd_dir)) - open_before == len(ds._open_fds)
