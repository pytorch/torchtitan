# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from array import array

import torch

from torchtitan.hf_datasets.pretokenized import PreTokenizedTextDataset


def _write_tokens(path, tokens: list[int]) -> None:
    token_array = array("I", tokens)
    with path.open("wb") as f:
        token_array.tofile(f)


def _write_dataset(path, tokens: list[int]) -> None:
    path.mkdir()
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
    path.mkdir()
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


def test_pretokenized_dataset_packs_contiguous_tokens(tmp_path):
    _write_dataset(tmp_path, list(range(10)))

    ds = PreTokenizedTextDataset(
        dataset_path=str(tmp_path),
        seq_len=4,
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


def test_pretokenized_dataset_shards_sequences_by_data_parallel_rank(tmp_path):
    _write_dataset(tmp_path, list(range(17)))

    ds = PreTokenizedTextDataset(
        dataset_path=str(tmp_path),
        seq_len=4,
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
