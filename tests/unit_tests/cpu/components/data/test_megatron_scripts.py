# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the Megatron data tools in ``scripts/data``."""

import gzip
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from scripts.data import megatron_merge, megatron_preprocess
from scripts.data.megatron_merge import get_groups_by_sizes, merge_files
from scripts.data.megatron_preprocess import collect_files, convert_file

from torchtitan.components.data.megatron import (
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
)
from torchtitan.components.data.megatron.bin import get_bin_path
from torchtitan.components.data.megatron.indexed_dataset import get_idx_path

_EOS = 1


class _CharTokenizer:
    """``ord(c)`` per character; a vocab under 65500 selects uint16 storage."""

    eos_id = _EOS

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def get_vocab_size(self) -> int:
        return 256


def _tokens(text: str) -> list[int]:
    return [ord(c) for c in text]


def _documents(prefix: str) -> list[list[int]]:
    return [item.tolist() for item in MMapIndexedDataset(prefix)]


def _write(prefix: str, documents: list[list[int]]) -> None:
    builder = MMapIndexedDatasetBuilder(get_bin_path(prefix))
    for document in documents:
        builder.add_item(torch.tensor(document))
        builder.end_document()
    builder.finalize(get_idx_path(prefix))


# ------------------------------------------------ megatron_preprocess


def test_convert_jsonl_appends_eod_and_skips_bad_lines(tmp_path) -> None:
    input_file = tmp_path / "data.jsonl"
    input_file.write_text(
        json.dumps({"text": "ab"})
        + "\n"
        + "not json\n"
        + json.dumps({"text": "cde"})
        + "\n"
    )
    output_prefix = str(tmp_path / "out" / "data")

    skipped = convert_file(
        _CharTokenizer(), str(input_file), output_prefix, append_eos_token=True
    )

    assert skipped == 1
    assert _documents(f"{output_prefix}_text") == [
        _tokens("ab") + [_EOS],
        _tokens("cde") + [_EOS],
    ]
    assert MMapIndexedDataset(f"{output_prefix}_text").index.dtype == np.uint16


def test_convert_writes_one_dataset_per_json_key(tmp_path) -> None:
    input_file = tmp_path / "data.jsonl"
    input_file.write_text(json.dumps({"question": "q", "answer": "aa"}) + "\n")
    output_prefix = str(tmp_path / "data")

    convert_file(
        _CharTokenizer(),
        str(input_file),
        output_prefix,
        json_keys=["question", "answer"],
        append_eos_token=False,
    )

    assert _documents(f"{output_prefix}_question") == [_tokens("q")]
    assert _documents(f"{output_prefix}_answer") == [_tokens("aa")]


def test_convert_empty_document_is_dropped(tmp_path) -> None:
    input_file = tmp_path / "data.jsonl"
    input_file.write_text(
        json.dumps({"text": ""}) + "\n" + json.dumps({"text": "x"}) + "\n"
    )
    output_prefix = str(tmp_path / "data")

    convert_file(_CharTokenizer(), str(input_file), output_prefix)

    assert _documents(f"{output_prefix}_text") == [_tokens("x") + [_EOS]]


def test_convert_json_gz(tmp_path) -> None:
    input_file = tmp_path / "data.json.gz"
    with gzip.open(input_file, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"text": "gz"}) + "\n")
    output_prefix = str(tmp_path / "data")

    convert_file(_CharTokenizer(), str(input_file), output_prefix)

    assert _documents(f"{output_prefix}_text") == [_tokens("gz") + [_EOS]]


def test_convert_parquet(tmp_path) -> None:
    input_file = tmp_path / "data.parquet"
    pq.write_table(pa.table({"text": ["pa", "rq"]}), input_file)
    output_prefix = str(tmp_path / "data")

    convert_file(_CharTokenizer(), str(input_file), output_prefix)

    assert _documents(f"{output_prefix}_text") == [
        _tokens("pa") + [_EOS],
        _tokens("rq") + [_EOS],
    ]


def test_convert_pretokenized_arrow(tmp_path) -> None:
    input_file = tmp_path / "data.arrow"
    schema = pa.schema([("tokens", pa.int64())])
    with pa.ipc.new_file(str(input_file), schema) as writer:
        writer.write_batch(pa.record_batch([pa.array([5, 6, 7])], schema=schema))
        writer.write_batch(pa.record_batch([pa.array([8])], schema=schema))
    output_prefix = str(tmp_path / "data")

    convert_file(_CharTokenizer(), str(input_file), output_prefix)

    # Arrow input is already tokenized; each record batch is one document.
    assert _documents(f"{output_prefix}_text") == [[5, 6, 7, _EOS], [8, _EOS]]


def test_convert_rejects_subset_for_file_input(tmp_path) -> None:
    input_file = tmp_path / "data.jsonl"
    input_file.write_text(json.dumps({"text": "a"}) + "\n")
    with pytest.raises(ValueError):
        convert_file(
            _CharTokenizer(), str(input_file), str(tmp_path / "data"), subset="x"
        )


def test_collect_files_mirrors_directory_layout(tmp_path) -> None:
    input_dir = tmp_path / "in"
    (input_dir / "sub").mkdir(parents=True)
    (input_dir / "a.jsonl").write_text("")
    (input_dir / "sub" / "b.jsonl.zst").write_bytes(b"")
    (input_dir / ".hidden").write_text("")
    output_dir = tmp_path / "out"

    class _Args:
        input = str(input_dir)
        output_prefix = str(output_dir)

    files = collect_files(_Args())

    assert files == [
        (str(input_dir / "a.jsonl"), str(output_dir / "a")),
        # The two-part .jsonl.zst extension is stripped whole.
        (str(input_dir / "sub" / "b.jsonl.zst"), str(output_dir / "sub" / "b")),
    ]
    assert (output_dir / "sub").is_dir()


def test_collect_files_single_file(tmp_path) -> None:
    input_file = tmp_path / "a.jsonl"
    input_file.write_text("")

    class _Args:
        input = str(input_file)
        output_prefix = str(tmp_path / "out")

    assert collect_files(_Args()) == [(str(input_file), str(tmp_path / "out"))]


def test_preprocess_main_single_file(tmp_path, monkeypatch) -> None:
    input_file = tmp_path / "data.jsonl"
    input_file.write_text(json.dumps({"text": "hi"}) + "\n")
    output_prefix = str(tmp_path / "out" / "data")
    monkeypatch.setattr(
        megatron_preprocess,
        "HuggingFaceTokenizer",
        lambda tokenizer_path: _CharTokenizer(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "megatron_preprocess.py",
            "--input",
            str(input_file),
            "--tokenizer",
            "unused",
            "--append-eod",
            "--output-prefix",
            output_prefix,
        ],
    )

    megatron_preprocess.main()

    assert _documents(f"{output_prefix}_text") == [_tokens("hi") + [_EOS]]


# ------------------------------------------------ megatron_merge


def test_merge_files_keeps_order_and_dtype(tmp_path) -> None:
    prefix1, prefix2 = str(tmp_path / "a"), str(tmp_path / "b")
    _write(prefix1, [[1, 2], [3]])
    _write(prefix2, [[4, 5, 6]])
    merged = str(tmp_path / "merged")

    merge_files([prefix1, prefix2], merged)

    assert _documents(merged) == [[1, 2], [3], [4, 5, 6]]
    assert (
        MMapIndexedDataset(merged).index.dtype
        == MMapIndexedDataset(prefix1).index.dtype
    )


def test_groups_by_sizes_uses_relative_subdirs(tmp_path) -> None:
    (tmp_path / "in" / "sub").mkdir(parents=True)
    _write(str(tmp_path / "in" / "x"), [[1]])
    _write(str(tmp_path / "in" / "sub" / "y"), [[2]])
    _write(str(tmp_path / "in" / "sub" / "z"), [[3]])

    groups = dict(get_groups_by_sizes(str(tmp_path / "in")))

    assert groups == {
        ".": [[str(tmp_path / "in" / "x")]],
        "sub": [
            [str(tmp_path / "in" / "sub" / "y"), str(tmp_path / "in" / "sub" / "z")]
        ],
    }


def test_groups_by_sizes_splits_at_max_size(tmp_path) -> None:
    (tmp_path / "in").mkdir()
    for name in ("x", "y"):
        _write(str(tmp_path / "in" / name), [[1]])

    # Every file exceeds a 0 GB limit, so each forms its own group.
    groups = dict(get_groups_by_sizes(str(tmp_path / "in"), max_size=0))

    assert groups == {".": [[str(tmp_path / "in" / "x")], [str(tmp_path / "in" / "y")]]}


def test_merge_main_directory(tmp_path, monkeypatch) -> None:
    (tmp_path / "in" / "sub").mkdir(parents=True)
    _write(str(tmp_path / "in" / "sub" / "y"), [[2]])
    _write(str(tmp_path / "in" / "sub" / "z"), [[3, 4]])
    output_prefix = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "megatron_merge.py",
            "--input-directory",
            str(tmp_path / "in"),
            "--output-prefix",
            str(output_prefix),
        ],
    )

    megatron_merge.main()

    assert _documents(str(output_prefix / "sub" / "0")) == [[2], [3, 4]]
    file_map = json.loads((output_prefix / "sub" / "file_map-0.json").read_text())
    assert file_map["subdir"] == "sub" and len(file_map["group"]) == 2


def test_merge_main_requires_exactly_one_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["megatron_merge.py", "--output-prefix", str(tmp_path / "out")]
    )
    with pytest.raises(SystemExit):
        megatron_merge.main()


def test_merge_main_input_prefixes_creates_parent(tmp_path, monkeypatch) -> None:
    _write(str(tmp_path / "a"), [[7]])
    output_prefix = tmp_path / "nested" / "merged"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "megatron_merge.py",
            "--input-prefixes",
            str(tmp_path / "a"),
            "--output-prefix",
            str(output_prefix),
        ],
    )

    megatron_merge.main()

    assert os.path.exists(get_idx_path(str(output_prefix)))
    assert _documents(str(output_prefix)) == [[7]]
