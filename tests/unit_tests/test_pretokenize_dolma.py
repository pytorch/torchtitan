# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "pre-processing" / "pretokenize_dolma.py"
)
_SPEC = importlib.util.spec_from_file_location("pretokenize_dolma", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
pretokenize_dolma = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pretokenize_dolma)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def test_shard_local_document_shuffle_is_deterministic(tmp_path):
    records = [
        {"id": 0, "text": "A"},
        {"id": 1, "text": "A"},
        {"id": 2, "text": "A"},
        {"id": 3, "text": "B"},
        {"id": 4, "text": "B"},
    ]
    path = tmp_path / "part.jsonl"
    _write_jsonl(path, records)

    first = pretokenize_dolma._shuffled_records(
        path, shuffle_seed=7, source_file_idx=0
    )
    second = pretokenize_dolma._shuffled_records(
        path, shuffle_seed=7, source_file_idx=0
    )

    first_ids = [record["id"] for record in first]
    assert first_ids == [record["id"] for record in second]
    assert first_ids != list(range(len(records)))
    assert sorted(first_ids) == list(range(len(records)))
    shuffled_text = [record["text"] for record in first]
    assert all(
        shuffled_text[start : start + 3] != ["A", "A", "A"]
        for start in range(len(shuffled_text) - 2)
    )


def test_shard_local_document_shuffle_uses_source_index(tmp_path):
    records = [{"id": idx, "text": str(idx)} for idx in range(20)]
    path = tmp_path / "part.jsonl"
    _write_jsonl(path, records)

    first = pretokenize_dolma._shuffled_records(
        path, shuffle_seed=7, source_file_idx=0
    )
    second = pretokenize_dolma._shuffled_records(
        path, shuffle_seed=7, source_file_idx=1
    )

    assert [record["id"] for record in first] != [
        record["id"] for record in second
    ]


def test_completed_metadata_requires_document_shuffle_version(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "format_version": pretokenize_dolma.FORMAT_VERSION - 1,
                "shuffle_seed": 7,
            }
        )
    )

    with pytest.raises(ValueError, match="--overwrite"):
        pretokenize_dolma._validate_completed_metadata(
            metadata_path, shuffle_seed=7
        )

    metadata_path.write_text(
        json.dumps(
            {
                "format_version": pretokenize_dolma.FORMAT_VERSION,
                "shuffle_seed": 7,
            }
        )
    )
    pretokenize_dolma._validate_completed_metadata(metadata_path, shuffle_seed=7)
