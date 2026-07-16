# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local-file sources for the grain dataloader.

A source is anything with `__len__` + `__getitem__` (`RandomAccessSource`); a `SourceConfig`
builds one and states its stable identity (`fingerprint`) for deterministic resume.
"""

import glob
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RandomAccessSource(Protocol):
    """Grain's structural source contract: length + deterministic integer indexing."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Any:
        ...


class SourceConfig(Protocol):
    """Builds a `RandomAccessSource`; `fingerprint` is its stable identity for resume."""

    def build(self) -> RandomAccessSource:
        ...

    def fingerprint(self) -> str:
        ...


@dataclass(frozen=True, slots=True)
class PathRewrite:
    """Regex substitution applied to a data path before it is opened.

    Example:

        PathRewrite(pattern=r"^/producer/root", replacement="/mnt/download")
        # "/producer/root/shard-00.jsonl" -> "/mnt/download/shard-00.jsonl"
    """

    pattern: str
    replacement: str

    def apply(self, path: str) -> str:
        return re.sub(self.pattern, self.replacement, path)


def rewrite_path(path: str, path_rewrites: tuple[PathRewrite, ...]) -> str:
    """Apply each rewrite in order to `path`."""
    for path_rewrite in path_rewrites:
        path = path_rewrite.apply(path)
    return path


def expand_paths(
    patterns: tuple[str, ...], path_rewrites: tuple[PathRewrite, ...]
) -> tuple[str, ...]:
    """Rewrite each glob pattern, expand it, and return the per-pattern-sorted concatenation.

    Example:

        expand_paths(
            patterns=("/producer/data_*.jsonl",),
            path_rewrites=(PathRewrite(pattern="^/producer", replacement="/mnt"),),
        )
        # -> ("/mnt/data_0.jsonl", "/mnt/data_1.jsonl")
    """
    paths: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(rewrite_path(pattern, path_rewrites)))
        if not matched:
            raise FileNotFoundError(f"pattern matched no files: {pattern!r}")
        # resolve() so the same file reached via symlink/second pattern is caught below
        paths.extend(str(Path(match).resolve()) for match in matched)
    if len(paths) != len(set(paths)):
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        raise ValueError(
            f"patterns resolve to the same file more than once: {duplicates}"
        )
    return tuple(paths)


def fingerprint_files(paths: tuple[str, ...]) -> str:
    """sha256 over each file's (basename, size); contents are never read.

    Basenames (not full paths) keep the fingerprint stable when identical data is
    mounted under a different root and reached via `PathRewrite`.
    """
    digest = hashlib.sha256()
    for path in paths:
        file_path = Path(path)
        digest.update(file_path.name.encode())
        digest.update(b"\0")
        digest.update(str(file_path.stat().st_size).encode())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, kw_only=True, slots=True)
class JsonlSourceConfig:
    """Small local JSONL files (prompt sets, SFT sets, test corpora), loaded fully into memory.

    Large corpora belong in a user-defined `SourceConfig` over pre-tokenized data.
    TODO(data-indexed-jsonl): offset-indexed random access for large raw JSONL.

    Example:

        source = JsonlSourceConfig(patterns=("tests/assets/c4_test/data.json",)).build()
        source[0]  # -> {"text": "Beginners BBQ Class Taking Place in Missoula! ..."}
    """

    patterns: tuple[str, ...]
    path_rewrites: tuple[PathRewrite, ...] = ()

    def build(self) -> "InMemoryJsonlSource":
        return InMemoryJsonlSource(
            paths=expand_paths(self.patterns, self.path_rewrites)
        )

    def fingerprint(self) -> str:
        return fingerprint_files(expand_paths(self.patterns, self.path_rewrites))


class InMemoryJsonlSource:
    """Rows of every file, concatenated in `paths` order, held in memory."""

    def __init__(self, *, paths: tuple[str, ...]) -> None:
        self._rows: list[dict[str, Any]] = []
        for path in paths:
            with open(path) as lines:
                self._rows.extend(json.loads(line) for line in lines if line.strip())

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._rows[index]
