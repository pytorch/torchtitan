# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path


class StorageAdapter(ABC):
    """Interface for storing and retrieving precompiled artifacts.

    Keys are flat, opaque strings (no hierarchy or path separators).
    Implementations map keys to storage locations internally — callers
    should not embed "/" or other filesystem-specific characters in keys.
    """

    @abstractmethod
    def save(self, key: str, data: bytes) -> str:
        """Save data under the given key. Returns the path/URI of the saved artifact.

        If an artifact already exists for the key, it is silently overwritten.
        """
        ...

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Load data for the given key.

        Raises FileNotFoundError if no artifact exists for the key.
        """
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if an artifact exists for the given key."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete an artifact for the given key. No-op if it doesn't exist."""
        ...


class DiskStorageAdapter(StorageAdapter):
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir).resolve()

    def _path_for(self, key: str) -> Path:
        path = (self.base_dir / f"{key}.bin").resolve()
        if not path.is_relative_to(self.base_dir):
            raise ValueError(f"Key {key!r} resolves outside base directory")
        return path

    def save(self, key: str, data: bytes) -> str:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then atomically rename to avoid
        # leaving partial files if the process crashes mid-write.
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            try:
                f.write(data)
                f.flush()
                Path(f.name).replace(path)
            except BaseException:
                Path(f.name).unlink(missing_ok=True)
                raise
        return str(path)

    def load(self, key: str) -> bytes:
        path = self._path_for(key)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found for key {key!r} at {path}")
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path_for(key).exists()

    def delete(self, key: str) -> None:
        self._path_for(key).unlink(missing_ok=True)
