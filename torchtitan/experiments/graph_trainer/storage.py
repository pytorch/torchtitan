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
    @abstractmethod
    def save(self, key: str, data: bytes) -> str:
        """Save data under the given key. Returns the path/URI of the saved artifact."""
        ...

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Load data for the given key."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if an artifact exists for the given key."""
        ...


class DiskStorageAdapter(StorageAdapter):
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def _path_for(self, key: str) -> Path:
        return self.base_dir / f"{key}.bin"

    def save(self, key: str, data: bytes) -> str:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then atomically rename to avoid
        # leaving partial files if the process crashes mid-write.
        fd, tmp_path = tempfile.mkstemp(dir=path.parent)
        try:
            with open(fd, "wb") as f:
                f.write(data)
            Path(tmp_path).replace(path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        return str(path)

    def load(self, key: str) -> bytes:
        path = self._path_for(key)
        if not path.exists():
            raise FileNotFoundError(
                f"Precompile artifact not found at {path}. "
                f"Run the precompile save phase first."
            )
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path_for(key).exists()
