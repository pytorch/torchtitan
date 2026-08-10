# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def archive_local_path(local_path: Path, *, archive_name: str | None = None) -> bytes:
    """Pack a file or directory with its basename as the archive root."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.add(
            local_path,
            arcname=archive_name or local_path.name,
            recursive=True,
        )
    return buffer.getvalue()


def materialize_archive(data: bytes, local_path: Path) -> None:
    """Safely materialize a single-root archive at an exact local path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            members = archive.getmembers()
            _validate_members(members)
            if sys.version_info >= (3, 12):
                archive.extractall(root, filter="fully_trusted")
            else:  # pragma: no cover - Python 3.10 and 3.11 only
                archive.extractall(root)

        top_level = {PurePosixPath(member.name).parts[0] for member in members}
        if len(top_level) != 1:
            raise ValueError(
                f"sandbox download archive must have one root; got {sorted(top_level)}"
            )
        extracted = root / next(iter(top_level))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() or local_path.is_symlink():
            if local_path.is_dir() and not local_path.is_symlink():
                shutil.rmtree(local_path)
            else:
                local_path.unlink()
        shutil.move(str(extracted), str(local_path))


def _validate_members(members: list[tarfile.TarInfo]) -> None:
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe path in sandbox archive: {member.name!r}")
        if member.issym() or member.islnk():
            target = PurePosixPath(member.linkname)
            if target.is_absolute() or ".." in target.parts:
                raise ValueError(
                    f"unsafe link target in sandbox archive: {member.linkname!r}"
                )
