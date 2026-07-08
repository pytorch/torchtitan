# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Filesystem helpers that transparently support local paths and remote fsspec
URIs (e.g. ``gs://``, ``s3://``) for checkpoint IO.

A path is treated as remote iff it contains ``"://"`` -- the same heuristic
PyTorch DCP uses in ``FileSystem.validate_checkpoint_id``. Local paths keep
using ``os``/``shutil`` unchanged; only remote paths go through fsspec, so the
battle-tested local code path is behaviorally identical.
"""

from __future__ import annotations

import os
import shutil


def is_remote(path: str | os.PathLike) -> bool:
    """Whether ``path`` is a remote fsspec URI rather than a local path."""
    return "://" in str(path)


def _resolve(path: str | os.PathLike):
    """Return ``(fs, path)`` for a remote URI via fsspec.

    Imported lazily so pure-local environments never need fsspec, and so a
    missing backend driver (e.g. ``gcsfs`` for ``gs://``) surfaces fsspec's own
    ``ImportError`` at the point of use.
    """
    from fsspec.core import url_to_fs

    return url_to_fs(path)


def exists(path: str | os.PathLike) -> bool:
    if is_remote(path):
        fs, p = _resolve(path)
        return fs.exists(p)
    return os.path.exists(path)


def isdir(path: str | os.PathLike) -> bool:
    if is_remote(path):
        fs, p = _resolve(path)
        return fs.isdir(p)
    return os.path.isdir(path)


def isfile(path: str | os.PathLike) -> bool:
    if is_remote(path):
        fs, p = _resolve(path)
        return fs.isfile(p)
    return os.path.isfile(path)


def listdir(path: str | os.PathLike) -> list[str]:
    """List directory entries as basenames, matching ``os.listdir``.

    fsspec's ``ls`` returns full paths, so remote entries are reduced to their
    basename to keep downstream ``step-(\\d+)`` matching and ``os.path.join``
    logic unchanged.
    """
    if is_remote(path):
        fs, p = _resolve(path)
        return [os.path.basename(entry.rstrip("/")) for entry in fs.ls(p, detail=False)]
    return os.listdir(path)


def rmtree(path: str | os.PathLike) -> None:
    if is_remote(path):
        fs, p = _resolve(path)
        # Mirror ``shutil.rmtree(..., ignore_errors=True)`` for the common
        # already-deleted case. Other errors (permissions, transient backend
        # failures) propagate so the caller can decide how to handle them.
        try:
            fs.rm(p, recursive=True)
        except FileNotFoundError:
            pass
    else:
        shutil.rmtree(path, ignore_errors=True)


def join(base: str, folder: str) -> str:
    """Join ``base`` and ``folder``, treating a remote ``folder`` as absolute.

    A remote ``folder`` (e.g. ``gs://bucket/run/ckpt``) is returned unchanged so
    it is not prefixed with a local ``base`` such as ``dump_folder``.
    """
    if is_remote(folder):
        return folder
    return os.path.join(base, folder)
