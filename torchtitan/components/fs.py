# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import posixpath

from fsspec.core import url_to_fs


def join_path(path: str, *parts: str) -> str:
    if not parts:
        return path
    if "://" in parts[0]:
        return posixpath.join(parts[0].rstrip("/"), *parts[1:])
    if not path:
        return posixpath.join(*parts)
    return posixpath.join(path.rstrip("/"), *parts)


def basename(path: str) -> str:
    return posixpath.basename(path.rstrip("/"))


def exists(path: str) -> bool:
    filesystem, fs_path = url_to_fs(path)
    return filesystem.exists(fs_path)


def isdir(path: str) -> bool:
    filesystem, fs_path = url_to_fs(path)
    if filesystem.isfile(fs_path):
        return False
    return filesystem.isdir(fs_path) or bool(ls(path))


def ls(path: str) -> list[str]:
    filesystem, fs_path = url_to_fs(path)
    try:
        return [os.fspath(entry) for entry in filesystem.ls(fs_path, detail=False)]
    except (FileNotFoundError, NotADirectoryError):
        return []


def rm(path: str, *, recursive: bool = False) -> None:
    filesystem, fs_path = url_to_fs(path)
    try:
        filesystem.rm(fs_path, recursive=recursive)
    except FileNotFoundError:
        pass
