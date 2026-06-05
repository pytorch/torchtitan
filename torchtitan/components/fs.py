# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import posixpath
from typing import cast, IO

from fsspec.core import url_to_fs


def join_path(path: str, *parts: str) -> str:
    if not parts:
        return path
    if not path:
        return posixpath.join(*parts)
    return posixpath.join(path.rstrip("/"), *parts)


def basename(path: str) -> str:
    return posixpath.basename(path.rstrip("/"))


def open_file(path: str, mode: str) -> IO[bytes]:
    fs, fs_path = url_to_fs(path)
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        parent = posixpath.dirname(fs_path)
        if parent:
            fs.makedirs(parent, exist_ok=True)
    return cast(IO[bytes], fs.open(fs_path, mode))


def exists(path: str) -> bool:
    fs, fs_path = url_to_fs(path)
    return fs.exists(fs_path)


def isdir(path: str) -> bool:
    fs, fs_path = url_to_fs(path)
    if fs.isfile(fs_path):
        return False
    return fs.isdir(fs_path) or bool(ls(path))


def ls(path: str) -> list[str]:
    fs, fs_path = url_to_fs(path)
    try:
        return [os.fspath(entry) for entry in fs.ls(fs_path, detail=False)]
    except (FileNotFoundError, NotADirectoryError):
        return []


def rm(path: str, *, recursive: bool = False) -> None:
    fs, fs_path = url_to_fs(path)
    try:
        fs.rm(fs_path, recursive=recursive)
    except FileNotFoundError:
        pass
