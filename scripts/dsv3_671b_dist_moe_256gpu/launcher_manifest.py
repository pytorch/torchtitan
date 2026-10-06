#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Create or verify a complete hash manifest for a launcher payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any


_FETCH_CHECKSUMS = re.compile(r".+:[0-9a-f]{7}\.CHECKSUMS")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_entries(
    root: Path,
    manifest_path: Path,
    *,
    allow_fetch_metadata: bool,
) -> list[dict[str, Any]]:
    entries = []
    manifest_path = manifest_path.resolve()
    for path in sorted(root.rglob("*")):
        if path.absolute() == manifest_path:
            continue
        relative_path = path.relative_to(root).as_posix()
        if allow_fetch_metadata and (
            relative_path == "METADATA" or _FETCH_CHECKSUMS.fullmatch(relative_path)
        ):
            continue
        path_stat = path.lstat()
        mode = stat.S_IMODE(path_stat.st_mode)
        if path.is_symlink():
            entries.append(
                {
                    "kind": "symlink",
                    "mode": f"{mode:04o}",
                    "path": relative_path,
                    "target": path.readlink().as_posix(),
                }
            )
        elif path.is_file():
            entries.append(
                {
                    "kind": "file",
                    "mode": f"{mode:04o}",
                    "path": relative_path,
                    "sha256": _file_sha256(path),
                    "size": path_stat.st_size,
                }
            )
        elif path.is_dir():
            entries.append(
                {
                    "kind": "directory",
                    "mode": f"{mode:04o}",
                    "path": relative_path,
                }
            )
    return entries


def create_manifest(root: Path, manifest_path: Path) -> None:
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    payload = {
        "format_version": 1,
        "entries": _payload_entries(
            root,
            manifest_path,
            allow_fetch_metadata=False,
        ),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_manifest(
    root: Path,
    manifest_path: Path,
    *,
    allow_fetch_metadata: bool,
) -> None:
    payload = json.loads(manifest_path.read_text())
    if payload.get("format_version") != 1 or not isinstance(
        payload.get("entries"), list
    ):
        raise ValueError(f"invalid launcher manifest: {manifest_path}")
    actual_entries = _payload_entries(
        root,
        manifest_path,
        allow_fetch_metadata=allow_fetch_metadata,
    )
    if actual_entries != payload["entries"]:
        expected_by_path = {entry["path"]: entry for entry in payload["entries"]}
        actual_by_path = {entry["path"]: entry for entry in actual_entries}
        missing = sorted(expected_by_path.keys() - actual_by_path.keys())
        unexpected = sorted(actual_by_path.keys() - expected_by_path.keys())
        changed = sorted(
            path
            for path in expected_by_path.keys() & actual_by_path.keys()
            if expected_by_path[path] != actual_by_path[path]
        )
        raise RuntimeError(
            "launcher manifest mismatch: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--allow-fbpkg-fetch-metadata",
        action="store_true",
        help="Ignore the METADATA and CHECKSUMS files added by fbpkg fetch.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not manifest_path.is_relative_to(root):
        raise ValueError("manifest must be inside the launcher root")
    if args.action == "create":
        if args.allow_fbpkg_fetch_metadata:
            parser.error("--allow-fbpkg-fetch-metadata is verify-only")
        create_manifest(root, manifest_path)
    else:
        verify_manifest(
            root,
            manifest_path,
            allow_fetch_metadata=args.allow_fbpkg_fetch_metadata,
        )


if __name__ == "__main__":
    main()
