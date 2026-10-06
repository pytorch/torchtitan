#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import hashlib
import sys
from pathlib import Path, PurePosixPath


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_tree_hashes(
    roots: list[Path], *, prefix_roots: bool = True
) -> dict[str, str]:
    """Return stable payload hashes, keyed by root name and relative path.

    Python bytecode is intentionally excluded. A ``.pyc`` embeds source paths
    and may be regenerated during package relocation or a normal import, so it
    is not a stable payload-integrity record. Source files, native extensions,
    and every other regular file remain covered, including non-bytecode files
    that happen to live under ``__pycache__``.
    """
    root_names = [root.name for root in roots]
    if prefix_roots and len(root_names) != len(set(root_names)):
        raise ValueError("runtime tree root names must be unique")
    if not prefix_roots and len(roots) != 1:
        raise ValueError("an unprefixed manifest requires exactly one root")

    hashes: dict[str, str] = {}
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"runtime tree root is not a directory: {root}")
        num_root_files = 0
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix == ".pyc":
                continue
            relative_path = path.relative_to(root)
            if prefix_roots:
                relative_path = Path(root.name) / relative_path
            relative_path = relative_path.as_posix()
            if "\n" in relative_path:
                raise ValueError(f"runtime tree path contains a newline: {path}")
            hashes[relative_path] = _sha256(path)
            num_root_files += 1
        if num_root_files == 0:
            raise ValueError(f"runtime tree has no stable payload files: {root}")
    return hashes


def write_manifest(
    output: Path, roots: list[Path], *, prefix_roots: bool = True
) -> None:
    hashes = runtime_tree_hashes(roots, prefix_roots=prefix_roots)
    output.write_text("".join(f"{digest} {path}\n" for path, digest in hashes.items()))


def read_manifest(manifest: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        try:
            digest, path = line.split(" ", 1)
        except ValueError as error:
            raise ValueError(
                f"invalid manifest line {line_number}: expected SHA256 and path"
            ) from error
        if not path or path.startswith(" "):
            raise ValueError(f"invalid path on manifest line {line_number}")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid SHA256 on manifest line {line_number}")
        if path in hashes:
            raise ValueError(f"duplicate manifest path: {path}")
        if PurePosixPath(path).suffix == ".pyc":
            raise ValueError(f"manifest must not contain Python bytecode: {path}")
        hashes[path] = digest
    if not hashes:
        raise ValueError("runtime tree manifest has no stable payload files")
    return hashes


def verify_manifest(
    manifest: Path, roots: list[Path], *, prefix_roots: bool = True
) -> list[str]:
    expected = read_manifest(manifest)
    actual = runtime_tree_hashes(roots, prefix_roots=prefix_roots)
    errors = [f"missing: {path}" for path in sorted(expected.keys() - actual.keys())]
    errors.extend(
        f"unexpected: {path}" for path in sorted(actual.keys() - expected.keys())
    )
    errors.extend(
        f"mismatch: {path}"
        for path in sorted(expected.keys() & actual.keys())
        if expected[path] != actual[path]
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or verify relocation-stable runtime tree hashes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--unprefixed", action="store_true")
    create_parser.add_argument("manifest", type=Path)
    create_parser.add_argument("roots", nargs="+", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--unprefixed", action="store_true")
    verify_parser.add_argument("manifest", type=Path)
    verify_parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()

    try:
        if args.command == "create":
            write_manifest(args.manifest, args.roots, prefix_roots=not args.unprefixed)
            return 0
        errors = verify_manifest(
            args.manifest, args.roots, prefix_roots=not args.unprefixed
        )
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Verified {len(read_manifest(args.manifest))} stable runtime files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
