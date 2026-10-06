# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dsv3_671b_dist_moe_256gpu.runtime_tree_manifest import (
    read_manifest,
    runtime_tree_hashes,
    verify_manifest,
    write_manifest,
)


def test_manifest_ignores_only_python_bytecode(tmp_path: Path) -> None:
    root = tmp_path / "package"
    cache = root / "__pycache__"
    cache.mkdir(parents=True)
    (root / "module.py").write_text("VALUE = 1\n")
    (root / "extension.so").write_bytes(b"native")
    (cache / "module.cpython-312.pyc").write_bytes(b"relocatable bytecode")
    (cache / "payload.dat").write_bytes(b"stable payload")
    manifest = tmp_path / "runtime.sha256"

    write_manifest(manifest, [root], prefix_roots=False)
    hashes = read_manifest(manifest)

    module_digest = hashlib.sha256(b"VALUE = 1\n").hexdigest()
    assert f"{module_digest} module.py\n" in manifest.read_text()
    assert f"{module_digest}  module.py\n" not in manifest.read_text()
    assert set(hashes) == {
        "__pycache__/payload.dat",
        "extension.so",
        "module.py",
    }
    (cache / "module.cpython-312.pyc").write_bytes(b"rewritten bytecode")
    assert verify_manifest(manifest, [root], prefix_roots=False) == []


def test_manifest_reports_stable_payload_changes(tmp_path: Path) -> None:
    root = tmp_path / "package"
    root.mkdir()
    payload = root / "module.py"
    payload.write_text("VALUE = 1\n")
    (root / "extension.so").write_bytes(b"native")
    manifest = tmp_path / "runtime.sha256"
    write_manifest(manifest, [root])

    payload.write_text("VALUE = 2\n")

    assert verify_manifest(manifest, [root]) == ["mismatch: package/module.py"]

    payload.unlink()
    assert verify_manifest(manifest, [root]) == ["missing: package/module.py"]

    (root / "unexpected.so").write_bytes(b"unexpected")
    assert verify_manifest(manifest, [root]) == [
        "missing: package/module.py",
        "unexpected: package/unexpected.so",
    ]


def test_runtime_tree_root_names_must_be_unique(tmp_path: Path) -> None:
    first = tmp_path / "first" / "package"
    second = tmp_path / "second" / "package"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    with pytest.raises(ValueError, match="root names must be unique"):
        runtime_tree_hashes([first, second])


def test_multiple_roots_use_deterministic_prefixes(tmp_path: Path) -> None:
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    (alpha / "z.py").write_text("z\n")
    (beta / "a.so").write_bytes(b"a")

    assert list(runtime_tree_hashes([beta, alpha])) == ["beta/a.so", "alpha/z.py"]


def test_manifest_rejects_python_bytecode_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "runtime.sha256"
    manifest.write_text(f"{'0' * 64} package/module.pyc\n")

    with pytest.raises(ValueError, match="must not contain Python bytecode"):
        read_manifest(manifest)


def test_empty_stable_tree_and_manifest_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "package"
    root.mkdir()
    (root / "module.pyc").write_bytes(b"bytecode")

    with pytest.raises(ValueError, match="no stable payload files"):
        runtime_tree_hashes([root])

    manifest = tmp_path / "empty.sha256"
    manifest.write_text("")
    with pytest.raises(ValueError, match="no stable payload files"):
        read_manifest(manifest)


def test_cli_exit_codes(tmp_path: Path) -> None:
    tool = (
        Path(__file__).parents[3]
        / "scripts/dsv3_671b_dist_moe_256gpu/runtime_tree_manifest.py"
    )
    root = tmp_path / "package"
    root.mkdir()
    payload = root / "module.py"
    payload.write_text("VALUE = 1\n")
    manifest = tmp_path / "runtime.sha256"

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(tool), *args],
            check=False,
            capture_output=True,
            text=True,
        )

    assert run("create", str(manifest), str(root)).returncode == 0
    assert run("verify", str(manifest), str(root)).returncode == 0

    payload.write_text("VALUE = 2\n")
    assert run("verify", str(manifest), str(root)).returncode == 1

    manifest.write_text("malformed\n")
    assert run("verify", str(manifest), str(root)).returncode == 2
    assert run("verify", str(tmp_path / "missing.sha256"), str(root)).returncode == 2
