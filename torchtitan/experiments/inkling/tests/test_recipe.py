# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil
from pathlib import Path

import pytest

from torchtitan_recipes.models.inkling import verify_checkpoint_source


_SOURCE_FILES = (
    "model.safetensors",
    "tokenizer.json",
    "mtp.safetensors",
    "hf_quant_config.json",
)


@pytest.mark.parametrize("changed", _SOURCE_FILES)
def test_resume_requires_original_checkpoint_contents(tmp_path: Path, changed: str):
    source = tmp_path / "source"
    source.mkdir()
    for name in _SOURCE_FILES:
        (source / name).write_bytes(b"aaa")
    run = tmp_path / "run"
    verify_checkpoint_source(source, run)
    (run / "checkpoint" / "step-10").mkdir(parents=True)

    relocated = tmp_path / "relocated"
    shutil.copytree(source, relocated)
    verify_checkpoint_source(relocated, run)
    (relocated / changed).write_bytes(b"bbb")
    with pytest.raises(ValueError, match="different checkpoint contents"):
        verify_checkpoint_source(relocated, run)
    verify_checkpoint_source(source, run)


def test_refuses_to_adopt_unverified_native_checkpoint(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    run = tmp_path / "run"
    (run / "checkpoint" / "step-1").mkdir(parents=True)
    with pytest.raises(ValueError, match="missing source manifest"):
        verify_checkpoint_source(source, run)
    assert not (run / "source-manifest.json").exists()
