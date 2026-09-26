# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config import CompileConfig


def test_compile_config_default() -> None:
    config = CompileConfig()
    assert config.components == ["loss"]


def test_compile_config_loss_only() -> None:
    config = CompileConfig(components=["loss"])
    assert config.components == ["loss"]


def test_compile_config_empty_components() -> None:
    config = CompileConfig(components=[])
    assert config.components == []


def test_compile_config_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match=r"foo.*allowed values are.*loss"):
        CompileConfig(components=["foo"])


def test_compile_config_rejects_model_component() -> None:
    with pytest.raises(ValueError, match=r"model.*allowed values are.*loss"):
        CompileConfig(components=["model"])
