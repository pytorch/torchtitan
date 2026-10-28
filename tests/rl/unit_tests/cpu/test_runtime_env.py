# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from torchtitan.rl._env import apply_env_defaults, RL_ENV_DEFAULTS


def test_apply_env_defaults() -> None:
    with patch.dict(os.environ, {}, clear=True):
        apply_env_defaults()

        assert dict(os.environ) == RL_ENV_DEFAULTS


def test_apply_env_defaults_preserves_existing_values() -> None:
    with patch.dict(
        os.environ,
        {"NCCL_DEBUG": "INFO", "UNRELATED": "value"},
        clear=True,
    ):
        apply_env_defaults()

        assert os.environ["NCCL_DEBUG"] == "INFO"
        assert os.environ["UNRELATED"] == "value"


@pytest.mark.parametrize("key", ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
@pytest.mark.parametrize("value", ["", "expandable_segments:False"])
def test_apply_env_defaults_preserves_allocator_alias(key: str, value: str) -> None:
    with patch.dict(os.environ, {key: value}, clear=True):
        apply_env_defaults()
        apply_env_defaults()

        assert os.environ["PYTORCH_ALLOC_CONF"] == value
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == value


def test_apply_env_defaults_preserves_distinct_allocator_aliases() -> None:
    explicit = {
        "PYTORCH_ALLOC_CONF": "expandable_segments:False",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    }
    with patch.dict(os.environ, explicit, clear=True):
        apply_env_defaults()

        for key, value in explicit.items():
            assert os.environ[key] == value


def test_apply_env_defaults_warns_before_copying_allocator_alias() -> None:
    with (
        patch.dict(
            os.environ, {"PYTORCH_ALLOC_CONF": "expandable_segments:False"}, clear=True
        ),
        patch.dict("sys.modules", {"torch": object()}),
        pytest.warns(UserWarning, match="torch.*already been imported"),
    ):
        apply_env_defaults()
