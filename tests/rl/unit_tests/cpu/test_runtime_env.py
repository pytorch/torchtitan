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
        apply_env_defaults()

        assert dict(os.environ) == {
            **RL_ENV_DEFAULTS,
            "PYTORCH_CUDA_ALLOC_CONF": RL_ENV_DEFAULTS["PYTORCH_ALLOC_CONF"],
        }


def test_apply_env_defaults_preserves_existing_values() -> None:
    with patch.dict(
        os.environ,
        {"NCCL_DEBUG": "INFO", "UNRELATED": "value"},
        clear=True,
    ):
        apply_env_defaults()

        assert os.environ["NCCL_DEBUG"] == "INFO"
        assert os.environ["UNRELATED"] == "value"


@pytest.mark.parametrize("value", ["", "expandable_segments:False"])
def test_apply_env_defaults_mirrors_explicit_allocator(value: str) -> None:
    with patch.dict(os.environ, {"PYTORCH_ALLOC_CONF": value}, clear=True):
        apply_env_defaults()
        apply_env_defaults()

        assert os.environ["PYTORCH_ALLOC_CONF"] == value
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == value


@pytest.mark.parametrize("canonical", [None, "", "expandable_segments:False"])
def test_apply_env_defaults_rejects_legacy_or_conflicting_allocator(canonical) -> None:
    explicit = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if canonical is not None:
        explicit["PYTORCH_ALLOC_CONF"] = canonical
    with patch.dict(os.environ, explicit, clear=True):
        with pytest.raises(ValueError, match="Use PYTORCH_ALLOC_CONF"):
            apply_env_defaults()
        assert dict(os.environ) == explicit
