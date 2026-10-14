# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from unittest.mock import patch

from torchtitan.rl._runtime import (
    ALLOCATOR_ENV_DEFAULTS,
    apply_env_defaults,
    RL_ENV_DEFAULTS,
    XPU_LAUNCH_ENV,
)


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


def test_apply_env_defaults_skips_allocator_conf_on_xpu() -> None:
    with patch.dict(os.environ, {XPU_LAUNCH_ENV: "0,1"}, clear=True):
        apply_env_defaults()

        for key in ALLOCATOR_ENV_DEFAULTS:
            assert key not in os.environ
        assert os.environ["NCCL_DEBUG"] == "WARN"


def test_apply_env_defaults_keeps_explicit_allocator_conf_on_xpu() -> None:
    with patch.dict(
        os.environ,
        {XPU_LAUNCH_ENV: "0,1", "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
        clear=True,
    ):
        apply_env_defaults()

        assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
