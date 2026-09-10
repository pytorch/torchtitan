# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
import torch

from torchtitan.components.dist_moe.backend import _DistMoeRuntime


def _runtime(prefetch: Any) -> _DistMoeRuntime:
    return _DistMoeRuntime(
        config=cast(Any, object()),
        group=cast(Any, object()),
        prefetch=prefetch,
    )


def test_dist_moe_runtime_closes_prefetch_after_initialization_failure():
    prefetch = Mock()
    runtime = _runtime(prefetch)

    with (
        patch(
            "torchtitan.components.dist_moe.backend.create_context",
            side_effect=RuntimeError("context creation failed"),
        ),
        pytest.raises(RuntimeError, match="context creation failed"),
    ):
        runtime.initialize(torch.device("cuda"))

    prefetch.close.assert_called_once_with()
    assert runtime.prefetch is None


def test_dist_moe_runtime_close_releases_pending_prefetch():
    prefetch = Mock()
    runtime = _runtime(prefetch)

    runtime.close()
    runtime.close()

    prefetch.close.assert_called_once_with()
    assert runtime.prefetch is None
