# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer-loop tests. Skipped pending a rewrite for the async rollout-buffer / episode-batcher loop."""

from __future__ import annotations

import pytest

pytest.skip(
    "TODO: rewrite for the async rollout-buffer / episode-batcher loop",
    allow_module_level=True,
)
