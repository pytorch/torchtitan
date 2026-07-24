# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class ContextParallelMethod(str, Enum):
    """Context-parallel attention method. ``str`` mixin (not ``enum.StrEnum``,
    which is Python 3.11+; torchtitan targets 3.10)."""

    ALLGATHER = "allgather"
    ULYSSES = "ulysses"
