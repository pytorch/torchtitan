# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .shard import per_param_placements, Shard

__all__ = [
    "per_param_placements",
    "Shard",
]
