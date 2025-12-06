# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .moe import FeedForward, MoE, MoEArgs, build_moe

try:
    from .moe_deepep import MoEWithDeepEP
    from torchtitan.distributed.deepep import MoEFlexTokenDispatcher
    HAS_DEEPEP = True
    __all__ = [
        "FeedForward", "MoE", "MoEArgs", "build_moe",
        "MoEWithDeepEP", 
        "MoEFlexTokenDispatcher",
    ]
except ImportError:
    HAS_DEEPEP = False
    __all__ = ["FeedForward", "MoE", "MoEArgs", "build_moe"]
