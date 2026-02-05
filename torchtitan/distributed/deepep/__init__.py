# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepEP distributed communication primitives for MoE."""

from .deepep import combine_tokens, dispatch_tokens, DispatchState
from .fused_a2a import autotune_deepep, get_tuned_configs, set_tuned_configs
from .utils import run_deepep_autotune_if_enabled

__all__ = [
    "dispatch_tokens",
    "combine_tokens",
    "DispatchState",
    "autotune_deepep",
    "run_deepep_autotune_if_enabled",
    "get_tuned_configs",
    "set_tuned_configs",
]
