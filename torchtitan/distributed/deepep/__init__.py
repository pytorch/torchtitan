# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepEP distributed communication primitives for MoE."""

from .deepep import (
    autotune_deepep,
    combine_tokens,
    dispatch_tokens,
    DispatchState,
    get_tuned_configs,
    run_deepep_autotune_if_enabled,
    set_tuned_configs,
    sync_combine,
)

__all__ = [
    "autotune_deepep",
    "combine_tokens",
    "dispatch_tokens",
    "DispatchState",
    "get_tuned_configs",
    "run_deepep_autotune_if_enabled",
    "set_tuned_configs",
    "sync_combine",
]
