# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepEP distributed communication primitives for MoE."""

from .deepep import combine_tokens, dispatch_tokens, DispatchState, sync_combine

__all__ = [
    "dispatch_tokens",
    "combine_tokens",
    "sync_combine",
    "DispatchState",
]
