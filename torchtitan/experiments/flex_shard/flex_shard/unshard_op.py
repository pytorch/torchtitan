# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch
from torch.library import Library


_LIB = Library("flex_shard", "DEF")
_LIB.define("unshard_bucket(Tensor[] full_params) -> Tensor[]")


def _unshard_bucket_impl(full_params: list[torch.Tensor]) -> list[torch.Tensor]:
    return [torch.ops.aten.alias.default(param) for param in full_params]


_LIB.impl(
    "unshard_bucket",
    _unshard_bucket_impl,
    "CompositeImplicitAutograd",
)

UNSHARD_BUCKET_OP = torch.ops.flex_shard.unshard_bucket.default


def mark_unshard_bucket(full_params: list[torch.Tensor]) -> list[torch.Tensor]:
    """Mark full params as coming from a FlexShard semantic unshard."""
    return list(torch.ops.flex_shard.unshard_bucket(full_params))


def is_unshard_bucket_op(func: Any) -> bool:
    """Return whether ``func`` is the FlexShard semantic unshard op."""
    return func is UNSHARD_BUCKET_OP
