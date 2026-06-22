# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch.distributed as dist

if TYPE_CHECKING:
    from .bucket_storage import ParamInfo


GradientReduceOp = Literal["avg", "sum"]


def validate_gradient_reduce_op(op: str) -> GradientReduceOp:
    if op not in ("avg", "sum"):
        raise ValueError(
            "FlexShard gradient_reduce_op must be either 'avg' or 'sum', "
            f"but got {op!r}."
        )
    return cast(GradientReduceOp, op)


def gradient_reduce_op_from_infos(infos: list[ParamInfo]) -> GradientReduceOp:
    if not infos:
        raise AssertionError("Expected at least one ParamInfo.")
    op = infos[0].gradient_reduce_op
    for info in infos[1:]:
        if info.gradient_reduce_op != op:
            raise ValueError(
                "FlexShard requires one gradient_reduce_op per communication "
                f"bucket, but {infos[0].fqn!r} uses {op!r} and {info.fqn!r} "
                f"uses {info.gradient_reduce_op!r}."
            )
    return validate_gradient_reduce_op(op)


def dist_reduce_op(op: GradientReduceOp) -> dist.ReduceOp.RedOpType:
    if op == "avg":
        return dist.ReduceOp.AVG
    return dist.ReduceOp.SUM
