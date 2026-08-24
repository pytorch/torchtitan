# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for modules with physically fused gate and up weights."""

from collections.abc import Callable

import torch


def make_fused_gate_up_init(
    gate_init: Callable,
    up_init: Callable,
    *,
    gate_up_axis: int,
) -> Callable:
    """Build one initializer from separate gate and up initializers."""

    def _init(tensor: torch.Tensor) -> None:
        gate_index: list[int | slice] = [slice(None)] * tensor.ndim
        up_index: list[int | slice] = [slice(None)] * tensor.ndim
        gate_index[gate_up_axis] = 0
        up_index[gate_up_axis] = 1
        gate_init(tensor[tuple(gate_index)])
        up_init(tensor[tuple(up_index)])

    return _init
