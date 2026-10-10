# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA launchers behind the DeepSeek V3 learned-router operator."""

from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import cast

from torch.utils.cpp_extension import load_inline

_FORWARD_DECLARATION = r"""
#include <ATen/core/Tensor.h>
#include <optional>
#include <vector>
std::vector<at::Tensor> learned_router_forward_cuda(
    const at::Tensor&, const std::optional<at::Tensor>&);
"""
_BACKWARD_DECLARATION = r"""
#include <ATen/core/Tensor.h>
#include <optional>
at::Tensor learned_router_backward_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, const std::optional<at::Tensor>&);
"""


@lru_cache(maxsize=2)
def _extension(direction) -> ModuleType:
    declaration = (
        _FORWARD_DECLARATION if direction == "forward" else _BACKWARD_DECLARATION
    )
    return cast(
        ModuleType,
        load_inline(
            name=f"torchtitan_dsv3_router_{direction}_cuda",
            cpp_sources=declaration,
            cuda_sources=Path(__file__)
            .with_name(f"_dsv3_router_{direction}.cu")
            .read_text(),
            functions=[f"learned_router_{direction}_cuda"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                "--fmad=false",
                "-gencode=arch=compute_103,code=sm_103",
            ],
        ),
    )


def prepare():
    """Build both extensions before torch.compile or CUDA graph capture."""
    _extension("forward")
    _extension("backward")


def forward(logits_TE, expert_bias_E):
    """One cooperative launch for routing, frequencies, and the raw loss."""
    return tuple(
        _extension("forward").learned_router_forward_cuda(
            logits_TE.resolve_neg(),
            expert_bias_E.resolve_neg() if expert_bias_E is not None else None,
        )
    )


def backward(
    scores_TE,
    row_norm_T1,
    expert_ids_TK,
    selected_scores_TK,
    route_denominator_T1,
    norm_denominator_T1,
    frequencies_E,
    grad_weights_TK,
    grad_raw_sum,
):
    """One CUDA launch for the routing and auxiliary score gradients."""
    return _extension("backward").learned_router_backward_cuda(
        scores_TE,
        row_norm_T1,
        expert_ids_TK,
        selected_scores_TK,
        route_denominator_T1,
        norm_denominator_T1,
        frequencies_E,
        grad_weights_TK,
        grad_raw_sum,
    )
