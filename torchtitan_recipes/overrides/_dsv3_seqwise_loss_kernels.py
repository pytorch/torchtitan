# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA implementation behind the DeepSeek V3 load-balance operator."""

from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import cast

import torch
from torch.utils.cpp_extension import load_inline

_DECLARATIONS = r"""
#include <ATen/core/Tensor.h>
#include <tuple>
std::tuple<at::Tensor, at::Tensor> seqwise_load_balance_forward_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&);
at::Tensor seqwise_load_balance_backward_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&);
"""


@lru_cache(maxsize=1)
def prepare() -> ModuleType:
    """Build the extension before compiling a model or capturing a CUDA graph."""
    return cast(
        ModuleType,
        load_inline(
            name="torchtitan_dsv3_seqwise_loss_cuda",
            cpp_sources=_DECLARATIONS,
            cuda_sources=Path(__file__).with_suffix(".cu").read_text(),
            functions=[
                "seqwise_load_balance_forward_cuda",
                "seqwise_load_balance_backward_cuda",
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                "--fmad=false",
                "-gencode=arch=compute_103,code=sm_103",
            ],
        ),
    )


def forward(
    scores_TE: torch.Tensor,
    routing_map_TE: torch.Tensor,
    arrival_counter: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the raw token-sum loss and saved per-expert frequencies."""
    return prepare().seqwise_load_balance_forward_cuda(
        scores_TE, routing_map_TE, arrival_counter
    )


def backward(
    grad_raw_sum: torch.Tensor,
    scores_TE: torch.Tensor,
    frequencies_E: torch.Tensor,
) -> torch.Tensor:
    """Return the score gradient, with the native reduction and rounding order."""
    return prepare().seqwise_load_balance_backward_cuda(
        grad_raw_sum.resolve_neg(), scores_TE, frequencies_E
    )
