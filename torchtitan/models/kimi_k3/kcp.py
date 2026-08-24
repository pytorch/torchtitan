# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KCP: KDA Context Parallelism (report sec 5.1.2).

    Two cross-rank dependencies with different shapes. The recurrence needs each rank's
    true incoming state, which does NOT decompose by summation -- the delta rule applies a
    token-dependent transition, so a prefix scan over (cumulative transition, zero-started
    state) fragments recovers it. The short convolutions need only the previous rank's
    tail, one fixed-size exchange.

    See ``phase13_k3like_48b_posttrain/KCP_DESIGN.md``.
    """

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def conv_with_halo(
    conv, x_local: torch.Tensor, cp_context, activation: str | None = None
) -> torch.Tensor:
    """Run a depthwise causal conv on a sequence-sharded input, exactly.

    Thin adapter over fla's ``causal_conv1d_cp``: unpack the depthwise weight
    the way ``ShortConvolution.forward`` does and hand over the CP context,
    which must have been built with ``conv1d_kernel_size`` set.

    ``activation`` defaults to reading ``conv.activation``, which fla's
    ``ShortConvolution`` carries. A plain ``nn.Conv1d`` does not -- the upstream
    K3 model applies its SiLU outside the conv -- so those call sites pass the
    name explicitly rather than getting a second copy of this function.

    The weight and bias are unwrapped to local first. Under TP the KDA layers are
    NoParallel, so these are DTensor(Replicate), and handing a DTensor to fla's
    triton kernel does not raise anything legible -- it surfaces as
    ``CUBLAS_STATUS_INTERNAL_ERROR`` or an illegal memory access from inside the
    kernel. The Ulysses path unwraps them in its own ``conv_subset``; this one did
    not, which is why KCP worked in every cell that had no TP and broke every cell
    that had both.
    """
    from einops import rearrange
    from fla.modules.conv.cp.ops import causal_conv1d_cp

    weight = conv.weight
    if isinstance(weight, DTensor):
        weight = weight.to_local()
    bias = conv.bias
    if bias is not None and isinstance(bias, DTensor):
        bias = bias.to_local()

    return causal_conv1d_cp(
        x=x_local,
        weight=rearrange(weight, "d 1 w -> d w"),
        bias=bias,
        activation=getattr(conv, "activation", None)
        if activation is None
        else activation,
        cp_context=cp_context,
    )


def build_kcp_context(
    seq_len_local: int,
    group,
    device,
    conv1d_kernel_size: int | None = None,
    cu_seqlens: "torch.Tensor | None" = None,
) -> object:
    """fla CP context for one evenly-split sequence.

    ``chunk_kda`` needs the GLOBAL cu_seqlens of the packed sequence plus the
    process group; ``build_cp_context`` derives each rank's slice from them.
    ``conv1d_kernel_size`` is required by ``causal_conv1d_cp`` and otherwise
    unused, so it is optional here.

    ``cu_seqlens`` defaults to ``[0, seq_len_local * world]``, i.e. ONE document
    spanning the whole sequence. Pass real boundaries to describe a packed
    (multi-document) sequence -- they must be GLOBAL, since that is what fla
    slices per rank.

    Whether the default is right is a property of the caller, not of this
    helper, and worth stating plainly: nothing in this repo hands KDA document
    boundaries in ANY mode. Both non-CP call sites pass ``cu_seqlens=None`` to
    ``chunk_kda``, so a packed SFT batch already carries the delta-rule state
    across document boundaries with or without CP. The default here matches that
    behaviour rather than introducing a hole of its own; fixing it means
    threading the dataloader's boundaries through every KDA call site, not
    changing this default.
    """
    from fla.ops.cp.context import build_cp_context

    if cu_seqlens is None:
        world = dist.get_world_size(group)
        total = seq_len_local * world
        cu_seqlens = torch.tensor([0, total], dtype=torch.int32, device=device)
    return build_cp_context(
        cu_seqlens, group=group, conv1d_kernel_size=conv1d_kernel_size
    )
