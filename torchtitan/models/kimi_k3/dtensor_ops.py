# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Leaving DTensor land for the fla-core triton kernels.

Both K3 attention kinds call kernels that do not dispatch through DTensor, so
both have to unwrap at the kernel call site. The two spellings differ only in
the gradient placement they hand back, and that difference is the whole point of
keeping them next to each other -- see the docstrings.
"""

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate


__all__ = ["to_local_if_dtensor", "to_local_partial_grad"]


def to_local_if_dtensor(t):
    """Strip DTensor wrapping for fla-core triton kernels.

    fla-core's chunk_kda / fused_kda_gate / ShortConvolution are Triton
    kernels that don't dispatch through DTensor. Under TP, KDA's
    self_attn is NoParallel-wrapped (params become DTensor(Replicate)
    on tp_mesh) and incoming x is also DTensor at the parent's
    boundary. KDA forward stashes the DTensor mesh+placements, strips
    DTensor from x and from each weight at the kernel call site, runs
    the kernels on plain tensors (each rank computes redundantly under
    Replicate), and re-DTensors at the end so the parent NoParallel
    output hook composes correctly.

    isinstance(t, DTensor) is the safe check that dynamo's fake-tensor
    mode honors (``hasattr(t, "to_local")`` is unreliable: dynamo's
    type tracking can elide attribute lookups on DTensor parameters).

    ``grad_placements`` is passed rather than left to default, which the
    distributed rules ask for on every ``to_local``. It is the forward
    placement, which is also what the default would pick -- and that is the
    right answer only because every rank does the SAME work with the unwrapped
    value. When ranks diverge the gradient of a replicated value is their sum,
    and :func:`to_local_partial_grad` is the spelling for that; stating this
    one explicitly is what makes the pair readable as a choice.
    """
    if isinstance(t, DTensor):
        return t.to_local(grad_placements=list(t.placements))
    return t


def to_local_partial_grad(t):
    """``to_local`` for a value each rank then consumes DIFFERENTLY.

    ``to_local()`` defaults the incoming gradient's placement to the forward
    placement. For a Replicate value that is correct only when every rank does the
    SAME work with it -- which is exactly KDA's redundant kernels, and why
    ``to_local_if_dtensor`` keeps the default.

    It is wrong when the ranks diverge. MLA's CP path expands the replicated
    ``k_rot`` onto this rank's head subset, so each rank's gradient is one partial
    contribution and the gradient of the replicated value is their sum: Partial,
    not Replicate. Keeping the default drops that all-reduce silently, because the
    placement still reads Replicate afterwards.

    Measured on ``kimi_k3_debugmodel_report_arch`` at tp2 x cp2: all four MLA
    layers' ``kv_a_proj_with_mqa`` gradients differed across the tp pair by 1-6%
    relative on every step, while tp2 alone was bit-identical -- the non-CP path
    never leaves DTensor, so DTensor reduces it there.
    """
    if not isinstance(t, DTensor):
        return t
    return t.to_local(
        grad_placements=[
            Partial() if isinstance(p, Replicate) else p for p in t.placements
        ]
    )
