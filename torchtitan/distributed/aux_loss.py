# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-agnostic machinery for model-internal auxiliary losses.

An auxiliary objective computed deep inside the model (MoE load-balance, DSA
indexer KL, ...) cannot be routed to the trainer's loss function under pipeline
parallelism, the single-tensor module-output contract, or fullgraph compile.
Instead its *gradient* is injected locally via an identity-forward
autograd.Function, and any cross-rank statistic it needs is reduced with a
single dispatched helper.

Everything here is loss-agnostic and parallel-backend-agnostic. The loss math
(which tensors, which axes are Partial, the denominator) lives in the calling
module; this module only provides the injection and the reduction primitive.
"""

import spmd_types as spmd
import torch
from torch.distributed.tensor import DTensor, Partial, Replicate

from torchtitan.distributed.spmd_types import current_spmd_mesh

__all__ = ["AuxLossInjection", "inject_aux_loss", "reduce_to_replicate"]


@spmd.register_autograd_function
class AuxLossInjection(torch.autograd.Function):
    """Inject an auxiliary-loss gradient without changing the forward value.

    Forward returns ``carrier`` unchanged, so the host module keeps a
    single-tensor return (PP/compile-safe; no value flows to the model output).
    ``aux_loss`` is a scalar built differentiably from module-local tensors; its
    backward seed of 1 lets ordinary autograd propagate the aux-loss gradient
    back into those tensors during the normal model backward. ``carrier`` must
    be on the live autograd graph to the model loss so this backward fires.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, carrier: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(aux_loss)
        return carrier

    @staticmethod
    def typecheck_forward(
        carrier: torch.Tensor, aux_loss: torch.Tensor
    ) -> torch.Tensor:
        # Identity in forward: the output carries the same SPMD type as
        # ``carrier``; ``aux_loss`` does not constrain the output type.
        out = AuxLossInjection.apply(carrier, aux_loss)
        spmd.assert_type(out, dict(spmd.get_local_type(carrier)))
        return out

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx, grad_carrier: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (aux_loss,) = ctx.saved_tensors
        return grad_carrier, torch.ones_like(aux_loss)


def inject_aux_loss(carrier: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
    """Attach ``aux_loss``'s gradient to ``carrier`` (identity in value)."""
    return AuxLossInjection.apply(carrier, aux_loss)


def reduce_to_replicate(x: torch.Tensor, *, axes: list[str]) -> torch.Tensor:
    """Differentiably all-reduce a ``Partial`` statistic to ``Replicate`` over
    ``axes`` (e.g. ``["cp", "tp"]``), preserving placement on every other axis.

    Which axes are ``Partial`` is a property of the parallelism config and is
    chosen by the caller; this helper hides *how* the reduction is realized.
    Only the ``full_dtensor`` and ``spmd_types`` backends are supported (both
    carry the data/sequence axes in-band): a ``DTensor`` input takes the
    ``full_dtensor`` path, anything else the ``spmd_types`` path. The ``default``
    backend is unsupported (CP/DP are out-of-band there); enabling the aux loss
    under it is rejected at config time. Axes of size 1, or axes on which ``x``
    is not ``Partial``, are no-ops, so single-process / unit-test calls (plain
    tensor, no current SPMD mesh) are pure-local.
    """
    if not axes:
        return x
    if isinstance(x, DTensor):
        return _reduce_full_dtensor(x, axes)
    return _reduce_spmd_types(x, axes)


def _reduce_spmd_types(x: torch.Tensor, axes: list[str]) -> torch.Tensor:
    mesh = current_spmd_mesh()
    if mesh is None:
        return x
    names = mesh.mesh_dim_names or ()
    for axis in axes:
        if axis in names and mesh.size(names.index(axis)) > 1:
            x = spmd.redistribute(
                x,
                mesh.get_group(axis),
                src=spmd.P,
                dst=spmd.R,
                backward_options={"op_dtype": x.dtype},
            )
    return x


def _reduce_full_dtensor(x: torch.Tensor, axes: list[str]) -> torch.Tensor:
    # All data/seq axes are in-band on the DTensor's own mesh; redistribute the
    # Partial axes in ``axes`` to Replicate and leave the rest (e.g. Shard(0) on
    # the batch axes, which FSDP reduces) untouched.
    if not isinstance(x, DTensor):
        return x
    names = x.device_mesh.mesh_dim_names or ()
    new = [
        Replicate() if (names[i] in axes and isinstance(p, Partial)) else p
        for i, p in enumerate(x.placements)
    ]
    return x.redistribute(placements=new) if new != list(x.placements) else x
