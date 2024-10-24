# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed._tensor import DTensor, Replicate
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)


@torch.no_grad()
def clip_grad_norm_(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
    pp_mesh: Optional[DeviceMesh] = None,
) -> torch.Tensor:
    """
    Adapted from torch v2.4.0 torch.nn.utils.clip_grad.clip_grad_norm_

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if pp_mesh is None:
        # fallback to the original implementation
        return torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    """
    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    first_device = grads[0].device
    grouped_grads: Dict[
        Tuple[torch.device, torch.dtype], Tuple[List[List[torch.Tensor]], List[int]]
    ] = _group_tensors_by_device_and_dtype(
        [grads]
    )  # type: ignore[assignment]

    norms: List[torch.Tensor] = []
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_grads, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    # ----- start modified from torch.nn.utils.clip_grad_norm_ -----
    if isinstance(total_norm, DTensor):
        # if total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`
        # we can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor
        total_norm = total_norm.redistribute(
            placements=[Replicate()] * total_norm.device_mesh.ndim
        ).to_local()

    total_norm **= norm_type
    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
    total_norm **= 1.0 / norm_type
    # ----- end modified from torch.nn.utils.clip_grad_norm_ -----

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)

    return total_norm
