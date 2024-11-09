# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Optional, Union

import torch
from torch import distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed._tensor import DTensor, Replicate


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
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

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

    total_norm = torch.nn.utils.get_total_norm(
        parameters, norm_type, error_if_nonfinite, foreach
    )

    if pp_mesh is not None and isinstance(total_norm, DTensor):
        # if total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`
        # we can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor
        total_norm = total_norm.redistribute(
            placements=[Replicate()] * total_norm.device_mesh.ndim
        ).to_local()

        total_norm **= norm_type
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
        total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
