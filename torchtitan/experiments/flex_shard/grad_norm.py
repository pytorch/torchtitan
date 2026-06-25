# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed import utils as dist_utils


_FLEX_SHARD_CLIP_PATCHED_ATTR = "_flex_shard_clip_grad_norm_patched"


def install_flex_shard_grad_norm_clipping() -> None:
    """Install FlexShard-local grad clipping support for TorchTitan Trainer.

    TorchTitan's EP grad-norm path assumes DTensor parameters. The experimental
    FlexShard DSV3 path uses local shard tensors, so it installs this adapter
    only when that path is selected instead of changing the shared utility.
    """
    if getattr(dist_utils.clip_grad_norm_, _FLEX_SHARD_CLIP_PATCHED_ATTR, False):
        return

    original_clip_grad_norm = dist_utils.clip_grad_norm_

    @functools.wraps(original_clip_grad_norm)
    def clip_grad_norm_(
        parameters: torch.Tensor | Iterable[torch.Tensor],
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
        pp_mesh: DeviceMesh | None = None,
        ep_enabled: bool = False,
    ) -> torch.Tensor:
        params = _materialize_parameters(parameters)
        if not _has_flex_shard_grad(params):
            return original_clip_grad_norm(
                params,
                max_norm,
                norm_type,
                error_if_nonfinite,
                foreach,
                pp_mesh,
                ep_enabled,
            )
        return _clip_grad_norm_with_flex_shard(
            params,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            pp_mesh,
            ep_enabled,
        )

    setattr(clip_grad_norm_, _FLEX_SHARD_CLIP_PATCHED_ATTR, True)
    dist_utils.clip_grad_norm_ = clip_grad_norm_


def _materialize_parameters(
    parameters: torch.Tensor | Iterable[torch.Tensor],
) -> list[torch.Tensor]:
    if isinstance(parameters, torch.Tensor):
        return [parameters]
    return list(parameters)


def _has_flex_shard_grad(parameters: list[torch.Tensor]) -> bool:
    return any(
        _is_flex_shard_param(param) and param.grad is not None
        for param in parameters
    )


@torch.no_grad()
def _clip_grad_norm_with_flex_shard(
    parameters: list[torch.Tensor],
    max_norm: float,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
    pp_mesh: DeviceMesh | None,
    ep_enabled: bool,
) -> torch.Tensor:
    ep_params: list[torch.Tensor] = []
    ep_grads: list[torch.Tensor] = []
    non_ep_params: list[torch.Tensor] = []
    non_ep_grads: list[torch.Tensor] = []
    flex_shard_params: list[torch.Tensor] = []
    flex_shard_grads: list[torch.Tensor] = []

    for param in parameters:
        if param.grad is None:
            continue
        if _is_flex_shard_param(param):
            flex_shard_params.append(param)
            flex_shard_grads.append(param.grad)
            continue

        if not isinstance(param, DTensor) or not isinstance(param.grad, DTensor):
            raise TypeError(
                "FlexShard grad clipping expects non-FlexShard parameters to be "
                f"DTensor, but got param={type(param).__name__}, "
                f"grad={type(param.grad).__name__}."
            )
        if ep_enabled and _dtensor_uses_ep_mesh(param):
            ep_params.append(param)
            ep_grads.append(param.grad)
        else:
            non_ep_params.append(param)
            non_ep_grads.append(param.grad)

    ep_total_norm = _full_tensor_norm(
        ep_grads,
        norm_type,
        error_if_nonfinite,
        foreach,
    )
    non_ep_total_norm = _full_tensor_norm(
        non_ep_grads,
        norm_type,
        error_if_nonfinite,
        foreach,
    )
    flex_shard_total_norm = torch.nn.utils.get_total_norm(
        flex_shard_grads,
        norm_type,
        error_if_nonfinite,
        foreach,
    )
    flex_shard_total_norm = _reduce_flex_shard_norm(flex_shard_total_norm, norm_type)

    norm_device = flex_shard_total_norm.device
    ep_total_norm = ep_total_norm.to(norm_device)
    non_ep_total_norm = non_ep_total_norm.to(norm_device)

    if math.isinf(norm_type):
        total_norm = torch.maximum(ep_total_norm, non_ep_total_norm)
        total_norm = torch.maximum(total_norm, flex_shard_total_norm)
    else:
        total_norm = (
            ep_total_norm**norm_type
            + non_ep_total_norm**norm_type
            + flex_shard_total_norm**norm_type
        )
        total_norm **= 1.0 / norm_type

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach)
    torch.nn.utils.clip_grads_with_norm_(
        flex_shard_params,
        max_norm,
        total_norm,
        foreach,
    )
    return total_norm


def _full_tensor_norm(
    grads: list[torch.Tensor],
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
) -> torch.Tensor:
    total_norm = torch.nn.utils.get_total_norm(
        grads,
        norm_type,
        error_if_nonfinite,
        foreach,
    )
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    return total_norm


def _reduce_flex_shard_norm(total_norm: torch.Tensor, norm_type: float) -> torch.Tensor:
    # The experimental FlexShard DSV3 path rejects PP, TP, CP, and HSDP, so the
    # local FlexShard shards are unique across the default data-parallel group.
    if not dist.is_available() or not dist.is_initialized():
        return total_norm
    if math.isinf(norm_type):
        dist.all_reduce(total_norm, op=dist.ReduceOp.MAX)
    else:
        total_norm **= norm_type
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)
        total_norm **= 1.0 / norm_type
    return total_norm


def _dtensor_uses_ep_mesh(tensor: DTensor) -> bool:
    mesh_dim_names = tensor.device_mesh.mesh_dim_names
    return mesh_dim_names is not None and "ep" in mesh_dim_names


def _is_flex_shard_param(tensor: torch.Tensor) -> bool:
    return hasattr(tensor, "_placements") and hasattr(tensor, "_mesh")
