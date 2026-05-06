# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn


_active_parametrization = True


@contextmanager
def disable_active_parametrization() -> Generator[None, None, None]:
    """Disable parametrization forward (returns raw sharded tensor).

    Use during initialization, checkpointing, or any context where
    parameter access should not trigger collective communication.
    """
    global _active_parametrization
    try:
        _active_parametrization = False
        yield
    finally:
        _active_parametrization = True


class _MixedPrecisionCast(torch.autograd.Function):
    """Cast with decoupled forward/backward dtype control.

    Forward casts to param_dtype (compute dtype). Backward casts to
    reduce_dtype (gradient reduction dtype).
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        param_dtype: torch.dtype | None,
        reduce_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        ctx.reduce_dtype = reduce_dtype
        if param_dtype is not None and x.dtype != param_dtype:
            return x.to(param_dtype)
        return x

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        if ctx.reduce_dtype is not None and grad.dtype != ctx.reduce_dtype:
            return grad.to(ctx.reduce_dtype), None, None
        return grad, None, None


class ShardParametrization(nn.Module):
    """Parametrization for Shard placement."""

    def __init__(
        self,
        shard_dim: int,
        group_name: str,
        world_size: int,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        compute_device: torch.device | None = None,
        padded_shard_size: int | None = None,
        global_dim_size: int | None = None,
    ):
        super().__init__()
        self.shard_dim = shard_dim
        self.group_name = group_name
        self.world_size = world_size
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.compute_device = compute_device
        self.padded_shard_size = padded_shard_size
        self.global_dim_size = global_dim_size

    def forward(self, local_shard: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return local_shard
        if (
            self.compute_device is not None
            and local_shard.device != self.compute_device
        ):
            local_shard = local_shard.to(self.compute_device, non_blocking=True)

        if self.padded_shard_size is not None:
            local_size = local_shard.shape[self.shard_dim]
            pad_size = self.padded_shard_size - local_size
            if pad_size > 0:
                pad_shape = list(local_shard.shape)
                pad_shape[self.shard_dim] = pad_size
                padding = local_shard.new_zeros(pad_shape)
                local_shard = torch.cat([local_shard, padding], dim=self.shard_dim)

        full = torch.ops._c10d_functional.all_gather_into_tensor(
            local_shard, self.world_size, self.group_name
        )
        full = torch.ops._c10d_functional.wait_tensor(full)
        if full.requires_grad and torch.is_grad_enabled():
            full.register_hook(lambda grad: grad / self.world_size)
        if self.shard_dim != 0:
            chunks = full.chunk(self.world_size, dim=0)
            full = torch.cat(chunks, dim=self.shard_dim)

        if self.global_dim_size is not None:
            full = full.narrow(self.shard_dim, 0, self.global_dim_size)

        if self.param_dtype is not None or self.reduce_dtype is not None:
            full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
        return full


class FlatShardParametrization(nn.Module):
    """Parametrization for FlatShard placement."""

    def __init__(
        self,
        group_name: str,
        world_size: int,
        original_shape: torch.Size,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        compute_device: torch.device | None = None,
        padded_shard_size: int | None = None,
        global_numel: int | None = None,
    ):
        super().__init__()
        self.group_name = group_name
        self.world_size = world_size
        self.original_shape = original_shape
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.compute_device = compute_device
        self.padded_shard_size = padded_shard_size
        self.global_numel = global_numel

    def forward(self, flat_shard: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return flat_shard
        if self.compute_device is not None and flat_shard.device != self.compute_device:
            flat_shard = flat_shard.to(self.compute_device, non_blocking=True)

        if self.padded_shard_size is not None:
            pad_size = self.padded_shard_size - flat_shard.numel()
            if pad_size > 0:
                padding = flat_shard.new_zeros(pad_size)
                flat_shard = torch.cat([flat_shard, padding])

        full_flat = torch.ops._c10d_functional.all_gather_into_tensor(
            flat_shard, self.world_size, self.group_name
        )
        full_flat = torch.ops._c10d_functional.wait_tensor(full_flat)
        if full_flat.requires_grad and torch.is_grad_enabled():
            full_flat.register_hook(lambda grad: grad / self.world_size)

        if self.global_numel is not None:
            full_flat = full_flat[: self.global_numel]

        full = full_flat.view(self.original_shape)
        if self.param_dtype is not None or self.reduce_dtype is not None:
            full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
        return full


class _OwnedBroadcast(torch.autograd.Function):
    """Differentiable broadcast for Owned placement."""

    @staticmethod
    def forward(
        ctx: Any,
        param: torch.Tensor,
        owner_rank: int,
        group_name: str,
        world_size: int,
    ) -> torch.Tensor:
        ctx.group_name = group_name
        ctx.world_size = world_size
        result = torch.ops._c10d_functional.broadcast(param, owner_rank, group_name)
        return torch.ops._c10d_functional.wait_tensor(result)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        reduced = torch.ops._c10d_functional.all_reduce(grad, "sum", ctx.group_name)
        reduced = torch.ops._c10d_functional.wait_tensor(reduced)
        return reduced / ctx.world_size, None, None, None


class OwnedParametrization(nn.Module):
    """Parametrization for Owned placement."""

    def __init__(
        self,
        owner_rank: int,
        group_name: str,
        world_size: int,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        compute_device: torch.device | None = None,
    ):
        super().__init__()
        self.owner_rank = owner_rank
        self.group_name = group_name
        self.world_size = world_size
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.compute_device = compute_device

    def forward(self, param: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return param
        if self.compute_device is not None and param.device != self.compute_device:
            param = param.to(self.compute_device, non_blocking=True)
        full = _OwnedBroadcast.apply(
            param, self.owner_rank, self.group_name, self.world_size
        )
        if self.param_dtype is not None or self.reduce_dtype is not None:
            full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
        return full


class RaggedShardParametrization(nn.Module):
    """Parametrization for RaggedShard placement."""

    def __init__(
        self,
        shard_dim: int,
        split_sizes: list[int],
        group_name: str,
        world_size: int,
        param_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype | None = None,
        compute_device: torch.device | None = None,
    ):
        super().__init__()
        self.shard_dim = shard_dim
        self.split_sizes = split_sizes
        self.max_shard_size = max(split_sizes)
        self.group_name = group_name
        self.world_size = world_size
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.compute_device = compute_device

    def forward(self, local_shard: torch.Tensor) -> torch.Tensor:
        if not _active_parametrization:
            return local_shard
        if (
            self.compute_device is not None
            and local_shard.device != self.compute_device
        ):
            local_shard = local_shard.to(self.compute_device, non_blocking=True)

        local_size = local_shard.shape[self.shard_dim]
        pad_size = self.max_shard_size - local_size
        if pad_size > 0:
            pad_shape = list(local_shard.shape)
            pad_shape[self.shard_dim] = pad_size
            padding = local_shard.new_zeros(pad_shape)
            local_shard = torch.cat([local_shard, padding], dim=self.shard_dim)

        full = torch.ops._c10d_functional.all_gather_into_tensor(
            local_shard, self.world_size, self.group_name
        )
        full = torch.ops._c10d_functional.wait_tensor(full)
        if full.requires_grad and torch.is_grad_enabled():
            full.register_hook(lambda grad: grad / self.world_size)

        chunks = full.chunk(self.world_size, dim=0)
        real_chunks = [
            chunk.narrow(self.shard_dim, 0, self.split_sizes[r])
            for r, chunk in enumerate(chunks)
        ]
        full = torch.cat(real_chunks, dim=self.shard_dim)
        if self.param_dtype is not None or self.reduce_dtype is not None:
            full = _MixedPrecisionCast.apply(full, self.param_dtype, self.reduce_dtype)
        return full


__all__ = [
    "disable_active_parametrization",
    "FlatShardParametrization",
    "OwnedParametrization",
    "RaggedShardParametrization",
    "ShardParametrization",
]
