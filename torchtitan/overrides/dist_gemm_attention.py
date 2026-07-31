# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed linear projections for TP+SP attention.

The reusable pieces are :class:`DistFusedQKVLinear` and
:class:`DistRowwiseLinear`. They preserve the stock parameter layouts, but move
the TP collective into an eager autograd function:

* ``DistFusedQKVLinear`` uses ``fused_all_gather_matmul`` in forward.
* ``DistRowwiseLinear`` uses ``fused_matmul_reduce_scatter`` in forward.

The ``dist_gemm_attention`` override only wires these blocks into
:class:`GQAttention` and removes the parent attention-boundary all-gather.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from torchtitan.config import derive, override
from torchtitan.models.common.attention import FusedQKVLinear, GQAttention
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.distributed.parallel_dims import ParallelDims


_SYMM_MEM_ENABLED_GROUPS: set[str] = set()


def _default_group() -> dist.ProcessGroup:
    return dist.distributed_c10d._get_default_group()


def _ensure_symm_mem_enabled(group: dist.ProcessGroup) -> str:
    group_name = group.group_name
    if group_name not in _SYMM_MEM_ENABLED_GROUPS:
        import torch.distributed._symmetric_memory as symm_mem

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`enable_symm_mem_for_group` is deprecated.*",
                category=FutureWarning,
            )
            symm_mem.enable_symm_mem_for_group(group_name)
        _SYMM_MEM_ENABLED_GROUPS.add(group_name)
    return group_name


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _tp_axis(dtensor: DTensor) -> int | None:
    mesh_dim_names = dtensor.device_mesh.mesh_dim_names
    if mesh_dim_names is None or "tp" not in mesh_dim_names:
        return None
    return tuple(mesh_dim_names).index("tp")


def _is_tp_sequence_sharded(tensor: torch.Tensor) -> bool:
    if not isinstance(tensor, DTensor):
        return False
    tp_axis = _tp_axis(tensor)
    if tp_axis is None:
        return False
    placement = tensor.placements[tp_axis]
    return isinstance(placement, Shard) and placement.dim == 1


def _is_tp_feature_sharded(tensor: torch.Tensor) -> bool:
    if not isinstance(tensor, DTensor):
        return False
    tp_axis = _tp_axis(tensor)
    if tp_axis is None:
        return False
    placement = tensor.placements[tp_axis]
    if not isinstance(placement, Shard):
        return False
    dim = placement.dim + tensor.ndim if placement.dim < 0 else placement.dim
    return dim == tensor.ndim - 1


def _tp_head_placements(input_dtensor: DTensor) -> tuple:
    placements = list(input_dtensor.placements)
    tp_axis = _tp_axis(input_dtensor)
    if tp_axis is None:
        raise RuntimeError("DistFusedQKVLinear requires a named TP mesh axis")
    placements[tp_axis] = Shard(2)
    return tuple(placements)


def _tp_sequence_placements(input_dtensor: DTensor) -> tuple:
    placements = list(input_dtensor.placements)
    tp_axis = _tp_axis(input_dtensor)
    if tp_axis is None:
        raise RuntimeError("DistRowwiseLinear requires a named TP mesh axis")
    placements[tp_axis] = Shard(1)
    return tuple(placements)


def _mm_with_optional_fp32_out(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except TypeError:
        return torch.mm(a.float(), b.float())


class _AllGatherLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_shard_m: torch.Tensor,
        w_shard_n: torch.Tensor,
        bias_shard_n: torch.Tensor | None,
        group: dist.ProcessGroup,
        group_name: str,
    ) -> torch.Tensor:
        if not x_shard_m.is_contiguous():
            x_shard_m = x_shard_m.contiguous()

        x_full, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x_shard_m,
            [w_shard_n.T],
            0,
            group_name,
        )
        y_shard_n = outputs[0]
        if bias_shard_n is not None:
            y_shard_n = y_shard_n + bias_shard_n

        rank = group.rank()
        world_size = group.size()
        x_shard_k = torch.chunk(x_full, world_size, dim=1)[rank].contiguous()

        ctx.save_for_backward(x_shard_k, w_shard_n)
        ctx.group_name = group_name
        ctx.has_bias = bias_shard_n is not None
        return y_shard_n

    @staticmethod
    def backward(ctx, grad_y_shard_n: torch.Tensor):
        x_shard_k, w_shard_n = ctx.saved_tensors
        if not grad_y_shard_n.is_contiguous():
            grad_y_shard_n = grad_y_shard_n.contiguous()

        grad_x_shard_m = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            grad_y_shard_n,
            w_shard_n,
            "sum",
            0,
            ctx.group_name,
        )

        # AG(X_k.T) @ dY produces dW.T. This mirrors the usual AG-linear wgrad
        # dual without depending on a higher-level distributed-linear package.
        _, grad_w_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            x_shard_k.T.contiguous(),
            [grad_y_shard_n],
            0,
            ctx.group_name,
        )
        grad_w_shard_n = grad_w_outputs[0].T.contiguous()
        grad_bias = grad_y_shard_n.sum(dim=0) if ctx.has_bias else None
        return grad_x_shard_m, grad_w_shard_n, grad_bias, None, None


def _all_gather_linear(
    x_shard_m: torch.Tensor,
    w_shard_n: torch.Tensor,
    bias_shard_n: torch.Tensor | None,
    *,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    group_name = _ensure_symm_mem_enabled(group)
    return _AllGatherLinear.apply(
        x_shard_m,
        w_shard_n,
        bias_shard_n,
        group,
        group_name,
    )


class _RowwiseLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_shard_k: torch.Tensor,
        w_shard_k: torch.Tensor,
        bias: torch.Tensor | None,
        group: dist.ProcessGroup,
        group_name: str,
    ) -> torch.Tensor:
        if not x_shard_k.is_contiguous():
            x_shard_k = x_shard_k.contiguous()

        y_shard_m = torch.ops.symm_mem.fused_matmul_reduce_scatter(
            x_shard_k,
            w_shard_k.T,
            "sum",
            1,
            group_name,
        )
        if bias is not None:
            y_shard_m = y_shard_m + bias

        ctx.save_for_backward(x_shard_k, w_shard_k)
        ctx.group = group
        ctx.group_name = group_name
        ctx.has_bias = bias is not None
        return y_shard_m

    @staticmethod
    def backward(ctx, grad_y_shard_m: torch.Tensor):
        x_shard_k, w_shard_k = ctx.saved_tensors
        if not grad_y_shard_m.is_contiguous():
            grad_y_shard_m = grad_y_shard_m.contiguous()

        grad_y, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            grad_y_shard_m,
            [w_shard_k],
            1,
            ctx.group_name,
        )
        grad_x_shard_k = outputs[0]

        grad_y_2d = grad_y.flatten(0, -2)
        x_2d = x_shard_k.flatten(0, -2)
        grad_w_shard_k = _mm_with_optional_fp32_out(grad_y_2d.T, x_2d)
        if grad_w_shard_k.dtype != w_shard_k.dtype:
            grad_w_shard_k = grad_w_shard_k.to(dtype=w_shard_k.dtype)

        grad_bias = None
        if ctx.has_bias:
            reduce_dims = tuple(range(grad_y_shard_m.ndim - 1))
            grad_bias = grad_y_shard_m.sum(dim=reduce_dims)
            dist.all_reduce(grad_bias, group=ctx.group)

        return grad_x_shard_k, grad_w_shard_k, grad_bias, None, None


def _rowwise_linear(
    x_shard_k: torch.Tensor,
    w_shard_k: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    group_name = _ensure_symm_mem_enabled(group)
    return _RowwiseLinear.apply(
        x_shard_k,
        w_shard_k,
        bias,
        group,
        group_name,
    )


class DistFusedQKVLinear(FusedQKVLinear):
    """Fused QKV projection whose forward all-gathers the TP sequence shard."""

    @dataclass(kw_only=True, slots=True)
    class Config(FusedQKVLinear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.tp_group: dist.ProcessGroup | None = None

    def parallelize(self, parallel_dims: "ParallelDims") -> None:
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is not None:
            self.tp_group = tp_mesh.get_group("tp")
        super().parallelize(parallel_dims)

    def forward(  # pyrefly: ignore[bad-override]
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.tp_group is None or not _is_tp_sequence_sharded(x):
            return super().forward(x)

        assert isinstance(x, DTensor)
        x_local = x.to_local()
        bsz, _, dim = x_local.shape
        qkv_flat = _all_gather_linear(
            x_local.reshape(-1, dim).contiguous(),
            _to_local(self.wqkv.weight),
            _to_local(self.wqkv.bias) if self.wqkv.bias is not None else None,
            group=self.tp_group,
        )

        full_seqlen = qkv_flat.shape[0] // bsz
        qkv = qkv_flat.view(
            bsz,
            full_seqlen,
            -1,
            self.r_dim,
            self.head_dim,
        )
        xq, xk, xv = torch.split(qkv, [self.heads_per_kv, 1, 1], dim=-2)
        xq = xq.reshape(bsz, full_seqlen, -1, self.head_dim).contiguous()
        xk = xk.reshape(bsz, full_seqlen, -1, self.head_dim).contiguous()
        xv = xv.reshape(bsz, full_seqlen, -1, self.head_dim).contiguous()

        placements = _tp_head_placements(x)
        return (
            DTensor.from_local(xq, x.device_mesh, placements, run_check=False),
            DTensor.from_local(xk, x.device_mesh, placements, run_check=False),
            DTensor.from_local(xv, x.device_mesh, placements, run_check=False),
        )


class DistRowwiseLinear(Linear):
    """Rowwise linear whose forward performs matmul + reduce-scatter."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.tp_group: dist.ProcessGroup | None = None

    def parallelize(self, parallel_dims: "ParallelDims") -> None:
        tp_mesh = parallel_dims.get_optional_mesh("tp")
        if tp_mesh is not None:
            self.tp_group = tp_mesh.get_group("tp")
        super().parallelize(parallel_dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.tp_group is None or not _is_tp_feature_sharded(input):
            return super().forward(input)

        assert isinstance(input, DTensor)
        y_local = _rowwise_linear(
            input.to_local(),
            _to_local(self.weight),
            _to_local(self.bias) if self.bias is not None else None,
            group=self.tp_group,
        )
        return DTensor.from_local(
            y_local,
            input.device_mesh,
            _tp_sequence_placements(input),
            run_check=False,
        )


class DistGemmGQAttention(GQAttention):
    """Stock GQA attention wired to distributed QKV and WO projections."""

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        qkv_linear: DistFusedQKVLinear.Config
        wo: DistRowwiseLinear.Config

    def parallelize(self, parallel_dims: "ParallelDims") -> None:
        # DistFusedQKVLinear owns the attention input all-gather explicitly.
        self._sharding_config = None
        super().parallelize(parallel_dims)


@override(
    target=FusedQKVLinear.Config,
    description="Use symm_mem fused all-gather matmul for fused-QKV projection.",
    exact=True,
)
def dist_fused_qkv(cfg: FusedQKVLinear.Config) -> DistFusedQKVLinear.Config:
    return derive(cfg, DistFusedQKVLinear.Config)


@override(
    target=Linear.Config,
    description="Use symm_mem fused matmul reduce-scatter for rowwise linear.",
    exact=True,
)
def dist_rowwise_linear(cfg: Linear.Config) -> DistRowwiseLinear.Config:
    base = cfg.sharding_config
    state_shardings = base.state_shardings if base is not None else {}
    return derive(
        cfg,
        DistRowwiseLinear.Config,
        sharding_config=ShardingConfig(state_shardings=state_shardings),
    )


@override(
    target=GQAttention.Config,
    description="Use distributed QKV and WO projections inside GQA attention.",
    exact=True,
)
def dist_gemm_attention(cfg: GQAttention.Config) -> DistGemmGQAttention.Config:
    if not isinstance(cfg.qkv_linear, FusedQKVLinear.Config):
        raise TypeError(
            "dist_gemm_attention requires GQAttention.qkv_linear to be "
            f"FusedQKVLinear.Config, got {type(cfg.qkv_linear).__name__}"
        )
    return derive(
        cfg,
        DistGemmGQAttention.Config,
        sharding_config=None,
        qkv_linear=dist_fused_qkv(cfg.qkv_linear),
        wo=dist_rowwise_linear(cfg.wo),
    )


__all__ = [
    "DistFusedQKVLinear",
    "DistGemmGQAttention",
    "DistRowwiseLinear",
    "dist_fused_qkv",
    "dist_gemm_attention",
    "dist_rowwise_linear",
]
