# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 all-gather on GroupedRaggedShard with block-wise quantization.

``GroupedRaggedShard`` cuts a bucket into byte-balanced ranges. By forcing those
cuts onto whole ``block_size x block_size`` quantization-tile boundaries
(``alignment = block_size * suffix_numel``), every rank owns *complete* tiles, so
it can quantize its shard to fp8 **locally** -- no cross-rank ``amax``. The
all-gather then moves **fp8 + tiny scales** (~half the bytes), and the gathered
fp8 weight is **bit-identical** to "all-gather bf16, then block-quantize".

The master weight stays bf16-sharded (storage is unchanged); fp8 is materialized
only by the unshard collective (the torchao/float8-FSDP model, ported to the
FlexShard placement contract). See
``fp8_allgather_grouped_ragged_shard_plan.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from typing_extensions import override

from ..flex_shard.placement_contract import (
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)
from ..flex_shard.utils import (
    _record_comm_if_eager,
    _record_copy_in_if_eager,
    _record_copy_out_if_eager,
)
from .ragged_shard import GroupedRaggedShard

try:
    from torchao.prototype.blockwise_fp8_training.kernels import (
        BLOCKWISE_1X128_SCALING_TYPE as _TORCHAO_BLOCKWISE_1X128_SCALING_TYPE,
        BLOCKWISE_128X128_SCALING_TYPE as _TORCHAO_BLOCKWISE_128X128_SCALING_TYPE,
        triton_fp8_blockwise_act_quant_lhs as _torchao_act_quant_lhs,
        triton_fp8_gemm_1x128_128x128 as _torchao_gemm_1x128_128x128,
    )
    from torchao.prototype.blockwise_fp8_training.linear import (
        Float8BlockwiseLinear as TorchAOFloat8BlockwiseLinear,
        _run_blockwise_mm as _torchao_run_blockwise_mm,
    )

    TORCHAO_BLOCKWISE_FP8_AVAILABLE = True
except ImportError:
    TorchAOFloat8BlockwiseLinear = None
    _TORCHAO_BLOCKWISE_1X128_SCALING_TYPE = None
    _TORCHAO_BLOCKWISE_128X128_SCALING_TYPE = None
    _torchao_act_quant_lhs = None
    _torchao_gemm_1x128_128x128 = None
    _torchao_run_blockwise_mm = None
    TORCHAO_BLOCKWISE_FP8_AVAILABLE = False

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import ParamInfo, PlacementFn
    from ..flex_shard.placement_contract import Placement


EPS = 1e-12


def _as_block_pair(block_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(block_size, int):
        return block_size, block_size
    block_m, block_n = block_size
    return block_m, block_n


def blockwise_quant_weight(
    weight: torch.Tensor,
    block_size: int | tuple[int, int],
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference block-wise fp8 weight quantization."""
    if weight.ndim != 2:
        raise ValueError(
            f"blockwise_quant_weight expects 2D, got {tuple(weight.shape)}"
        )
    m, n = weight.shape
    block_m, block_n = _as_block_pair(block_size)
    if m % block_m != 0 or n % block_n != 0:
        raise ValueError(
            f"dims {tuple(weight.shape)} must be divisible by block "
            f"({block_m}, {block_n})"
        )
    fp8_max = torch.finfo(fp8_dtype).max
    tiles = (
        weight.reshape(m // block_m, block_m, n // block_n, block_n)
        .permute(0, 2, 1, 3)
        .reshape(-1, block_m * block_n)
    )
    amax = tiles.abs().amax(dim=1, keepdim=True).clamp(min=EPS).to(torch.float64)
    scale = (fp8_max / amax).to(torch.float32)
    quant = (tiles.to(torch.float32) * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    quant = (
        quant.reshape(m // block_m, n // block_n, block_m, block_n)
        .permute(0, 2, 1, 3)
        .reshape(m, n)
    )
    recip_scale = (1.0 / scale).reshape(m // block_m, n // block_n).to(torch.float32)
    return quant.contiguous(), recip_scale.contiguous()


def blockwise_dequant_weight(
    quant: torch.Tensor,
    recip_scale: torch.Tensor,
    block_size: int | tuple[int, int],
) -> torch.Tensor:
    """Inverse of :func:`blockwise_quant_weight`."""
    m, n = quant.shape
    block_m, block_n = _as_block_pair(block_size)
    tiles = (
        quant.reshape(m // block_m, block_m, n // block_n, block_n)
        .permute(0, 2, 1, 3)
        .reshape(-1, block_m * block_n)
        .to(torch.float32)
    )
    deq = tiles * recip_scale.reshape(-1, 1).to(torch.float32)
    return (
        deq.reshape(m // block_m, n // block_n, block_m, block_n)
        .permute(0, 2, 1, 3)
        .reshape(m, n)
    )


def promote_to_square_block(block_m: int, block_n: int) -> int:
    """Smallest square block that preserves both requested tile dimensions."""
    return math.lcm(block_m, block_n)


def blockwise_transpose(
    quant: torch.Tensor,
    recip_scale: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transpose a block-wise fp8 weight to the other matmul orientation.

    For **square** ``block_size x block_size`` tiles, quantization commutes with
    transpose: tile ``(i, j)`` of ``W`` and tile ``(j, i)`` of ``Wᵀ`` are the same
    elements, so ``fp8(Wᵀ) == fp8(W)ᵀ`` and ``scale(Wᵀ) == scale(W)ᵀ`` *exactly*.
    A single gathered ``W`` can therefore serve both the forward (RHS ``Wᵀ``) and the
    backward dgrad (RHS ``W``) with no re-quantization and no second all-gather.

    Returns zero-copy transposed views ``(quant.t(), recip_scale.t())``: a row-major
    ``(M, N)`` weight becomes a **column-major** ``(N, M)`` view -- the layout the fp8
    GEMM RHS expects for the forward pass.

    GUARD: this identity holds only for square tiles. ``recip_scale`` must have
    shape ``(M // block_size, N // block_size)``; non-square tiling regroups
    elements under transpose and is rejected.
    """
    if quant.ndim != 2:
        raise ValueError(
            f"blockwise_transpose expects a 2D weight, got {tuple(quant.shape)}"
        )
    m, n = quant.shape
    expected = (m // block_size, n // block_size)
    if (
        m % block_size != 0
        or n % block_size != 0
        or tuple(recip_scale.shape) != expected
    ):
        raise ValueError(
            "blockwise_transpose requires square block_size x block_size tiling: "
            f"weight {tuple(quant.shape)} with block_size {block_size} expects scale "
            f"shape {expected}, but got {tuple(recip_scale.shape)}. The "
            "fp8(W^T) == fp8(W)^T transpose-invariance holds only for square tiles."
        )
    return quant.t(), recip_scale.t()


class BlockwiseFp8Weight(torch.Tensor):
    """Blockwise FP8 weight returned by the FlexShard all-gather."""

    __slots__ = ["_data", "_recip_scale", "_block_size", "_orig_dtype"]

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        recip_scale: torch.Tensor,
        block_size: int,
        orig_dtype: torch.dtype = torch.bfloat16,
        requires_grad: bool = False,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=orig_dtype,
            layout=data.layout,
            device=data.device,
            requires_grad=requires_grad,
        )

    def __init__(
        self,
        data: torch.Tensor,
        recip_scale: torch.Tensor,
        block_size: int,
        orig_dtype: torch.dtype = torch.bfloat16,
        requires_grad: bool = False,
    ) -> None:
        _ = requires_grad
        if data.ndim != 2:
            raise ValueError(f"BlockwiseFp8Weight expects 2D data, got {data.shape}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        out_dim, in_dim = data.shape
        if out_dim % block_size != 0 or in_dim % block_size != 0:
            raise ValueError(
                f"data shape {tuple(data.shape)} must be divisible by block_size "
                f"{block_size}"
            )
        expected_scale = (out_dim // block_size, in_dim // block_size)
        if tuple(recip_scale.shape) != expected_scale:
            raise ValueError(
                f"scale shape must be {expected_scale} for data {tuple(data.shape)} "
                f"and block_size {block_size}, got {tuple(recip_scale.shape)}"
            )
        if data.device != recip_scale.device:
            raise ValueError(
                f"data and scale must be on the same device, got {data.device} and "
                f"{recip_scale.device}"
            )
        self._data = data
        self._recip_scale = recip_scale
        self._block_size = block_size
        self._orig_dtype = orig_dtype

    @property
    def fp8_data(self) -> torch.Tensor:
        return self._data

    @property
    def recip_scale(self) -> torch.Tensor:
        return self._recip_scale

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def orig_dtype(self) -> torch.dtype:
        return self._orig_dtype

    def transpose_for_forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return blockwise_transpose(self._data, self._recip_scale, self._block_size)

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        result = blockwise_dequant_weight(
            self._data,
            self._recip_scale,
            self._block_size,
        )
        return result.to(dtype or self._orig_dtype)

    def __tensor_flatten__(self) -> tuple[list[str], dict[str, Any]]:
        return ["_data", "_recip_scale"], {
            "block_size": self._block_size,
            "orig_dtype": self._orig_dtype,
            "requires_grad": self.requires_grad,
        }

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any],
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
    ) -> "BlockwiseFp8Weight":
        _ = outer_size, outer_stride
        return BlockwiseFp8Weight(
            inner_tensors["_data"],
            inner_tensors["_recip_scale"],
            metadata["block_size"],
            metadata["orig_dtype"],
            requires_grad=metadata["requires_grad"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.detach.default:
            tensor = args[0]
            return BlockwiseFp8Weight(
                tensor._data.detach(),
                tensor._recip_scale.detach(),
                tensor._block_size,
                tensor._orig_dtype,
                requires_grad=False,
            )

        def unwrap(t: "BlockwiseFp8Weight") -> torch.Tensor:
            return t.dequantize()

        args, kwargs = pytree.tree_map_only(
            BlockwiseFp8Weight,
            unwrap,
            (args, kwargs or {}),
        )
        return func(*args, **kwargs)

    __torch_function__ = torch._C._disabled_torch_function_impl


_FlexShardFloat8BlockwiseLinearBase = (
    TorchAOFloat8BlockwiseLinear
    if TorchAOFloat8BlockwiseLinear is not None
    else nn.Linear
)


class _PrequantizedBlockwiseFp8Linear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: BlockwiseFp8Weight,
        bias: torch.Tensor | None,
        block_size: int,
        out_dtype: torch.dtype,
        use_triton: bool,
    ) -> torch.Tensor:
        if (
            _torchao_act_quant_lhs is None
            or _torchao_run_blockwise_mm is None
            or _torchao_gemm_1x128_128x128 is None
        ):
            raise ImportError(
                "torchao is required to consume BlockwiseFp8Weight with "
                "FlexShardFloat8BlockwiseLinear."
            )
        if block_size != weight.block_size:
            raise ValueError(
                f"linear block_size {block_size} does not match gathered weight "
                f"block_size {weight.block_size}"
            )
        if block_size != 128:
            raise AssertionError("torchao Float8BlockwiseLinear only supports 128")

        x_orig_shape = x.shape
        x_2d = x.reshape(-1, x_orig_shape[-1])
        x_fp8, x_scale = _torchao_act_quant_lhs(x_2d, block_size)
        weight_t_fp8, weight_t_scale = weight.transpose_for_forward()
        out = _torchao_run_blockwise_mm(
            use_triton=use_triton,
            triton_kernel=_torchao_gemm_1x128_128x128,
            mat_a=x_fp8,
            mat_b=weight_t_fp8,
            scale_a=x_scale,
            scale_recipe_a=_TORCHAO_BLOCKWISE_1X128_SCALING_TYPE,
            scale_b=weight_t_scale,
            scale_recipe_b=_TORCHAO_BLOCKWISE_128X128_SCALING_TYPE,
            out_dtype=out_dtype,
        )
        out = out.reshape(*x_orig_shape[:-1], out.shape[-1])
        if bias is not None:
            out = out + bias
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
        x, weight = ctx.saved_tensors
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        weight_hp = weight.dequantize(dtype=x.dtype)

        grad_x = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output_2d.to(x.dtype).matmul(weight_hp).reshape_as(x)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output_2d.t().matmul(x_2d).to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(dim=0)
        return grad_x, grad_weight, grad_bias, None, None, None


class FlexShardFloat8BlockwiseLinear(_FlexShardFloat8BlockwiseLinearBase):
    """torchao Float8BlockwiseLinear that consumes FlexShard FP8 all-gather output."""

    def __init__(self, *args, **kwargs) -> None:
        if TorchAOFloat8BlockwiseLinear is None:
            raise ImportError(
                "torchao is not installed. Please install torchao to use "
                "FlexShardFloat8BlockwiseLinear."
            )
        super().__init__(*args, **kwargs)

    @classmethod
    def from_float(cls, mod):
        if TorchAOFloat8BlockwiseLinear is None:
            raise ImportError(
                "torchao is not installed. Please install torchao to use "
                "FlexShardFloat8BlockwiseLinear."
            )
        return TorchAOFloat8BlockwiseLinear.from_float.__func__(cls, mod)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        # Without FlexShard, self.weight is the module's normal high-precision
        # Parameter and torchao quantizes it in forward. With the fp8 FlexShard
        # placement, the dynamic parameter getter returns the already-gathered
        # BlockwiseFp8Weight, so skip weight quantization and consume it directly.
        if not isinstance(weight, BlockwiseFp8Weight):
            return super().forward(x)
        return _PrequantizedBlockwiseFp8Linear.apply(
            x,
            weight,
            self.bias,
            self.block_size,
            self.dtype,
            self.use_triton,
        )


def convert_to_flex_shard_float8_blockwise_linear(
    module: nn.Module,
    filter_fn: Callable[[str, nn.Linear], bool] | None = None,
) -> nn.Module:
    """Replace matching nn.Linear children with the FlexShard torchao wrapper."""
    if TorchAOFloat8BlockwiseLinear is None:
        raise ImportError(
            "torchao is not installed. Please install torchao to use "
            "FlexShardFloat8BlockwiseLinear."
        )

    def should_convert(fqn: str, child: nn.Linear) -> bool:
        return filter_fn(fqn, child) if filter_fn is not None else True

    def convert_children(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            fqn = f"{prefix}.{name}" if prefix else name
            if isinstance(child, FlexShardFloat8BlockwiseLinear):
                continue
            if isinstance(child, nn.Linear) and should_convert(fqn, child):
                setattr(parent, name, FlexShardFloat8BlockwiseLinear.from_float(child))
            else:
                convert_children(child, fqn)

    convert_children(module)
    return module


class Fp8BlockwiseGroupedRaggedShard(GroupedRaggedShard):
    """``GroupedRaggedShard`` that all-gathers block-wise fp8 instead of bf16.

    The byte cut is aligned to whole ``block_size x block_size`` tiles
    (``alignment = block_size * suffix_numel``), so each rank owns complete tiles
    and quantizes its shard locally; the gathered fp8 + scales are bit-identical to
    gathering bf16 then quantizing. Storage stays bf16 (the optimizer's master
    weight); only the unshard produces fp8. ``finish_prepared_unshard`` returns a
    :class:`BlockwiseFp8Weight` for an fp8-aware consumer. Currently requires even
    ``local_units``.
    """

    @dataclass(frozen=True)
    class _Fp8UnshardState:
        infos: list[ParamInfo]
        pg: Any
        debug_fqn: str | None
        rank_offsets: tuple[int, ...]
        rank_numels: tuple[int, ...]

    def __init__(
        self,
        dims: tuple[int, ...] = (0,),
        local_units: tuple[int, ...] = (1,),
        block_size: int = 128,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        super().__init__(dims, local_units)
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")
        self.block_size = block_size
        self.fp8_dtype = fp8_dtype

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fp8BlockwiseGroupedRaggedShard):
            return False
        return (
            self.dims == other.dims
            and self.local_units == other.local_units
            and self.block_size == other.block_size
            and self.fp8_dtype == other.fp8_dtype
        )

    def __hash__(self) -> int:
        return hash(
            (type(self), self.dims, self.local_units, self.block_size, self.fp8_dtype)
        )

    def __repr__(self) -> str:
        return (
            "Fp8BlockwiseGroupedRaggedShard("
            f"dims={self.dims}, local_units={self.local_units}, "
            f"block_size={self.block_size})"
        )

    def _require_even_units(self) -> None:
        if len(set(self.local_units)) != 1:
            raise ValueError(
                "Fp8BlockwiseGroupedRaggedShard currently requires even local_units "
                f"(all equal) so every rank's shard is the same size; got "
                f"{self.local_units}."
            )

    @override
    def _param_alignment_numel(
        self,
        named_params: list[tuple[str, nn.Parameter]],
    ) -> int:
        # Align the byte cut to whole block x block tiles: every rank then owns
        # complete tiles and can quantize locally. Requires both the sharded
        # (prefix) and the kept (suffix) dims to be multiples of block_size.
        alignment = 1
        for fqn, param in named_params:
            prefix_numel = self._prefix_numel(param.shape)
            suffix_numel = self._suffix_numel(param.shape)
            if (
                prefix_numel % self.block_size != 0
                or suffix_numel % self.block_size != 0
            ):
                raise ValueError(
                    "Fp8BlockwiseGroupedRaggedShard requires both the sharded prefix "
                    f"({prefix_numel}) and the kept suffix ({suffix_numel}) of {fqn!r} "
                    f"to be divisible by block_size ({self.block_size}) so the "
                    f"{self.block_size}x{self.block_size} tiles are whole."
                )
            alignment = math.lcm(alignment, self.block_size * suffix_numel)
        return alignment

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Quantize each local shard to fp8 + scales and pack the send buffers."""
        self._require_even_units()
        bs = self.block_size
        tile_numel = bs * bs
        bucket_layout = self._bucket_layout(infos[0])
        rank = mesh.get_local_rank()
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        rank_offset = bucket_layout.rank_offsets[rank]
        rank_numel = bucket_layout.rank_numels[rank]

        with _record_copy_in_if_eager():
            local_fp8 = torch.zeros(rank_numel, dtype=self.fp8_dtype, device=device)
            local_scale = torch.zeros(
                rank_numel // tile_numel, dtype=torch.float32, device=device
            )
            for tensor, info in zip(tensors, infos, strict=True):
                if info.local_numel == 0:
                    continue
                quant, recip_scale = blockwise_quant_weight(
                    tensor.reshape(info.local_shape), bs, self.fp8_dtype
                )
                offset = self._param_layout(info).local_global_offset - rank_offset
                local_fp8[offset : offset + info.local_numel].copy_(quant.reshape(-1))
                local_scale[
                    offset // tile_numel : offset // tile_numel
                    + info.local_numel // tile_numel
                ].copy_(recip_scale.reshape(-1))

        gathered_fp8 = torch.empty(
            bucket_layout.global_numel, dtype=self.fp8_dtype, device=device
        )
        gathered_scale = torch.empty(
            bucket_layout.global_numel // tile_numel, dtype=torch.float32, device=device
        )
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[local_fp8, local_scale, gathered_fp8, gathered_scale],
            placement_state=Fp8BlockwiseGroupedRaggedShard._Fp8UnshardState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                rank_offsets=bucket_layout.rank_offsets,
                rank_numels=bucket_layout.rank_numels,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        """All-gather the fp8 data (as bytes) and the fp32 scales.

        Uses the list ``dist.all_gather`` (op ``c10d.allgather_``, like the bf16
        ``GroupedRaggedShard``), which reshard-after-forward already tags
        ``MUST_RECOMPUTE`` -- so the fp8 unshard is freed after forward and
        re-quantized + re-gathered in backward. fp8 is viewed as uint8 so the
        collective is dtype-agnostic across NCCL versions (fp8 is 1 byte, 1:1).
        """
        state = prepared.placement_state
        if not isinstance(state, Fp8BlockwiseGroupedRaggedShard._Fp8UnshardState):
            raise AssertionError(
                "Expected Fp8BlockwiseGroupedRaggedShard._Fp8UnshardState, "
                f"got {type(state).__name__}"
            )
        local_fp8, local_scale, gathered_fp8, gathered_scale = prepared.buffers
        tile_numel = self.block_size * self.block_size
        # Per-rank destination views: gathered[rank r range] receives rank r's shard.
        fp8_views = [
            gathered_fp8[offset : offset + numel].view(torch.uint8)
            for offset, numel in zip(state.rank_offsets, state.rank_numels, strict=True)
        ]
        scale_views = [
            gathered_scale[
                offset // tile_numel : offset // tile_numel + numel // tile_numel
            ]
            for offset, numel in zip(state.rank_offsets, state.rank_numels, strict=True)
        ]
        with _record_comm_if_eager("FlexShard::all_gather", state.debug_fqn):
            dist.all_gather(fp8_views, local_fp8.view(torch.uint8), group=state.pg)
            dist.all_gather(scale_views, local_scale, group=state.pg)

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Slice each param's full fp8 + scales from the gathered buffers."""
        if not isinstance(
            prepared.placement_state, Fp8BlockwiseGroupedRaggedShard._Fp8UnshardState
        ):
            raise AssertionError(
                "Expected Fp8BlockwiseGroupedRaggedShard._Fp8UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        gathered_fp8 = prepared.buffers[2]
        gathered_scale = prepared.buffers[3]
        bs = self.block_size
        tile_numel = bs * bs
        full_params: list[torch.Tensor] = []
        with _record_copy_out_if_eager():
            for info in prepared.placement_state.infos:
                param_offset = self._param_layout(info).param_offset
                out_dim, in_dim = info.global_shape
                data = gathered_fp8[
                    param_offset : param_offset + info.global_numel
                ].view(info.global_shape)
                num_scales = (out_dim // bs) * (in_dim // bs)
                scale = gathered_scale[
                    param_offset // tile_numel : param_offset // tile_numel + num_scales
                ].view(out_dim // bs, in_dim // bs)
                full_params.append(
                    BlockwiseFp8Weight(
                        data,
                        scale,
                        bs,
                        orig_dtype=info.unsharded_dtype,
                        requires_grad=info.requires_grad,
                    )
                )
        return PlacementUnshardResult(full_params, prepared.buffers)


def make_fp8_blockwise_grouped_ragged_placement_fn(
    *,
    block_size: int = 128,
    local_units: tuple[int, ...],
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> PlacementFn:
    """Return a placement_fn assigning one fp8 GroupedRaggedShard per bucket."""

    def fp8_blockwise_placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        if len(local_units) != mesh.size():
            raise ValueError(
                "Fp8BlockwiseGroupedRaggedShard local_units length must match mesh "
                f"size: got {len(local_units)} local units for mesh size {mesh.size()}."
            )
        placement = Fp8BlockwiseGroupedRaggedShard(
            dims=(0,),
            local_units=local_units,
            block_size=block_size,
            fp8_dtype=fp8_dtype,
        )
        return {fqn: (placement,) for fqn, _ in named_params}

    return fp8_blockwise_placements


def make_flex_shard_float8_blockwise_all_gather_placement_fn(
    *,
    block_size: int = 128,
    local_units: tuple[int, ...],
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> PlacementFn:
    """Placement fn for Float8BlockwiseLinear + FlexShard FP8 all-gather."""
    return make_fp8_blockwise_grouped_ragged_placement_fn(
        block_size=block_size,
        local_units=local_units,
        fp8_dtype=fp8_dtype,
    )


class Fp8TwoOrientationGroupedRaggedShard(GroupedRaggedShard):
    """Non-square block-wise fp8 all-gather with one buffer per orientation."""

    @dataclass(frozen=True)
    class _TwoOrientationState:
        infos: list[ParamInfo]
        pg: Any
        debug_fqn: str | None
        rank_offsets: tuple[int, ...]
        rank_numels: tuple[int, ...]

    def __init__(
        self,
        dims: tuple[int, ...] = (0,),
        local_units: tuple[int, ...] = (1,),
        block_size: int = 128,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        super().__init__(dims, local_units)
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")
        self.block_size = block_size
        self.fp8_dtype = fp8_dtype

    def __eq__(self, other: object) -> bool:
        if type(other) is not Fp8TwoOrientationGroupedRaggedShard:
            return False
        return (
            self.dims == other.dims
            and self.local_units == other.local_units
            and self.block_size == other.block_size
            and self.fp8_dtype == other.fp8_dtype
        )

    def __hash__(self) -> int:
        return hash(
            (type(self), self.dims, self.local_units, self.block_size, self.fp8_dtype)
        )

    def __repr__(self) -> str:
        return (
            "Fp8TwoOrientationGroupedRaggedShard("
            f"dims={self.dims}, local_units={self.local_units}, "
            f"block_size={self.block_size})"
        )

    def _require_even_units(self) -> None:
        if len(set(self.local_units)) != 1:
            raise ValueError(
                "Fp8TwoOrientationGroupedRaggedShard currently requires even "
                f"local_units (all equal); got {self.local_units}."
            )

    @override
    def _param_alignment_numel(
        self,
        named_params: list[tuple[str, nn.Parameter]],
    ) -> int:
        alignment = 1
        for fqn, param in named_params:
            prefix_numel = self._prefix_numel(param.shape)
            suffix_numel = self._suffix_numel(param.shape)
            if (
                prefix_numel % self.block_size != 0
                or suffix_numel % self.block_size != 0
            ):
                raise ValueError(
                    "Fp8TwoOrientationGroupedRaggedShard requires both the sharded "
                    f"prefix ({prefix_numel}) and the kept suffix ({suffix_numel}) of "
                    f"{fqn!r} to be divisible by block_size ({self.block_size})."
                )
            alignment = math.lcm(alignment, self.block_size * suffix_numel)
        return alignment

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        self._require_even_units()
        bs = self.block_size
        bucket_layout = self._bucket_layout(infos[0])
        rank = mesh.get_local_rank()
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        rank_offset = bucket_layout.rank_offsets[rank]
        rank_numel = bucket_layout.rank_numels[rank]

        with _record_copy_in_if_eager():
            local_fwd = torch.zeros(rank_numel, dtype=self.fp8_dtype, device=device)
            local_bwd = torch.zeros(rank_numel, dtype=self.fp8_dtype, device=device)
            local_scale_fwd = torch.zeros(
                rank_numel // bs, dtype=torch.float32, device=device
            )
            local_scale_bwd = torch.zeros(
                rank_numel // bs, dtype=torch.float32, device=device
            )
            for tensor, info in zip(tensors, infos, strict=True):
                if info.local_numel == 0:
                    continue
                shard = tensor.reshape(info.local_shape)
                q_fwd, s_fwd = blockwise_quant_weight(shard, (1, bs), self.fp8_dtype)
                q_bwd, s_bwd = blockwise_quant_weight(shard, (bs, 1), self.fp8_dtype)
                offset = self._param_layout(info).local_global_offset - rank_offset
                local_fwd[offset : offset + info.local_numel].copy_(q_fwd.reshape(-1))
                local_bwd[offset : offset + info.local_numel].copy_(q_bwd.reshape(-1))
                scale_offset = offset // bs
                scale_numel = info.local_numel // bs
                local_scale_fwd[scale_offset : scale_offset + scale_numel].copy_(
                    s_fwd.reshape(-1)
                )
                local_scale_bwd[scale_offset : scale_offset + scale_numel].copy_(
                    s_bwd.reshape(-1)
                )

        gathered_fwd = torch.empty(
            bucket_layout.global_numel, dtype=self.fp8_dtype, device=device
        )
        gathered_bwd = torch.empty(
            bucket_layout.global_numel, dtype=self.fp8_dtype, device=device
        )
        gathered_scale_fwd = torch.empty(
            bucket_layout.global_numel // bs, dtype=torch.float32, device=device
        )
        gathered_scale_bwd = torch.empty(
            bucket_layout.global_numel // bs, dtype=torch.float32, device=device
        )
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[
                local_fwd,
                local_bwd,
                local_scale_fwd,
                local_scale_bwd,
                gathered_fwd,
                gathered_bwd,
                gathered_scale_fwd,
                gathered_scale_bwd,
            ],
            placement_state=Fp8TwoOrientationGroupedRaggedShard._TwoOrientationState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                rank_offsets=bucket_layout.rank_offsets,
                rank_numels=bucket_layout.rank_numels,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        state = prepared.placement_state
        if not isinstance(
            state, Fp8TwoOrientationGroupedRaggedShard._TwoOrientationState
        ):
            raise AssertionError(
                "Expected Fp8TwoOrientationGroupedRaggedShard._TwoOrientationState, "
                f"got {type(state).__name__}"
            )
        (
            local_fwd,
            local_bwd,
            local_scale_fwd,
            local_scale_bwd,
            gathered_fwd,
            gathered_bwd,
            gathered_scale_fwd,
            gathered_scale_bwd,
        ) = prepared.buffers
        bs = self.block_size

        def _data_views(gathered: torch.Tensor) -> list[torch.Tensor]:
            return [
                gathered[offset : offset + numel].view(torch.uint8)
                for offset, numel in zip(
                    state.rank_offsets, state.rank_numels, strict=True
                )
            ]

        def _scale_views(gathered: torch.Tensor) -> list[torch.Tensor]:
            return [
                gathered[offset // bs : offset // bs + numel // bs]
                for offset, numel in zip(
                    state.rank_offsets, state.rank_numels, strict=True
                )
            ]

        with _record_comm_if_eager("FlexShard::all_gather", state.debug_fqn):
            dist.all_gather(
                _data_views(gathered_fwd), local_fwd.view(torch.uint8), group=state.pg
            )
            dist.all_gather(
                _data_views(gathered_bwd), local_bwd.view(torch.uint8), group=state.pg
            )
            dist.all_gather(
                _scale_views(gathered_scale_fwd), local_scale_fwd, group=state.pg
            )
            dist.all_gather(
                _scale_views(gathered_scale_bwd), local_scale_bwd, group=state.pg
            )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        state = prepared.placement_state
        if not isinstance(
            state, Fp8TwoOrientationGroupedRaggedShard._TwoOrientationState
        ):
            raise AssertionError(
                "Expected Fp8TwoOrientationGroupedRaggedShard._TwoOrientationState, "
                f"got {type(state).__name__}"
            )
        gathered_fwd = prepared.buffers[4]
        gathered_bwd = prepared.buffers[5]
        gathered_scale_fwd = prepared.buffers[6]
        gathered_scale_bwd = prepared.buffers[7]
        bs = self.block_size
        full_params: list[torch.Tensor] = []
        with _record_copy_out_if_eager():
            for info in state.infos:
                param_offset = self._param_layout(info).param_offset
                out_dim, in_dim = info.global_shape
                num = info.global_numel
                data_fwd = gathered_fwd[param_offset : param_offset + num].view(
                    info.global_shape
                )
                data_bwd = gathered_bwd[param_offset : param_offset + num].view(
                    info.global_shape
                )
                scale_start = param_offset // bs
                scale_fwd = gathered_scale_fwd[
                    scale_start : scale_start + num // bs
                ].view(out_dim, in_dim // bs)
                scale_bwd = gathered_scale_bwd[
                    scale_start : scale_start + num // bs
                ].view(out_dim // bs, in_dim)
                data_fwd._scale_forward = scale_fwd
                data_fwd._fp8_backward = data_bwd
                data_fwd._scale_backward = scale_bwd
                data_fwd._block_size = bs
                full_params.append(data_fwd)
        return PlacementUnshardResult(full_params, prepared.buffers)


def make_fp8_two_orientation_grouped_ragged_placement_fn(
    *,
    block_size: int = 128,
    local_units: tuple[int, ...],
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> PlacementFn:
    """Return a placement_fn assigning one two-orientation fp8 shard per bucket."""

    def fp8_two_orientation_placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        if len(local_units) != mesh.size():
            raise ValueError(
                "Fp8TwoOrientationGroupedRaggedShard local_units length must match mesh "
                f"size: got {len(local_units)} local units for mesh size {mesh.size()}."
            )
        placement = Fp8TwoOrientationGroupedRaggedShard(
            dims=(0,),
            local_units=local_units,
            block_size=block_size,
            fp8_dtype=fp8_dtype,
        )
        return {fqn: (placement,) for fqn, _ in named_params}

    return fp8_two_orientation_placements


__all__ = [
    "BlockwiseFp8Weight",
    "blockwise_dequant_weight",
    "blockwise_quant_weight",
    "blockwise_transpose",
    "convert_to_flex_shard_float8_blockwise_linear",
    "FlexShardFloat8BlockwiseLinear",
    "Fp8BlockwiseGroupedRaggedShard",
    "Fp8TwoOrientationGroupedRaggedShard",
    "make_flex_shard_float8_blockwise_all_gather_placement_fn",
    "make_fp8_blockwise_grouped_ragged_placement_fn",
    "make_fp8_two_orientation_grouped_ragged_placement_fn",
    "promote_to_square_block",
    "TORCHAO_BLOCKWISE_FP8_AVAILABLE",
    "TorchAOFloat8BlockwiseLinear",
]
