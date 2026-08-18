# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernels.

Tensor suffix: ``TNH`` = tokens, heads, head dimension.
"""

from dataclasses import dataclass, fields
from typing import cast, ClassVar, Literal, TYPE_CHECKING

import spmd_types as spmd

import torch
import torch.distributed as dist

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.spmd_types import current_spmd_mesh

from torchtitan.models.common.attention import (
    BaseAttention,
    FlexAttention,
    VarlenAttention,
    VarlenMetadata,
)
from torchtitan.protocols.module import Module

if TYPE_CHECKING:
    from torchtitan.trainer import Trainer

__all__ = [
    "ContextParallelKernel",
    "AllGatherCPFlexAttention",
    "AllGatherCPVarlenAttention",
    "UlyssesCPKernel",
    "UlyssesCPFlexAttention",
    "UlyssesCPVarlenAttention",
    "use_cp_kernel",
]

_SEQ_DIM = 0
_HEAD_DIM = 1


def use_cp_kernel(config: "Trainer.Config", kernel: type[Module]) -> None:
    """Replace each attention kernel config while preserving its fields.

    TODO: Support multiple CP kernels for different attention types.
    """
    if not issubclass(kernel, ContextParallelKernel):
        raise ValueError(f"{kernel.__qualname__} must inherit ContextParallelKernel.")
    assert config.model_spec is not None, "model_spec is required"
    for _, traversed, _, _ in config.model_spec.model.traverse(BaseAttention.Config):
        # traverse returns the base config type.
        attention = cast(BaseAttention.Config, traversed)
        existing = attention.inner_attention
        if not issubclass(kernel.Config, type(existing)):
            raise ValueError(
                f"{kernel.__qualname__}.Config must inherit "
                f"{type(existing).__qualname__}."
            )
        attention.inner_attention = kernel.Config(
            **{f.name: getattr(existing, f.name) for f in fields(existing)}
        )


class ContextParallelKernel:
    """Mixin for attention kernels that own their CP collectives.

    Note that Flux doesn't use this kernel.
    """

    @property
    def cp_group(self) -> dist.ProcessGroup:
        """Return the active multi-rank CP process group."""
        mesh = current_spmd_mesh()
        if mesh is None:
            raise RuntimeError(
                f"{type(self).__name__} requires an active SPMD mesh context."
            )
        mesh_axis_names = mesh.mesh_dim_names or ()
        cp_group = mesh.get_group("cp") if "cp" in mesh_axis_names else None
        if cp_group is None or cp_group.size() == 1:
            raise RuntimeError(f"{type(self).__name__} requires an active CP mesh.")
        return cp_group


def _all_gather_kv(
    k_TNH: torch.Tensor,
    v_TNH: torch.Tensor,
    cp_group: dist.ProcessGroup,
    reduce_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K and V over the token dim; the backward reduce-scatters.

    ``reduce_dtype`` picks the dtype of that reduction; None keeps the input
    dtype.
    """
    k_TNH, v_TNH = (
        spmd.redistribute(
            x_TNH,
            cp_group,
            src=spmd.S(_SEQ_DIM),
            dst=spmd.R,
            backward_options={"op_dtype": reduce_dtype or x_TNH.dtype},
        )
        for x_TNH in (k_TNH, v_TNH)
    )
    return k_TNH, v_TNH


class AllGatherCPFlexAttention(ContextParallelKernel, FlexAttention):
    """FlexAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        reduce_dtype: Literal["float32", "bfloat16"] | None = None
        """Dtype of the backward reduce-scatter. None keeps the input dtype."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = (
            TORCH_DTYPE_MAP[config.reduce_dtype] if config.reduce_dtype else None
        )

    def forward(
        self,
        q_TNH: torch.Tensor,
        k_TNH: torch.Tensor,
        v_TNH: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        k_TNH, v_TNH = _all_gather_kv(k_TNH, v_TNH, self.cp_group, self.reduce_dtype)
        return super().forward(q_TNH, k_TNH, v_TNH, **kwargs)


class AllGatherCPVarlenAttention(ContextParallelKernel, VarlenAttention):
    """VarlenAttention with K/V all-gathered across the context-parallel mesh.

    Same shape as the Flex kernel above: Q stays sequence-sharded and the
    kernel sees every key and value. Varlen packs documents into one sequence,
    so gathering is not enough -- the gathered K/V still carry the other ranks'
    query regions, which this rank must not attend to. ``CPVarlenMetadata``
    carries a gather index that picks out the visible region, and the selection
    happens after the packed reshape, where the index applies.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):
        requires_causal_mask: ClassVar[bool] = True
        """CPVarlenMetadata's right-aligned construction only holds for causal."""

        reduce_dtype: Literal["float32", "bfloat16"] | None = None
        """Dtype of the backward reduce-scatter. None keeps the input dtype."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = (
            TORCH_DTYPE_MAP[config.reduce_dtype] if config.reduce_dtype else None
        )

    def _select_visible_kv(
        self,
        k_TNH: torch.Tensor,
        v_TNH: torch.Tensor,
        attention_masks: VarlenMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torchtitan.distributed.context_parallel.varlen_cp import CPVarlenMetadata

        if not isinstance(attention_masks, CPVarlenMetadata):
            raise ValueError(
                f"{type(self).__name__} needs CPVarlenMetadata, which cp_shard "
                f"builds from the global VarlenMetadata; got "
                f"{type(attention_masks).__name__}."
            )
        # The right-aligned causal construction in CPVarlenMetadata only holds
        # for causal masks.
        if self.window_size != (-1, 0):
            raise ValueError(
                "Varlen attention under context parallel only supports causal "
                f"masking (window_size=(-1, 0)); got {self.window_size}."
            )
        # A rank-identical local re-pack: the index carries no mesh type, so
        # shield it from the spmd_types checker like the varlen_attn call it
        # feeds. The output is re-typed from Q on the way out.
        with spmd.no_typecheck():
            indices = attention_masks.k_global_gather_indices
            return k_TNH.index_select(0, indices), v_TNH.index_select(0, indices)

    def forward(
        self,
        q_TNH: torch.Tensor,
        k_TNH: torch.Tensor,
        v_TNH: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        k_TNH, v_TNH = _all_gather_kv(k_TNH, v_TNH, self.cp_group, self.reduce_dtype)
        return super().forward(q_TNH, k_TNH, v_TNH, **kwargs)


class UlyssesCPKernel(ContextParallelKernel):
    """DeepSpeed-Ulysses style CP: an all-to-all trades sequence for heads."""

    @staticmethod
    def _reshard(
        x: torch.Tensor, cp_group: dist.ProcessGroup, *, src: int, dst: int
    ) -> torch.Tensor:
        """Move the CP sharding of ``x`` from tensor dim ``src`` to ``dst``."""
        return spmd.redistribute(
            x.contiguous(),
            cp_group,
            src=spmd.S(src),
            dst=spmd.S(dst),
        )

    def forward(
        self,
        q_TNH: torch.Tensor,
        k_TNH: torch.Tensor,
        v_TNH: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        cp_group = self.cp_group
        # (T/cp, N, H) -> (T, N/cp, H)
        q_TNH, k_TNH, v_TNH = (
            self._reshard(x, cp_group, src=_SEQ_DIM, dst=_HEAD_DIM)
            for x in (q_TNH, k_TNH, v_TNH)
        )
        # The wrapped kernel supplies forward through the MRO of the concrete
        # subclass, which this mixin cannot see on its own.
        # pyrefly: ignore [missing-attribute]
        out_TNH = super().forward(q_TNH, k_TNH, v_TNH, **kwargs)
        # (T, N/cp, H) -> (T/cp, N, H)
        return self._reshard(out_TNH, cp_group, src=_HEAD_DIM, dst=_SEQ_DIM)


class UlyssesCPFlexAttention(UlyssesCPKernel, FlexAttention):
    """FlexAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        shard_attention_mask: ClassVar[bool] = False
        """The kernel sees the whole sequence, so its mask must stay global."""


class UlyssesCPVarlenAttention(UlyssesCPKernel, VarlenAttention):
    """VarlenAttention under Ulysses CP."""

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):
        shard_attention_mask: ClassVar[bool] = False
        """The kernel sees the whole sequence, so its mask must stay global."""
