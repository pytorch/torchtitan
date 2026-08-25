# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernels.

Tensor suffix: ``TNH`` = tokens, heads, head dimension.
"""

from dataclasses import dataclass, fields
from typing import cast, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.tensor.experimental._context_parallel import flex_cp_allgather

from torchtitan.distributed.spmd_types import current_spmd_mesh

from torchtitan.models.common.attention import BaseAttention, FlexAttention
from torchtitan.protocols.module import Module

if TYPE_CHECKING:
    from torchtitan.trainer import Trainer

__all__ = [
    "ContextParallelKernel",
    "AllGatherCPFlexAttention",
    "use_cp_kernel",
]

_SEQ_DIM = 0


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


class AllGatherCPFlexAttention(ContextParallelKernel, FlexAttention):
    """FlexAttention with sharded Q and all-gathered K/V."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        pass

    def forward(
        self,
        q_TNH: torch.Tensor,
        k_TNH: torch.Tensor,
        v_TNH: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        pg_name = dist._get_process_group_name(self.cp_group)
        k_TNH, v_TNH = flex_cp_allgather(
            k_TNH.contiguous(), v_TNH.contiguous(), _SEQ_DIM, pg_name
        )
        return super().forward(q_TNH, k_TNH, v_TNH, **kwargs)
