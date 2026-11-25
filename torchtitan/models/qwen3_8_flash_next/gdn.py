# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GDN that reads its ``SplitLinear`` ``in_proj``'s two outputs directly.

Flash-Next's GDN ``in_proj`` is a ``SplitLinear``: two GEMMs, ``projection``
(q, k, v per key head) and ``gates`` (z, a, b per key head). Calling
``SplitLinear`` interleaves their outputs per key head into the single-``Linear``
layout the shared ``GDN`` expects, and ``GDN`` immediately splits that layout
again. With 2 key heads::

    projection  [q0 k0 v0 | q1 k1 v1]
    gates       [z0 a0 b0 | z1 a1 b1]
    SplitLinear [q0 k0 v0 z0 a0 b0 | q1 k1 v1 z1 a1 b1]    <- copy skipped here
    GDN split   mixed_qkv [q0 q1 | k0 k1 | v0 v1], z, a, b

``SplitInProjGDN`` builds ``mixed_qkv``, ``a``, ``b`` and the gate straight from
the two GEMM outputs. Parameters, GEMMs and every tensor the GDN kernel sees
are the shared ``GDN``'s, so numerics are bitwise identical; only the
interleaved copy (and its backward split) is gone.
"""

# T = tokens, C = projection channels, H = value heads, V = value head dim,
# D = model dimension.

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.linear import maybe_gather_tp_input

from .attention import SplitLinear


class SplitInProjGDN(GDN):
    """``GDN`` that reads a ``SplitLinear`` ``in_proj``'s two outputs directly."""

    @dataclass(kw_only=True, slots=True)
    class Config(GDN.Config):
        def __post_init__(self):
            GDN.Config.__post_init__(self)
            if not isinstance(self.in_proj, SplitLinear.Config):
                raise ValueError("SplitInProjGDN requires a SplitLinear in_proj")
            q_size, k_size, v_size, *_ = self.in_proj_split_sizes
            projection_rows = self.n_key_heads * (q_size + k_size + v_size)
            if (
                self.in_proj.projection.out_features != projection_rows
                or self.in_proj.num_groups != self.n_key_heads
            ):
                raise ValueError(
                    "SplitInProjGDN requires one in_proj group per key head and "
                    f"in_proj.projection to hold exactly q, k and v ({projection_rows} "
                    f"rows), got {self.in_proj.num_groups} groups and "
                    f"{self.in_proj.projection.out_features} rows"
                )

    @staticmethod
    def _split_projections(
        projection_TC: torch.Tensor,
        gates_TC: torch.Tensor,
        split_sizes: list[int],
        value_head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(mixed_qkv_TC, a_TH, b_TH, gate_THV)`` from the two GEMM outputs.

        Same tensors as the shared GDN's ``_split_grouped_in_proj`` on the
        interleaved layout.
        """
        num_tokens = projection_TC.shape[0]
        q_size, k_size, v_size, z_size, a_size, b_size = split_sizes
        query, key, value = projection_TC.unflatten(
            -1, (-1, q_size + k_size + v_size)
        ).split([q_size, k_size, v_size], dim=-1)
        gate, a, b = gates_TC.unflatten(-1, (-1, z_size + a_size + b_size)).split(
            [z_size, a_size, b_size], dim=-1
        )
        mixed_qkv_TC = torch.cat(
            [query.flatten(1), key.flatten(1), value.flatten(1)], dim=-1
        )
        gate_THV = gate.reshape(num_tokens, -1, value_head_dim)
        return (
            mixed_qkv_TC,
            a.flatten(1).contiguous(),
            b.flatten(1).contiguous(),
            gate_THV.contiguous(),
        )

    # Same region name and Inductor options as the shared GDN split: both cat limits
    # at 0 make Inductor copy each piece into its slice, not one kernel that picks a
    # source per element; combo_kernels launches the remaining copies as one kernel.
    @staticmethod
    @local_compile(
        "gdn_split_in_proj",
        batch_invariant=True,
        options={
            "max_pointwise_cat_inputs": 0,
            "max_complex_pointwise_cat_inputs": 0,
            "combo_kernels": True,
        },
    )
    def _split_projections_with_grad(
        projection_TC: torch.Tensor,
        gates_TC: torch.Tensor,
        split_sizes: list[int],
        value_head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return SplitInProjGDN._split_projections(
            projection_TC, gates_TC, split_sizes, value_head_dim
        )

    @staticmethod
    @local_compile(
        "gdn_split_in_proj", batch_invariant=True, options={"combo_kernels": True}
    )
    def _split_projections_no_grad(
        projection_TC: torch.Tensor,
        gates_TC: torch.Tensor,
        split_sizes: list[int],
        value_head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return SplitInProjGDN._split_projections(
            projection_TC, gates_TC, split_sizes, value_head_dim
        )

    @staticmethod
    @spmd.local_map(
        # Both GEMM outputs hold whole key heads on each TP rank.
        in_types=(
            spmd.PartitionSpec(("dp", "cp"), "tp"),
            spmd.PartitionSpec(("dp", "cp"), "tp"),
            [None] * 6,
            None,
        ),
        out_types=(
            spmd.PartitionSpec(("dp", "cp"), "tp"),  # mixed_qkv_TC
            spmd.PartitionSpec(("dp", "cp"), "tp"),  # a_TH
            spmd.PartitionSpec(("dp", "cp"), "tp"),  # b_TH
            spmd.PartitionSpec(("dp", "cp"), "tp", None),  # gate_THV
        ),
    )
    def _split_in_proj(
        projection_TC: torch.Tensor,
        gates_TC: torch.Tensor,
        split_sizes: list[int],
        value_head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if projection_TC.requires_grad or gates_TC.requires_grad:
            return SplitInProjGDN._split_projections_with_grad(
                projection_TC, gates_TC, split_sizes, value_head_dim
            )
        return SplitInProjGDN._split_projections_no_grad(
            projection_TC, gates_TC, split_sizes, value_head_dim
        )

    def forward(self, x_TD, attention_metadata=None):
        x_TD = maybe_gather_tp_input(self, x_TD)
        num_tokens = x_TD.shape[0]
        projection_TC = self.in_proj.projection(x_TD)
        gates_TC = self.in_proj.gates(x_TD)
        # The split copies the GEMM outputs with bare ops.
        remat.recompute_needs_tensor(projection_TC, gates_TC)
        mixed_qkv_TC, a_TH, b_TH, gate_THV = self._split_in_proj(
            projection_TC, gates_TC, self.in_proj_split_sizes, self.value_head_dim
        )
        # From here on identical to GDN.forward; keep the two in sync.
        output_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            mixed_qkv_TC,
            a_TH,
            b_TH,
            self.conv_q.weight,
            self.conv_k.weight,
            self.conv_v.weight,
            self.A_log,
            self.dt_bias,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            attention_metadata=attention_metadata,
        )
        # The gated norm reads the inner_attention and split outputs with bare ops.
        remat.recompute_needs_tensor(output_THV, gate_THV)
        output_THV = self.norm(output_THV, gate_THV)
        return self.out_proj(output_THV.reshape(num_tokens, -1))
