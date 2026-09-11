# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel KDA stages backed by Attention Gym."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from attn_gym.linear.context_parallel import (
    context_parallel_conv_history,
    ContextParallelPlan,
    ContextParallelRouting,
)
from attn_gym.linear.kda.context_parallel import context_parallel_kda

from torchtitan.distributed.context_parallel.api import ContextParallelLoadBalancer
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.models.common.cp_attention import CPInnerAttention

from .kda import InnerKDA

# TODO(acisseJZhong): Define a common CP partition interface so model-specific
# kernels can support new load balancers without duplicating partition logic.
def _partition_fragments(
    num_tokens: int,
    world_size: int,
    load_balancer: str | None,
) -> list[list[tuple[int, int]]]:
    """Return the global token ranges owned by each CP rank.

    Ranges are half-open and follow the order of each rank's local tensor.

    Example with eight tokens and two ranks::

        contiguous = [
            [(0, 4)],
            [(4, 8)],
        ]
        headtail = [
            [(0, 2), (6, 8)],
            [(2, 4), (4, 6)],
        ]

    The token count must be divisible by the number of equal-sized blocks.
    """
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if load_balancer not in (None, "headtail"):
        raise ValueError(
            "KDA context parallelism supports only contiguous or headtail "
            f"token partitions, got {load_balancer!r}."
        )
    num_blocks = world_size if load_balancer is None else 2 * world_size
    if num_tokens % num_blocks:
        raise ValueError(
            f"KDA context parallelism requires the token count ({num_tokens}) "
            f"to be divisible by {num_blocks} for load_balancer={load_balancer!r}."
        )
    block_size = num_tokens // num_blocks
    owned_blocks = (
        [[rank] for rank in range(world_size)]
        if load_balancer is None
        else [[rank, num_blocks - 1 - rank] for rank in range(world_size)]
    )
    return [
        [(block * block_size, (block + 1) * block_size) for block in rank_blocks]
        for rank_blocks in owned_blocks
    ]


def _build_kda_context_parallel_routing(
    *,
    cu_seqlens_global: Sequence[int],
    conv_kernel_size: int,
    load_balancer: str | None,
    device: torch.device,
    group: dist.ProcessGroup,
) -> ContextParallelRouting:
    """Build KDA routing from global packed-document offsets."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    num_tokens = cu_seqlens_global[-1]
    plan = ContextParallelPlan.from_fragments(
        cu_seqlens_global,
        _partition_fragments(num_tokens, world_size, load_balancer),
        rank,
    )
    return plan.routing(device, conv_history=conv_kernel_size - 1)


class ContextParallelInnerKDA(CPInnerAttention, InnerKDA):
    """Inner KDA with distributed convolution and recurrent-state plumbing."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, InnerKDA.Config):
        def cp_shard_metadata(
            self,
            input_dict: dict[str, Any],
            load_balancer: ContextParallelLoadBalancer,
        ) -> dict[str, Any]:
            """Attach rank-local KDA routing for the configured CP layout."""
            tokens_T = input_dict.get("input")
            if not isinstance(tokens_T, torch.Tensor):
                raise ValueError("KDA context parallelism requires tensor input.")

            group = load_balancer.cp_mesh.get_group()
            attention_masks = input_dict.get("attention_masks")
            if attention_masks is None:
                num_tokens = tokens_T.shape[0] * dist.get_world_size(group)
                cu_seqlens_global = [0, num_tokens]
            else:
                if isinstance(attention_masks, VarlenMetadata):
                    kda_metadata = attention_masks
                elif isinstance(attention_masks, Mapping):
                    kda_metadata = attention_masks.get("kda")
                else:
                    kda_metadata = None
                if not isinstance(kda_metadata, VarlenMetadata):
                    raise ValueError(
                        "KDA context parallelism requires VarlenMetadata in "
                        "attention_masks or attention_masks['kda']."
                    )
                cu_seqlens_global = kda_metadata.cu_seq_q.tolist()

            # Kimi K3 validation rejects PTRR, so a non-None balancer is head-tail.
            input_dict["kda_cp_routing"] = _build_kda_context_parallel_routing(
                cu_seqlens_global=cu_seqlens_global,
                conv_kernel_size=self.conv_kernel_size,
                load_balancer=load_balancer.load_balancer_type,
                device=tokens_T.device,
                group=group,
            )
            return input_dict

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        # The shared KDA interface passes global packed-sequence offsets. CP
        # stages instead use the rank-local boundaries stored in routing.
        del cu_seqlens
        if routing is None:
            raise ValueError("KDA context parallelism requires per-batch routing.")
        if routing.tail_sources.shape[1] != conv_q_weight_C1W.shape[-1] - 1:
            raise ValueError(
                "KDA context-parallel routing convolution history must match "
                "the model's convolution width minus one."
            )
        return self.run_stages(
            query_TC,
            key_TC,
            value_TC,
            raw_gate_THK,
            raw_beta_TH,
            conv_q_weight_C1W,
            conv_k_weight_C1W,
            conv_v_weight_C1W,
            A_log_H,
            dt_bias_HK,
            cu_seqlens=routing.cu_seqlens,
            routing=routing,
        )

    def short_convolution(
        self,
        qkv_1TC: torch.Tensor,
        conv_weight_C1W: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        assert routing is not None, "CP forward must validate routing."
        assert (
            initial_state is None
        ), "KDA context parallelism constructs the convolution history."
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "KDA context parallelism requires an active multi-rank CP mesh axis."
            )
        initial_state = context_parallel_conv_history(
            qkv_1TC,
            routing,
            cp_group,
        )
        return super().short_convolution(
            qkv_1TC,
            conv_weight_C1W,
            initial_state,
            cu_seqlens=cu_seqlens,
            routing=routing,
        )

    def kda_core(
        self,
        q_1THK: torch.Tensor,
        k_1THK: torch.Tensor,
        v_1THV: torch.Tensor,
        raw_gate_1THK: torch.Tensor,
        raw_beta_1TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        # CP routing contains the rank-local sequence boundaries. Ignore the
        # global packed-sequence offsets passed through the regular KDA interface.
        del cu_seqlens
        assert routing is not None, "CP forward must validate routing."
        q_1THK, k_1THK, gate_1THK, beta_1TH = self.kernel.prepare_inputs(
            q_1THK,
            k_1THK,
            raw_gate_1THK,
            raw_beta_1TH,
            A_log_H,
            dt_bias_HK,
        )
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is None:
            raise RuntimeError(
                "KDA context parallelism requires an active multi-rank CP mesh axis."
            )
        output_1THV, _ = context_parallel_kda(
            q_1THK,
            k_1THK,
            v_1THV,
            gate_1THK,
            beta_1TH,
            routing=routing,
            group=cp_group,
        )
        return output_1THV
