# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel KDA stages backed by Attention Gym."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from attn_gym.linear.context_parallel import (
    context_parallel_conv_history,
    ContextParallelPlan,
    ContextParallelRouting,
)
from attn_gym.linear.kda.context_parallel import context_parallel_kda

from torchtitan.models.common.cp_attention import ContextParallelKernel
from torchtitan.transforms.base import retype_node

from .kda import InnerKDA, KDA


def partition_fragments(
    num_tokens: int,
    world_size: int,
    load_balancer: str | None,
) -> list[list[tuple[int, int]]]:
    """Return each CP rank's global token fragments in local span order.

    Contiguous assigns rank ``r`` block ``r`` of ``W`` equal blocks. Headtail
    assigns blocks ``r`` and ``2W - 1 - r`` of ``2W`` equal blocks, matching
    TorchTitan's head-tail load balancer. The token count must be evenly
    divisible by the number of blocks.
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


def build_kda_context_parallel_routing(
    *,
    num_tokens: int,
    conv_kernel_size: int,
    load_balancer: str | None,
    device: torch.device,
    group: dist.ProcessGroup,
) -> ContextParallelRouting:
    """Build Attention Gym routing matching TorchTitan's CP partition."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    # TODO(acisseJZhong): Use global packed-sequence offsets once Kimi K3
    # supports sample packing.
    plan = ContextParallelPlan.from_fragments(
        (0, num_tokens),
        partition_fragments(num_tokens, world_size, load_balancer),
        rank,
    )
    return plan.routing(device, conv_history=conv_kernel_size - 1)


class ContextParallelInnerKDA(ContextParallelKernel, InnerKDA):
    """Inner KDA with distributed convolution and recurrent-state plumbing."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerKDA.Config):
        pass

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
        # CP routing owns the local fragment boundaries used by the stages below.
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
        initial_state = context_parallel_conv_history(
            qkv_1TC,
            routing,
            self.cp_group,
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
        # The CP core reads local fragment boundaries from routing.
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
        output_1THV, _ = context_parallel_kda(
            q_1THK,
            k_1THK,
            v_1THV,
            gate_1THK,
            beta_1TH,
            routing=routing,
            group=self.cp_group,
        )
        return output_1THV


def set_kda_context_parallel(config: KDA.Config) -> None:
    """Replace the local InnerKDA stages with their CP implementation."""
    config.inner_kda = retype_node(config.inner_kda, ContextParallelInnerKDA)
