# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
import torch_remat as remat

from torchtitan.models.common.activation import UnaryActivationFn
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)

from torchtitan.models.muse_spark.model import MuseSparkRMSNorm


# Shape suffix legend:
# T = num tokens, D = model dimension, C = compressed expert dimension,
# E = num experts, K = top-k.


class MuseSparkRouterSigmoid(UnaryActivationFn):
    @dataclass(kw_only=True, slots=True)
    class Config(UnaryActivationFn.Config):
        logit_multiplier: float = 1.0

    def __init__(self, config: Config):
        self.logit_multiplier = config.logit_multiplier

    def __call__(
        self,
        logits_TE: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        with torch.autocast(device_type=logits_TE.device.type, enabled=False):
            return torch.sigmoid(logits_TE * self.logit_multiplier)


class MuseSparkMoE(MoE):
    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pooler: Linear.Config
        pre_expert_norm: MuseSparkRMSNorm.Config
        unpooler: Linear.Config
        expert_bias_init: float = 0.0

        def __post_init__(self) -> None:
            MoE.Config.__post_init__(self)
            dispatcher = getattr(self.routed_experts, "token_dispatcher", None)
            if dispatcher is None:
                owner = self.routed_experts._owner
                if owner is None or not issubclass(owner, DistMoeRoutedExperts):
                    raise ValueError(
                        f"{type(self.routed_experts).__qualname__} has no token dispatcher"
                    )
            elif dispatcher._owner not in (
                LocalTokenDispatcher,
                AllToAllTokenDispatcher,
            ):
                raise ValueError(
                    "Muse Spark MoE requires LocalTokenDispatcher or "
                    "AllToAllTokenDispatcher, "
                    f"got {type(dispatcher).__qualname__}"
                )

    def __init__(self, config: Config):
        if config.shared_experts is not None:
            raise ValueError("Muse Spark MoE does not support shared experts")
        super().__init__(config)
        self.num_experts = config.num_experts
        self.expert_bias_init = config.expert_bias_init
        self.pooler = config.pooler.build()
        self.pre_expert_norm = config.pre_expert_norm.build()
        self.unpooler = config.unpooler.build()
        if self.expert_bias_E is None:
            del self.expert_bias_E
            self.register_buffer(
                "expert_bias_E",
                torch.full(
                    (self.num_experts,),
                    self.expert_bias_init,
                    dtype=torch.float32,
                ),
                persistent=True,
            )
        else:
            self.expert_bias_E.fill_(self.expert_bias_init)

    def forward(
        self,
        x_TD: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        # EP uses TP token shards independently of dense sequence parallelism.
        x_TD, padding_mask_T = self._maybe_shard_routed_branch_inputs_across_tp(
            x_TD, padding_mask_T
        )
        (topk_scores_TK, topk_expert_ids_TK, routing_map_TE,) = remat.region(
            self.router,
            self.remat_region_name("router"),
            recompute=self.remat_should_recompute("router"),
        )(
            x_TD,
            self.expert_bias_E,
            padding_mask_T=padding_mask_T,
            aux_loss_denominator=aux_loss_denominator,
            **router_kwargs,
        )
        remat.recompute_needs_tensor(routing_map_TE)
        num_local_tokens_per_expert_E = routing_map_TE.sum(dim=0)

        expert_input_TC = self.pooler(x_TD)
        expert_input_TC = remat.region(
            self.pre_expert_norm,
            self.remat_region_name("pre_expert_norm"),
            recompute=self.remat_should_recompute("pre_expert_norm"),
        )(expert_input_TC)
        # Local token dispatch indexes these inputs outside a remat region.
        remat.recompute_needs_tensor(
            expert_input_TC, topk_scores_TK, topk_expert_ids_TK
        )
        out_TC = self.routed_experts(
            expert_input_TC,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
            padding_mask_T=padding_mask_T,
        )
        output_TD = self.unpooler(out_TC)
        # Replicated unpooler weights produce a token shard, not a row-linear partial.
        output_TD = self._maybe_zero_fill_routed_output_to_tp_partial(output_TD)
        return self._maybe_all_reduce_moe_output_across_tp(output_TD)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        super()._init_self_buffers(buffer_device=buffer_device)
        if buffer_device is None:
            buffer_device = self.router.tokens_per_expert_E.device
        with torch.device(buffer_device):
            self.expert_bias_E = torch.full(
                (self.num_experts,),
                self.expert_bias_init,
                dtype=torch.float32,
            )
