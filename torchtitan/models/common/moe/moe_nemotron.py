# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Nemotron-compatible MoE implementation."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.protocols.module import Module

from .moe import MoE


def relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


def get_activation_fn(name: str):
    if name == "relu2":
        return relu2
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise NotImplementedError(f"Unsupported activation: {name}")


class NemotronMoEExpert(nn.Module):
    """Nemotron expert MLP: down_proj(act(up_proj(x)))."""

    def __init__(self, dim: int, hidden_dim: int, activation: str, bias: bool):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.act_fn = get_activation_fn(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))

    def init_weights(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.up_proj.weight, mean=0.0, std=0.02)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        nn.init.trunc_normal_(self.down_proj.weight, mean=0.0, std=init_std)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)


class NemotronTopkRouter(nn.Module):
    """Top-k router matching Nemotron behavior."""

    def __init__(self, config: MoE.Config, dim: int):
        super().__init__()
        self.top_k = config.top_k
        self.n_routed_experts = config.num_experts
        self.routed_scaling_factor = config.route_scale
        self.n_group = config.num_expert_groups or 1
        self.topk_group = config.num_limited_groups or self.n_group
        self.norm_topk_prob = config.route_norm
        self.score_func = config.score_func
        self.hidden_size = dim

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.hidden_size)))
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(self.n_routed_experts), persistent=True
        )

    @torch.no_grad()
    def get_topk_indices(self, scores: torch.Tensor) -> torch.Tensor:
        scores_for_choice = scores.view(-1, self.n_routed_experts)
        scores_for_choice = scores_for_choice + self.e_score_correction_bias.unsqueeze(0)

        if self.n_group > 1:
            if self.n_routed_experts % self.n_group != 0:
                raise ValueError(
                    f"num_experts ({self.n_routed_experts}) must be divisible by num_expert_groups ({self.n_group})"
                )
            group_scores = (
                scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(-1, self.n_routed_experts)
            )
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        return topk_indices

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = F.linear(
            hidden_states.to(torch.float32), self.weight.to(torch.float32)
        )

        if self.score_func == "sigmoid":
            scores = torch.sigmoid(router_logits)
        elif self.score_func == "softmax":
            scores = F.softmax(router_logits, dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)

        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def init_weights(self, init_std: float = 0.02) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=init_std)


class NemotronMoE(MoE):
    """MoE matching Nemotron routing/expert behavior."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        activation: str = "silu"
        bias: bool = False
        shared_hidden_dim: int | None = None

    def __init__(self, config: Config, *, dim: int):
        # Keep inheritance relationship for integration while preserving exact behavior.
        Module.__init__(self)
        shared_hidden_dim = (
            config.hidden_dim * config.num_shared_experts
            if config.shared_hidden_dim is None
            else config.shared_hidden_dim
        )

        self.experts = nn.ModuleList(
            [
                NemotronMoEExpert(
                    dim=dim,
                    hidden_dim=config.hidden_dim,
                    activation=config.activation,
                    bias=config.bias,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.router = NemotronTopkRouter(config=config, dim=dim)
        self.shared_experts = NemotronMoEExpert(
            dim=dim,
            hidden_dim=shared_hidden_dim,
            activation=config.activation,
            bias=config.bias,
        )

        # Keep these fields for compatibility with optimizer/infra checks.
        self.load_balance_coeff = None
        self.expert_bias = None
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=False,
        )

    def moe_forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = F.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx, expert in enumerate(self.experts):
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)
            else:
                # Keep no-op path so all experts participate in autograd graph.
                dummy_out = expert(torch.zeros_like(hidden_states[0]).unsqueeze(0))
                final_hidden_states = final_hidden_states + dummy_out * 0

        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape

        topk_indices, topk_weights = self.router(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe_forward(hidden_states, topk_indices, topk_weights)
        hidden_states = hidden_states.view(*orig_shape)

        return hidden_states + self.shared_experts(residuals)

    def init_weights(self, **kwargs) -> None:
        init_std = kwargs.get("init_std")
        if init_std is None:
            raise ValueError("NemotronMoE.init_weights requires init_std")
        self.router.init_weights(0.02)
        for expert in self.experts:
            expert.init_weights(init_std)
        self.shared_experts.init_weights(init_std)
