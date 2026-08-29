# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter


def _build_hash_routing_table(
    vocab_size: int,
    num_experts: int,
    top_k: int,
    device=None,
    chunk_size: int = 8192,
) -> torch.Tensor:
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
    tid2eid = torch.empty((vocab_size, top_k), dtype=torch.long, device=device)
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        tid2eid[start:end] = (
            torch.rand((end - start, num_experts), device=device)
            .topk(top_k, dim=-1)
            .indices
        )
    return tid2eid


class DeepSeekV4Router(TokenChoiceTopKRouter):
    """DeepSeek V4 router with optional hash-based expert selection."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        vocab_size: int
        n_hash_layers: int = 3
        layer_id: int = 0

    def __init__(self, config: Config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.n_hash_layers = config.n_hash_layers
        self.layer_id = config.layer_id
        self.hash = config.layer_id < config.n_hash_layers
        if self.hash:
            self.register_buffer(
                "tid2eid",
                _build_hash_routing_table(
                    self.vocab_size,
                    self.num_experts,
                    self.top_k,
                ),
                persistent=True,
            )

    def _init_self_buffers(self, *, buffer_device=None):
        if self.hash:
            if buffer_device is None:
                buffer_device = self.tid2eid.device
            with torch.device(buffer_device):
                self.tid2eid = _build_hash_routing_table(
                    self.vocab_size,
                    self.num_experts,
                    self.top_k,
                    device=buffer_device,
                )

    def forward(
        self,
        x_TD: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        *,
        input_ids_T: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=x_TD.device.type, dtype=torch.float32):
            scores_TE = self.gate(x_TD)

        if self.score_func == "sqrtsoftplus":
            scores_TE = torch.sqrt(F.softplus(scores_TE))
        elif self.score_func == "sigmoid":
            scores_TE = torch.sigmoid(scores_TE)
        elif self.score_func == "softmax":
            scores_TE = F.softmax(scores_TE, dim=-1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self.hash:
            if input_ids_T is None:
                raise ValueError("input_ids_T is required for DeepSeek V4 hash routing.")
            topk_expert_ids_TK = self.tid2eid.to(input_ids_T.device)[input_ids_T]
        else:
            scores_for_choice_TE = (
                scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
            )
            if self.num_expert_groups is not None:
                scores_for_choice_TE = self._get_node_limited_routing_scores(
                    scores_for_choice_TE
                )
            _, topk_expert_ids_TK = torch.topk(
                scores_for_choice_TE,
                k=self.top_k,
                dim=-1,
                sorted=False,
            )

        topk_scores_TK = scores_TE.gather(dim=-1, index=topk_expert_ids_TK)
        if self._debug_force_load_balance:
            topk_expert_ids_TK, topk_scores_TK = self._debug_force_load_balance_routing(
                scores_TE
            )
        if self.route_norm:
            topk_scores_TK = topk_scores_TK / (
                topk_scores_TK.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_scores_TK = topk_scores_TK * self.route_scale
        return topk_scores_TK, topk_expert_ids_TK, scores_TE


class DeepSeekV4MoE(MoE):
    """DeepSeek V4 MoE that forwards token IDs to hash-routing layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass

    def forward(self, x_TD: torch.Tensor, input_ids_T: torch.Tensor) -> torch.Tensor:
        topk_scores_TK, topk_expert_ids_TK, scores_TE = self.router(
            x_TD,
            self.expert_bias_E,
            input_ids_T=input_ids_T,
        )
        routing_map_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter_(
            -1,
            topk_expert_ids_TK,
            True,
        )
        num_local_tokens_per_expert_E = routing_map_TE.sum(dim=0)
        if self.training:
            with torch.no_grad():
                self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)

        out_TD = self.routed_experts(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        shared_out_TD = (
            self.shared_experts(x_TD) if self.shared_experts is not None else None
        )
        if shared_out_TD is not None:
            out_TD = out_TD + shared_out_TD
        return out_TD
