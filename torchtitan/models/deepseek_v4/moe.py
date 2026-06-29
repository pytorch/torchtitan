# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter


def _softplus_stable(x):
    return torch.log1p(torch.exp(-x.abs())) + torch.relu(x)


def _build_hash_routing_table(vocab_size, num_experts, top_k, device=None, chunk_size=8192):
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
                    self.vocab_size, self.num_experts, self.top_k
                ),
                persistent=True,
            )

    def _init_self_buffers(self, *, buffer_device=None):
        if self.hash:
            if buffer_device is None:
                buffer_device = self.tid2eid.device
            with torch.device(buffer_device):
                self.tid2eid = _build_hash_routing_table(
                    self.vocab_size, self.num_experts, self.top_k,
                    device=buffer_device,
                )

    def _hash_lookup(self, input_ids, scores):
        if isinstance(input_ids, DTensor):
            input_ids = input_ids.to_local()

        tid2eid = self.tid2eid
        if isinstance(tid2eid, DTensor):
            tid2eid = tid2eid.to_local()

        selected_experts_indices = tid2eid.to(input_ids.device)[input_ids]
        if isinstance(scores, DTensor):
            selected_experts_indices = DTensor.from_local(
                selected_experts_indices,
                scores.device_mesh,
                scores.placements,
                run_check=False,
            )
        else:
            selected_experts_indices = selected_experts_indices.to(scores.device)
        return selected_experts_indices

    def forward(self, x_BLD, expert_bias_E=None, *, input_ids=None):
        scores = self.gate(x_BLD)
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=-1)
        elif self.score_func == "sqrtsoftplus":
            scores = _softplus_stable(scores.to(torch.float32)).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self.hash:
            if input_ids is None:
                raise ValueError("input_ids is required for DeepSeek V4 hash routing.")
            selected_experts_indices = self._hash_lookup(input_ids, scores)
        else:
            scores_for_choice = (
                scores if expert_bias_E is None else scores + expert_bias_E
            )
            selected_experts_indices = scores_for_choice.topk(self.top_k, dim=-1)[1]

        top_scores = scores.gather(dim=-1, index=selected_experts_indices)

        if self._debug_force_load_balance:
            selected_experts_indices, top_scores = self._debug_force_load_balance_routing(
                scores
            )

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        return top_scores, selected_experts_indices, scores


class DeepSeekV4MoE(MoE):
    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass

    def forward(self, x_BLD: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(x_BLD, input_ids=input_ids)
