# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.models.common.moe import TokenChoiceTopKRouter


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

    def _select_experts(
        self,
        scores_BLE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        if self.hash:
            if input_ids is None:
                raise ValueError("input_ids is required for DeepSeek V4 hash routing.")
            return self.tid2eid.to(input_ids.device)[input_ids]
        return super()._select_experts(
            scores_BLE, expert_bias_E, **router_kwargs
        )
