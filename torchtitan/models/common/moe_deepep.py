# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE with DeepEP backend for efficient expert-parallel communication."""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor, Partial

from torchtitan.distributed.deepep.deepep import sync_combine

from .moe import MoE


class DeepEPMoE(MoE):
    """
    Mixture of Experts with DeepEP/HybridEP communication.

    Overrides forward() to insert shared_experts between the async combine
    (inside experts.forward → token_dispatcher.combine) and sync_combine(),
    overlapping shared_experts compute with combine communication.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert DTensor to local tensor for MoE-internal computation.
        # grad_placements=(Partial(),) ensures x.grad is Partial on the tp_mesh
        # in backward, so gradient reduction (reduce-scatter from Partial to
        # Shard(1)) happens once at the MoE boundary rather than being
        # duplicated inside the MoE.
        if isinstance(x, DTensor):
            assert (
                x.device_mesh.ndim == 1
            ), f"Expected 1D mesh, got {x.device_mesh.ndim}D mesh"
            assert x.device_mesh.mesh_dim_names == (
                "tp",
            ), f"Expected TP mesh, got mesh_dim_names={x.device_mesh.mesh_dim_names}"
            x = x.to_local(grad_placements=(Partial(),))
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x, self.expert_bias
        )

        if self.load_balance_coeff is not None:
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # Dispatch + expert computation + async combine all inside experts.forward().
        # DeepEPTokenDispatcher.combine() returns immediately (async),
        # so routed_output is not yet ready to read.
        routed_output = self.experts(
            x, num_tokens_per_expert, top_scores, selected_experts_indices
        )

        # shared_experts runs in parallel with the async combine communication.
        out = self.shared_experts(x) if self.shared_experts is not None else None

        # Wait for the async combine to complete before using routed_output.
        sync_combine()

        if out is None:
            return routed_output.reshape(bs, slen, dim)
        return (out + routed_output).reshape(bs, slen, dim)
