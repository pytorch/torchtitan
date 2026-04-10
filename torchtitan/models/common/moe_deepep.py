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
    Mixture of Experts with DeepEP communication.

    Inherits from MoE but overrides forward() to pass routing info to experts,
    letting DeepEPExpertParallel hooks handle dispatch/combine.

    The forward pass is structured to overlap shared_experts computation with
    the DeepEP combine communication:
    1. Router computes expert assignments
    2. DeepEP dispatches tokens to experts (sync)
    3. Experts process tokens
    4. DeepEP combine starts (async) - returns immediately
    5. shared_experts runs IN PARALLEL with combine communication
    6. sync_combine() waits for combine to complete
    7. Addition of shared_experts output and routed_output
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        # DeepEP doesn't use reorderer - routing handled by DeepEPExpertParallel
        self.reorderer = None  # pyrefly: ignore [bad-assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with DeepEP communication and overlapped shared_experts.

        DeepEPExpertParallel hooks intercept experts() call and handle
        dispatch/combine via deepep functions. The combine operation runs
        asynchronously, allowing shared_experts to overlap with the
        combine all-to-all communication.
        """
        # Convert DTensor to local tensor for MoE-internal computation.
        # grad_placements=(Partial(),) ensures x.grad is Partial on the tp_mesh
        # in backward, so gradient reduction (reduce-scatter from Partial to
        # Shard(1)) happens once at the MoE boundary rather than being
        # duplicated inside the MoE.
        #
        # Why grad(x) is Partial on the tp_mesh across all parallelism:
        # - TP only / TP+EP with ETP=TP: TP-sharded expert weights (Colwise on
        #   w1/w3, Rowwise on w2) produce Partial output gradients.
        # - TP+EP with ETP=1: each TP rank processes a disjoint token subset
        #   (via ReordererSequenceParallel), so grad(x) is non-zero only at
        #   each rank's token positions(Partial).
        #
        # This holds for all MoE components (router.gate, routed experts, shared
        # experts) and regardless of score_before_experts.
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

        # Call experts with routing info - hooks handle DeepEP dispatch/combine.
        # The combine operation returns asynchronously, allowing overlap with
        # shared_experts computation below.
        routed_output = self.experts(
            x,
            num_tokens_per_expert,
            selected_experts_indices,
            top_scores,
            self.experts.num_experts,
        )

        # shared_experts runs in parallel with combine communication.
        # This is the key optimization - we overlap compute with communication.
        out = self.shared_experts(x) if self.shared_experts is not None else None

        # Sync the combine operation before using routed_output.
        # This inserts a CUDA stream wait, ensuring combine is complete before
        # the subsequent addition or reshape operations read routed_output.
        sync_combine()

        if out is None:
            return routed_output.reshape(bs, slen, dim)
        return (out + routed_output).reshape(bs, slen, dim)
