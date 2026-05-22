# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.tensor import DTensor

import spmd_types as spmd
from torchtitan.distributed.spmd_state import current_mesh, set_current_mesh
from torchtitan.models.common.moe import GroupedExperts, MoE
from torchtitan.protocols.module import Module


@spmd.register_autograd_function
class ScaleBiasForward(torch.autograd.Function):
    """
    Custom autograd function that scales bias in forward pass but not in backward.

    For tensor parallel MoE, we need to scale the bias by 1/tp_degree in forward
    to cancel the extra reduction effect, but keep the gradient unchanged in backward.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, bias, tp_degree):
        ctx.tp_degree = tp_degree
        if tp_degree > 1:
            return bias / tp_degree
        return bias

    @staticmethod
    def typecheck_forward(bias, tp_degree):
        out = ScaleBiasForward.apply(bias, tp_degree)
        local_type = dict(spmd.get_local_type(bias)) if spmd.has_local_type(bias) else {}
        mesh_axis_names = spmd.current_mesh_names() or {}
        if tp_degree > 1 and "tp" in mesh_axis_names:
            local_type[mesh_axis_names["tp"]] = spmd.V
        if local_type:
            spmd.set_local_type(out, local_type)
        return out

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # Don't scale the gradient - pass it through as-is
        return grad_output, None


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


def _tp_degree_from_current_mesh() -> int:
    mesh = current_mesh()
    mesh_axis_names = spmd.current_mesh_names() or {}
    if mesh is None or "tp" not in mesh_axis_names:
        return 1
    assert mesh.mesh_dim_names is not None
    return mesh.size(mesh.mesh_dim_names.index("tp"))


class GptOssGroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        swiglu_limit: float = 7.0

    def __init__(self, config: Config):
        super().__init__()
        dim = config.dim
        hidden_dim = config.hidden_dim
        num_experts = config.num_experts
        self.num_experts = num_experts
        self.swiglu_limit = config.swiglu_limit

        self.mlp1_weight = nn.Parameter(
            torch.empty((num_experts, hidden_dim * 2, dim))
        )  # (num_experts, out_dim, in_dim)
        self.mlp1_bias = nn.Parameter(torch.empty((num_experts, hidden_dim * 2)))
        self.mlp2_weight = nn.Parameter(
            torch.empty((num_experts, dim, hidden_dim))
        )  # (num_experts, out_dim, in_dim)
        self.mlp2_bias = nn.Parameter(torch.empty((num_experts, dim)))

        self.token_dispatcher = config.token_dispatcher.build()

    def _experts_forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine."""
        if isinstance(self.mlp1_weight, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            mlp1_weight = self.mlp1_weight.to_local()
            # pyrefly: ignore [missing-attribute]
            mlp1_bias = self.mlp1_bias.to_local()
            # pyrefly: ignore [missing-attribute]
            mlp2_weight = self.mlp2_weight.to_local()
            # pyrefly: ignore [missing-attribute]
            mlp2_bias = self.mlp2_bias.to_local()
        else:
            mlp1_weight = self.mlp1_weight
            mlp1_bias = self.mlp1_bias
            mlp2_weight = self.mlp2_weight
            mlp2_bias = self.mlp2_bias

        # Determine tp_degree from device mesh if available. Local SPMD typed
        # params are plain tensors, so use the active mesh before falling back
        # to DTensor metadata.
        tp_degree = _tp_degree_from_current_mesh()
        if isinstance(self.mlp1_weight, DTensor):
            mesh_dim_names = self.mlp1_weight.device_mesh.mesh_dim_names
            # pyrefly: ignore[not-iterable]
            if "tp" in mesh_dim_names:
                # pyrefly: ignore [missing-attribute]
                tp_dim_idx = mesh_dim_names.index("tp")
                tp_degree = self.mlp1_weight.device_mesh.size(tp_dim_idx)

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        # Pad num_tokens_per_expert with tail slack so that repeat_interleave
        # with output_size=x.shape[0] directly produces a static-shaped output,
        # avoiding the D2H sync that repeat_interleave incurs without output_size.
        tail_slack = (
            (x.shape[0] - offsets[-1]).unsqueeze(0).to(num_tokens_per_expert.dtype)
        )
        num_tokens_per_expert_long = torch.cat(
            [num_tokens_per_expert, tail_slack]
        ).long()

        h = torch._grouped_mm(
            x.bfloat16(), mlp1_weight.transpose(-2, -1).bfloat16(), offs=offsets
        )

        b1 = torch.cat([mlp1_bias, mlp1_bias.new_zeros(1, mlp1_bias.shape[-1])])
        b1 = b1.repeat_interleave(
            num_tokens_per_expert_long, dim=0, output_size=x.shape[0]
        )
        h = h + b1.to(h.dtype)

        h = swiglu(h, limit=self.swiglu_limit)
        h = torch._grouped_mm(h, mlp2_weight.transpose(-2, -1).bfloat16(), offs=offsets)

        # Apply custom autograd function to scale bias in forward but not in backward
        b2 = torch.cat([mlp2_bias, mlp2_bias.new_zeros(1, mlp2_bias.shape[-1])])
        b2 = b2.repeat_interleave(
            num_tokens_per_expert_long, dim=0, output_size=x.shape[0]
        )
        b2 = ScaleBiasForward.apply(b2, tp_degree)
        return h + b2.to(h.dtype)

    def forward(
        self,
        x: torch.Tensor,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        combine_base: torch.Tensor | None = None,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        """Dispatch tokens to experts, compute, combine, and scatter_add."""
        routed_input, num_tokens_local, metadata = self.token_dispatcher.dispatch(
            x, top_scores, selected_experts_indices
        )
        sparse_mesh = getattr(self.token_dispatcher, "sparse_mesh", None)
        with set_current_mesh(sparse_mesh):
            routed_output = self._experts_forward(routed_input, num_tokens_local)
        return self.token_dispatcher.combine(
            routed_output,
            metadata,
            x,
            combine_base=combine_base,
            shared_experts=shared_experts,
        )

    def parallelize(self, parallel_dims) -> None:
        """Parallelize experts and wire dispatcher meshes.

        Mirrors ``GroupedExperts.parallelize``: after the base
        ``Module.parallelize`` distributes the expert weight params, install
        the EP / TP meshes on the non-Module ``token_dispatcher`` child via
        ``wire_meshes``. ``GptOssGroupedExperts`` inherits ``Module``
        directly (not ``GroupedExperts``) so it needs its own override.
        """
        super().parallelize(parallel_dims)
        self.token_dispatcher.wire_meshes(
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
        )
        if parallel_dims.ep_enabled:
            self.token_dispatcher.sparse_mesh = parallel_dims.get_activated_mesh(
                ["dp_replicate", "efsdp", "ep"]
            )


class GptOssMoE(MoE):
    """GptOss MoE implementation that inherits from the base MoE class."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        swiglu_limit: float = 7.0

    def __init__(self, config: Config):
        # Initialize the base MoE class
        super().__init__(config)

        # Override the base GroupedExperts with GptOssGroupedExperts. Forward
        # every Module.Config slot from ``config.experts`` so the rebuilt
        # config carries ``sharding_config`` (set by
        # ``set_moe_sharding_config``) into the new instance.
        gptoss_experts_config = GptOssGroupedExperts.Config(
            dim=config.experts.dim,
            hidden_dim=config.experts.hidden_dim,
            num_experts=config.experts.num_experts,
            swiglu_limit=config.swiglu_limit,
            param_init=config.experts.param_init,
            sharding_config=config.experts.sharding_config,
            token_dispatcher=config.experts.token_dispatcher,
        )
        self.experts = gptoss_experts_config.build()
