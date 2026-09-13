# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch_remat as remat

from torchtitan.models.common.activation import ActivationFn, SwiGLU
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

# Shape suffix legend:
#   T = token dimensions, D = model dimension, F = feed-forward hidden dimension

__all__ = ["FeedForward", "SigmoidGatedFeedForward", "compute_ffn_hidden_dim"]


def compute_ffn_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 1,
    ffn_dim_multiplier: float | None = None,
) -> int:
    """Compute the SwiGLU hidden dimension for Llama3/4-style models.

    This applies the 2/3 scaling, optional multiplier, and rounds up to multiple_of.
    """
    hidden_dim = int(2 * 4 * dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class FeedForward(Module):
    """SwiGLU feed-forward with one physical gate-and-up projection.

    ``w13`` stores the interleaved gate and up projections. Config takes the
    **final** hidden_dim (no internal 2/3 scaling). Use
    compute_ffn_hidden_dim() for Llama3/4-style dim computation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w13: Linear.Config
        w2: Linear.Config
        activation_fn: ActivationFn.Config = field(default_factory=SwiGLU.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.w13 = config.w13.build()
        self.w2 = config.w2.build()
        self.activation_fn = config.activation_fn.build()
        self.register_state_dict_post_hook(self._split_w13_on_save)
        self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

    @staticmethod
    def _split_w13_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Expose fused parameters under the logical w1/w3 checkpoint keys."""
        for param_name in ("weight", "bias"):
            fused_key = f"{prefix}w13.{param_name}"
            if fused_key not in state_dict:
                continue
            gate_up = state_dict.pop(fused_key).unflatten(0, (-1, 2))
            state_dict[f"{prefix}w1.{param_name}"] = gate_up[:, 0].contiguous()
            state_dict[f"{prefix}w3.{param_name}"] = gate_up[:, 1].contiguous()

    @staticmethod
    def _merge_w13_on_load(module, state_dict, prefix, *args) -> None:
        """Pack logical w1/w3 checkpoint entries into the fused parameter."""
        for param_name in ("weight", "bias"):
            gate_key = f"{prefix}w1.{param_name}"
            up_key = f"{prefix}w3.{param_name}"
            if gate_key not in state_dict or up_key not in state_dict:
                continue
            state_dict[f"{prefix}w13.{param_name}"] = torch.stack(
                [state_dict.pop(gate_key), state_dict.pop(up_key)], dim=1
            ).flatten(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up_TF = remat.region(
            self.w13,
            self.remat_region_name("w13"),
            recompute=self.remat_should_recompute("w13"),
        )(x)
        gate_TF, up_TF = gate_up_TF.unflatten(-1, (-1, 2)).unbind(-1)
        remat.recompute_needs_tensor(gate_TF, up_TF)
        out_TD = remat.region(
            self.w2,
            self.remat_region_name("w2"),
            recompute=self.remat_should_recompute("w2"),
        )(self.activation_fn(gate_TF, up_TF))
        remat.recompute_needs_tensor(out_TD)
        return out_TD


class SigmoidGatedFeedForward(FeedForward):
    """SwiGLU feed-forward with a per-token sigmoid gate.

    The output is ``sigmoid(gate(x)) * ffn(x)``. It uses FeedForward's fused
    ``w13`` and ``w2`` projections and adds a separate ``gate`` projection.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        gate: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate = config.gate.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_TD = super().forward(x)
        gate_out_TD = remat.region(
            self.gate,
            self.remat_region_name("gate"),
            recompute=self.remat_should_recompute("gate"),
        )(x)
        remat.recompute_needs_tensor(out_TD, gate_out_TD)
        return torch.sigmoid(gate_out_TD) * out_TD
