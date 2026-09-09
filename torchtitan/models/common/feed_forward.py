# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.models.common.linear import _build_interleaved_linear, Linear
from torchtitan.protocols.module import Module

# Shape suffix legend:
#   T = token dimensions, D = model dimension, F = feed-forward hidden dimension

__all__ = ["FeedForward", "SigmoidGatedFeedForward", "compute_ffn_hidden_dim"]


def _interleaved_init(
    first_init: Callable, second_init: Callable
) -> Callable[[torch.Tensor], None]:
    """Initialize two logical output slices of one interleaved parameter."""

    def init(param: torch.Tensor) -> None:
        logical_param = param.unflatten(0, (-1, 2))
        first_init(logical_param[:, 0])
        second_init(logical_param[:, 1])

    return init


def _merge_gate_up_param_init(
    w1: Linear.Config, w3: Linear.Config
) -> dict[str, Callable] | None:
    """Preserve the logical w1 and w3 initializers in an interleaved w13."""
    if w1.param_init is None and w3.param_init is None:
        return None
    if w1.param_init is None or w3.param_init is None:
        raise ValueError("w1 and w3 must either both define param_init or neither")
    if w1.param_init.keys() != w3.param_init.keys():
        raise ValueError("w1 and w3 param_init must initialize the same parameters")
    return {
        name: _interleaved_init(w1.param_init[name], w3.param_init[name])
        for name in w1.param_init
    }


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

    ``w1`` and ``w3`` remain separate logical configs and checkpoint keys, but
    build into one interleaved ``w13`` Linear. Config takes the **final**
    hidden_dim (no internal 2/3 scaling). Use compute_ffn_hidden_dim() for
    Llama3/4-style dim computation.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w1: Linear.Config
        w2: Linear.Config
        w3: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.w13 = _build_interleaved_linear(
            config.w1,
            config.w3,
            logical_names=("w1", "w3"),
            param_init=_merge_gate_up_param_init(config.w1, config.w3),
        )
        self.w2 = config.w2.build()
        self.register_state_dict_post_hook(self._split_w13_on_save)
        self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_TD = self._forward_swiglu(x)
        remat.recompute_needs_tensor(out_TD)
        return out_TD

    def _forward_swiglu(self, x: torch.Tensor) -> torch.Tensor:
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
        )(self._activation(gate_TF, up_TF))
        return out_TD

    def _activation(self, gate_TF: torch.Tensor, up_TF: torch.Tensor) -> torch.Tensor:
        return F.silu(gate_TF) * up_TF

    @staticmethod
    def _split_w13_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Expose the physical w13 parameter as logical w1 and w3 keys."""
        for param_name in ("weight", "bias"):
            key = f"{prefix}w13.{param_name}"
            if key not in state_dict:
                continue
            param = state_dict.pop(key).unflatten(0, (-1, 2))
            state_dict[f"{prefix}w1.{param_name}"] = param[:, 0].contiguous()
            state_dict[f"{prefix}w3.{param_name}"] = param[:, 1].contiguous()

        # LoRA uses this hook to expose separate w1 and w3 adapter keys.
        if hasattr(module.w13, "_expose_logical_state_dict"):
            module.w13._expose_logical_state_dict(
                state_dict,
                physical_prefix=f"{prefix}w13.",
                logical_prefix=prefix,
            )

    @staticmethod
    def _merge_w13_on_load(module, state_dict, prefix, *args) -> None:
        """Merge logical w1 and w3 checkpoint keys into the physical w13."""
        for param_name in ("weight", "bias"):
            w1_key = f"{prefix}w1.{param_name}"
            w3_key = f"{prefix}w3.{param_name}"
            if w1_key in state_dict and w3_key in state_dict:
                state_dict[f"{prefix}w13.{param_name}"] = torch.stack(
                    [state_dict.pop(w1_key), state_dict.pop(w3_key)], dim=1
                ).flatten(0, 1)

        # LoRA uses this hook to restore separate w1 and w3 adapter keys.
        if hasattr(module.w13, "_restore_logical_state_dict"):
            module.w13._restore_logical_state_dict(
                state_dict,
                physical_prefix=f"{prefix}w13.",
                logical_prefix=prefix,
            )

        native_key = f"{prefix}w13"
        if native_key in state_dict:
            state_dict[f"{prefix}w13.weight"] = state_dict.pop(native_key).flatten(0, 1)


class SigmoidGatedFeedForward(FeedForward):
    """SwiGLU feed-forward with a per-token sigmoid gate.

    The output is ``sigmoid(gate(x)) * ffn(x)``. It retains FeedForward's
    logical ``w1``/``w2``/``w3`` checkpoint layout.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        gate: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate = config.gate.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_TD = self._forward_swiglu(x)
        gate_out_TD = remat.region(
            self.gate,
            self.remat_region_name("gate"),
            recompute=self.remat_should_recompute("gate"),
        )(x)
        remat.recompute_needs_tensor(out_TD, gate_out_TD)
        return torch.sigmoid(gate_out_TD) * out_TD
