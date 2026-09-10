# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

import torch

from torchtitan.models.common.activation import ActivationFn, SwiGLU
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

# Shape suffix legend:
#   T = token dimensions, D = model dimension, F = feed-forward hidden dimension

__all__ = ["FeedForward", "SigmoidGatedFeedForward", "compute_ffn_hidden_dim"]


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

    merged_param_init = {}
    for name in w1.param_init:
        gate_init = w1.param_init[name]
        up_init = w3.param_init[name]

        def init(
            param: torch.Tensor,
            gate_init: Callable = gate_init,
            up_init: Callable = up_init,
        ) -> None:
            logical_param = param.unflatten(0, (-1, 2))
            gate_init(logical_param[:, 0])
            up_init(logical_param[:, 1])

        merged_param_init[name] = init
    return merged_param_init


def _validate_fused_gate_up_configs(
    w1: Linear.Config,
    w3: Linear.Config,
) -> None:
    """Validate that w1 and w3 can build one physical projection."""
    config_type = type(w1)
    if type(w3) is not config_type:
        raise ValueError(
            "Cannot fuse w1 and w3 with different implementations: "
            f"w1 uses {type(w1).__qualname__}, but w3 uses "
            f"{type(w3).__qualname__}."
        )
    if w1.out_features != w3.out_features:
        raise ValueError("Fused w1 and w3 must have matching out_features")

    comparable_fields = {
        field.name
        for field in fields(config_type)
        if field.init and field.name not in ("out_features", "param_init")
    }
    for field_name in comparable_fields:
        if getattr(w3, field_name) != getattr(w1, field_name):
            raise ValueError(f"Fused w1 and w3 have different {field_name} values")


def _build_fused_gate_up_linear(
    w1: Linear.Config,
    w3: Linear.Config,
    *,
    param_init: dict[str, Callable] | None,
) -> Any:
    """Build logical w1 and w3 configs as one interleaved w13 Linear."""
    _validate_fused_gate_up_configs(w1, w3)

    config_type = type(w1)
    config_kwargs = {
        field.name: getattr(w1, field.name)
        for field in fields(config_type)
        if field.init
    }
    config_kwargs["out_features"] = w1.out_features + w3.out_features
    config_kwargs["param_init"] = param_init
    w13 = config_type(**config_kwargs).build()
    w13._logical_output_slices = (
        ("w1", w1.out_features),
        ("w3", w3.out_features),
    )
    return w13


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
        activation_fn: ActivationFn.Config = field(
            default_factory=lambda: ActivationFn.Config(
                fn=SwiGLU()  # pyrefly: ignore[bad-argument-type]
            )
        )

    def __init__(self, config: Config):
        super().__init__()
        self.w13 = _build_fused_gate_up_linear(
            config.w1,
            config.w3,
            param_init=_merge_gate_up_param_init(config.w1, config.w3),
        )
        self.w2 = config.w2.build()
        self.activation_fn = config.activation_fn.build()
        self.register_state_dict_post_hook(self._split_w13_on_save)
        self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_TF, up_TF = self.w13(x).unflatten(-1, (-1, 2)).unbind(-1)
        return self.w2(self._activation(gate_TF, up_TF))

    def _activation(self, gate_TF: torch.Tensor, up_TF: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(gate_TF, up_TF)

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

        native_key = f"{prefix}w13"
        if native_key in state_dict:
            state_dict[f"{prefix}w13.weight"] = state_dict.pop(native_key).flatten(0, 1)


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
        out = super().forward(x)
        return torch.sigmoid(self.gate(x)) * out
