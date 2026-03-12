# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3 model copy for kernel ablation studies.

This module provides a Qwen3 model variant where individual ops can be
swapped to vLLM's fused kernels to measure their performance impact.

Usage:
    In config_registry.py, swap model_spec to use this model:

        from torchtitan.experiments.rl.unified.models.qwen3_ablation import ablation_model_registry
        model_spec=ablation_model_registry("1.7B")

Ablation 1: Replace nn.RMSNorm with vLLM's fused RMSNorm kernel.
Ablation 2: Replace F.silu(w1(x)) * w3(x) with vLLM's fused SiluAndMul kernel.
"""

from dataclasses import fields

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.utils import trunc_normal_
from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.model import Qwen3Model, Qwen3TransformerBlock
from torchtitan.protocols.model_spec import ModelSpec

# vLLM's fused kernels
from vllm.model_executor.layers.activation import SiluAndMul as VLLMSiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm


class FeedForwardVLLMSiluAndMul(FeedForward):
    """FeedForward with vLLM's fused SiluAndMul kernel.

    Original: return self.w2(F.silu(self.w1(x)) * self.w3(x))
      - 3 kernel launches: silu, mul, (implicit in w2)
    Fused:    return self.w2(silu_and_mul(cat(w1(x), w3(x))))
      - 1 fused kernel launch for silu+mul
    """

    def __init__(self, config: FeedForward.Config, *, dim: int):
        super().__init__(config, dim=dim)
        self.silu_and_mul = VLLMSiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate gate (w1) and up (w3) projections, then apply fused silu+mul
        gate_up = torch.cat([self.w1(x), self.w3(x)], dim=-1)
        return self.w2(self.silu_and_mul(gate_up))


class Qwen3TransformerBlockAblation(Qwen3TransformerBlock):
    """Qwen3TransformerBlock with vLLM's fused RMSNorm and SiluAndMul."""

    def __init__(
        self,
        config: Qwen3TransformerBlock.Config,
        *,
        layer_id: int,
        dim: int,
        n_layers: int,
    ):
        # Call grandparent __init__ to skip Qwen3TransformerBlock's nn.RMSNorm creation.
        TransformerBlock.__init__(self)

        self.attention = config.attention.build(dim=dim)

        self.moe_enabled = config.moe_enabled
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build(dim=dim)
        else:
            assert config.feed_forward is not None
            # ── Ablation: use FeedForward with vLLM's fused SiluAndMul ──
            self.feed_forward = FeedForwardVLLMSiluAndMul(
                config.feed_forward, dim=dim
            )

        # ── Ablation: use vLLM's fused RMSNorm ──
        self.attention_norm = VLLMRMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = VLLMRMSNorm(dim, eps=config.norm_eps)

        if config.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5


class Qwen3ModelAblation(Qwen3Model):
    """Qwen3Model with vLLM's fused kernels for ablation studies.

    Fused ops:
    - RMSNorm: vLLM's Triton-based fused kernel (all layers + final norm)
    - SiluAndMul: vLLM's fused activation kernel (all FFN layers)
    """

    def __init__(self, config: Qwen3Model.Config):
        super().__init__(config)

        # Replace final norm with vLLM's fused version
        self.norm = VLLMRMSNorm(config.dim, eps=config.norm_eps)

        # Replace all layers with the ablation block
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = Qwen3TransformerBlockAblation(
                config.layer,
                layer_id=layer_id,
                dim=config.dim,
                n_layers=config.n_layers,
            )


def ablation_model_registry(flavor: str) -> ModelSpec:
    """Return a ModelSpec identical to qwen3's but using the ablation model."""
    base_spec = model_registry(flavor)

    # _owner is a ClassVar set by Configurable.__init_subclass__.
    # We can't modify it on a slots dataclass instance. Instead, create a
    # thin Config subclass whose _owner points to our ablation model class.
    # This makes config.build() instantiate Qwen3ModelAblation.
    base_config = base_spec.model

    class _AblationConfig(type(base_config)):
        _owner = Qwen3ModelAblation

    # Copy all field values from the base config into a new _AblationConfig instance
    field_values = {}
    for f in fields(base_config):
        if f.init:
            field_values[f.name] = getattr(base_config, f.name)
    config = _AblationConfig(**field_values)

    return ModelSpec(
        name=base_spec.name,
        flavor=base_spec.flavor,
        model=config,
        parallelize_fn=base_spec.parallelize_fn,
        pipelining_fn=base_spec.pipelining_fn,
        build_loss_fn=base_spec.build_loss_fn,
        post_optimizer_build_fn=base_spec.post_optimizer_build_fn,
        state_dict_adapter=base_spec.state_dict_adapter,
    )
