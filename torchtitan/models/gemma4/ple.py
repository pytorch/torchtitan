# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Gemma-4 Per-Layer Embedding (PLE) Module
# Implements auxiliary per-layer embedding table, projection, gating, and injection
# for Gemma-4 Edge architectures (e2b, e4b).

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common import Embedding, Linear, RMSNorm
from torchtitan.protocols.module import Module


class Gemma4PerLayerEmbedding(Module):
    """Model-level Per-Layer Embedding (PLE) generator for Gemma-4 Edge models.

    Computes the packed per-layer conditioning tensor by fusing:
    1. Token-identity lookup from `tok_embeddings_per_layer`
    2. Context-aware projection from `per_layer_model_projection`
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        vocab_size: int
        dim: int
        num_layers: int
        ple_dim: int = 256
        tok_embeddings_per_layer: Embedding.Config
        per_layer_model_projection: Linear.Config
        per_layer_projection_norm: RMSNorm.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.dim = config.dim
        self.num_layers = config.num_layers
        self.ple_dim = config.ple_dim

        self.tok_embeddings_per_layer = config.tok_embeddings_per_layer.build()
        self.per_layer_model_projection = config.per_layer_model_projection.build()
        self.per_layer_projection_norm = config.per_layer_projection_norm.build()

    def forward(
        self, tokens: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Compute the combined [..., num_layers, ple_dim] per-layer conditioning tensor.

        Args:
            tokens: Input token IDs of shape [...] (1D packed tokens or 2D batch x seq).
            inputs_embeds: Base embedding representations of shape [..., dim].

        Returns:
            per_layer_inputs tensor of shape [..., num_layers, ple_dim].
        """
        # 1. Token-identity lookup scaled by sqrt(ple_dim)
        ple_tokens = self.tok_embeddings_per_layer(tokens) * math.sqrt(self.ple_dim)
        ple_tokens_shape = list(tokens.shape) + [self.num_layers, self.ple_dim]
        ple_tokens = ple_tokens.view(*ple_tokens_shape)

        # 2. Context-aware projection scaled by 1/sqrt(dim) and normalized
        proj = self.per_layer_model_projection(inputs_embeds) / math.sqrt(self.dim)
        proj = proj.view(*ple_tokens_shape)
        proj = self.per_layer_projection_norm(proj)

        # 3. Fused per-layer inputs scaled by 1/sqrt(2)
        return (ple_tokens + proj) * (1.0 / math.sqrt(2.0))


class Gemma4LayerPLE(Module):
    """Layer-level Gated Per-Layer Embedding (PLE) injection for Gemma-4 Edge decoder layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        ple_dim: int = 256
        per_layer_input_gate: Linear.Config
        per_layer_projection: Linear.Config
        post_per_layer_input_norm: RMSNorm.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.dim = config.dim
        self.ple_dim = config.ple_dim
        self.per_layer_input_gate = config.per_layer_input_gate.build()
        self.per_layer_projection = config.per_layer_projection.build()
        self.post_per_layer_input_norm = config.post_per_layer_input_norm.build()

    def forward(
        self, x: torch.Tensor, layer_ple_input: torch.Tensor
    ) -> torch.Tensor:
        """Inject gated per-layer embedding conditioning into layer residual stream.

        Args:
            x: Current hidden state tensor of shape [..., dim].
            layer_ple_input: Layer slice of shape [..., ple_dim].

        Returns:
            Updated hidden state tensor of shape [..., dim].
        """
        gate = F.gelu(self.per_layer_input_gate(x), approximate="tanh")
        ple_val = self.per_layer_projection(gate * layer_ple_input)
        return x + self.post_per_layer_input_norm(ple_val)
