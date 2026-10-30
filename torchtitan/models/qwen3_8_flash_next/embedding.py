# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hashed n-gram embeddings and per-layer lexical injection."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.models.base.module import Module
from torchtitan.models.common import Conv1d, Embedding, Linear

from .residual import GroupedOffsetRMSNorm

# T = packed tokens, N = hash heads, E = embedding dimension,
# S = residual streams, D = model dimension, C = S * D, W = kernel width.


def ngram_vocab_sizes(
    base: int, num_heads: int, *, ple_layer_index: int
) -> tuple[int, ...]:
    """Use successive distinct primes, including the PLE layer's head offset."""
    sizes = []
    candidate = base
    while len(sizes) < (ple_layer_index + 1) * num_heads:
        if candidate >= 2 and all(
            candidate % divisor for divisor in range(2, math.isqrt(candidate) + 1)
        ):
            sizes.append(candidate)
        candidate += 1
    return tuple(sizes[ple_layer_index * num_heads :])


def ngram_multipliers(
    vocab_size: int, ngram_size: int, *, ple_layer_index: int, seed: int
) -> tuple[int, ...]:
    mask = (1 << 64) - 1
    half_bound = max(1, ((1 << 63) - 1) // vocab_size // 2)
    result = []
    for position in range(ngram_size):
        value = (
            seed + 10007 * ple_layer_index + 0x9E3779B97F4A7C15 * (position + 1)
        ) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        result.append(2 * (value % half_bound) + 1)
    return tuple(result)


def document_starts(positions_T: torch.Tensor) -> torch.Tensor:
    indices_T = torch.arange(positions_T.numel(), device=positions_T.device)
    return torch.where(positions_T == 0, indices_T, 0).cummax(0).values


class NGramEmbedding(Module):
    """XOR hashes of document-local history into a table sharded across features."""

    layer_multipliers: torch.Tensor
    ngram_heads_vocab_sizes: torch.Tensor
    ngram_heads_offsets: torch.Tensor

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ngram_size: int
        heads_per_ngram: int
        vocab_size: int
        head_vocab_sizes: tuple[int, ...]
        ngram_embedding: Embedding.Config
        eos_token_id: int
        ple_layer_index: int = 0
        seed: int = 1234
        split_ngram_parts: int = 1

        def __post_init__(self):
            if self.ngram_size < 2 or self.heads_per_ngram < 1:
                raise ValueError("ngram_size must be >= 2 and heads_per_ngram positive")
            if (
                len(self.head_vocab_sizes)
                != (self.ngram_size - 1) * self.heads_per_ngram
            ):
                raise ValueError("head_vocab_sizes must contain one size per hash head")
            if (
                self.split_ngram_parts < 1
                or self.ngram_embedding.num_embeddings % self.split_ngram_parts
            ):
                raise ValueError("split_ngram_parts must divide the padded table rows")

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.ngram_size = config.ngram_size
        self.heads_per_ngram = config.heads_per_ngram
        self.eos_token_id = config.eos_token_id
        self.ngram_embedding = config.ngram_embedding.build()
        self.register_buffer("layer_multipliers", None)
        self.register_buffer("ngram_heads_vocab_sizes", None)
        self.register_buffer("ngram_heads_offsets", None)
        self._init_self_buffers()

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        cfg = self.config
        with torch.device(buffer_device or self.ngram_embedding.weight.device):
            self.layer_multipliers = torch.tensor(
                ngram_multipliers(
                    cfg.vocab_size,
                    cfg.ngram_size,
                    ple_layer_index=cfg.ple_layer_index,
                    seed=cfg.seed,
                ),
                dtype=torch.int64,
            )
            self.ngram_heads_vocab_sizes = torch.tensor(
                cfg.head_vocab_sizes, dtype=torch.int64
            )
            self.ngram_heads_offsets = (
                self.ngram_heads_vocab_sizes.cumsum(0) - self.ngram_heads_vocab_sizes
            )

    def hash_ids(
        self, input_ids_T: torch.Tensor, *, document_positions_T: torch.Tensor
    ) -> torch.Tensor:
        input_ids_T = input_ids_T.long()
        indices_T = torch.arange(input_ids_T.numel(), device=input_ids_T.device)
        starts_T = document_starts(document_positions_T)
        eos_T = (
            torch.where(input_ids_T == self.eos_token_id, indices_T + 1, 0)
            .cummax(0)
            .values
        )
        previous_eos_T = F.pad(eos_T[:-1], (1, 0))
        starts_T = torch.maximum(starts_T, previous_eos_T)
        shifted_T: list[torch.Tensor] = []
        for shift in range(self.ngram_size):
            source_T = indices_T - shift
            shifted_T.append(
                torch.where(
                    source_T >= starts_T,
                    input_ids_T[source_T.clamp_min(0)],
                    self.eos_token_id,
                )
            )
        hashes_TN = []
        mixed_T = shifted_T[0] * self.layer_multipliers[0]
        for position in range(1, self.ngram_size):
            mixed_T = torch.bitwise_xor(
                mixed_T, shifted_T[position] * self.layer_multipliers[position]
            )
            begin = (position - 1) * self.heads_per_ngram
            end = begin + self.heads_per_ngram
            hashes_TN.append(
                mixed_T.unsqueeze(-1).remainder(self.ngram_heads_vocab_sizes[begin:end])
                + self.ngram_heads_offsets[begin:end]
            )
        return torch.cat(hashes_TN, -1)

    def forward(
        self, input_ids_T: torch.Tensor, *, document_positions_T: torch.Tensor
    ) -> torch.Tensor:
        hashes_TN = self.hash_ids(
            input_ids_T, document_positions_T=document_positions_T
        )
        return self.ngram_embedding(hashes_TN).flatten(-2)


class PLELayer(Module):
    """Inject lexical values into each stream with a dilated causal convolution."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_streams: int
        ple_embedding: NGramEmbedding.Config
        key_proj: Linear.Config
        value_proj: Linear.Config
        norm_key: GroupedOffsetRMSNorm.Config
        norm_query: GroupedOffsetRMSNorm.Config
        norm_conv: GroupedOffsetRMSNorm.Config
        conv1d: Conv1d.Config

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.num_streams = config.num_streams
        self.dilation = config.ple_embedding.ngram_size
        self.ple_embedding = config.ple_embedding.build()
        self.key_proj = config.key_proj.build()
        self.value_proj = config.value_proj.build()
        self.norm_key = config.norm_key.build()
        self.norm_query = config.norm_query.build()
        self.norm_conv = config.norm_conv.build()
        self.conv1d = config.conv1d.build()

    def forward(
        self,
        x_TC: torch.Tensor,
        *,
        input_ids_T: torch.Tensor,
        document_positions_T: torch.Tensor,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding_TE = self.ple_embedding(
            input_ids_T, document_positions_T=document_positions_T
        )
        key_TSD = self.norm_key(self.key_proj(embedding_TE)).unflatten(
            -1, (self.num_streams, self.dim)
        )
        query_TSD = self.norm_query(x_TC).unflatten(-1, (self.num_streams, self.dim))
        score_TS = (key_TSD * query_TSD).sum(-1) * self.dim**-0.5
        score_TS = score_TS.abs().clamp_min(1e-6).sqrt() * score_TS.sign()
        value_TD = self.value_proj(embedding_TE)
        remat.recompute_needs_tensor(value_TD)
        gated_TC = (score_TS.sigmoid().unsqueeze(-1) * value_TD.unsqueeze(-2)).flatten(
            -2
        )
        normed_TC = self.norm_conv(gated_TC)
        remat.recompute_needs_tensor(normed_TC)
        if padding_mask_T is not None:
            gated_TC = gated_TC * (~padding_mask_T).unsqueeze(-1)
            normed_TC = normed_TC * (~padding_mask_T).unsqueeze(-1)
        indices_T = torch.arange(x_TC.shape[0], device=x_TC.device)
        starts_T = document_starts(document_positions_T)
        conv_TC = torch.zeros_like(normed_TC, dtype=torch.float32)
        # Cross-correlation's last coefficient multiplies the current token.
        # Reset every shift at document boundaries, including under packed PP/SP.
        for tap in range(self.conv1d.kernel_size[0]):
            source_T = (
                indices_T - (self.conv1d.kernel_size[0] - 1 - tap) * self.dilation
            )
            shifted_TC = normed_TC[source_T.clamp_min(0)].float() * (
                source_T >= starts_T
            ).unsqueeze(-1)
            conv_TC = conv_TC + shifted_TC * self.conv1d.weight[:, 0, tap].float()
        return gated_TC + F.silu(conv_TC.to(normed_TC.dtype))
