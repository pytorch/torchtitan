# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Token normalization, document-local n-gram hashing, and gated Engram reads.

T: tokens, L: Engram layers, Q: hash columns, C: HC branches, D: hidden width,
E: embedding features.
"""

from dataclasses import dataclass
from math import isqrt

import numpy as np

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear

from .quantization import fp8_activation


class FeatureParallelEmbedding(Embedding):
    """Read a feature shard of every Engram row without vocabulary offsets."""

    @dataclass(kw_only=True, slots=True)
    class Config(Embedding.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.Embedding.forward(self, input)


def compressed_token_map(tokenizer) -> tuple[tuple[int, ...], int]:
    """Use the author's Rust-tokenizer normalization, including byte tokens."""
    from tokenizers import normalizers, Regex

    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), "\ue000"),
            normalizers.Strip(),
            normalizers.Replace("\ue000", " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    normalized_ids = {}
    lookup = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        lookup.append(normalized_ids.setdefault(key, len(normalized_ids)))
    return tuple(lookup), len(normalized_ids)


def hash_primes(
    *, num_layers: int, max_ngram_size: int, num_heads: int, bucket_size: int
):
    """Draw the same globally unique prime bucket sizes as the reference."""
    seen = set()
    layers = []
    for _ in range(num_layers):
        orders = []
        for _ in range(max_ngram_size - 1):
            current = bucket_size - 1
            heads = []
            for _ in range(num_heads):
                current += 1
                while (
                    current in seen
                    or current < 2
                    or any(
                        current % divisor == 0
                        for divisor in range(2, isqrt(current) + 1)
                    )
                ):
                    current += 1
                heads.append(current)
                seen.add(current)
            orders.append(tuple(heads))
        layers.append(tuple(orders))
    return tuple(layers)


class EngramHash(Module):
    """Stateless hashing; history never crosses a document or an image span.

    Full checkpoints require their tokenizer's normalized map. A metadata-only
    configuration may omit the map, but cannot hash tokens until one is supplied."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_ids: tuple[int, ...]
        primes: tuple[tuple[tuple[int, ...], ...], ...]
        max_ngram_size: int
        compressed_vocab_size: int
        pad_id: int
        token_map: tuple[int, ...] = ()

        def __post_init__(self):
            if (
                self.token_map
                and len(set(self.token_map)) != self.compressed_vocab_size
            ):
                raise ValueError(
                    "Engram token map does not match compressed_vocab_size."
                )

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.register_buffer(
            "token_map",
            torch.empty(len(config.token_map), dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "primes",
            torch.empty(
                len(config.layer_ids),
                config.max_ngram_size - 1,
                len(config.primes[0][0]),
                dtype=torch.int64,
            ),
            persistent=False,
        )
        self.register_buffer(
            "multipliers",
            torch.empty(
                len(config.layer_ids), config.max_ngram_size, dtype=torch.int64
            ),
            persistent=False,
        )
        self.register_buffer(
            "offsets",
            torch.empty(
                len(config.layer_ids),
                (config.max_ngram_size - 1) * len(config.primes[0][0]),
                dtype=torch.int64,
            ),
            persistent=False,
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None):
        config = self.config
        device = buffer_device or self.token_map.device
        self.token_map = torch.tensor(
            config.token_map, dtype=torch.int64, device=device
        )
        self.primes = torch.tensor(config.primes, dtype=torch.int64, device=device)
        bound = max(1, (np.iinfo(np.int64).max // config.compressed_vocab_size) // 2)
        self.multipliers = torch.tensor(
            np.stack(
                [
                    np.random.default_rng(10007 * layer_id).integers(
                        0, bound, config.max_ngram_size, dtype=np.int64
                    )
                    * 2
                    + 1
                    for layer_id in config.layer_ids
                ]
            ),
            device=device,
        )
        sizes_LQ = self.primes.flatten(-2)
        self.offsets = sizes_LQ.cumsum(-1) - sizes_LQ

    @torch.no_grad()
    def forward(self, tokens_T, positions_T, *, token_mask_T=None):
        if not self.config.token_map:
            raise ValueError(
                "Engram hashing requires the checkpoint tokenizer. Pass "
                "tokenizer_path to the model flavor, or supply engram_hashes to forward."
            )
        compressed_T = self.token_map[tokens_T]
        alive_T = (
            torch.ones_like(tokens_T, dtype=torch.bool)
            if token_mask_T is None
            else token_mask_T
        )
        pad = self.token_map[self.config.pad_id]
        slots_T = torch.arange(tokens_T.numel(), device=tokens_T.device)
        blocked_T = torch.zeros_like(alive_T)
        history = []
        for shift in range(self.config.max_ngram_size):
            source_T = (slots_T - shift).clamp_min(0)
            blocked_T = blocked_T | (positions_T < shift) | ~alive_T[source_T]
            history.append(torch.where(blocked_T, pad, compressed_T[source_T]))
        tokens_TN = torch.stack(history, -1)
        products_TLN = tokens_TN.unsqueeze(1) * self.multipliers
        rolling_TL = products_TLN[..., 0]
        hashes = []
        for shift in range(1, self.config.max_ngram_size):
            rolling_TL = torch.bitwise_xor(rolling_TL, products_TLN[..., shift])
            hashes.append(rolling_TL.unsqueeze(-1) % self.primes[:, shift - 1])
        return torch.cat(hashes, -1) + self.offsets


class Engram(Module):
    """Retrieve n-gram features and write a context-gated value to each HC branch."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hc_mult: int
        norm_eps: float
        embed: Embedding.Config
        wkv: Linear.Config
        quantize: bool = True

    def __init__(self, config: Config):
        super().__init__()
        self.dim, self.hc_mult, self.eps = config.dim, config.hc_mult, config.norm_eps
        self.quantize = config.quantize
        self.embed = config.embed.build()
        self.wkv = config.wkv.build()
        self.q_weight = torch.nn.Parameter(torch.empty(config.hc_mult, config.dim))
        self.k_weight = torch.nn.Parameter(torch.empty(config.hc_mult, config.dim))

    def forward(self, x_TCD, hashes_TQ, *, token_mask_T=None):
        features_TQE = self.embed(hashes_TQ)
        if self.quantize:
            features_TQE = fp8_activation(features_TQE)
        kv_TD = self.wkv(features_TQE.flatten(-2))
        remat.recompute_needs_tensor(kv_TD)
        key_TC_D, value_TD = kv_TD.split([self.hc_mult * self.dim, self.dim], -1)
        key_TCD = key_TC_D.float().unflatten(-1, (self.hc_mult, self.dim))
        h_TCD = x_TCD.float()
        weight_CD = self.q_weight.float() * self.k_weight.float()
        rstd_TC = torch.rsqrt(h_TCD.square().mean(-1) + self.eps) * torch.rsqrt(
            key_TCD.square().mean(-1) + self.eps
        )
        dot_TC = (h_TCD * weight_CD * key_TCD).sum(-1) * rstd_TC * self.dim**-0.5
        gate_TC = torch.sigmoid(
            torch.copysign(dot_TC.abs().clamp_min(1e-6).sqrt(), dot_TC)
        )
        if token_mask_T is not None:
            tp_group = spmd_mesh_group("tp")
            if tp_group is not None and spmd_dense_sp_enabled():
                token_mask_T = spmd.redistribute(
                    token_mask_T, tp_group, src=spmd.R, dst=spmd.S(0)
                )
            gate_TC = gate_TC.masked_fill(~token_mask_T.unsqueeze(-1), 0)
        return (h_TCD + gate_TC.unsqueeze(-1) * value_TD.float().unsqueeze(-2)).to(
            x_TCD.dtype
        )
