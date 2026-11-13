# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hashed n-gram embeddings and per-layer lexical injection."""

import math
from dataclasses import dataclass
from typing import cast

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from torchtitan.distributed.fsdp import resolve_fsdp_mesh
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.row_sharded_embedding import (
    row_sharded_embedding,
    RowShardedEmbeddingMetadata,
)
from torchtitan.models.base.module import Module
from torchtitan.models.common import Conv1d, Embedding, Linear

from .norm import GroupedOffsetRMSNorm

# T = packed tokens, N = hash heads, E = embedding dimension,
# S = residual streams, D = model dimension, C = S * D, W = kernel width,
# R = n-gram table rows.


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
            # The reference advances once in the layer builder and once in SplitMix64.
            seed
            + 10007 * ple_layer_index
            + 0x9E3779B97F4A7C15 * (position + 2)
        ) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        result.append(2 * (value % half_bound) + 1)
    return tuple(result)


def document_starts(positions_T: torch.Tensor) -> torch.Tensor:
    """Return each packed token's document-start index, with shape [T]."""
    indices_T = torch.arange(positions_T.numel(), device=positions_T.device)
    return torch.where(positions_T == 0, indices_T, 0).cummax(0).values


class NGramTable(Embedding):
    """Lexical embedding with persistent rows owned by data-parallel ranks."""

    @dataclass(kw_only=True, slots=True)
    class Config(Embedding.Config):
        pass

    row_mesh: DeviceMesh | None

    def __init__(self, config: Config):
        super().__init__(config)
        self.row_mesh = None

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        global_shape = torch.Size((self.num_embeddings, self.embedding_dim))
        global_stride = self.weight.stride()
        self.row_mesh = parallelism_context.get_optional_mesh("dp")
        if self.row_mesh is not None and self.num_embeddings % self.row_mesh.size():
            # SPMD meta allocation pads uneven shards; owners store only valid rows.
            raise ValueError("PLE requires num_embeddings divisible by the DP degree")
        super()._parallelize(parallelism_context)
        if self.row_mesh is not None:
            # Use the backbone storage mesh to preserve its norm reduction order.
            storage_mesh, _ = resolve_fsdp_mesh(parallelism_context)
            assert storage_mesh.mesh_dim_names is not None
            assert self._sharding_config is not None
            layout = self._sharding_config.state_shardings["weight"]
            placements = tuple(
                spmd.spmd_type_to_dtensor_placement(
                    layout.local_type[
                        "dp" if axis in {"dp_replicate", "dp_shard"} else axis
                    ]
                )
                for axis in storage_mesh.mesh_dim_names
            )
            self.weight = torch.nn.Parameter(
                DTensor.from_local(
                    self.weight,
                    storage_mesh,
                    placements,
                    shape=global_shape,
                    stride=global_stride,
                    run_check=False,
                ),
                requires_grad=self.weight.requires_grad,
            )

    def _init_self_parameters(self) -> None:
        if self.row_mesh is None:
            super()._init_self_parameters()
            return
        # Initialize the local view to preserve owner-seeded RNG without DTensor offsets.
        devices = [] if self.weight.device.type == "cpu" else [self.weight.device]
        weight_RE = self.weight
        assert isinstance(weight_RE, DTensor)
        with torch.random.fork_rng(
            devices=devices, device_type=self.weight.device.type
        ):
            torch.manual_seed(torch.initial_seed() + self.row_mesh.get_local_rank())
            self._init_param(
                "weight",
                torch.nn.Parameter(
                    weight_RE.to_local(grad_placements=weight_RE.placements),
                    requires_grad=False,
                ),
            )

    def forward(
        self,
        input: torch.Tensor,
        *,
        routing: RowShardedEmbeddingMetadata | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self.row_mesh is None:
            return F.embedding(input, self.weight).to(dtype=dtype or self.weight.dtype)
        if routing is None:
            raise ValueError(
                "Row-owned PLE requires lookup routing from preprocess_inputs"
            )
        # _parallelize stores the weight as a Shard(0) DTensor over row_mesh.
        return row_sharded_embedding(cast(DTensor, self.weight), routing, dtype=dtype)


class NGramEmbedding(Module):
    """Map XOR hashes of document-local token history to lexical embeddings."""

    layer_multipliers: torch.Tensor
    ngram_heads_vocab_sizes: torch.Tensor
    ngram_heads_offsets: torch.Tensor

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ngram_size: int
        heads_per_ngram: int
        vocab_size: int
        head_vocab_sizes: tuple[int, ...]
        ngram_embedding: NGramTable.Config
        eos_token_id: int
        # Number of HF checkpoint parts of the table; read by the state dict adapter.
        split_ngram_parts: int

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
                # The only PLE layer uses the reference's hash layer 0 and seed.
                ngram_multipliers(
                    cfg.vocab_size, cfg.ngram_size, ple_layer_index=0, seed=1234
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
        """Hash each token's document-local n-grams into table rows, ``[T, N]``.

        Padding is hashed like any token. The data pipeline starts padding at
        document position 0, so it never enters a real document's history;
        like the GDN layers, PLE does not skip padding inside a document.
        """
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
        self,
        input_ids_T: torch.Tensor,
        *,
        document_positions_T: torch.Tensor,
        routing: RowShardedEmbeddingMetadata | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        hashes_TN = (
            self.hash_ids(input_ids_T, document_positions_T=document_positions_T)
            if routing is None
            else routing.input_ids
        )
        return self.ngram_embedding(hashes_TN, routing=routing, dtype=dtype).flatten(-2)


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
        routing: RowShardedEmbeddingMetadata | None = None,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding_TE = self.ple_embedding(
            input_ids_T,
            document_positions_T=document_positions_T,
            routing=routing,
            dtype=x_TC.dtype,
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
