# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
from torch import nn

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    FlexAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module


class Attention(BaseAttention):
    """
    Multi-head latent attention (MLA) module.

    This is DeepSeek V3-specific and NOT shared with other models.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        dim: int
        wq: Linear.Config | None = None
        wq_a: Linear.Config | None = None
        wq_b: Linear.Config | None = None
        wkv_a: Linear.Config
        wkv_b: Linear.Config
        wo: Linear.Config
        q_lora_rank: int = 0
        kv_lora_rank: int = 512
        q_norm: RMSNorm.Config
        kv_norm: RMSNorm.Config
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        rope: RoPE.Config
        inner_attention: Module.Config = field(default_factory=FlexAttention.Config)
        mscale: float = 1.0

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        if self.q_lora_rank == 0:
            assert config.wq is not None, "wq is required when q_lora_rank == 0"
            self.wq = config.wq.build()
        else:
            assert (
                config.wq_a is not None and config.wq_b is not None
            ), "wq_a and wq_b are required when q_lora_rank > 0"
            self.wq_a = config.wq_a.build()
            self.q_norm = config.q_norm.build()
            self.wq_b = config.wq_b.build()

        # TODO(fegin): revisit
        # https://github.com/pytorch/torchtitan/pull/2785#discussion_r3034078575
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.wo = config.wo.build()
        self.softmax_scale = self.qk_head_dim**-0.5

        if config.rope.max_seq_len > config.rope.original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.inner_attention = config.inner_attention.build()
        self.rope = config.rope.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))

        # TODO(pianpwk): same QKV:S(2) unflatten case handled by even sharding
        with spmd.local():
            q = q.view(bsz, seqlen, -1, self.qk_head_dim)
            if get_spmd_backend() == "spmd_types":
                spmd.assert_type(
                    q,
                    {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                )

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Key-value projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        q_pe, k_pe = self.rope(q_pe, k_pe.unsqueeze(2), positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.wkv_b(self.kv_norm(kv))

        with spmd.local():  # QKV even shard unflatten, but the expand is truly local SPMD
            kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k = torch.cat([k_nope, k_pe.expand(-1, -1, k_nope.size(2), -1)], dim=-1)
            if get_spmd_backend() == "spmd_types":
                for t in [k, v]:
                    spmd.assert_type(
                        t,
                        {"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.S(2)},
                    )

        output = self.inner_attention(
            q, k, v, attention_masks=attention_masks, scale=self.softmax_scale
        ).contiguous()
        output = output.view(bsz, seqlen, -1)
        return self.wo(output)


class DeepSeekV3TransformerBlock(TransformerBlock):
    """
    DeepSeek V3 Transformer block with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(self.attention_norm(x), attention_masks, positions)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x


class MTPTransformerBlock(DeepSeekV3TransformerBlock):
    """Multi-token prediction transformer block for DeepSeek-V3."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3TransformerBlock.Config):
        enorm: RMSNorm.Config
        hnorm: RMSNorm.Config
        eh_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.enorm = config.enorm.build()
        self.hnorm = config.hnorm.build()
        self.eh_proj = config.eh_proj.build()

    def forward(
        self,
        input_offset: torch.Tensor,
        prev_embed: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        input_offset = self.enorm(input_offset)
        prev_embed = self.hnorm(prev_embed)
        h = self.eh_proj(torch.cat([input_offset, prev_embed], dim=-1))

        h = h + self.attention(self.attention_norm(h), attention_masks, positions)
        if self.moe_enabled:
            h = h + self.moe(self.ffn_norm(h))
        else:
            h = h + self.feed_forward(self.ffn_norm(h))
        return h


class DeepSeekV3Model(Decoder):
    """
    DeepSeek-V3 Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        vocab_size: int = 102400
        n_main_layers: int = 0
        num_mtp_modules: int = 0

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism
            training = getattr(config, "training", None)
            num_mtp_modules = (
                getattr(training, "num_mtp_modules", 0)
                if training is not None
                else self.num_mtp_modules
            )
            self.layers = [
                layer
                for layer in self.layers
                if not isinstance(layer, MTPTransformerBlock.Config)
            ]
            self.num_mtp_modules = num_mtp_modules
            self.n_main_layers = len(self.layers)

            if num_mtp_modules > 0:
                if parallelism.pipeline_parallel_degree > 1:
                    raise NotImplementedError(
                        "DeepSeek-V3 MTP does not support pipeline parallelism yet."
                    )
                if self.n_main_layers == 0:
                    raise ValueError("DeepSeek-V3 MTP requires at least one layer.")

                ref_layer = self.layers[-1]
                for _ in range(num_mtp_modules):
                    self.layers.append(
                        MTPTransformerBlock.Config(
                            attention=copy.deepcopy(ref_layer.attention),
                            attention_norm=copy.deepcopy(ref_layer.attention_norm),
                            ffn_norm=copy.deepcopy(ref_layer.ffn_norm),
                            feed_forward=copy.deepcopy(ref_layer.feed_forward),
                            moe=copy.deepcopy(ref_layer.moe),
                            enorm=RMSNorm.Config(
                                normalized_shape=self.dim,
                                param_init=copy.deepcopy(
                                    ref_layer.ffn_norm.param_init
                                ),
                            ),
                            hnorm=RMSNorm.Config(
                                normalized_shape=self.dim,
                                param_init=copy.deepcopy(
                                    ref_layer.ffn_norm.param_init
                                ),
                            ),
                            eh_proj=Linear.Config(
                                in_features=2 * self.dim,
                                out_features=self.dim,
                                param_init=copy.deepcopy(self.lm_head.param_init),
                            ),
                        )
                    )

            from torchtitan.models.deepseek_v3.sharding import (
                set_deepseek_v3_sharding_config,
            )

            set_deepseek_v3_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:

            assert isinstance(self.layers[0].attention, Attention.Config)
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layers[0].attention.n_heads,
                self.layers[0].attention.qk_nope_head_dim
                + self.layers[0].attention.qk_rope_head_dim
                + self.layers[0].attention.v_head_dim,
                seq_len,
            )

    def _forward_with_mtp(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        num_mtp_modules = self.config.num_mtp_modules
        n_main_layers = self.config.n_main_layers
        seq_len = tokens.shape[1] - num_mtp_modules
        if seq_len <= 0:
            raise ValueError(
                f"Token sequence length ({tokens.shape[1]}) must be greater than "
                f"num_mtp_modules ({num_mtp_modules})."
            )

        main_tokens = tokens[:, :seq_len]
        main_positions = (
            positions[:, :seq_len]
            if positions is not None and positions.shape[1] != seq_len
            else positions
        )

        h = (
            self.tok_embeddings(main_tokens)
            if self.tok_embeddings is not None
            else main_tokens
        )
        layers = list(self.layers.values())
        for layer in layers[:n_main_layers]:
            h = layer(h, attention_masks, main_positions)

        prev_embed = h
        h = self.norm(h) if self.norm is not None else h
        output = self.lm_head(h) if self.lm_head is not None else h
        output_list = [output]

        if self.tok_embeddings is None:
            raise ValueError("DeepSeek-V3 MTP requires token embeddings.")

        h = prev_embed
        for mtp_idx, layer in enumerate(layers[n_main_layers:], 1):
            input_offset = self.tok_embeddings(tokens[:, mtp_idx : mtp_idx + seq_len])
            h = layer(input_offset, h, attention_masks, main_positions)
            mtp_h = self.norm(h) if self.norm is not None else h
            mtp_output = self.lm_head(mtp_h) if self.lm_head is not None else mtp_h
            output_list.append(mtp_output)

        return output_list

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        if self.config.num_mtp_modules > 0:
            return self._forward_with_mtp(tokens, positions, attention_masks)
        return super().forward(tokens, positions, attention_masks)
