# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
from dataclasses import dataclass, field

import torch
from torch import nn

from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    ScaledDotProductAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.rope import apply_rotary_emb_single_complex
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


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
        inner_attention: Module.Config = field(
            default_factory=ScaledDotProductAttention.Config
        )
        mask_type: str = "causal"
        mscale: float = 1.0
        rope_factor: float = 1.0
        rope_max_seq_len: int = 4096
        rope_original_seq_len: int = 4096

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

        if config.rope_max_seq_len > config.rope_original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb_single_complex(q_pe, freqs_cis, positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # Key-value projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pe = apply_rotary_emb_single_complex(k_pe.unsqueeze(2), freqs_cis, positions)

        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

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
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x


class MTPTransformerBlock(DeepSeekV3TransformerBlock):
    """Multi-Token Prediction (MTP) transformer block.

    Each MTP block takes an offset token embedding and the previous layer's
    hidden state, projects them via enorm/hnorm/eh_proj, then runs through a
    standard attention + FFN/MoE block.
    """

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
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        input_offset = self.enorm(input_offset)
        prev_embed = self.hnorm(prev_embed)
        h = torch.cat([input_offset, prev_embed], dim=-1)
        h = self.eh_proj(h)

        h = h + self.attention(
            self.attention_norm(h), freqs_cis, attention_masks, positions
        )
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
            trainer_config,
            **kwargs,
        ) -> None:

            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            # Read num_mtp_modules from training config (single source of truth)
            num_mtp = getattr(training, "num_mtp_modules", 0)
            self.num_mtp_modules = num_mtp

            if num_mtp > 0:
                # MTP layers are appended dynamically from training config.
                # n_main_layers = number of layers already in the config
                # (built by _build_dsv3_layers, which only creates main layers).
                self.n_main_layers = len(self.layers)

                ref_layer = self.layers[-1]
                ref_attn = ref_layer.attention
                ref_attn_norm = ref_layer.attention_norm
                ref_ffn_norm = ref_layer.ffn_norm
                ref_ffn = ref_layer.feed_forward
                ref_moe = ref_layer.moe

                # Create and append MTP layer configs
                for mtp_idx in range(num_mtp):
                    mtp_cfg = MTPTransformerBlock.Config(
                        attention=ref_attn,
                        attention_norm=ref_attn_norm,
                        ffn_norm=ref_ffn_norm,
                        feed_forward=ref_ffn,
                        moe=ref_moe,
                        enorm=ref_ffn_norm,
                        hnorm=ref_ffn_norm,
                        eh_proj=Linear.Config(
                            in_features=2 * self.dim,
                            out_features=self.dim,
                            param_init=self.lm_head.param_init,
                        ),
                    )
                    self.layers.append(mtp_cfg)

            # Sync rope fields to attention for all layers (main + MTP).
            # Mutate in-place — simpler than replacing each config in the list.
            for layer_cfg in self.layers:
                assert isinstance(layer_cfg.attention, Attention.Config)
                layer_cfg.attention.rope_max_seq_len = seq_len
                layer_cfg.attention.rope_factor = self.rope.rope_factor
                layer_cfg.attention.rope_original_seq_len = self.rope.original_seq_len

            for layer_cfg in self.layers:
                if layer_cfg.moe is not None:
                    layer_cfg.moe.router._debug_force_load_balance = (
                        debug.moe_force_load_balance
                    )
                    comm_backend = getattr(
                        layer_cfg.moe.experts.token_dispatcher,
                        "comm_backend",
                        "standard",
                    )
                    if (
                        comm_backend in ("deepep", "hybridep")
                        and parallelism.expert_parallel_degree == 1
                    ):
                        raise ValueError(
                            f"{comm_backend.upper()} requires expert parallelism "
                            "(expert_parallel_degree > 1)."
                        )

            if parallelism.context_parallel_degree > 1 and not isinstance(
                self.layers[0].attention.inner_attention,
                ScaledDotProductAttention.Config,
            ):
                raise NotImplementedError(
                    "Context Parallel for DeepSeek V3 only supports "
                    "ScaledDotProductAttention. Got "
                    f"{type(self.layers[0].attention.inner_attention).__name__}."
                )

            from torchtitan.models.deepseek_v3.sharding import (
                set_deepseek_v3_sharding_config,
            )

            set_deepseek_v3_sharding_config(
                self,
                loss_parallel=not parallelism.disable_loss_parallel,
                enable_sp=parallelism.enable_sequence_parallel,
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
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        num_mtp = self.config.num_mtp_modules
        n_main = self.config.n_main_layers

        if self.tok_embeddings is not None:
            seq_len = tokens.shape[1] - num_mtp
            assert seq_len > 0, (
                f"Token sequence length ({tokens.shape[1]}) must be greater than "
                f"num_mtp_modules ({num_mtp})"
            )
            if positions is not None and positions.shape[1] != seq_len:
                main_positions = positions[:, :seq_len]
            else:
                main_positions = positions
            h = self.tok_embeddings(tokens[:, :seq_len])
        else:
            h = tokens
            seq_len = h.shape[1]
            main_positions = positions

        # Main layers
        for i, layer in enumerate(self.layers.values()):
            if i >= n_main:
                break
            h = layer(h, self.freqs_cis, attention_masks, main_positions)

        # PP mid-stage without norm/lm_head: return hidden state
        if n_main < len(self.layers) and self.norm is None and self.lm_head is None:
            return h
        
        prev_embed = h
        h = self.norm(h) if self.norm is not None else h
        output = self.lm_head(h.float()) if self.lm_head is not None else h

        if n_main >= len(self.layers):
            return output

        # MTP layers from the unified list (indices n_main .. end)
        assert self.tok_embeddings is not None, (
            "MTP layers require tok_embeddings for offset token embedding."
        )
        output_list = [output]
        for mtp_idx, layer in enumerate(
            list(self.layers.values())[n_main:]
        ):
            token_offset_id = mtp_idx + 1
            token_end_idx = token_offset_id + seq_len
            input_offset = self.tok_embeddings(
                tokens[:, token_offset_id:token_end_idx]
            )
            h = layer(
                input_offset, prev_embed, self.freqs_cis,
                attention_masks, main_positions,
            )
            prev_embed = h
            h = self.norm(h) if self.norm is not None else h
            mtp_output = self.lm_head(h.float()) if self.lm_head is not None else h
            output_list.append(mtp_output)

        return output_list


    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        if self.config.num_mtp_modules > 0:
            return self._forward_with_mtp(tokens, attention_masks, positions)
        return super().forward(tokens, attention_masks, positions)