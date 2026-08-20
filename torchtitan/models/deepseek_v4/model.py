# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses as dc
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.protocols.module import Module

from .mhc import HcHead, HcPost, HcPre


class DeepSeekV4TransformerBlock(TransformerBlock):
    """Transformer block with HC pre/post mixing around attention and FFN."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config

    def __init__(self, config: Config):
        super().__init__()
        cfg = config

        self.attention = cfg.attention.build()
        self.attention_norm = (
            cfg.attention_norm.build() if cfg.attention_norm is not None else None
        )
        self.ffn_norm = cfg.ffn_norm.build() if cfg.ffn_norm is not None else None
        if cfg.moe is not None:
            self.moe = cfg.moe.build()
            self.feed_forward = None
            self.moe_enabled = True
        else:
            self.moe = None
            self.feed_forward = (
                cfg.feed_forward.build() if cfg.feed_forward is not None else None
            )
            self.moe_enabled = False

        self.hc_attn_pre = cfg.hc_attn_pre.build()
        self.hc_ffn_pre = cfg.hc_ffn_pre.build()
        self.hc_post = cfg.hc_post.build()

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """Run one DeepSeek V4 decoder block.

        Args:
            x: Hidden states of shape ``[B, L, hc_mult, D]``.
            input_ids: Token IDs of shape ``[B, L]`` used by hash routing.
            attention_masks: Optional decoder mask handle; sparse attention may
                ignore it and build masks internally.
            positions: Optional position IDs of shape ``[B, L]``.

        Returns:
            Hidden states of shape ``[B, L, hc_mult, D]``.
        """
        residual = x
        x, post, comb = self.hc_attn_pre(x)
        x = self.attention(self.attention_norm(x), attention_masks, positions)
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_ffn_pre(x)
        if self.moe_enabled:
            x = self.moe(self.ffn_norm(x), input_ids)
        else:
            x = self.feed_forward(self.ffn_norm(x))
        x = self.hc_post(x, residual, post, comb)
        return x


class MTPBlock(Module):
    """Auxiliary multi-token prediction block for DeepSeek V4."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        block: DeepSeekV4TransformerBlock.Config
        dim: int
        hc_mult: int = 4
        norm_eps: float = 1e-6
        eps: float = 1e-6
        e_proj: Linear.Config
        h_proj: Linear.Config
        enorm: RMSNorm.Config
        hnorm: RMSNorm.Config
        norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.block = cfg.block.build()
        self.e_proj = cfg.e_proj.build()
        self.h_proj = cfg.h_proj.build()
        self.enorm = cfg.enorm.build()
        self.hnorm = cfg.hnorm.build()
        self.norm = cfg.norm.build()
        self.hc_mult = cfg.hc_mult
        self.dim = cfg.dim
        self.norm_eps = cfg.norm_eps
        self.eps = cfg.eps
        hc_dim = self.hc_mult * self.dim
        self.hc_head_fn = torch.nn.Parameter(
            torch.empty(self.hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = torch.nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32)
        )
        self.hc_head_scale = torch.nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.embed: Linear | None = None
        self.head: Linear | None = None

    def _merge_hc(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, self.hc_head_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.embed is None or self.head is None:
            raise ValueError("MTPBlock requires embed and head references")

        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = self.block(x, input_ids, attention_masks, positions)
        x = self._merge_hc(x)
        x = self.norm(x)
        return self.head(x.float())


class DeepSeekV4Model(Decoder):
    """DeepSeek V4 decoder model with HC branches and sparse attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int
        vocab_size: int
        hc_mult: int = 4
        n_mtp_layers: int = 0
        compress_ratios: tuple[int, ...] = (1, 1, 4, 4)
        n_layers: int = 4
        norm_eps: float = 1e-6
        hc_head: HcHead.Config
        mtp_layers: list[MTPBlock.Config] | None = None

        def update_from_config(self, *, config, **kwargs):
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            if hasattr(config, "training"):
                seq_len = config.training.seq_len
                for layer_cfg in self.layers:
                    attention = layer_cfg.attention
                    if attention.compressor is not None:
                        attention.compressor.rope = dc.replace(
                            attention.compressor.rope,
                            max_seq_len=seq_len,
                        )
                    if attention.compressor_128 is not None:
                        attention.compressor_128.rope = dc.replace(
                            attention.compressor_128.rope,
                            max_seq_len=seq_len,
                        )
                    if attention.indexer is not None:
                        attention.indexer.rope = dc.replace(
                            attention.indexer.rope,
                            max_seq_len=seq_len,
                        )
                        attention.indexer.compressor.rope = dc.replace(
                            attention.indexer.compressor.rope,
                            max_seq_len=seq_len,
                        )

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                for i in range(self.n_layers):
                    layer_cfg = self.layers[i]
                    n_heads = layer_cfg.attention.n_heads
                    if n_heads % tp != 0:
                        raise ValueError(
                            f"n_heads ({n_heads}) must be divisible by tp ({tp})"
                        )
                    n_groups = layer_cfg.attention.n_groups
                    if n_groups % tp != 0:
                        raise ValueError(
                            f"n_groups ({n_groups}) must be divisible by tp ({tp})"
                        )

            if parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "Context Parallel is not yet supported for DeepSeek V4 sparse attention."
                )

            from .sharding import set_deepseek_v4_sharding_config

            set_deepseek_v4_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(self, model, seq_len):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_embed_params = sum(
                p.numel()
                for n, p in model.named_parameters()
                if p.requires_grad and "tok_embeddings" not in n and "lm_head" not in n
            )
            n_layers = self.n_layers + self.n_mtp_layers
            head_dim = self.layers[0].attention.head_dim
            n_heads = self.layers[0].attention.n_heads
            flops_per_token = (
                6 * non_embed_params + 12 * n_layers * n_heads * head_dim * seq_len
            )
            return total_params, int(flops_per_token)

    def __init__(self, config: Config):
        super().__init__(config)
        cfg = config

        self.hc_mult = cfg.hc_mult
        self.n_mtp_layers = cfg.n_mtp_layers
        self.compress_ratios = list(cfg.compress_ratios)[: cfg.n_layers]
        self.n_main_layers = cfg.n_layers

        self.hc_head = cfg.hc_head.build()
        self.mtp_layers = torch.nn.ModuleList()
        if cfg.mtp_layers is not None:
            self.mtp_layers = torch.nn.ModuleList(
                mtp_layer.build() for mtp_layer in cfg.mtp_layers
            )
            for mtp_layer in self.mtp_layers:
                mtp_layer.embed = self.tok_embeddings
                mtp_layer.head = self.lm_head

    def get_attention_masks(self, positions):
        return None

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        """Run the DeepSeek V4 decoder.

        Args:
            tokens: Token IDs of shape ``[B, L]`` when embeddings are enabled,
                or hidden states when embeddings are skipped.
            positions: Optional position IDs of shape ``[B, L]``.
            attention_masks: Optional decoder attention mask handle.

        Returns:
            Logits of shape ``[B, L, vocab_size]`` unless the LM head is
            skipped, in which case hidden states of shape ``[B, L, D]`` are
            returned.
        """
        input_ids = tokens.detach().long()
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        for i in range(self.n_main_layers):
            layer = self.layers[str(i)]
            h = layer(h, input_ids, attention_masks, positions)

        h = self.hc_head(h)
        h = self.norm(h) if self.norm is not None else h
        if self._skip_lm_head:
            return h
        output = self.lm_head(h.float()) if self.lm_head is not None else h
        return output

    def mtp_forward(
        self,
        h: torch.Tensor,
        input_ids: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Run all auxiliary MTP blocks and return their logits."""
        return [
            mtp_block(h, input_ids, attention_masks, positions)
            for mtp_block in self.mtp_layers
        ]

