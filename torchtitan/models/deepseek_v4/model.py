# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import cast, TYPE_CHECKING

import torch
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.deepseek_v3.mtp import roll_mtp_sequence
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)

from .mhc import HcHead, HcPost, HcPre

if TYPE_CHECKING:
    from .attention import Attention
    from .moe import DeepSeekV4MoE
    from .mtp import MTPBlock


class DeepSeekV4TransformerBlock(TransformerBlock):
    """Transformer block with HC pre/post mixing around attention and FFN."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # Redeclared with the DeepSeek V4 specific types so sharding and MTP
        # build helpers can access V4-only fields (e.g. router.layer_id).
        attention: "Attention.Config"  # pyrefly: ignore [bad-override]
        moe: "DeepSeekV4MoE.Config | None" = None  # pyrefly: ignore [bad-override]
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config

    def __init__(self, config: Config):
        super().__init__()
        cfg = config

        self.attention = cfg.attention.build()
        self.attention_norm = cfg.attention_norm.build()
        self.ffn_norm = cfg.ffn_norm.build()
        if cfg.moe is not None:
            assert cfg.moe is not None
            self.moe = cfg.moe.build()
            self.moe_enabled = True
        else:
            assert cfg.feed_forward is not None
            self.moe = None
            self.feed_forward = cfg.feed_forward.build()
            self.moe_enabled = False

        self.hc_attn_pre = cfg.hc_attn_pre.build()
        self.hc_ffn_pre = cfg.hc_ffn_pre.build()
        self.hc_post = cfg.hc_post.build()

    def forward(
        self,
        x: torch.Tensor,
        input_ids_T: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """Run one DeepSeek V4 decoder block.

        Args:
            x: Hidden states of shape ``[T, hc_mult, D]``.
            input_ids_T: Token IDs of shape ``[T]`` used by hash routing.
            attention_masks: Optional decoder mask handle; sparse attention may
                ignore it and build masks internally.
            positions: Optional position IDs of shape ``[T]``.

        Returns:
            Hidden states of shape ``[T, hc_mult, D]``.
        """
        residual = x
        x, post, comb = self.hc_attn_pre(x)
        x = self.attention(self.attention_norm(x), attention_masks, positions)
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_ffn_pre(x)
        if self.moe_enabled:
            ffn_input = self.ffn_norm(x)
            if getattr(self.moe.router, "hash", False):
                x = self.moe(ffn_input, input_ids_T=input_ids_T)
            else:
                x = self.moe(ffn_input)
        else:
            x = self.feed_forward(self.ffn_norm(x))
        x = self.hc_post(x, residual, post, comb)
        return x


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
        mtp_layers: list["MTPBlock.Config"] | None = None

        def update_from_config(self, *, config, **kwargs):
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            if self.mtp_layers is not None and parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "DeepSeek V4 MTP does not support pipeline parallelism yet."
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

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            """Estimate DeepSeek V4 training FLOPs from the final model config."""
            deepseek_v4_model = cast(DeepSeekV4Model, model)
            nparams, active_nparams = get_nparams_and_active_nparams(deepseek_v4_model)

            attention_op_flops = 0
            for layers in (self.layers, self.mtp_layers or ()):
                for layer in layers:
                    attention = layer.attention
                    inner_attention = attention.inner_attention
                    attention_op_flops += quadratic_attention_flops_per_token(
                        num_heads=attention.n_heads,
                        qk_head_dim=attention.head_dim,
                        v_head_dim=attention.head_dim,
                        seq_len=seq_len,
                        sliding_window_size=inner_attention.window_size,
                    )

                    if attention.compress_ratio > 1:
                        compressed_seq_len = seq_len // attention.compress_ratio
                        if attention.compress_ratio == 4:
                            attention_op_flops += (
                                6
                                * attention.index_n_heads
                                * attention.index_head_dim
                                * compressed_seq_len
                            )
                            compressed_seq_len = min(
                                compressed_seq_len, inner_attention.index_topk
                            )
                        attention_op_flops += quadratic_attention_flops_per_token(
                            num_heads=attention.n_heads,
                            qk_head_dim=attention.head_dim,
                            v_head_dim=attention.head_dim,
                            seq_len=compressed_seq_len,
                        )

            active_nparams += len(deepseek_v4_model.mtp_layers) * sum(
                param.numel() for param in deepseek_v4_model.lm_head.parameters()
            )
            active_nparams += (self.hc_mult - 1) * sum(
                param.numel()
                for mtp_layer in deepseek_v4_model.mtp_layers
                for param in cast("MTPBlock", mtp_layer).h_proj.parameters()
            )

            return nparams, 6 * active_nparams + attention_op_flops

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

    def get_attention_masks(self, positions):
        return None

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ):
        """Run the DeepSeek V4 decoder."""
        if len(self.mtp_layers) > 0 and self.tok_embeddings is None:
            raise ValueError("DeepSeek V4 MTP forward requires token embeddings.")
        if len(self.mtp_layers) > 0 and self._skip_lm_head:
            raise ValueError(
                "DeepSeek V4 MTP cannot skip the LM head because chunked "
                "cross entropy is not supported."
            )

        input_ids_T = tokens.detach().long()
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        h = h.unsqueeze(1).repeat(1, self.hc_mult, 1)

        for i in range(self.n_main_layers):
            layer = self.layers[str(i)]
            h = layer(h, input_ids_T, attention_masks, positions)

        prev_hc_hidden = h
        main_hidden = self.hc_head(h)
        main_hidden = self.norm(main_hidden) if self.norm is not None else main_hidden

        if len(self.mtp_layers) == 0:
            if self._skip_lm_head or self.lm_head is None:
                return main_hidden
            return self.lm_head(main_hidden)

        outputs = [main_hidden] + self.mtp_forward(
            prev_hc_hidden,
            tokens,
            attention_masks,
            positions,
        )
        return [
            self.lm_head(item) if self.lm_head is not None else item for item in outputs
        ]

    def mtp_forward(
        self,
        prev_hc_hidden: torch.Tensor,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Run auxiliary MTP depths and return prediction hidden states."""
        mtp_outputs = []
        for depth, mtp_block in enumerate(self.mtp_layers, 1):
            mtp_tokens, valid_mask = roll_mtp_sequence(
                tokens,
                shift=depth,
                fill_value=0,
                positions=positions,
                return_valid_mask=True,
            )
            prev_hc_hidden, prediction_hidden = mtp_block(
                self.tok_embeddings(mtp_tokens),
                prev_hc_hidden,
                mtp_tokens.detach().long(),
                valid_mask,
                attention_masks,
                positions,
            )
            mtp_outputs.append(prediction_hidden)
        return mtp_outputs
