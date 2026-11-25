# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffixes: T = tokens, M = HC branches (hc_mult), D = hidden width.

from dataclasses import dataclass, field
from typing import Any, cast, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.base.model import FlopsEstimator
from torchtitan.models.base.module import Module, ModuleList
from torchtitan.models.common.attention import AttentionMetadataMap
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.mtp import MTPDecoder

from .mhc import HcHead, HcPost, HcPre
from .state_dict_adapter import DeepSeekV4StateDictAdapter

if TYPE_CHECKING:
    from .attention import (
        DeepSeekV4Attention,
        DeepSeekV4CompressedSparseAttentionMetadata,
    )
    from .mtp import DeepSeekV4MTPBlock


class DeepSeekV4TransformerBlock(TransformerBlock):
    """Transformer block with HC pre/post mixing around attention and FFN."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        attention: "DeepSeekV4Attention.Config"  # pyrefly: ignore [bad-override]
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config

    def __init__(self, config: Config):
        super().__init__()
        cfg = config

        self.attention = cfg.attention.build()
        self.attention_norm = cfg.attention_norm.build()
        self.ffn_norm = cfg.ffn_norm.build()
        self.feed_forward = cfg.feed_forward.build()

        self.hc_attn_pre = cfg.hc_attn_pre.build()
        self.hc_ffn_pre = cfg.hc_ffn_pre.build()
        self.hc_post = cfg.hc_post.build()

    def forward(
        self,
        x: torch.Tensor,
        input_ids_T: torch.Tensor | None,
        attention_metadata: "DeepSeekV4CompressedSparseAttentionMetadata | None",
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ):
        """Run one DeepSeek V4 decoder block.

        Args:
            x: Hidden states of shape ``[T, hc_mult, D]``.
            input_ids_T: Token IDs of shape ``[T]``; required by hash-routed
                layers.
            attention_metadata: Optional document offsets consumed by sparse
                attention.
            positions: Optional position IDs of shape ``[T]``.

        Returns:
            Hidden states of shape ``[T, hc_mult, D]``.
        """
        residual = x
        x, post, comb = self.hc_attn_pre(x)
        x = self.attention(
            self.attention_norm(x),
            attention_metadata,
            positions,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask=padding_mask,
        )
        # hc_post reads the attention output (wo_b projection) with bare ops.
        remat.recompute_needs_tensor(x)
        x = self.hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = self.hc_ffn_pre(x)
        if isinstance(self.feed_forward, MoE):
            ffn_input = self.ffn_norm(x)
            if getattr(self.feed_forward.router, "hash", False):
                x = self.feed_forward(
                    ffn_input,
                    padding_mask_T=padding_mask,
                    aux_loss_denominator=aux_loss_denominator,
                    input_ids_T=input_ids_T,
                )
            else:
                x = self.feed_forward(
                    ffn_input,
                    padding_mask_T=padding_mask,
                    aux_loss_denominator=aux_loss_denominator,
                )
        else:
            x = self.feed_forward(self.ffn_norm(x))
        # hc_post reads the MoE / feed-forward output with bare ops.
        remat.recompute_needs_tensor(x)
        x = self.hc_post(x, residual, post, comb)
        return x


class DeepSeekV4Model(MTPDecoder):
    state_dict_adapter_cls = DeepSeekV4StateDictAdapter

    """DeepSeek V4 decoder model with HC branches and sparse attention."""

    mtp_layers: "ModuleList[DeepSeekV4MTPBlock]"  # pyrefly: ignore [bad-override]

    pipeline_last_stage_module_fqns = ("hc_head",)

    def pipeline(self, **kwargs):
        """Partition the decoder into pipeline stages."""
        pp_mesh = kwargs["parallelism_context"].get_mesh("pp")
        if torch.distributed.get_backend(pp_mesh.get_group("pp")) == "fake":
            # TODO: the static stage metadata built for fake process groups
            # describes a [T, dim] hidden state, but V4 stages exchange
            # [T, hc_mult, dim].
            raise ValueError(
                "DeepSeek V4 does not support pipeline parallelism on a fake "
                "process group."
            )
        return super().pipeline(**kwargs)

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)

    @dataclass(kw_only=True, slots=True)
    class Config(MTPDecoder.Config):
        dim: int
        vocab_size: int
        # pyrefly: ignore [bad-override]
        layers: list[DeepSeekV4TransformerBlock.Config]
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=[
                    "loss",
                    "fused_binary_activation",
                    "fp32_to_bf16_split",
                    "rope",
                ]
            )
        )
        hc_mult: int = 4
        n_mtp_layers: int = 0
        compress_ratios: tuple[int, ...] = (1, 1, 4, 4)
        n_layers: int = 4
        norm_eps: float = 1e-6
        hc_head: HcHead.Config
        # pyrefly: ignore [bad-override]
        mtp_layers: list["DeepSeekV4MTPBlock.Config"] = field(default_factory=list)

        def build_flops_estimator(
            self,
            model: Module,
            *,
            seq_len: int,
            excluded_submodules: tuple[Module | None, ...] = (),
        ) -> FlopsEstimator:
            mtp_decoder_estimate = super(
                DeepSeekV4Model.Config, self
            ).build_flops_estimator(
                model, seq_len=seq_len, excluded_submodules=excluded_submodules
            )
            model = cast(DeepSeekV4Model, model)
            # MTPDecoder counts the repeated LM head; each MTP depth also runs
            # h_proj on hc_mult branches, which the parameter count sees once.
            repeated_h_proj_flops_per_token = (
                6
                * (self.hc_mult - 1)
                * sum(
                    p.numel()
                    for layer in model.mtp_layers
                    for p in layer.h_proj.parameters()
                )
            )
            return lambda input_dict: (
                mtp_decoder_estimate(input_dict)
                + repeated_h_proj_flops_per_token
                * int((~input_dict["padding_mask"]).sum())
            )

    def __init__(self, config: Config):
        from torchtitan.distributed.spmd_types import spmd_mesh_size

        tp = spmd_mesh_size("tp")
        if tp > 1:
            for layer in config.layers[: config.n_layers]:
                num_heads = layer.attention.n_heads
                if num_heads % tp != 0:
                    raise ValueError(
                        f"n_heads ({num_heads}) must be divisible by tp ({tp})"
                    )
                num_groups = layer.attention.n_groups
                if num_groups % tp != 0:
                    raise ValueError(
                        f"n_groups ({num_groups}) must be divisible by tp ({tp})"
                    )
        if spmd_mesh_size("cp") > 1:
            raise NotImplementedError(
                "Context Parallel is not yet supported for DeepSeek V4 sparse attention."
            )
        super().__init__(config)
        cfg = config

        self.hc_mult = cfg.hc_mult
        self.n_mtp_layers = cfg.n_mtp_layers
        self.compress_ratios = list(cfg.compress_ratios)[: cfg.n_layers]

        self.hc_head = cfg.hc_head.build()

    def preprocess_inputs(self, input_dict: dict[str, Any], **kwargs: Any):
        """Build the MTPDecoder batch and pass token IDs on for hash routing."""
        inputs, labels, model_kwargs, loss_kwargs = super().preprocess_inputs(
            input_dict, **kwargs
        )
        # `inputs` holds the token IDs on every rank here. Under PP, later
        # stages' forward receives the previous stage's hidden state instead,
        # but model kwargs reach every stage unchanged.
        model_kwargs["input_ids_T"] = inputs[0] if isinstance(inputs, tuple) else inputs
        return inputs, labels, model_kwargs, loss_kwargs

    def forward(
        self,
        tokens: torch.Tensor | tuple[torch.Tensor, ...],
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
        *,
        input_ids_T: torch.Tensor | None = None,
        mtp_input_valid_masks: tuple[torch.Tensor, ...] | None = None,
    ):
        """Run the DeepSeek V4 decoder.

        On a later PP stage, ``tokens`` is the previous stage's ``[T, M, D]``
        hidden state and ``input_ids_T`` carries the token IDs. With MTP layers,
        ``tokens`` holds the main tokens and one prepared token tensor per MTP
        depth, and ``mtp_input_valid_masks`` one validity mask per depth (see
        ``preprocess_inputs``).
        """
        if attention_metadata is None:
            raise ValueError("DeepSeek V4 requires attention metadata.")
        mtp_input_tokens: tuple[torch.Tensor, ...] = ()
        if len(self.mtp_layers) > 0:
            if self.tok_embeddings is None:
                raise ValueError("DeepSeek V4 MTP forward requires token embeddings.")
            if self._skip_lm_head:
                raise ValueError(
                    "DeepSeek V4 MTP cannot skip the LM head because chunked "
                    "cross entropy is not supported."
                )
            if (
                not isinstance(tokens, tuple)
                or len(tokens) != len(self.mtp_layers) + 1
                or mtp_input_valid_masks is None
                or len(mtp_input_valid_masks) != len(self.mtp_layers)
            ):
                raise ValueError(
                    "DeepSeek V4 MTP forward requires the main tokens plus one "
                    "prepared token tensor and one validity mask per MTP layer."
                )
            tokens, *rest = tokens
            mtp_input_tokens = tuple(rest)
        else:
            tokens = cast(torch.Tensor, tokens)

        if self.tok_embeddings is not None:
            if input_ids_T is None:
                input_ids_T = tokens.detach().long()
            h_TMD = self.tok_embeddings(tokens).unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            h_TMD = tokens

        with spmd.no_typecheck():
            main_aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        for layer in self.layers.values():
            layer_attention_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.get(type(layer.attention.inner_attention))
            )
            h_TMD = layer(
                h_TMD,
                input_ids_T,
                layer_attention_metadata,
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=main_aux_loss_denominator,
            )

        # Only the last PP stage owns hc_head and norm.
        if self.norm is None:
            return h_TMD
        h_TD = self._pre_lm_head_norm_and_gather(self.hc_head(h_TMD))

        if len(self.mtp_layers) == 0:
            if self._skip_lm_head or self.lm_head is None:
                return h_TD
            return self.lm_head(h_TD)

        assert mtp_input_valid_masks is not None
        outputs = [h_TD] + self.mtp_forward(
            h_TMD,
            mtp_input_tokens,
            mtp_input_valid_masks,
            attention_metadata,
            positions,
            padding_mask,
            aux_loss_denominators,
        )
        return tuple(
            self.lm_head(item) if self.lm_head is not None else item for item in outputs
        )

    def mtp_forward(
        self,
        prev_hc_hidden: torch.Tensor,
        mtp_input_tokens: tuple[torch.Tensor, ...],
        mtp_input_valid_masks: tuple[torch.Tensor, ...],
        attention_metadata: AttentionMetadataMap | None = None,
        positions: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Run auxiliary MTP depths on their prepared tokens and return
        prediction hidden states."""
        mtp_outputs = []
        for depth, (mtp_block, mtp_tokens, valid_mask) in enumerate(
            zip(self.mtp_layers, mtp_input_tokens, mtp_input_valid_masks, strict=True),
            1,
        ):
            layer_attention_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.get(type(mtp_block.attention.inner_attention))
            )
            with spmd.no_typecheck():
                aux_loss_denominator = (
                    None
                    if aux_loss_denominators is None
                    else aux_loss_denominators[depth]
                )
            prev_hc_hidden, prediction_hidden = mtp_block(
                self.tok_embeddings(mtp_tokens),
                prev_hc_hidden,
                mtp_tokens.detach().long(),
                valid_mask,
                layer_attention_metadata,
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
            mtp_outputs.append(prediction_hidden)
        return mtp_outputs
