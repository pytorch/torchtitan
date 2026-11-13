# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffixes: T = tokens, M = HC branches (hc_mult), D = hidden width.

from dataclasses import dataclass, field
from math import lcm
from typing import Any, cast, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.base.model import FlopsEstimator
from torchtitan.models.base.module import Module, ModuleList
from torchtitan.models.common.attention import (
    AttentionMetadataMap,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3.mtp import (
    apply_fsdp_to_mtp_decoder,
    roll_mtp_sequence,
)

from .mhc import HcHead, HcPost, HcPre
from .state_dict_adapter import DeepSeekV4StateDictAdapter

if TYPE_CHECKING:
    from .attention import DeepSeekV4Attention
    from .mtp import MTPBlock


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
        attention_metadata: VarlenAttentionMetadata | None,
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


class DeepSeekV4Model(Decoder):
    state_dict_adapter_cls = DeepSeekV4StateDictAdapter

    """DeepSeek V4 decoder model with HC branches and sparse attention."""

    mtp_layers: "ModuleList[MTPBlock]"

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

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        precision: PrecisionConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torchtitan.distributed.fsdp import (
            resolve_fsdp_mesh,
            resolve_sparse_fsdp_mesh,
        )

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)
        apply_fsdp_to_mtp_decoder(
            # DeepSeek V4 has the decoder and MTP layer structure required by
            # this helper, but uses its own MTP implementation.
            self,  # pyrefly: ignore [bad-argument-type]
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[precision.param_dtype],
            reduce_dtype=TORCH_DTYPE_MAP[precision.reduce_dtype],
            pp_enabled=parallelism_context.pp_enabled,
            cpu_offload=parallelism.fsdp_enable_cpu_offloading,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallelism_context.ep,
            edp_mesh=edp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh_dims=edp_mesh_dims,
            symm_mem_scope=parallelism.fsdp_symm_mem_scope,
            unshard_in_current_stream=parallelism.fsdp_unshard_in_current_stream,
        )

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
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
        mtp_layers: list["MTPBlock.Config"] | None = None

        @property
        def per_sample_pad_multiple(self) -> int | None:
            """Multiple every packed document segment must have, if any.

            Compression groups are ``compress_ratio`` tokens wide, so a document
            segment has to end on a boundary every variant respects. None means
            no layer compresses and packing needs no alignment.
            """
            alignment = lcm(
                *(
                    max(1, layer.attention.compress_ratio)
                    for layer in (*self.layers, *(self.mtp_layers or ()))
                )
            )
            return None if alignment == 1 else alignment

        def __post_init__(self) -> None:
            Decoder.Config.__post_init__(self)
            alignment = self.per_sample_pad_multiple
            if alignment is not None and self.max_context_length % alignment != 0:
                raise ValueError(
                    f"document alignment ({alignment}) must divide "
                    f"max_context_length ({self.max_context_length})."
                )

        def build_flops_estimator(
            self,
            model: Module,
            *,
            seq_len: int,
            excluded_submodules: tuple[Module | None, ...] = (),
        ) -> FlopsEstimator:
            decoder_estimate = super(
                DeepSeekV4Model.Config, self
            ).build_flops_estimator(
                model, seq_len=seq_len, excluded_submodules=excluded_submodules
            )
            model = cast(DeepSeekV4Model, model)
            repeated_mtp_flops_per_token = 6 * (
                len(model.mtp_layers)
                * sum(p.numel() for p in model.lm_head.parameters())
                + (self.hc_mult - 1)
                * sum(
                    p.numel()
                    for layer in model.mtp_layers
                    for p in layer.h_proj.parameters()
                )
            )
            return lambda input_dict: (
                decoder_estimate(input_dict)
                + repeated_mtp_flops_per_token * input_dict["input"].numel()
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
        self.mtp_layers = ModuleList()
        if cfg.mtp_layers is not None:
            self.mtp_layers = ModuleList(
                mtp_layer.build() for mtp_layer in cfg.mtp_layers
            )

    def preprocess_inputs(self, input_dict: dict[str, Any], **kwargs: Any):
        """Build the decoder batch and pass token IDs on for hash routing."""
        inputs, labels, model_kwargs, loss_kwargs = super().preprocess_inputs(
            input_dict, **kwargs
        )
        # `inputs` holds the token IDs on every rank here. Under PP, later
        # stages' forward receives the previous stage's hidden state instead,
        # but model kwargs reach every stage unchanged.
        model_kwargs["input_ids_T"] = inputs
        return inputs, labels, model_kwargs, loss_kwargs

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
        *,
        input_ids_T: torch.Tensor | None = None,
    ):
        """Run the DeepSeek V4 decoder.

        On a later PP stage, ``tokens`` is the previous stage's ``[T, M, D]``
        hidden state and ``input_ids_T`` carries the token IDs.
        """
        if len(self.mtp_layers) > 0 and self.tok_embeddings is None:
            raise ValueError("DeepSeek V4 MTP forward requires token embeddings.")
        if len(self.mtp_layers) > 0 and self._skip_lm_head:
            raise ValueError(
                "DeepSeek V4 MTP cannot skip the LM head because chunked "
                "cross entropy is not supported."
            )

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
        h_TD = self.norm(self.hc_head(h_TMD))

        if len(self.mtp_layers) == 0:
            if self._skip_lm_head or self.lm_head is None:
                return h_TD
            return self.lm_head(h_TD)

        outputs = [h_TD] + self.mtp_forward(
            h_TMD,
            tokens,
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
        tokens: torch.Tensor,
        attention_metadata: AttentionMetadataMap | None = None,
        positions: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Run auxiliary MTP depths and return prediction hidden states."""
        mtp_outputs = []
        for depth, mtp_block in enumerate(self.mtp_layers, 1):
            mtp_tokens, valid_mask = roll_mtp_sequence(
                tokens,
                shift=depth,
                fill_value=0,
                positions=positions,
                padding_mask=padding_mask,
                return_valid_mask=True,
            )
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
