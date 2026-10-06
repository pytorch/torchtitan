# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import spmd_types as spmd
import torch
import torch_remat as remat
from spmd_types import SpmdType
from torch import nn

from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.local_compile import local_compile
from torchtitan.distributed.parallelism_context import MeshAxisName, ParallelismContext
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    spmd_local_context,
)
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import (
    AttentionMetadataMap,
    BaseAttention,
    FlexAttentionMetadata,
    local_head_split,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.linear import maybe_gather_tp_input
from torchtitan.models.common.multimodal import (
    add_zero_vision_dependency,
    build_dummy_vision_inputs,
    get_vision_positions,
    MultimodalModel,
    scatter_vision_embeds,
)
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding
from torchtitan.models.utils import (
    delta_rule_flops_per_token,
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module

from .gdn import GatedDeltaNet, InnerGatedDeltaNet
from .rope import MRoPE
from .state_dict_adapter import Qwen35StateDictAdapter
from .vision_encoder import Qwen35VisionEncoder

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels,
# H = attention heads,
# K = query/key head dimension, V = value head dimension,
# R = rotary dimension, P = non-rotary dimension.


class OffsetRMSNorm(Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``.

    Weight is zero-initialized so the norm starts as identity-scaled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    @local_compile("offset_rmsnorm", batch_invariant=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability in pow/rsqrt
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)


class Qwen35Attention(BaseAttention):
    """Full attention with output gating and partial RoPE for Qwen3.5.

    Differences from GQAttention:
    - wq is 2x wider: produces both query and sigmoid gate
    - Partial RoPE: only first ``rotary_dim`` elements get RoPE
    - Output gating: ``attn_output * sigmoid(gate)`` before ``wo``
    - QK norm uses OffsetRMSNorm

    Uses separate ``wq``/``wk``/``wv`` instead of the common fused ``qkv_linear``
    (so this subclasses ``BaseAttention``, not ``GQAttention``): the 2x-wide,
    gated ``wq`` doesn't fit a fused QKV projection that TP-shards by head.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        n_kv_heads: int
        head_dim: int
        rotary_dim: int
        rope: MRoPE.Config
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        wo: Linear.Config
        q_norm: OffsetRMSNorm.Config
        k_norm: OffsetRMSNorm.Config
        inner_attention: Module.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.wo = config.wo.build()

        self.rope = config.rope.build()

        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()

        self.scaling = self.head_dim**-0.5

        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The query, key, and value projections all consume x. Gather once
        # at their common attention boundary.
        x_TD = maybe_gather_tp_input(self, x_TD)

        num_tokens = x_TD.shape[0]

        # wq is 2x wider: produces query + gate
        xq_gate_THC = local_head_split(self.wq(x_TD), self.head_dim * 2)
        xq_THK, gate_THV = xq_gate_THC.chunk(2, dim=-1)
        xk_THK = local_head_split(self.wk(x_TD), self.head_dim)
        xv_THV = local_head_split(self.wv(x_TD), self.head_dim)

        # QK norm (before RoPE). The norms read the wq and wk projection outputs
        # with bare ops.
        remat.recompute_needs_tensor(xq_THK, xk_THK)
        xq_THK = self.q_norm(xq_THK)
        xk_THK = self.k_norm(xk_THK)

        xq_THK, xk_THK = self._partial_rope(xq_THK, xk_THK, positions)

        out_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            xq_THK,
            xk_THK,
            xv_THV,
            attention_metadata=attention_metadata,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        )
        # The output gating reads the inner_attention and wq gate outputs with bare
        # ops.
        remat.recompute_needs_tensor(out_THV, gate_THV)
        out_THV = out_THV.contiguous()

        # Output gating
        out_THV = out_THV * torch.sigmoid(gate_THV)
        out_TD = out_THV.view(num_tokens, -1)
        return self.wo(out_TD)

    # TODO: consider moving this to rope directly.
    @local_compile("partial_rope", batch_invariant=True)
    def _partial_rope(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to the first ``rotary_dim`` channels of each head; keep the rest."""
        assert self.rotary_dim <= self.head_dim
        xq_THR, xq_THP = (
            xq_THK[..., : self.rotary_dim],
            xq_THK[..., self.rotary_dim :],
        )
        xk_THR, xk_THP = (
            xk_THK[..., : self.rotary_dim],
            xk_THK[..., self.rotary_dim :],
        )
        xq_THR, xk_THR = self.rope(xq_THR, xk_THR, positions)
        xq_THK = torch.cat([xq_THR, xq_THP], dim=-1)
        xk_THK = torch.cat([xk_THR, xk_THP], dim=-1)
        return xq_THK, xk_THK


class Qwen35TransformerBlock(Module):
    """Hybrid transformer block for Qwen3.5.

    Each layer uses either full attention (Qwen35Attention) or linear
    attention (GatedDeltaNet), determined by which config is provided.
    Both types share the same FFN/MoE structure.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: Qwen35Attention.Config | None = None
        delta_net: GatedDeltaNet.Config | None = None
        feed_forward: Module.Config | None = None
        moe: Module.Config | None = None
        attention_norm: OffsetRMSNorm.Config
        ffn_norm: OffsetRMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.full_attn = config.attention is not None

        if self.full_attn:
            self.attn = config.attention.build()  # pyrefly: ignore [missing-attribute]
            self.attention_metadata_key = type(self.attn.inner_attention)
        else:
            assert config.delta_net is not None
            self.attn = config.delta_net.build()
            self.attention_metadata_key = type(self.attn.inner_gated_delta_net)

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            # pyrefly: ignore [missing-attribute]
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_TD = self.attention_norm(x_TD)
        if self.full_attn:
            h_TD = self.attn(h_TD, attention_metadata, positions)
        else:
            h_TD = self.attn(h_TD, attention_metadata)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(h_TD)
        x_TD = x_TD + h_TD

        h_TD = self.ffn_norm(x_TD)
        if self.moe_enabled:
            h_TD = self.moe(h_TD, padding_mask_T=padding_mask)
        else:
            h_TD = self.feed_forward(h_TD)
        # The residual add reads the MoE / feed-forward output with bare ops.
        remat.recompute_needs_tensor(h_TD)
        return x_TD + h_TD


class Qwen35Model(MultimodalModel):
    state_dict_adapter_cls = Qwen35StateDictAdapter
    multimodal_encoder_fqns = ("vision_encoder",)

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)

    pipeline_first_stage_module_fqns = ("vision_encoder",)

    """Qwen3.5: Multimodal model with hybrid attention.

    Combines a hybrid decoder (GatedDeltaNet linear attention + full
    attention with output gating and partial RoPE) with a Vision
    Transformer encoder for multimodal understanding.

    Key architectural features:
    - Hybrid attention: 75% GatedDeltaNet (linear) + 25% full attention
    - Output gating on full attention: ``attn_out * sigmoid(gate)``
    - Partial RoPE: only first ``rotary_dim`` elements get positional encoding
    - OffsetRMSNorm: ``(1 + weight) * norm(x)`` with zero-init weight
    - MRoPE: 3D (temporal/height/width) position IDs for multimodal batches;
      text batches use the plain 1D positions
    - MoE variant: routed experts + shared expert with sigmoid gate

    MRoPE positions (shape ``(num_tokens, 3)``) are built by the dataloader as
    ``mrope_positions``. After building the attention masks from the 2D
    ``positions``, ``preprocess_inputs`` picks which one the RoPE layers see:
    ``mrope_positions`` when present (multimodal), else the 2D ``positions`` --
    the chosen tensor overwrites the single ``positions`` input. This keeps RoPE
    consistent across every pipeline stage even though the raw vision inputs
    (``pixel_values``/``grid_thw``) only reach the first stage. The per-layer
    MRoPE dispatches on the position rank.

    Forward pass flow::

        forward(tokens, pixel_values, grid_thw, positions, ...)
          │
          ├─ _prepare_multimodal_embeds
          │    ├─ tok_embeddings(tokens)              → text embeddings
          │    ├─ _get_vision_embeds(pixel_values)     → vision embeddings
          │    │    └─ vision_encoder(pixel_values)     → merge patches
          │    ├─ get_vision_positions              → locate vision regions
          │    └─ _scatter_vision_embeds                → scatter into text sequence
          │
          └─ transformer layers (hybrid), each given ``positions`` (3D or 2D)
               └─ for each layer:
                    ├─ full attention (every Nth):  QK-norm → partial RoPE → SDPA → gate
                    │    (the layer's MRoPE builds the cos/sin cache from positions)
                    └─ GatedDeltaNet (others):      Conv1d → gated delta rule → gated norm
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        vision_encoder: Qwen35VisionEncoder.Config | None = None
        local_compile_regions: list[str] = field(
            default_factory=lambda: [
                "loss",
                "swiglu",
                "gated_rmsnorm",
                "offset_rmsnorm",
                "partial_rope",
                "shared_expert_gate",
            ]
        )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            # The vision encoder cost scales with patches rather than text
            # sequence length, so this remains a decoder-only MFU estimate.
            qwen_model = cast("Qwen35Model", model)
            nparams, active_nparams = get_nparams_and_active_nparams(
                model,
                modules_excluded_from_active_params=(qwen_model.vision_encoder,),
            )
            attention_op_flops = 0
            for layer in self.layers:
                if isinstance(layer.attention, Qwen35Attention.Config):
                    attention = layer.attention
                    attention_op_flops += quadratic_attention_flops_per_token(
                        num_heads=attention.n_heads,
                        qk_head_dim=attention.head_dim,
                        v_head_dim=attention.head_dim,
                        seq_len=seq_len,
                    )
                elif isinstance(layer.delta_net, GatedDeltaNet.Config):
                    delta_net = layer.delta_net
                    num_value_heads = (
                        delta_net.in_proj_v.out_features // delta_net.value_head_dim
                    )
                    attention_op_flops += delta_rule_flops_per_token(
                        num_heads=num_value_heads,
                        key_head_dim=delta_net.key_head_dim,
                        v_head_dim=delta_net.value_head_dim,
                    )
            return nparams, 6 * active_nparams + attention_op_flops

        def set_sharding_(self, parallelism: ParallelismConfig) -> None:
            from .sharding import set_qwen35_sharding_config

            set_qwen35_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

    def __init__(self, config: Config):
        from torchtitan.distributed.spmd_types import spmd_mesh_size

        tp = spmd_mesh_size("tp")
        if tp > 1:
            delta_net = next(
                (layer.delta_net for layer in config.layers if layer.delta_net),
                None,
            )
            if delta_net is not None:
                num_key_heads = (
                    delta_net.in_proj_q.out_features // delta_net.key_head_dim
                )
                num_value_heads = (
                    delta_net.in_proj_v.out_features // delta_net.value_head_dim
                )
                if num_key_heads % tp != 0 or num_value_heads % tp != 0:
                    raise ValueError(
                        f"tensor parallel degree ({tp}) must divide "
                        f"num_key_heads ({num_key_heads}) and "
                        f"num_value_heads ({num_value_heads})."
                    )
        super().__init__(config)

        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )
        self.spatial_merge_size = (
            config.vision_encoder.spatial_merge_size
            if config.vision_encoder is not None
            else None
        )

    def parallelize(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        local_compile_regions: list[str],
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
        skip_dp: bool = False,
    ) -> Qwen35Model:
        if parallelism_context.cp_enabled:
            raise NotImplementedError(
                "Context Parallel is not yet supported for Qwen3.5. "
                "GatedDeltaNet requires full-sequence allgather, and multimodal "
                "CP needs vision scatter before CP sharding."
            )

        return super().parallelize(
            parallelism_context=parallelism_context,
            training=training,
            parallelism=parallelism,
            local_compile_regions=local_compile_regions,
            ac_config=ac_config,
            dump_folder=dump_folder,
            skip_dp=skip_dp,
        )

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context: ParallelismContext,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build masks, CP-shard, SPMD-wrap (+ deltanet annotation), and return."""
        del kwargs
        padding_mask = input_dict.get("padding_mask", None)

        # Attention masks are built from the 1D ``positions``.
        positions = input_dict.get("positions")
        if positions is not None:
            input_dict["attention_metadata"] = self._get_attention_metadata(
                positions=positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )

        input_shardings = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(),
        }

        # RoPE uses the 3D MRoPE positions when present (multimodal), else the
        # same 2D positions. Collapse both into the single ``positions`` input.
        mrope_positions = input_dict.pop("mrope_positions", None)
        if mrope_positions is None:
            rope_positions = positions
        else:
            rope_positions = mrope_positions
            # MRoPE positions fold to ``(tokens, 3)`` (2D); replicate the
            # trailing component axis instead of the 1D token layout.
            input_shardings["positions"] = SpmdType(
                {
                    MeshAxisName.DP: spmd.V,
                    MeshAxisName.CP: spmd.V,
                    MeshAxisName.TP: spmd.R,
                },
                partition_spec=spmd.PartitionSpec(
                    (MeshAxisName.DP, MeshAxisName.CP), None
                ),
            )
        assert rope_positions is not None, (
            "Qwen3.5 needs RoPE positions: the batch must provide "
            "'positions' or 'mrope_positions'."
        )
        input_dict["positions"] = rope_positions
        if parallelism_context.cp_enabled:
            input_dict = self._cp_shard(
                input_dict,
                input_shardings=input_shardings,
                parallelism_context=parallelism_context,
                parallelism=parallelism,
            )
        input_dict = annotate_input_spmd_types(
            parallelism_context, input_dict, input_shardings
        )
        # Plain-tensor inputs are typed above; the nested GatedDeltaNet cu_seq_q
        # must be annotated separately.
        attention_metadata = input_dict.get("attention_metadata")
        deltanet_metadata = (
            attention_metadata.get(InnerGatedDeltaNet)
            if attention_metadata is not None
            else None
        )
        if isinstance(deltanet_metadata, VarlenAttentionMetadata):
            deltanet_metadata.annotate_spmd_types()

        inputs = input_dict.pop("input")
        labels = input_dict.pop("labels")
        return inputs, labels, input_dict

    def _get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the vision encoder and return packed embeddings with token counts.

        Args:
            pixel_values: Packed patches ``(total_num_patches, patch_dim)``.
            grid_thw: Grid dimensions (num_items, 3) for [t, h, w]

        Returns:
            vision_embeds: Packed vision embeddings ``(total_tokens, dim)``.
            num_tokens_per_item: (num_items,) actual token count per item
        """
        if self.vision_encoder is None:
            raise ValueError("Vision inputs were provided without a vision encoder.")
        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)

        merge_unit = self.vision_encoder.spatial_merge_unit
        num_tokens_per_item = grid_thw.prod(-1) // merge_unit

        return vision_embeds, num_tokens_per_item

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
    ) -> torch.Tensor:
        """Embed tokens, run vision encoder, scatter vision into text.

        Args:
            tokens: Input token IDs ``(num_tokens,)``.
            pixel_values: Image patches or None
            pixel_values_videos: Video patches or None
            grid_thw: Grid dimensions for images or None
            grid_thw_videos: Grid dimensions for videos or None
            special_tokens: Special token definitions

        Returns:
            ``(num_tokens, dim)`` embeddings with vision tokens scattered in.
        """
        inputs_embeds = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )
        if self.vision_encoder is None:
            return inputs_embeds

        # TODO: Configure the job-wide modality set from the dataset so
        # single-modality jobs can avoid the second encoder call.
        image_is_dummy = pixel_values is None or grid_thw is None
        if image_is_dummy:
            grid_size = self.vision_encoder.spatial_merge_size
            pixel_values, grid_thw = build_dummy_vision_inputs(
                patch_dim=self.vision_encoder.patch_embed.in_features,
                grid_thw=(1, grid_size, grid_size),
                device=inputs_embeds.device,
            )
        vision_embeds, num_tokens = self._get_vision_embeds(
            pixel_values, grid_thw=grid_thw
        )
        if image_is_dummy:
            inputs_embeds = add_zero_vision_dependency(inputs_embeds, vision_embeds)
        else:
            if special_tokens is None:
                raise ValueError("special_tokens is required for image inputs")
            image_positions = get_vision_positions(
                tokens, num_tokens, special_tokens["image_id"]
            )
            if image_positions:
                inputs_embeds = scatter_vision_embeds(
                    inputs_embeds,
                    vision_embeds=vision_embeds,
                    vision_positions=image_positions,
                )

        video_is_dummy = pixel_values_videos is None or grid_thw_videos is None
        if video_is_dummy:
            grid_size = self.vision_encoder.spatial_merge_size
            pixel_values_videos, grid_thw_videos = build_dummy_vision_inputs(
                patch_dim=self.vision_encoder.patch_embed.in_features,
                grid_thw=(1, grid_size, grid_size),
                device=inputs_embeds.device,
            )
        vision_embeds, num_tokens = self._get_vision_embeds(
            pixel_values_videos, grid_thw=grid_thw_videos
        )
        if video_is_dummy:
            inputs_embeds = add_zero_vision_dependency(inputs_embeds, vision_embeds)
        else:
            if special_tokens is None:
                raise ValueError("special_tokens is required for video inputs")
            video_positions = get_vision_positions(
                tokens, num_tokens, special_tokens["video_id"]
            )
            if video_positions:
                inputs_embeds = scatter_vision_embeds(
                    inputs_embeds,
                    vision_embeds=vision_embeds,
                    vision_positions=video_positions,
                )

        return inputs_embeds

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        positions: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
    ):
        with spmd_local_context("dp"):
            if self.tok_embeddings is not None:
                x = self._prepare_multimodal_embeds(
                    tokens,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    grid_thw=grid_thw,
                    grid_thw_videos=grid_thw_videos,
                    special_tokens=special_tokens,
                )
            else:
                x = tokens

        if spmd.is_type_checking():
            spmd.assert_type(
                x,
                {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
                spmd.PartitionSpec(("dp", "cp"), None),
            )

        # ``positions`` is 3D MRoPE (batch, seq, 3) for multimodal batches and
        # 2D (batch, seq) for text; ``preprocess_inputs`` resolved which one to
        # forward. The per-layer MRoPE dispatches on rank.
        for layer in self.layers.values():
            x = layer(
                x,
                (
                    attention_metadata.get(
                        cast(Qwen35TransformerBlock, layer).attention_metadata_key
                    )
                    if attention_metadata is not None
                    else None
                ),
                positions,
                padding_mask=padding_mask,
            )

        x = self.norm(x) if self.norm is not None else x
        if self._skip_lm_head:
            return x
        return self.lm_head(x) if self.lm_head is not None else x
