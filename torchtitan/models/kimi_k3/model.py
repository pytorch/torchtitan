# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, cast

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    current_spmd_mesh,
)
from torchtitan.distributed.utils import get_spmd_backend

from torchtitan.hf_datasets.multimodal.mm_datasets import MMSamplePackingConfig
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    FlexAttention,
    VarlenAttention,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.multimodal import (
    get_vision_positions,
    scatter_vision_embeds,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding
from torchtitan.models.kimi_k3.sharding import (
    set_kimi_k3_sharding_config,
    set_tensor_parallel_sharding_config,
)
from torchtitan.models.utils import (
    delta_rule_flops_per_token,
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module

from .kda import KDA
from .moe import KimiFeedForward, KimiLatentMoE
from .vision_encoder import KimiK3VisionEncoder

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels, H = heads,
# K = query/key head dimension, V = value head dimension,
# N = attention-residual entries.


def _local_head_split(
    x_TE: torch.Tensor, num_tokens: int, heads: int, head_dim: int
) -> torch.Tensor:
    """Unflatten a TP-sharded ``[T, H*K]`` projection into ``[T, H, K]``.

    spmd_types cannot propagate a shard on the feature dim through a split of
    that dim, so the view runs in a local region and the result is re-typed as
    head-sharded on TP, the same shape core's ``local_qkv_head_split`` uses.
    """
    with spmd.local():
        x_TNH = x_TE.view(num_tokens, heads, head_dim)
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            spmd.assert_type(
                x_TNH, spmd.V, spmd.PartitionSpec(("dp", "cp"), "tp", None)
            )
    return x_TNH


class KimiMLAAttention(BaseAttention):
    """Kimi K3 multi-head latent attention.

    Unlike DeepSeek-V3 MLA, the released K3 configuration sets
    ``mla_use_nope=True``: the RoPE-sized query/key slices remain part of the
    projected head, but no rotary transform is applied, so this has no rope
    config at all. Attention delegates to the configured inner backend.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv_a: Linear.Config
        kv_norm: RMSNorm.Config
        wkv_b: Linear.Config
        gate: Linear.Config
        wo: Linear.Config
        inner_attention: Module.Config = field(default_factory=FlexAttention.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.scale = self.q_head_dim**-0.5

        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.gate = config.gate.build()
        self.wo = config.wo.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions

        num_tokens = x_TD.shape[0]
        # Head count derived from the projection width: under TP the colwise
        # projections hold n_heads/tp per rank, and under spmd_types they hand
        # back that local slice.
        q_proj_TE = self.wq_b(self.q_norm(self.wq_a(x_TD)))
        h_local = q_proj_TE.shape[-1] // self.q_head_dim
        q_THK = _local_head_split(q_proj_TE, num_tokens, h_local, self.q_head_dim)

        compressed_kv_TC = self.wkv_a(x_TD)
        kv_latent_TC, k_rope_TK = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_THC = _local_head_split(
            self.wkv_b(self.kv_norm(kv_latent_TC)),
            num_tokens,
            h_local,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope_THK, v_THV = torch.split(
            kv_THC,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        # The rotary slice is headless and replicated; expanding it onto the
        # local heads and joining it to the head-sharded nope part is a
        # local-region op, re-typed as head-sharded on TP afterwards.
        with spmd.local():
            k_rope_THK = k_rope_TK.view(num_tokens, 1, self.qk_rope_head_dim).expand(
                -1, h_local, -1
            )
            k_THK = torch.cat((k_nope_THK, k_rope_THK), dim=-1)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                spmd.assert_type(
                    k_THK, spmd.V, spmd.PartitionSpec(("dp", "cp"), "tp", None)
                )

        out_THV = self.inner_attention(
            q_THK,
            k_THK,
            v_THV,
            attention_masks=attention_masks,
            scale=self.scale,
        )
        out_TD = out_THV.reshape(num_tokens, h_local * self.v_head_dim)
        out_TD = out_TD * torch.sigmoid(self.gate(x_TD))
        return self.wo(out_TD)


def _apply_attention_residual(
    prefix_sum_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection: Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    """Apply Kimi's block-level attention residual in FP32.

    TODO: Add TP Support. The current implementation assumes that the input tensors are on a single device.
    """
    assert norm.eps is not None

    values_TND = torch.cat((block_residual_TND, prefix_sum_TD.unsqueeze(1)), dim=1)
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + norm.eps)
    score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    output_TD = torch.matmul(probs_T1N, values_float).squeeze(1)
    return output_TD.to(values_TND.dtype)


class KimiK3TransformerBlock(Module):
    """Hybrid KDA/MLA decoder block with Kimi attention residuals."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_id: int
        attn_res_block_size: int
        attention: KimiMLAAttention.Config | None
        delta_attention: KDA.Config | None
        feed_forward: KimiFeedForward.Config | None
        moe: KimiLatentMoE.Config | None
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config
        attention_res_norm: RMSNorm.Config | None
        attention_res_proj: Linear.Config | None
        ffn_res_norm: RMSNorm.Config
        ffn_res_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        if (config.attention is None) == (config.delta_attention is None):
            raise ValueError(
                "Exactly one of attention or delta_attention must be configured."
            )
        if (config.feed_forward is None) == (config.moe is None):
            raise ValueError("Exactly one of feed_forward or moe must be configured.")
        self.layer_id = config.layer_id
        self.attn_res_block_size = config.attn_res_block_size
        self.attention = (
            config.attention.build() if config.attention is not None else None
        )
        self.delta_attention = (
            config.delta_attention.build()
            if config.delta_attention is not None
            else None
        )
        self.feed_forward = (
            config.feed_forward.build() if config.feed_forward is not None else None
        )
        self.moe = config.moe.build() if config.moe is not None else None
        self.moe_enabled = self.moe is not None
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.attention_res_norm = (
            config.attention_res_norm.build()
            if config.attention_res_norm is not None
            else None
        )
        self.attention_res_proj = (
            config.attention_res_proj.build()
            if config.attention_res_proj is not None
            else None
        )
        self.ffn_res_norm = config.ffn_res_norm.build()
        self.ffn_res_proj = config.ffn_res_proj.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        block_residual_TND: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_sum_TD = x_TD

        if block_residual_TND.shape[1] > 0:
            assert self.attention_res_proj is not None
            assert self.attention_res_norm is not None
            x_TD = _apply_attention_residual(
                prefix_sum_TD,
                block_residual_TND,
                self.attention_res_proj,
                self.attention_res_norm,
            )

        opens_block = self.layer_id % self.attn_res_block_size == 0
        if opens_block:
            block_residual_TND = torch.cat(
                (
                    block_residual_TND,
                    prefix_sum_TD.unsqueeze(1),
                ),
                dim=1,
            )

        h_TD = self.attention_norm(x_TD)
        if self.attention is not None:
            h_TD = self.attention(h_TD, attention_masks, positions)
        else:
            assert self.delta_attention is not None
            h_TD = self.delta_attention(h_TD, None, positions)
        prefix_sum_TD = h_TD if opens_block else prefix_sum_TD + h_TD

        h_TD = _apply_attention_residual(
            prefix_sum_TD,
            block_residual_TND,
            self.ffn_res_proj,
            self.ffn_res_norm,
        )
        h_TD = self.ffn_norm(h_TD)
        if self.moe is not None:
            h_TD = self.moe(h_TD)
        else:
            assert self.feed_forward is not None
            h_TD = self.feed_forward(h_TD)
        return prefix_sum_TD + h_TD, block_residual_TND


class KimiK3Model(Decoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        layers: list[KimiK3TransformerBlock.Config]
        output_res_norm: RMSNorm.Config
        output_res_proj: Linear.Config
        vision_encoder: KimiK3VisionEncoder.Config | None = None

        def update_from_config(self, *, config, **kwargs) -> None:
            dataset = config.dataloader.dataset
            # TODO: Support sample packing by resetting the Q/K/V causal-convolution
            # and KDA recurrent states at document boundaries.
            if isinstance(dataset, MMSamplePackingConfig):
                raise ValueError("Kimi K3 does not yet support sample packing.")
            # No tensor parallelism on this branch: the declarations are issued
            # at tp = 1, where spmd_types still consumes every one of them.
            enable_sp = False
            set_kimi_k3_sharding_config(
                self,
                enable_ep=config.parallelism.expert_parallel_degree > 1,
                enable_sp=enable_sp,
            )
            # spmd_types consumes every declaration, so the dense-path
            # declarations are issued whenever it drives the model, not only
            # at tp > 1; partial_dtensor reads only their tp placements.
            spmd_types = config.parallelism.spmd_backend == "spmd_types"
            if spmd_types:
                set_tensor_parallel_sharding_config(
                    self,
                    enable_sp=enable_sp,
                    declare_vision_encoder=spmd_types,
                    spmd_types=spmd_types,
                )
            Decoder.Config.update_from_config(self, config=config, **kwargs)

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            kimi_model = cast("KimiK3Model", model)
            nparams, active_nparams = get_nparams_and_active_nparams(
                model,
                modules_excluded_from_active_params=(kimi_model.vision_encoder,),
            )
            attention_op_flops = 0
            for layer in self.layers:
                if isinstance(layer.attention, KimiMLAAttention.Config):
                    attention = layer.attention
                    attention_op_flops += quadratic_attention_flops_per_token(
                        num_heads=attention.n_heads,
                        qk_head_dim=(
                            attention.qk_nope_head_dim + attention.qk_rope_head_dim
                        ),
                        v_head_dim=attention.v_head_dim,
                        seq_len=seq_len,
                    )
                elif isinstance(layer.delta_attention, KDA.Config):
                    delta_attention = layer.delta_attention
                    attention_op_flops += delta_rule_flops_per_token(
                        num_heads=delta_attention.num_heads,
                        key_head_dim=delta_attention.head_dim,
                        v_head_dim=delta_attention.head_dim,
                    )
            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config):
        super().__init__(config)
        self.output_res_norm = config.output_res_norm.build()
        self.output_res_proj = config.output_res_proj.build()
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )

    def preprocess_inputs(  # pyrefly: ignore [bad-override]
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Decoder preprocessing plus SPMD layouts for the image inputs."""
        # Function-local import avoids a circular import
        # (context_parallel.api -> models.common -> decoder).
        from torchtitan.distributed.context_parallel.api import (
            prepare_context_parallel_input,
        )

        batch: dict[str, Any] = dict(input_dict)
        positions = batch.get("positions", None)
        if positions is not None:
            inner = self.config.first_full_attention_backend
            if isinstance(inner, (FlexAttention.Config, VarlenAttention.Config)):
                batch["attention_masks"] = self.get_attention_masks(positions=positions)

        # pixel_values and grid_thw are DP-local and TP-invariant, the layout
        # every VLM decoder shares; the vision encoder runs per rank.
        input_sharding = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(include_cp_axis=True),
        }
        if parallel_dims.cp_enabled:
            batch = prepare_context_parallel_input(
                batch,
                input_sharding,
                parallel_dims.get_mesh("cp"),
                parallelism.context_parallel_load_balancer,
                parallelism.context_parallel_ptrr_mask_key,
            )
        if parallelism.spmd_backend == "spmd_types":
            batch = annotate_input_spmd_types(parallel_dims, batch, input_sharding)

        inputs = batch.pop("input")
        labels = batch.pop("labels")
        return inputs, labels, batch

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
    ) -> torch.Tensor:
        embeddings_TD = self.tok_embeddings(tokens)
        if (pixel_values is None) != (grid_thw is None):
            raise ValueError(
                "pixel_values and grid_thw must either both be provided or "
                "both be omitted."
            )
        if pixel_values is None:
            return embeddings_TD
        assert grid_thw is not None
        if self.vision_encoder is None:
            raise ValueError("pixel_values were provided without a vision encoder.")
        if special_tokens is None:
            raise ValueError("special_tokens are required for multimodal inputs.")

        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)
        # MoonViT collapses time and merges spatially, so the text-side token
        # count per item is (h/kh)*(w/kw), independent of t.
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        num_tokens_per_item = (grid_thw[:, 1] // kernel_h) * (
            grid_thw[:, 2] // kernel_w
        )
        vision_positions = get_vision_positions(
            tokens,
            num_tokens_per_item,
            special_tokens["image_id"],
        )
        if isinstance(embeddings_TD, DTensor) and not isinstance(
            vision_embeds, DTensor
        ):
            # Under TP the embedding's output is a DTensor while the tower,
            # replicated and undeclared, returns a plain tensor; the splice's
            # copy_ refuses the mix. The tower's output is replicate-consistent
            # across the mesh (replicated weights, the same pixels everywhere),
            # so saying so is a wrap, not a transfer.
            vision_embeds = DTensor.from_local(
                vision_embeds,
                embeddings_TD.device_mesh,
                [Replicate()] * len(embeddings_TD.placements),
            )
        # Under SP the stream arrives sequence-sharded and the splice indexes
        # global token positions, so gather for the scatter and re-shard after.
        sp_placements = None
        if isinstance(embeddings_TD, DTensor) and any(
            p.is_shard() for p in embeddings_TD.placements
        ):
            sp_placements = embeddings_TD.placements
            embeddings_TD = embeddings_TD.redistribute(
                placements=tuple(
                    Replicate() if p.is_shard() else p for p in sp_placements
                )
            )
        spliced_TD = scatter_vision_embeds(
            embeddings_TD,
            vision_embeds=vision_embeds,
            vision_positions=vision_positions,
        )
        if sp_placements is not None and isinstance(spliced_TD, DTensor):
            spliced_TD = spliced_TD.redistribute(placements=sp_placements)
        return spliced_TD

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        if pixel_values_videos is not None or grid_thw_videos is not None:
            raise NotImplementedError("Kimi K3 v1 supports images but not videos.")
        if self.tok_embeddings is not None:
            h_TD = self._prepare_multimodal_embeds(
                tokens,
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                special_tokens=special_tokens,
            )
        else:
            h_TD = tokens

        num_tokens, D = h_TD.shape
        block_residual_TND = h_TD.new_zeros(num_tokens, 0, D)
        mesh = current_spmd_mesh()
        if mesh is not None and mesh.mesh_dim_names and "tp" in mesh.mesh_dim_names:
            # A fresh tensor reads as replicated on TP; the block stream it
            # opens is invariant there, and the stack concatenates the two.
            # Forward is a no-op; the type decides how gradients reduce.
            block_residual_TND = spmd.redistribute(
                block_residual_TND, mesh.get_group("tp"), src=spmd.R, dst=spmd.I
            )
        for layer in self.layers.values():
            h_TD, block_residual_TND = layer(
                h_TD,
                block_residual_TND,
                attention_masks,
                positions,
            )

        h_TD = _apply_attention_residual(
            h_TD,
            block_residual_TND,
            self.output_res_proj,
            self.output_res_norm,
        )
        h_TD = self.norm(h_TD) if self.norm is not None else h_TD
        if self._skip_lm_head:
            return h_TD
        return self.lm_head(h_TD) if self.lm_head is not None else h_TD
