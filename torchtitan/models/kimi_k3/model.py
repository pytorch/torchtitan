# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast, TypedDict

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.context_parallel import (
    ContextParallelPartitioner,
    HeadTailLoadBalancer,
)
from torchtitan.distributed.fsdp import add_zero_valued_dependency
from torchtitan.distributed.parallel_dims import MeshAxisName, ParallelDims
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    annotate_replicated_parameters,
    spmd_local_context,
    spmd_mesh_group,
)
from torchtitan.models.common import FeedForward, Linear
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_varlen_metadata_for_document,
    FlexInnerAttention,
    local_head_split,
    VarlenInnerAttention,
    VarlenMetadata,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import (
    decoder_input_sharding,
    dense_activation_placement,
    token_id_placement,
)
from torchtitan.models.common.multimodal import (
    build_vision_bank_indices,
    gather_vision_embeds,
    get_vision_positions,
    MultimodalModel,
    scatter_vision_embeds,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding
from torchtitan.models.kimi_k3.sharding import set_kimi_k3_sharding_config
from torchtitan.models.utils import (
    delta_rule_flops_per_token,
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.protocols.module import Module

from .kda import KDA, KDAAttentionMetadata
from .moe import KimiLatentMoE
from .state_dict_adapter import KimiK3StateDictAdapter
from .vision_encoder import enable_vision_cp_typecheck_rules, KimiK3VisionEncoder

logger = logging.getLogger(__name__)


class KimiK3AttentionMetadata(TypedDict):
    """Per-batch metadata for Kimi K3 attention backends."""

    quadratic_attention: BlockMask | VarlenMetadata | None
    kda: KDAAttentionMetadata


# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels, H = heads,
# K = query/key head dimension, V = value head dimension,
# N = attention-residual entries.


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
        inner_attention: Module.Config = field(
            default_factory=FlexInnerAttention.Config
        )

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

        q_THK = local_head_split(
            self.wq_b(self.q_norm(self.wq_a(x_TD))), self.q_head_dim
        )

        compressed_kv_TC = self.wkv_a(x_TD)
        kv_latent_TC, k_rope_TK = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        packed_kv_TC = self.wkv_b(self.kv_norm(kv_latent_TC))
        with spmd.local():
            kv_THC = local_head_split(
                packed_kv_TC,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            k_nope_THK, v_THV = torch.split(
                kv_THC,
                [self.qk_nope_head_dim, self.v_head_dim],
                dim=-1,
            )
            # The head-shared key is expanded only after the local projection.
            k_rope_THK = k_rope_TK.unsqueeze(1).expand(-1, k_nope_THK.shape[-2], -1)
            k_THK = torch.cat((k_nope_THK, k_rope_THK), dim=-1)
            if spmd.is_type_checking():
                spmd.assert_type(k_THK, {"dp": spmd.S(0), "tp": spmd.S(1)})

        out_THV = self.inner_attention(
            q_THK,
            k_THK,
            v_THV,
            attention_masks=attention_masks,
            scale=self.scale,
        )
        out_TD = out_THV.flatten(-2)
        out_TD = out_TD * torch.sigmoid(self.gate(x_TD))
        return self.wo(out_TD)


def _apply_attention_residual(
    prefix_sum_TD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    projection: Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    """Apply Kimi's block-level attention residual in FP32."""
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
        feed_forward: FeedForward.Config | None
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
        attention_metadata: KimiK3AttentionMetadata | None = None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
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
            layer_mask = (
                attention_metadata["quadratic_attention"]
                if attention_metadata is not None
                else None
            )
            h_TD = self.attention(h_TD, layer_mask, positions)
        else:
            assert self.delta_attention is not None
            kda_metadata = attention_metadata["kda"] if attention_metadata else None
            h_TD = self.delta_attention(
                h_TD,
                kda_metadata.varlen if kda_metadata is not None else None,
                positions,
                routing=(kda_metadata.cp_routing if kda_metadata is not None else None),
            )
        prefix_sum_TD = h_TD if opens_block else prefix_sum_TD + h_TD

        h_TD = _apply_attention_residual(
            prefix_sum_TD,
            block_residual_TND,
            self.ffn_res_proj,
            self.ffn_res_norm,
        )
        h_TD = self.ffn_norm(h_TD)
        if self.moe is not None:
            h_TD = self.moe(h_TD, padding_mask_T=padding_mask)
        else:
            assert self.feed_forward is not None
            h_TD = self.feed_forward(h_TD)
        return prefix_sum_TD + h_TD, block_residual_TND


def _build_cp_subgroups(cp_group) -> dict[int, dist.ProcessGroup]:
    """One sub-CP group layout per divisor of the CP size: ``{sub-group count: this rank's group}``.

    Which layout a step wants depends on how many large images its batch holds,
    and a group cannot be built per batch, so every layout is built here. The CP
    rank lists are all-gathered first because the enumeration each call takes has
    to cover the world and be identical on every rank.
    """
    if cp_group is None:
        return {}
    cp_ranks = dist.get_process_group_ranks(cp_group)
    cp_size = len(cp_ranks)
    if cp_size <= 1:
        return {}
    gathered: list[list[int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, cp_ranks)
    every_cp_group = sorted({tuple(entry) for entry in gathered if entry})
    out: dict[int, dist.ProcessGroup] = {1: cp_group}
    for n_sub in (d for d in range(2, cp_size + 1) if cp_size % d == 0):
        size = cp_size // n_sub
        mine, _ = dist.new_subgroups_by_enumeration(
            [
                list(ranks[s * size : (s + 1) * size])
                for ranks in every_cp_group
                for s in range(n_sub)
            ]
        )
        assert isinstance(mine, dist.ProcessGroup)
        out[n_sub] = mine
    return out


@spmd.register_local_autograd_function
class _PlainGradBoundary(torch.autograd.Function):
    """Identity forward; the incoming gradient leaves as a plain tensor.

    The vision tower's dynamic CP runs hand-written collectives whose
    transpose is a reduce-scatter with no DTensor sharding strategy;
    ``to_local()`` re-wraps the gradient with the forward placements and
    ``grad_placements`` only says which placements to re-wrap with. Only an
    autograd.Function can say "do not re-wrap".
    """

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        return grad.to_local() if isinstance(grad, DTensor) else grad


class KimiK3Model(MultimodalModel):
    state_dict_adapter_cls = KimiK3StateDictAdapter
    multimodal_encoder_fqns = ("vision_encoder",)

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallel_dims) -> None:
        from torchtitan.components.optimizer import register_moe_quantile_balancing_hook

        register_moe_quantile_balancing_hook(optimizers, model_parts, parallel_dims)

    supports_pipeline_parallel = False

    # The sub-CP groups the tower's dynamic partition can choose between, built
    # once by parallelize because building a group is collective.
    _vision_cp_subgroups: dict[int, dist.ProcessGroup] = {}

    def set_vision_cp_subgroups(self, subgroups: dict[int, dist.ProcessGroup]) -> None:
        """Hand the tower the sub-CP groups it may partition an image over."""
        self._vision_cp_subgroups = subgroups

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        layers: list[KimiK3TransformerBlock.Config]
        output_res_norm: RMSNorm.Config
        output_res_proj: Linear.Config
        vision_encoder: KimiK3VisionEncoder.Config | None = None
        # The smallest image worth partitioning across CP ranks (report sec
        # 5.2.3); below it the replicated encode is cheaper, since a split buys
        # one gather per layer.
        dynamic_cp_min_patches: int = 256

        def update_from_config(self, *, config, **kwargs) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism
            if parallelism.context_parallel_degree > 1:
                load_balancer_config = parallelism.context_parallel_load_balancer
                if load_balancer_config is not None and not isinstance(
                    load_balancer_config, HeadTailLoadBalancer.Config
                ):
                    raise ValueError(
                        "Kimi K3 KDA context parallelism supports only contiguous "
                        "or headtail token partitions."
                    )
                conv_kernel_sizes = {
                    layer.delta_attention.conv_kernel_size
                    for layer in self.layers
                    if layer.delta_attention is not None
                }
                if len(conv_kernel_sizes) != 1:
                    raise ValueError(
                        "Kimi K3 context parallelism requires every KDA layer "
                        "to use the same convolution kernel size."
                    )

            # Vision attention is also head-sharded; validate its head count.
            tp = parallelism.tensor_parallel_degree
            vision_heads = (
                self.vision_encoder.block.attn.num_heads
                if self.vision_encoder is not None
                else None
            )
            if tp > 1 and vision_heads is not None and vision_heads % tp != 0:
                raise ValueError(
                    f"tensor_parallel_degree ({tp}) must divide "
                    f"vision num_heads ({vision_heads})."
                )

            set_kimi_k3_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

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
        self.dynamic_cp_min_patches = config.dynamic_cp_min_patches
        self._dyncp_logged = False

    def parallelize(
        self,
        *,
        parallel_dims: ParallelDims,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        compile_config: CompileConfig | None,
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
        skip_dp: bool = False,
    ) -> KimiK3Model:
        unsupported = [
            name
            for name, enabled in (
                ("pipeline parallel", parallel_dims.pp_enabled),
            )
            if enabled
        ]
        if unsupported:
            raise NotImplementedError(
                f"Kimi K3 does not support {', '.join(unsupported)}."
            )
        if compile_config is not None and "model" in compile_config.components:
            raise NotImplementedError("Kimi K3 does not support model compilation yet.")

        from torchtitan.distributed.utils import get_spmd_context

        with get_spmd_context(parallel_dims=parallel_dims):
            annotate_replicated_parameters(self, parallel_dims)
            self._parallelize(parallel_dims)
            if parallel_dims.cp_enabled:
                # Building a process group is collective, so every rank runs this,
                # including a pipeline stage that holds no tower and never uses it.
                enable_vision_cp_typecheck_rules()
                self.set_vision_cp_subgroups(
                    _build_cp_subgroups(
                        parallel_dims.get_mesh(MeshAxisName.CP).get_group()
                    )
                )
            if ac_config is not None:
                policy = ac_config.build(dump_folder=dump_folder)
                policy.apply(self)
                if self.vision_encoder is not None:
                    policy.apply(self.vision_encoder)
            if not skip_dp:
                self._apply_fsdp(
                    parallel_dims=parallel_dims,
                    training=training,
                    parallelism=parallelism,
                )
        return self

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build masks and CP metadata, shard inputs, and annotate layouts."""
        del kwargs
        batch: dict[str, Any] = dict(input_dict)
        positions = batch.get("positions")
        padding_mask = batch.get("padding_mask", None)
        if positions is not None:
            inner = self.config.first_full_attention_backend
            if isinstance(
                inner, (FlexInnerAttention.Config, VarlenInnerAttention.Config)
            ):
                batch["attention_masks"] = self.get_attention_masks(
                    positions=positions,
                    padding_mask=padding_mask,
                    max_num_documents=max_num_documents,
                    max_context_length=max_context_length,
                )

        pixel_values = batch.get("pixel_values")
        grid_thw = batch.get("grid_thw")
        special_tokens = batch.get("special_tokens")
        if parallel_dims.cp_enabled and pixel_values is not None:
            if grid_thw is None:
                raise ValueError(
                    "pixel_values were provided but grid_thw was not provided."
                )
            if self.vision_encoder is None:
                raise ValueError("pixel_values were provided without a vision encoder.")
            if special_tokens is None or "image_id" not in special_tokens:
                raise ValueError(
                    "pixel_values require special_tokens with an 'image_id' entry."
                )
            if self.tok_embeddings is not None:
                batch["vision_bank_indices_T"] = build_vision_bank_indices(
                    batch["input"],
                    placeholder_id=special_tokens["image_id"],
                )
            batch.pop("special_tokens")

        input_sharding = {
            **decoder_input_sharding(),
            **multimodal_input_sharding(include_cp_axis=True),
        }
        input_sharding["vision_bank_indices_T"] = token_id_placement()
        if parallel_dims.cp_enabled:
            partitioner = ContextParallelPartitioner(
                input_dict=batch,
                input_shardings=input_sharding,
                cp_mesh=parallel_dims.get_mesh("cp"),
                load_balancer_config=parallelism.context_parallel_load_balancer,
            )
            batch = partitioner.shard_inputs(batch)
            batch = self._prepare_context_parallel_metadata(batch, partitioner)

        batch = annotate_input_spmd_types(parallel_dims, batch, input_sharding)

        inputs = batch.pop("input")
        labels = batch.pop("labels")
        return inputs, labels, batch

    def get_attention_masks(  # pyrefly: ignore [bad-override]
        self,
        positions: torch.Tensor,
        *,
        padding_mask: torch.Tensor | None = None,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
    ) -> KimiK3AttentionMetadata:
        attn_config = self.config.first_attention

        kda_metadata = create_varlen_metadata_for_document(
            positions,
            padding_mask=padding_mask,
            max_num_documents=max_num_documents,
            max_context_length=max_context_length,
        )

        if attn_config is None:
            quadratic_attention = None
        elif isinstance(attn_config.inner_attention, VarlenInnerAttention.Config):
            # Under varlen both consumers read the same document offsets.
            quadratic_attention = kda_metadata
        else:
            quadratic_attention = super().get_attention_masks(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
        return {
            "quadratic_attention": quadratic_attention,  # pyrefly: ignore [bad-assignment]
            "kda": KDAAttentionMetadata(varlen=kda_metadata),
        }

    def encode_images(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """The tower's forward on one micro-batch's images.

        The forward calls this; the pipeline's vision run-ahead calls it too,
        for a later micro-batch, so the two cannot drift. Under context
        parallelism the large images are partitioned across the ranks of a
        sub-CP group (report sec 5.2.3), the rest are encoded replicated.
        """
        assert self.vision_encoder is not None
        group_all = spmd_mesh_group(MeshAxisName.CP)
        subgroups = self._vision_cp_subgroups
        if group_all is None or not subgroups:
            pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
            return self.vision_encoder(pixel_values, grid_thw=grid_thw)
        return self._encode_images_partitioned(
            pixel_values, grid_thw, group_all, subgroups
        )

    def _tower_needs_collectives(self) -> bool:
        """Is the tower wrapped in something that issues per-forward collectives?

        True once FSDP has sharded it, which is when skipping it desynchronizes
        the process group; a replicated DTensor issues no all-gather to match,
        so the test is on the placement, not the type.
        """
        assert self.vision_encoder is not None
        return any(
            isinstance(p, DTensor) and any(pl.is_shard() for pl in p.placements)
            for p in self.vision_encoder.parameters()
        )

    def _tower_placeholder(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The smallest input the tower accepts, for a rank with no images."""
        assert self.vision_encoder is not None
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        device = next(self.parameters()).device
        grid = torch.tensor([[1, kernel_h, kernel_w]], dtype=torch.long, device=device)
        weight = self.vision_encoder.patch_embed.weight
        # A plain tensor: once FSDP has sharded the tower the weight is a
        # DTensor, and a placeholder inheriting that meets the tower's own
        # plain tensors as a mixed matmul.
        patches = torch.zeros(
            kernel_h * kernel_w, weight.shape[-1], dtype=weight.dtype, device=device
        )
        return patches, grid

    def _encode_images_partitioned(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        group_all: dist.ProcessGroup,
        subgroups: dict[int, dist.ProcessGroup],
    ) -> torch.Tensor:
        """Encode every image, partitioning the large ones (report sec 5.2.3).

        Every large image is encoded by one sub-CP group, its patches split
        across that sub-group's ranks with k/v gathered inside the group;
        images below the threshold, or whose grid height does not divide the
        merge kernel, stay whole and are encoded replicated.
        """
        import torch.distributed._functional_collectives as funcol

        from torchtitan.models.kimi_k3.vision_encoder import CPPatchPlan
        from torchtitan.models.kimi_k3.vit_cp_plan import (
            balance_images,
            classify,
            merged_tokens,
            row_partition,
            subgroup_layout,
        )

        assert self.vision_encoder is not None
        encoder = self.vision_encoder
        weight_dtype = encoder.patch_embed.weight.dtype
        grids = grid_thw.tolist()
        counts = [t * h * w for t, h, w in grids]
        kh, kw = encoder.merge_kernel_size
        offsets = [0]
        for c in counts:
            offsets.append(offsets[-1] + c)

        def _replicated(which: list[int]) -> dict[int, torch.Tensor]:
            out = {}
            for i in which:
                item = pixel_values[offsets[i] : offsets[i + 1]].to(weight_dtype)
                item_grid = torch.tensor(
                    [grids[i]], dtype=grid_thw.dtype, device=grid_thw.device
                )
                out[i] = encoder(item, grid_thw=item_grid)
            return out

        def _all_replicated() -> torch.Tensor:
            out = _replicated(list(range(len(counts))))
            return torch.cat([out[i] for i in range(len(counts))], dim=0)

        cp_size = dist.get_world_size(group_all)
        if cp_size <= 1:
            return _all_replicated()
        large = classify(counts, cp_size, min_patches=self.dynamic_cp_min_patches)
        # A grid height that does not divide the merge kernel cannot be cut
        # safely; such an image stays replicated.
        large = [i for i in large if grids[i][1] % kh == 0]
        if not large:
            return _all_replicated()
        n_sub, g = subgroup_layout(len(large), cp_size)
        group = subgroups.get(n_sub)
        if group is None or g <= 1:
            return _all_replicated()

        cp_rank = dist.get_rank(group_all)
        my_sub = cp_rank // g
        rank_in_sub = cp_rank % g
        group_of = balance_images([counts[i] for i in large], n_sub)
        my_large = [
            img for img, sub in zip(large, group_of, strict=True) if sub == my_sub
        ]
        if not self._dyncp_logged:
            self._dyncp_logged = True
            logger.info(
                "Dynamic CP: %d large image(s) of %d over %d sub-CP group(s) of "
                "%d rank(s); min_patches=%d.",
                len(large),
                len(counts),
                n_sub,
                g,
                self.dynamic_cp_min_patches,
            )

        out: dict[int, torch.Tensor] = {}
        # Every sub-group runs the same number of passes, or the collectives
        # inside them desynchronise; a sub-group with fewer images pads with
        # an empty pass whose output is discarded.
        per_sub = [sum(1 for s in group_of if s == k) for k in range(n_sub)]
        n_passes = max(per_sub) if per_sub else 0
        for p in range(n_passes):
            img = my_large[p] if p < len(my_large) else None
            if img is None:
                local = pixel_values.new_zeros(kh * kw, *pixel_values.shape[1:])
                local_grid = torch.tensor(
                    [[1, kh, kw]], dtype=grid_thw.dtype, device=grid_thw.device
                )
                plan = CPPatchPlan(
                    group=group,
                    valid_total=kh * kw * g,
                    full_grid=(1, kh * g, kw),
                    row_start=0,
                    band=kh,
                    real_rows=kh,
                )
            else:
                t, h, w = grids[img]
                shards = row_partition(t, h, w, kh=kh, group_size=g)
                sh = shards[rank_in_sub]
                bands = [s.row_end - s.row_start for s in shards]
                band = max(bands)
                if bands != sorted(bands, reverse=True):
                    raise AssertionError(
                        f"bands {bands} are not non-increasing; padding would land "
                        "inside the gathered token stream"
                    )
                flat = pixel_values[offsets[img] : offsets[img + 1]]
                # This rank's rows of every frame: the projector's temporal
                # mean spans all frames.
                pad_rows = band - (sh.row_end - sh.row_start)
                pieces = []
                for a, b in sh.ranges:
                    pieces.append(flat[a:b])
                    if pad_rows:
                        pieces.append(flat.new_zeros(pad_rows * w, *flat.shape[1:]))
                local = torch.cat(pieces, dim=0)
                local_grid = torch.tensor(
                    [[t, band, w]], dtype=grid_thw.dtype, device=grid_thw.device
                )
                plan = CPPatchPlan(
                    group=group,
                    valid_total=counts[img],
                    full_grid=(t, h, w),
                    row_start=sh.row_start,
                    band=band,
                    real_rows=sh.row_end - sh.row_start,
                )
            feats = encoder(local.to(weight_dtype), grid_thw=local_grid, cp_plan=plan)
            if isinstance(feats, DTensor):
                feats = feats.to_local()
            local_feat = _PlainGradBoundary.apply(feats)
            # The boundary on the output too: the gradient arrives from
            # downstream, and the gather's transpose must not see a DTensor.
            gathered = _PlainGradBoundary.apply(
                funcol.all_gather_tensor(
                    local_feat.contiguous(), gather_dim=0, group=group
                )
            )
            if img is not None:
                t, h, w = grids[img]
                # The projector collapses time: a video's token count carries no t.
                out[img] = gathered[: merged_tokens(h, w, kh, kw)]
        rest = [i for i in range(len(counts)) if i not in out]
        if rest:
            out.update(_replicated(rest))
        return torch.cat([out[i] for i in range(len(counts))], dim=0)

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
        vision_bank_indices_T: torch.Tensor | None,
    ) -> torch.Tensor:
        embeddings_TD = self.tok_embeddings(tokens)
        if (pixel_values is None) != (grid_thw is None):
            raise ValueError(
                "pixel_values and grid_thw must either both be provided or "
                "both be omitted."
            )
        if pixel_values is None:
            # An image-free batch is normal, but FSDP2 issues the tower's
            # all-gather from its pre-forward hook, so every rank must run it:
            # a zero-valued placeholder keeps the collectives and the DP average.
            if self.vision_encoder is not None and self._tower_needs_collectives():
                placeholder, placeholder_grid = self._tower_placeholder()
                unused = self.vision_encoder(placeholder, grid_thw=placeholder_grid)
                if isinstance(unused, DTensor):
                    unused = unused.to_local()
                return add_zero_valued_dependency(embeddings_TD, unused)
            return embeddings_TD
        assert grid_thw is not None
        if self.vision_encoder is None:
            raise ValueError("pixel_values were provided without a vision encoder.")

        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self.encode_images(pixel_values, grid_thw)
        # MoonViT collapses time and merges spatially, so the text-side token
        # count per item is (h/kh)*(w/kw), independent of t.
        kernel_h, kernel_w = self.vision_encoder.merge_kernel_size
        num_tokens_per_item = (grid_thw[:, 1] // kernel_h) * (
            grid_thw[:, 2] // kernel_w
        )
        if vision_bank_indices_T is not None:
            return gather_vision_embeds(
                embeddings_TD,
                vision_bank_VD=vision_embeds,
                vision_bank_indices_T=vision_bank_indices_T,
            )
        if special_tokens is None:
            raise ValueError("special_tokens are required for multimodal inputs.")
        vision_positions = get_vision_positions(
            tokens,
            num_tokens_per_item,
            special_tokens["image_id"],
        )
        return scatter_vision_embeds(
            embeddings_TD,
            vision_embeds=vision_embeds,
            vision_positions=vision_positions,
        )

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
        attention_masks: KimiK3AttentionMetadata | None = None,
        padding_mask: torch.Tensor | None = None,
        vision_bank_indices_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pixel_values_videos is not None or grid_thw_videos is not None:
            raise NotImplementedError("Kimi K3 v1 supports images but not videos.")
        if self.tok_embeddings is not None:
            with spmd_local_context("dp"):
                h_TD = self._prepare_multimodal_embeds(
                    tokens,
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                    special_tokens=special_tokens,
                    vision_bank_indices_T=vision_bank_indices_T,
                )
        else:
            h_TD = tokens

        if spmd.is_type_checking():
            # Vision fusion runs on a DP-local mesh. Restore the token layout
            # before constructing and propagating the attention residual state.
            spmd.assert_type(
                h_TD,
                dense_activation_placement(tp=spmd.I, cp=spmd.S(0)),
            )

        block_residual_TND = h_TD.unsqueeze(1)[:, :0]
        for layer in self.layers.values():
            h_TD, block_residual_TND = layer(
                h_TD,
                block_residual_TND,
                attention_metadata=attention_masks,
                positions=positions,
                padding_mask=padding_mask,
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
