# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 Flash training model with explicit cross-layer state.

T: tokens, C: HC branches, D: model width, S: global KV tokens,
K: selected keys, P: candidate keys, L: Engram layers, Q: hash columns,
F: concatenated backbone width.
"""

from dataclasses import dataclass, field
from typing import ClassVar, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    set_current_spmd_mesh,
    spmd_dense_sp_enabled,
    spmd_mesh_group,
)
from torchtitan.models.base.module import ModuleDict
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.flops import get_nparams_and_active_nparams
from torchtitan.models.common.multimodal import MultimodalModel
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v4.mhc import HcPost

from .attention import CSA2InnerAttention, DeepSeekV41Attention, rms_norm
from .engram import Engram, EngramHash
from .mhc import HcCoefficients, initial_pre_mix, mix_branches
from .moe import ModalityMoE, update_image_expert_bias
from .state_dict_adapter import DeepSeekV41StateDictAdapter
from .vision import Aligner, rotary_cache, VisionEncoder

if TYPE_CHECKING:
    from .dspark import DSpark


class DeepSeekV41TransformerBlock(TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        attention: DeepSeekV41Attention.Config  # pyrefly: ignore [bad-override]
        feed_forward: ModalityMoE.Config  # pyrefly: ignore [bad-override]
        attention_norm: RMSNorm.Config  # pyrefly: ignore [bad-override]
        ffn_norm: RMSNorm.Config  # pyrefly: ignore [bad-override]
        hc_attn: HcCoefficients.Config
        hc_ffn: HcCoefficients.Config
        hc_post: HcPost.Config
        engram: Engram.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        # pyrefly: ignore [bad-override]
        self.attention: DeepSeekV41Attention = config.attention.build()
        self.attention_norm: RMSNorm = config.attention_norm.build()
        self.ffn_norm: RMSNorm = config.ffn_norm.build()
        # pyrefly: ignore [bad-override]
        self.feed_forward: ModalityMoE = config.feed_forward.build()
        self.hc_attn: HcCoefficients = config.hc_attn.build()
        self.hc_ffn: HcCoefficients = config.hc_ffn.build()
        self.hc_post: HcPost = config.hc_post.build()
        self.engram = config.engram.build() if config.engram is not None else None

    def forward(
        self,
        x_TCD,
        pre_TC,
        global_k_SD,
        index_k_SD,
        indices_TK,
        scores_TK,
        candidates_TP,
        *,
        positions,
        attention_metadata=None,
        padding_mask=None,
        image_mask_T=None,
        engram_hashes_TQ=None,
        engram_token_mask_T=None,
        aux_loss_denominator=None,
        **attention_kwargs,
    ):
        if self.engram is not None:
            assert engram_hashes_TQ is not None
            x_TCD = self.engram(
                x_TCD, engram_hashes_TQ, token_mask_T=engram_token_mask_T
            )
        residual_TCD = x_TCD
        attn_pre_TC, post_TC, comb_TCC = self.hc_attn(x_TCD)
        attn_input_TD = rms_norm(self.attention_norm, mix_branches(x_TCD, pre_TC))
        (
            out_TD,
            global_k_SD,
            index_k_SD,
            indices_TK,
            scores_TK,
            candidates_TP,
        ) = self.attention(
            attn_input_TD,
            global_k_SD,
            index_k_SD,
            indices_TK,
            scores_TK,
            candidates_TP,
            positions=positions,
            attention_metadata=attention_metadata,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask=padding_mask,
            **attention_kwargs,
        )
        remat.recompute_needs_tensor(out_TD)
        x_TCD = self.hc_post(out_TD, residual_TCD, post_TC, comb_TCC)
        residual_TCD = x_TCD
        pre_TC, post_TC, comb_TCC = self.hc_ffn(x_TCD)
        out_TD = self.feed_forward(
            rms_norm(self.ffn_norm, mix_branches(x_TCD, attn_pre_TC)),
            padding_mask_T=padding_mask,
            image_mask_T=image_mask_T,
            aux_loss_denominator=aux_loss_denominator,
        )
        remat.recompute_needs_tensor(out_TD)
        x_TCD = self.hc_post(out_TD, residual_TCD, post_TC, comb_TCC)
        return (
            x_TCD,
            pre_TC,
            global_k_SD,
            index_k_SD,
            indices_TK,
            scores_TK,
            candidates_TP,
        )


class DeepSeekV41Model(MultimodalModel):
    """Teacher-forced training evaluates the complete causal encoder and decoder.

    Full layers replace the shared global pool. Reindex replaces selections;
    Reuse consumes both. Carried tensors keep autograd history, and are scoped to one forward/microbatch.
    """

    state_dict_adapter_cls = DeepSeekV41StateDictAdapter
    supports_pipeline_parallel: ClassVar[bool] = False
    multimodal_encoder_fqns = ("vision",)

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        layers: list[DeepSeekV41TransformerBlock.Config]
        hc_mult: int
        num_encoder_layers: int
        compress_ratios: tuple[int, ...]
        kv_source_layers: tuple[int, ...]
        index_source_layers: tuple[int, ...]
        engram_layer_ids: tuple[int, ...]
        engram_hash: EngramHash.Config
        vision: VisionEncoder.Config | None = None
        aligner: Aligner.Config | None = None
        image_embeddings: Embedding.Config | None = None
        drafter: "DSpark.Config | None" = None
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=["loss", "fused_binary_activation", "rope"]
            )
        )

        def __post_init__(self):
            super(DeepSeekV41Model.Config, self).__post_init__()
            if (self.vision is None) != (self.aligner is None) or (
                self.vision is None
            ) != (self.image_embeddings is None):
                raise ValueError(
                    "Vision encoder, aligner, and image markers must be configured together."
                )
            if len(self.layers) != len(self.compress_ratios):
                raise ValueError("Each backbone layer requires a compression ratio.")
            if not set(self.kv_source_layers).issubset(self.index_source_layers):
                raise ValueError(
                    "Every Full layer must also produce index keys and selections."
                )
            if not 0 < self.num_encoder_layers < len(self.layers):
                raise ValueError("V4.1 requires both encoder and decoder layers.")
            latest_ratio = 0
            for layer_id, ratio in enumerate(self.compress_ratios):
                if layer_id in self.kv_source_layers:
                    latest_ratio = ratio
                if ratio and ratio != latest_ratio:
                    raise ValueError(
                        "Shared KV consumers must use their producer's compression ratio."
                    )

        def get_nparams_and_flops(self, model, seq_len):
            nparams, active = get_nparams_and_active_nparams(
                model,
                modules_excluded_from_active_params=(
                    model.vision,
                    model.aligner,
                    model.drafter,
                ),
            )
            attention_flops = 0
            for layer in self.layers:
                attn = layer.attention
                slots = min(seq_len, attn.inner_attention.window_size)
                if attn.ratio:
                    slots += min(seq_len // attn.ratio, attn.inner_attention.index_topk)
                attention_flops += 12 * attn.n_heads * attn.head_dim * slots
                if (indexer := attn.indexer) is not None:
                    keys = seq_len // attn.ratio
                    if indexer.uses_candidates:
                        keys = min(
                            keys,
                            indexer.num_candidate_blocks * indexer.candidate_block_size,
                        )
                    attention_flops += 6 * indexer.num_heads * indexer.head_dim * keys
            return nparams, 6 * active + attention_flops

    def __init__(self, config: Config):
        super().__init__(config)
        self.config: DeepSeekV41Model.Config = config  # pyrefly: ignore [bad-override]
        # pyrefly: ignore [bad-override]
        self.layers: ModuleDict[DeepSeekV41TransformerBlock]
        self.engram_hash = config.engram_hash.build()
        self.vision = config.vision.build() if config.vision is not None else None
        self.aligner = config.aligner.build() if config.aligner is not None else None
        self.image_embeddings = (
            config.image_embeddings.build()
            if config.image_embeddings is not None
            else None
        )
        self.drafter = config.drafter.build() if config.drafter is not None else None
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.float()

    def parallelize(
        self,
        *,
        parallelism_context,
        training,
        parallelism,
        local_compile,
        activation_remat,
        dump_folder,
    ):
        """Apply the selected remat policy to the backbone, vision and drafter."""
        local_compile.apply()
        with parallelism_context.activate_spmd():
            self._parallelize(parallelism_context)
            if activation_remat is not None:
                for model in (self, self.vision, self.drafter):
                    if model is not None:
                        activation_remat.apply(model)
            self._apply_fsdp(
                parallelism_context=parallelism_context,
                training=training,
                parallelism=parallelism,
            )
        return self

    def _apply_fsdp(self, *, parallelism_context, training, parallelism):
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            fully_shard,
            MixedPrecisionPolicy,
            OffloadPolicy,
        )

        from torchtitan.config import TORCH_DTYPE_MAP
        from torchtitan.distributed.fsdp import (
            get_fsdp_reshard_after_forward_policy,
            linear_param_shard_placements,
            resolve_fsdp_mesh,
        )
        from .attention import Compressor
        from .dspark import ConfidenceHead
        from .moe import ModalityRouter

        fp32_parameter_ids = set()
        for module in self.modules():
            if (
                isinstance(module, (RMSNorm, HcCoefficients, ConfidenceHead))
                or isinstance(module, Compressor)
                and module.ratio == 2
            ):
                fp32_parameter_ids.update(
                    id(parameter) for parameter in module.parameters()
                )
            if isinstance(module, ModalityRouter):
                fp32_parameter_ids.update(
                    id(parameter) for parameter in module.gate.parameters()
                )
            if isinstance(module, DeepSeekV41Attention):
                fp32_parameter_ids.add(id(module.attn_sink.weight))

        def dtype_override(parameter):
            return torch.float32 if id(parameter) in fp32_parameter_ids else None

        self._fsdp_param_dtype_override_fn = dtype_override
        vision_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            param_dtype_override_fn=dtype_override,
        )
        for name in ("vision", "aligner"):
            if (encoder := getattr(self, name)) is not None:
                vision_mesh, vision_mesh_dims = resolve_fsdp_mesh(
                    parallelism_context, shard_on_cp_axis=False
                )
                fully_shard(
                    encoder,
                    mesh=vision_mesh,
                    mp_policy=vision_policy,
                    dp_mesh_dims=vision_mesh_dims,
                    shard_placement_fn=linear_param_shard_placements(encoder).get,
                    reshard_after_forward=get_fsdp_reshard_after_forward_policy(
                        parallelism.fsdp_reshard_after_forward,
                        pp_enabled=parallelism_context.pp_enabled,
                    ),
                    offload_policy=(
                        CPUOffloadPolicy()
                        if training.enable_cpu_offload
                        else OffloadPolicy()
                    ),
                )
        if self.drafter is not None:
            self.drafter._fsdp_param_dtype_override_fn = dtype_override
            self.drafter._apply_fsdp(
                parallelism_context=parallelism_context,
                training=training,
                parallelism=parallelism,
            )
        Decoder._apply_fsdp(
            self,
            parallelism_context=parallelism_context,
            training=training,
            parallelism=parallelism,
        )

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallelism_context):
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        drafters = [part.drafter for part in model_parts if part.drafter is not None]
        if drafters:
            type(drafters[0])._register_optimizer_hooks(
                optimizers, drafters, parallelism_context
            )
        if any(part.config.vision is not None for part in model_parts):
            optimizers.register_step_pre_hook(
                lambda *args, **kwargs: update_image_expert_bias(
                    model_parts, parallelism_context
                )
            )
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)

    def preprocess_inputs(
        self, input_dict, *, parallelism_context, parallelism, **kwargs
    ):
        from torchtitan.models.common.decoder_sharding import token_id_placement

        if parallelism_context.cp_enabled:
            raise ValueError(
                "CSA2 shared pools currently require context_parallel_degree=1."
            )
        extras = {
            name: input_dict.pop(name)
            for name in ("engram_hashes", "token_types", "pixel_values", "grid_thw")
            if name in input_dict
        }
        tokens_T, positions_T = input_dict["input"], input_dict["positions"]
        if "engram_hashes" not in extras:
            with spmd.no_typecheck():
                token_mask_T = (
                    None if "token_types" not in extras else extras["token_types"] < 0
                )
                if input_dict.get("padding_mask") is not None:
                    token_mask_T = (
                        ~input_dict["padding_mask"]
                        if token_mask_T is None
                        else token_mask_T & ~input_dict["padding_mask"]
                    )
                extras["engram_hashes"] = self.engram_hash(
                    tokens_T, positions_T, token_mask_T=token_mask_T
                )
        if "pixel_values" in extras:
            if (
                "grid_thw" not in extras
                or "token_types" not in extras
                or self.config.vision is None
            ):
                raise ValueError(
                    "Images require grid_thw, token_types, and a vision-enabled model."
                )
            with spmd.no_typecheck():
                grids = extras.pop("grid_thw").tolist()
                if any(
                    time != 1 or height <= 0 or width <= 0
                    for time, height, width in grids
                ):
                    raise ValueError(
                        "V4.1 vision expects positive image grids with time=1."
                    )
                extras["grid_hw"] = tuple((height, width) for _, height, width in grids)
                extras["vision_rope_cache"] = rotary_cache(
                    extras["grid_hw"],
                    head_dim=self.config.vision.dim // self.config.vision.num_heads,
                    theta=self.config.vision.rope_theta,
                    device=tokens_T.device,
                )
                visual_T = extras["token_types"] == 1
                extras["vision_bank_indices"] = (
                    visual_T.long().cumsum(0) - 1
                ).masked_fill(~visual_T, -1)
        elif "grid_thw" in extras:
            raise ValueError("grid_thw requires pixel_values.")
        inputs, labels, model_kwargs, loss_kwargs = super().preprocess_inputs(
            input_dict,
            parallelism_context=parallelism_context,
            parallelism=parallelism,
            **kwargs,
        )
        # These offsets describe each DP rank's documents, identically on TP.
        metadata = model_kwargs["attention_metadata"][CSA2InnerAttention]
        with set_current_spmd_mesh(parallelism_context.spmd_dense_mesh()):
            offset_layout = spmd.SpmdType({"dp": spmd.V, "cp": spmd.I, "tp": spmd.I})
            spmd.assert_type(metadata.cu_seq_q, offset_layout)
            spmd.assert_type(metadata.cu_seq_k, offset_layout)
        from torchtitan.models.common.vision_encoder_sharding import (
            multimodal_input_sharding,
        )
        from .sharding import engram_hash_placement

        input_layouts = {
            "engram_hashes": engram_hash_placement(),
            "token_types": token_id_placement(),
            "vision_bank_indices": token_id_placement(),
            **multimodal_input_sharding(),
        }
        input_layouts["vision_rope_cache"] = input_layouts["pixel_values"]
        extras = annotate_input_spmd_types(parallelism_context, extras, input_layouts)
        model_kwargs.update(extras)
        return inputs, labels, model_kwargs, loss_kwargs

    def _merge_images(
        self,
        hidden_TD,
        *,
        pixel_values,
        grid_hw,
        token_types,
        vision_rope_cache,
        vision_bank_indices,
    ):
        from torchtitan.models.common.multimodal import (
            add_zero_vision_dependency,
            build_dummy_vision_inputs,
            gather_vision_embeds,
        )

        if self.vision is None:
            if pixel_values is not None:
                raise ValueError("Images require a vision-enabled model.")
            return hidden_TD
        assert self.aligner is not None and self.image_embeddings is not None
        dummy = pixel_values is None
        if dummy:
            pixel_values, _ = build_dummy_vision_inputs(
                patch_dim=self.vision.patch_embed.in_features,
                grid_thw=(1, 3, 3),
                device=hidden_TD.device,
            )
            grid_hw = ((3, 3),)
        elif grid_hw is None or token_types is None:
            raise ValueError("Images require grid_hw and token_types.")
        encoded_TD = self.vision(
            pixel_values.to(self.vision.patch_embed.weight.dtype),
            grid_hw=grid_hw,
            rope_cache=vision_rope_cache,
        )
        features_VD = self.aligner(encoded_TD, grid_hw=grid_hw)
        if dummy:
            hidden_TD = add_zero_vision_dependency(hidden_TD, features_VD)
            return (
                hidden_TD + self.image_embeddings.weight.sum().to(hidden_TD.dtype) * 0
            )
        if vision_bank_indices is None:
            visual_T = token_types == 1
            vision_bank_indices = (visual_T.long().cumsum(0) - 1).masked_fill(
                ~visual_T, -1
            )
        if not torch.compiler.is_compiling():
            torch._assert_async(
                (token_types == 1).sum() == features_VD.size(0),
                "Image patch placeholders must match the aligned visual token count.",
            )
        hidden_TD = gather_vision_embeds(
            hidden_TD,
            vision_bank_VD=features_VD,
            vision_bank_indices_T=vision_bank_indices,
        )
        for marker, row in ((0, 0), (2, 1), (3, 2)):
            with spmd.local():
                merged_TD = torch.where(
                    (token_types == marker)[:, None],
                    self.image_embeddings.weight[row].to(hidden_TD.dtype),
                    hidden_TD,
                )
            if spmd.is_type_checking():
                spmd.assert_type_like(merged_TD, hidden_TD)
            hidden_TD = merged_TD
        return hidden_TD

    def forward(
        self,
        tokens,
        positions=None,
        attention_metadata=None,
        *,
        padding_mask=None,
        aux_loss_denominators=None,
        engram_hashes=None,
        token_types=None,
        pixel_values=None,
        grid_hw=None,
        vision_rope_cache=None,
        vision_bank_indices=None,
        return_draft_context=False,
    ):
        if return_draft_context and (
            self.tok_embeddings is None or self.norm is None or self.drafter is None
        ):
            raise ValueError(
                "Draft-context capture requires a complete model with DSpark enabled."
            )
        if positions is None:
            positions = torch.arange(tokens.size(0), device=tokens.device)
        image_mask_T = None if token_types is None else token_types >= 0
        token_mask_T = None if image_mask_T is None else ~image_mask_T
        if padding_mask is not None:
            token_mask_T = (
                ~padding_mask if token_mask_T is None else token_mask_T & ~padding_mask
            )
        if engram_hashes is None:
            engram_hashes = self.engram_hash(
                tokens, positions, token_mask_T=token_mask_T
            )
        hidden_TD = self.tok_embeddings(tokens)
        hidden_TD = self._merge_images(
            hidden_TD,
            pixel_values=pixel_values,
            grid_hw=grid_hw,
            token_types=token_types,
            vision_rope_cache=vision_rope_cache,
            vision_bank_indices=vision_bank_indices,
        )
        if self.config.vision is not None:
            if (tp_group := spmd_mesh_group("tp")) is not None:
                hidden_TD = spmd.redistribute(
                    hidden_TD,
                    tp_group,
                    src=spmd.R,
                    dst=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
                )
        h_TCD = hidden_TD[:, None, :].expand(-1, self.config.hc_mult, -1)
        pre_mix = initial_pre_mix(h_TCD)
        # Empty shared state is replicated across TP, unlike the residual.
        with spmd.no_typecheck():
            global_k = hidden_TD.new_empty(0, self.config.layers[0].attention.head_dim)
            first_indexer = self.config.layers[
                self.config.kv_source_layers[0]
            ].attention.indexer
            assert first_indexer is not None
            index_k = hidden_TD.new_empty(0, first_indexer.head_dim)
            selected_indices = torch.empty(
                tokens.size(0), 0, dtype=torch.int32, device=tokens.device
            )
            selected_scores = hidden_TD.new_empty(
                tokens.size(0), 0, dtype=torch.float32
            )
        candidates = selected_indices
        if spmd.is_type_checking():
            from torchtitan.models.common.decoder_sharding import (
                replicated_activation_placement,
            )

            spmd.assert_type(
                (global_k, index_k, selected_indices, selected_scores, candidates),
                replicated_activation_placement(),
            )
        with spmd.no_typecheck():
            denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        metadata = (
            None
            if attention_metadata is None
            else attention_metadata.get(CSA2InnerAttention)
        )
        draft_context = []
        for layer_id, layer in self.layers.items():
            if layer.engram is not None:
                assert engram_hashes is not None
                layer_hashes_TQ = engram_hashes[
                    :, self.config.engram_layer_ids.index(int(layer_id))
                ]
            else:
                layer_hashes_TQ = None
            if (
                return_draft_context
                and self.drafter is not None
                and int(layer_id) in self.drafter.config.target_layer_ids
            ):
                draft_context.append(h_TCD.mean(-2))
            (
                h_TCD,
                pre_mix,
                global_k,
                index_k,
                selected_indices,
                selected_scores,
                candidates,
            ) = layer(
                h_TCD,
                pre_mix,
                global_k,
                index_k,
                selected_indices,
                selected_scores,
                candidates,
                positions=positions,
                attention_metadata=metadata,
                padding_mask=padding_mask,
                image_mask_T=image_mask_T,
                engram_hashes_TQ=layer_hashes_TQ,
                engram_token_mask_T=token_mask_T,
                aux_loss_denominator=denominator,
            )
        hidden_TD = rms_norm(self.norm, mix_branches(h_TCD, pre_mix))
        output = hidden_TD if self._skip_lm_head else self.lm_head(hidden_TD)
        return (
            (output, torch.cat(draft_context, -1)) if return_draft_context else output
        )

    def forward_draft(
        self,
        main_features_TF,
        draft_embeddings_TD,
        previous_tokens_BS,
        anchors_B,
        *,
        main_positions_T,
        output_weight_VD,
    ):
        """Train the drafter using explicitly supplied frozen backbone projections."""
        if self.drafter is None:
            raise ValueError("Enable num_draft_layers=3 to use DSpark.")
        return self.drafter(
            main_features_TF,
            draft_embeddings_TD,
            previous_tokens_BS,
            anchors_B,
            main_positions_T=main_positions_T,
            output_weight_VD=output_weight_VD,
        )
