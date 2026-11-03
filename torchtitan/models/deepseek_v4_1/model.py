# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 Flash training model with explicit cross-layer state.

T: tokens, C: HC branches, D: model width, S: global KV tokens,
K: selected keys, P: candidate keys, L: Engram layers, Q: hash columns.
"""

from dataclasses import dataclass, field
from typing import ClassVar

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    set_current_spmd_mesh,
)
from torchtitan.models.base.module import ModuleDict
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.flops import get_nparams_and_active_nparams
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v4.mhc import HcPost

from .attention import CSA2InnerAttention, DeepSeekV41Attention, rms_norm
from .engram import Engram, EngramHash
from .mhc import HcCoefficients, initial_pre_mix, mix_branches
from .moe import ModalityMoE
from .state_dict_adapter import DeepSeekV41StateDictAdapter


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


class DeepSeekV41Model(Decoder):
    """Teacher-forced training evaluates the complete causal encoder and decoder.

    Full layers replace the shared global pool. Reindex replaces selections;
    Reuse consumes both. Carried tensors keep autograd history, and are scoped to one forward/microbatch.
    """

    state_dict_adapter_cls = DeepSeekV41StateDictAdapter
    supports_pipeline_parallel: ClassVar[bool] = False

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
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=["loss", "fused_binary_activation", "rope"]
            )
        )

        def __post_init__(self):
            super(DeepSeekV41Model.Config, self).__post_init__()
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
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.float()

    def _apply_fsdp(self, *, parallelism_context, training, parallelism):

        from .attention import Compressor
        from .moe import ModalityRouter

        fp32_parameter_ids = set()
        for module in self.modules():
            if (
                isinstance(module, (RMSNorm, HcCoefficients))
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
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)

    def preprocess_inputs(
        self, input_dict, *, parallelism_context, parallelism, **kwargs
    ):
        from .sharding import engram_hash_placement

        if parallelism_context.cp_enabled:
            raise ValueError(
                "CSA2 shared pools currently require context_parallel_degree=1."
            )
        extras = {}
        if "engram_hashes" in input_dict:
            extras["engram_hashes"] = input_dict.pop("engram_hashes")
        else:
            with spmd.no_typecheck():
                padding_mask_T = input_dict.get("padding_mask")
                extras["engram_hashes"] = self.engram_hash(
                    input_dict["input"],
                    input_dict["positions"],
                    token_mask_T=None if padding_mask_T is None else ~padding_mask_T,
                )
        inputs, labels, model_kwargs, loss_kwargs = super().preprocess_inputs(
            input_dict,
            parallelism_context=parallelism_context,
            parallelism=parallelism,
            **kwargs,
        )
        metadata = model_kwargs["attention_metadata"][CSA2InnerAttention]
        with set_current_spmd_mesh(parallelism_context.spmd_dense_mesh()):
            offset_layout = spmd.SpmdType({"dp": spmd.V, "cp": spmd.I, "tp": spmd.I})
            spmd.assert_type(metadata.cu_seq_q, offset_layout)
            spmd.assert_type(metadata.cu_seq_k, offset_layout)
        extras = annotate_input_spmd_types(
            parallelism_context, extras, {"engram_hashes": engram_hash_placement()}
        )
        model_kwargs.update(extras)
        return inputs, labels, model_kwargs, loss_kwargs

    def forward(
        self,
        tokens,
        positions=None,
        attention_metadata=None,
        *,
        padding_mask=None,
        aux_loss_denominators=None,
        engram_hashes=None,
    ):
        if positions is None:
            positions = torch.arange(tokens.size(0), device=tokens.device)
        token_mask_T = None
        if padding_mask is not None:
            token_mask_T = (
                ~padding_mask if token_mask_T is None else token_mask_T & ~padding_mask
            )
        if engram_hashes is None:
            engram_hashes = self.engram_hash(
                tokens, positions, token_mask_T=token_mask_T
            )
        hidden_TD = self.tok_embeddings(tokens)
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
        for layer_id, layer in self.layers.items():
            if layer.engram is not None:
                assert engram_hashes is not None
                layer_hashes_TQ = engram_hashes[
                    :, self.config.engram_layer_ids.index(int(layer_id))
                ]
            else:
                layer_hashes_TQ = None
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
                engram_hashes_TQ=layer_hashes_TQ,
                engram_token_mask_T=token_mask_T,
                aux_loss_denominator=denominator,
            )
        hidden_TD = rms_norm(self.norm, mix_branches(h_TCD, pre_mix))
        output = hidden_TD if self._skip_lm_head else self.lm_head(hidden_TD)
        return output
