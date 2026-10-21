# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-neutral GLM-5 model mathematics.

Read this file from the outside in:

1. ``Glm5Model`` supplies token embedding, decoder blocks, final norm, and the
   language-model head inherited from ``Decoder``.
2. ``Glm5TransformerBlock`` composes pre-norm DSA attention with either a
   dense FFN or the common TorchTitan MoE implementation.
3. ``Glm5Attention`` expands MLA's low-rank Q/KV representations, asks the
   frozen ``Glm5DsaIndexer`` which keys each query may attend to, and delegates
   score/value computation to ``Glm5FlexAttention`` in ``dsa.py``.
4. ``dsa.py`` contains the indexer and token-selected BlockMask construction.

The model uses token-first tensors because TorchTitan packs microbatches into a
single token dimension before model forward. Distributed placement is declared
separately in ``sharding.py``; the formulas below describe global logical
shapes. Distributed adaptation of this sparse path is a separate step.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch_remat as remat
from torch import nn

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.common import ComplexRoPE, Linear, RMSNorm
from torchtitan.models.common.attention import Attention, InnerAttention
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.flops import get_nparams_and_active_nparams
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3.mtp import MTPDecoder, roll_mtp_sequence

from .dsa import create_dsa_causal_mask, Glm5DsaIndexer
from .sharding import dsa_mask_placement
from .state_dict_adapter import Glm5StateDictAdapter

# Tensor dimensions used in this file:
# T/Q/K: token, query-token, and key-token dimensions.
# D/N/H/R/P/V: model, head-count, head, RoPE, pass-through, and value dimensions.


class Glm5Attention(Attention):
    """Multi-head latent attention (MLA) gated by DSA top-k indices.

    Q follows ``D -> q_lora_rank -> N * (nope + rope)``. KV follows
    ``D -> (kv_lora_rank + rope)`` and expands only after the low-rank norm.
    The indexer consumes the same token states and returns ``[T, topk]`` key
    indices. ``inner_attention`` applies token-selected sparse attention to
    the expanded Q/K/V tensors.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        dim: int
        q_lora_rank: int
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
        wo: Linear.Config
        rope: ComplexRoPE.Config
        indexer: Glm5DsaIndexer.Config | None
        inner_attention: InnerAttention.Config

        @property
        def qk_head_dim(self) -> int:
            return self.qk_nope_head_dim + self.qk_rope_head_dim

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.softmax_scale = config.qk_head_dim**-0.5
        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.wo = config.wo.build()
        self.rope = config.rope.build()
        self.indexer = config.indexer.build() if config.indexer is not None else None
        self.inner_attention = config.inner_attention.build()

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: torch.Tensor,
        positions: torch.Tensor | None = None,
        topk_indices_TS: torch.Tensor | None = None,
        *,
        return_indices: bool = False,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        T = x_TD.shape[0]
        if positions is None:
            positions = torch.arange(T, device=x_TD.device)

        # Query path: compress to LoRA rank R, normalize, then expand to N
        # heads. Split content (NoPE) from position-sensitive (RoPE) features.
        q_resid_TR = self.wq_a(x_TD)
        # RMSNorm reads the saved projection output with bare ops.
        remat.recompute_needs_tensor(q_resid_TR)
        q_resid_TR = self.q_norm(q_resid_TR)
        q_TNH = self.wq_b(q_resid_TR)
        # The head view/split reads the saved projection output with bare ops.
        remat.recompute_needs_tensor(q_TNH)
        q_TNH = q_TNH.view(T, self.n_heads, self.qk_head_dim)
        q_nope_TNP, q_rope_TNR = torch.split(
            q_TNH, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        # KV path: keep content in a compressed latent representation and keep
        # one shared rotary key. The expensive per-head expansion happens only
        # after normalization.
        compressed_kv_TC = self.wkv_a(x_TD)
        # The latent split and RMSNorm read this projection with bare ops.
        remat.recompute_needs_tensor(compressed_kv_TC)
        kv_TR, k_rope_TR = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_TR = self.kv_norm(kv_TR)
        q_rope_TNR, k_rope_T1R = self.rope(
            q_rope_TNR, k_rope_TR.unsqueeze(1), positions
        )
        q_TNH = torch.cat((q_nope_TNP, q_rope_TNR), dim=-1)
        kv_TNX = self.wkv_b(kv_TR)
        # The head view/split reads the saved projection output with bare ops.
        remat.recompute_needs_tensor(kv_TNX)
        kv_TNX = kv_TNX.view(T, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope_TNP, v_TNV = torch.split(
            kv_TNX, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_TNH = torch.cat(
            (k_nope_TNP, k_rope_T1R.expand(-1, self.n_heads, -1)),
            dim=-1,
        )
        # Selection and value computation are separate on purpose: the indexer
        # is frozen and discrete, while the attention path remains trainable.
        if topk_indices_TS is None:
            if self.indexer is None:
                raise ValueError("shared-indexer attention requires top-k indices")
            topk_indices_TS = self.indexer(
                x_TD, q_resid_TR, positions, attention_masks[0]
            )
        output_TNV = self.inner_attention(
            q_TNH,
            k_TNH,
            v_TNV,
            attention_metadata=attention_masks,
            topk_indices_QS=topk_indices_TS,
            scale=self.softmax_scale,
        )
        # The contiguous view reads the attention output with bare ops.
        remat.recompute_needs_tensor(output_TNV)
        output_TD: torch.Tensor = self.wo(output_TNV.contiguous().view(T, -1))
        assert topk_indices_TS is not None
        return (output_TD, topk_indices_TS) if return_indices else output_TD


class Glm5TransformerBlock(TransformerBlock):
    """GLM-5 decoder block with either a dense FFN or common MoE."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        preserve_output_for_mtp: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.preserve_output_for_mtp = config.preserve_output_for_mtp
        self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: torch.Tensor,
        positions: torch.Tensor | None = None,
        topk_indices_TS: torch.Tensor | None = None,
        *,
        return_indices: bool = False,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_indices:
            attention_TD, topk_indices_TS = self.attention(
                self.attention_norm(x_TD),
                attention_masks,
                positions,
                topk_indices_TS,
                return_indices=True,
            )
        else:
            attention_TD = self.attention(
                self.attention_norm(x_TD),
                attention_masks,
                positions,
                topk_indices_TS,
            )
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(attention_TD)
        x_TD = x_TD + attention_TD
        normalized_TD = self.ffn_norm(x_TD)
        if isinstance(self.feed_forward, MoE):
            ffn_output_TD = self.feed_forward(
                normalized_TD,
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            ffn_output_TD = self.feed_forward(normalized_TD)
        if self.preserve_output_for_mtp:
            # The saved residual add is a bare consumer of the MoE/FFN result.
            # MTP's auxiliary backward branch replays this boundary, so declare
            # the input dependency before entering the non-recomputed region.
            remat.recompute_needs_tensor(ffn_output_TD)
        # Trailing add, always saved: it saves nothing for backward, so replay skips
        # it and its inputs need no persisting, matching checkpoint early stop.
        output_TD: torch.Tensor = remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(x_TD, ffn_output_TD)
        if self.preserve_output_for_mtp:
            # MTP adds an extra backward branch through the main decoder. Keep
            # each saved residual boundary materialized for MTP flavors so
            # selective-AC replay never exposes a metadata placeholder to a
            # bare consumer. Non-MTP flavors leave this disabled.
            remat.recompute_needs_tensor(output_TD)
            output_TD = output_TD.clone()
        if return_indices:
            assert topk_indices_TS is not None
            return output_TD, topk_indices_TS
        return output_TD


class Glm5Model(MTPDecoder):
    """GLM-5 decoder with FlexAttention DSA attention.

    The inherited decoder accepts token ids on the first PP stage and hidden
    states on later stages. A PP stage may own zero decoder layers, which is why
    mask construction and forward preserve embedding-only/output-only stages.
    """

    state_dict_adapter_cls = Glm5StateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(MTPDecoder.Config):
        dim: int = 6144
        vocab_size: int = 154880
        index_sources: tuple[int, ...] = ()

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attention = self.layers[0].attention
            assert isinstance(attention, Glm5Attention.Config)
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            base_flops = 6 * active_nparams
            # The common parameter-based estimate charges every parameter for
            # forward and backward matmuls (6 FLOPs per parameter). GLM-5's
            # pretrained indexer is frozen and runs under no_grad, so it only
            # incurs its forward cost (2 FLOPs per parameter). Keep the total
            # parameter count unchanged while removing the nonexistent
            # indexer backward work from MFU accounting.
            nparams_indexer = sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if ".attention.indexer." in name
            )
            base_flops -= 4 * nparams_indexer
            # Per token, the indexer computes N dot products of width H against
            # the full sequence, followed by a weighted reduction over N heads.
            num_dsa_layers = len(self.layers) + len(self.mtp_layers or ())
            assert attention.indexer is not None
            dsa_flops = (
                2
                * num_dsa_layers
                * attention.indexer.n_heads
                * (attention.indexer.head_dim + 1)
                * seq_len
            )
            return nparams, base_flops + dsa_flops

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)

    def get_attention_masks(
        self,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        # Pipeline partitioning may create an embedding-only or output-only
        # stage. Such a stage has no attention operation and needs no mask.
        if len(self.layers) == 0:
            return None

        if positions.ndim != 1:
            raise ValueError("GLM-5 positions must have shape [T].")
        # Non-first pipeline stages have tok_embeddings pruned away, but each
        # PP stage containing attention builds its own mask. Every stage
        # receives the same positions, so any float parameter dtype gives the
        # same mask dtype; fall back to the first layer's norm weight when
        # embeddings are absent. layers is a ModuleDict whose keys keep their
        # original indices after pruning, so iterate values rather than
        # indexing [0].
        if self.tok_embeddings is not None:
            token_dtype = self.tok_embeddings.weight.dtype
        else:
            attention_norm = next(iter(self.layers.values())).attention_norm
            assert isinstance(attention_norm, nn.Module)
            token_dtype = next(attention_norm.parameters()).dtype
        return create_dsa_causal_mask(
            positions,
            dtype=token_dtype,
            padding_mask_T=padding_mask,
        )

    def __init__(self, config):
        super().__init__(config)
        self.index_sources = config.index_sources
        if self.index_sources:
            if len(self.index_sources) != len(config.layers):
                raise ValueError("index_sources must contain one source per layer")
            for layer, source in enumerate(self.index_sources):
                if not 0 <= source <= layer or self.index_sources[source] != source:
                    raise ValueError(
                        "index source must be a current/earlier producer layer"
                    )

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context,
        parallelism,
        global_routing_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        """Build the global DSA mask, then CP-shard only its query dimension."""
        del kwargs
        loss_kwargs = input_dict.pop("loss_kwargs", {})
        positions = input_dict.get("positions")
        if positions is not None:
            input_dict["attention_masks"] = self.get_attention_masks(
                positions,
                input_dict.get("padding_mask"),
            )

        input_shardings = decoder_input_sharding()
        input_shardings["attention_masks"] = dsa_mask_placement()
        depths = (
            range(1, len(self.mtp_layers) + 1)
            if self.mtp_layers is not None
            else range(0)
        )
        if self.mtp_layers is not None and positions is None:
            raise ValueError("GLM-5 MTP input preprocessing requires positions.")
        for depth in depths:
            assert positions is not None
            mtp_tokens, mtp_valid = roll_mtp_sequence(
                input_dict["input"],
                shift=depth,
                positions=positions,
                padding_mask=input_dict.get("padding_mask"),
                fill_value=0,
                return_valid_mask=True,
            )
            mtp_labels = roll_mtp_sequence(
                input_dict["labels"],
                shift=depth,
                positions=positions,
                padding_mask=input_dict.get("padding_mask"),
                fill_value=IGNORE_INDEX,
                return_valid_mask=False,
            )
            mtp_padding = ~mtp_valid
            if input_dict.get("padding_mask") is not None:
                mtp_padding = mtp_padding | input_dict["padding_mask"]
            input_dict[f"mtp_tokens_{depth}"] = mtp_tokens
            input_dict[f"mtp_labels_{depth}"] = mtp_labels
            input_dict[f"mtp_valid_{depth}"] = mtp_valid
            input_dict[f"mtp_attention_masks_{depth}"] = self.get_attention_masks(
                positions, mtp_padding
            )
            input_shardings[f"mtp_tokens_{depth}"] = input_shardings["input"]
            input_shardings[f"mtp_labels_{depth}"] = input_shardings["labels"]
            input_shardings[f"mtp_valid_{depth}"] = input_shardings["input"]
            input_shardings[f"mtp_attention_masks_{depth}"] = dsa_mask_placement()
        input_shardings.update(dict.fromkeys(loss_kwargs, input_shardings["labels"]))
        input_dict.update(loss_kwargs)
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

        loss_kwargs = {name: input_dict.pop(name) for name in loss_kwargs}
        inputs = input_dict.pop("input")
        labels = input_dict.pop("labels")
        has_aux_loss = next(self.config.traverse(AuxLoss.Config), None) is not None
        if has_aux_loss or self.mtp_layers:
            input_dict["aux_loss_denominators"] = global_routing_token_counts
        if self.mtp_layers is not None:
            inputs = (inputs, *(input_dict.pop(f"mtp_tokens_{d}") for d in depths))
            labels = (labels, *(input_dict.pop(f"mtp_labels_{d}") for d in depths))
            input_dict["mtp_input_valid_masks"] = tuple(
                input_dict.pop(f"mtp_valid_{d}") for d in depths
            )
            input_dict["mtp_attention_masks"] = tuple(
                input_dict.pop(f"mtp_attention_masks_{d}") for d in depths
            )
        return inputs, labels, input_dict, loss_kwargs

    def forward(  # pyrefly: ignore[bad-param-name-override, bad-override]
        self,
        tokens: torch.Tensor | tuple[torch.Tensor, ...],
        pipeline_indices_TS: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        attention_masks: torch.Tensor | None = None,
        mtp_input_valid_masks: tuple[torch.Tensor, ...] | None = None,
        mtp_attention_masks: tuple[torch.Tensor, ...] | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # Preserve Decoder's positional ``model(tokens, positions)`` API. PP
        # metadata is [T, S], while GLM positions are required to be [T].
        if pipeline_indices_TS is not None and pipeline_indices_TS.ndim == 1:
            if positions is not None:
                raise ValueError("positions were provided twice")
            positions = pipeline_indices_TS
            pipeline_indices_TS = None
        main_tokens = tokens[0] if isinstance(tokens, tuple) else tokens
        if positions is None:
            positions = torch.arange(main_tokens.shape[0], device=main_tokens.device)
        if attention_masks is None:
            attention_masks = self.get_attention_masks(positions, padding_mask)
        hidden_TD = (
            self.tok_embeddings(main_tokens)
            if self.tok_embeddings is not None
            else main_tokens
        )
        index_sources = self.index_sources or tuple(range(len(self.config.layers)))
        local_layers = {int(name) for name in self.layers}
        external_sources = {
            index_sources[layer]
            for layer in local_layers
            if index_sources[layer] not in local_layers
        }
        indices_by_source: dict[int, torch.Tensor] = {}
        aux_loss_denominator = (
            None if aux_loss_denominators is None else aux_loss_denominators[0]
        )
        if external_sources:
            if pipeline_indices_TS is None:
                raise ValueError("this GLM-5 PP stage requires DSA indices")
            source = next(iter(external_sources))
            indices_by_source[source] = pipeline_indices_TS
        for name, layer in self.layers.items():
            layer_id = int(name)
            source = index_sources[layer_id]
            hidden_TD, indices_TS = layer(
                hidden_TD,
                attention_masks,
                positions,
                indices_by_source.get(source),
                return_indices=True,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
            if source == layer_id:
                indices_by_source[source] = indices_TS
        pre_norm_hidden_TD = hidden_TD
        hidden_TD = self.norm(hidden_TD) if self.norm is not None else hidden_TD
        if self.mtp_layers is not None:
            if not isinstance(tokens, tuple) or len(tokens) != len(self.mtp_layers) + 1:
                raise ValueError(
                    "GLM-5 MTP requires one shifted token tensor per depth."
                )
            if mtp_input_valid_masks is None or mtp_attention_masks is None:
                raise ValueError("GLM-5 MTP requires validity masks and DSA masks.")
            if self.tok_embeddings is None or self.lm_head is None:
                raise ValueError("GLM-5 MTP is not supported on a pipeline partition.")
            mtp_outputs: list[torch.Tensor] = []
            prev_hidden = pre_norm_hidden_TD
            depth_inputs = zip(
                self.mtp_layers,
                tokens[1:],
                mtp_input_valid_masks,
                mtp_attention_masks,
                strict=True,
            )
            for depth, (
                layer,
                depth_tokens,
                valid_mask,
                depth_attention_mask,
            ) in enumerate(depth_inputs, 1):
                aux_denom = (
                    None
                    if aux_loss_denominators is None
                    else aux_loss_denominators[depth]
                )
                prev_hidden = layer(
                    self.tok_embeddings(depth_tokens),
                    prev_hidden,
                    valid_mask,
                    depth_attention_mask,
                    positions,
                    padding_mask=padding_mask,
                    aux_loss_denominator=aux_denom,
                )
                mtp_outputs.append(prev_hidden)
            outputs = (hidden_TD, *mtp_outputs)
            if self._skip_lm_head:
                return outputs
            return tuple(self.lm_head(item) for item in outputs)
        if self._skip_lm_head or self.lm_head is None:
            if local_layers:
                next_layer = max(local_layers) + 1
                if next_layer < len(index_sources):
                    source = index_sources[next_layer]
                    if source != next_layer:
                        return hidden_TD, indices_by_source[source]
            return hidden_TD
        return self.lm_head(hidden_TD)
