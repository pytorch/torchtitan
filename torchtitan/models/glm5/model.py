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
3. ``Glm5Attention`` prepares MLA's low-rank Q/KV representations, asks the
   ``Glm5DsaIndexer`` which keys each query may attend to, and delegates
   score/value computation to ``Glm5SparseAttention`` in ``dsa.py``.
4. ``dsa.py`` contains the indexer and selected-key gathering.

The model uses token-first tensors because TorchTitan packs microbatches into a
single token dimension before model forward. Distributed placement is declared
separately in ``sharding.py``; the formulas below describe global logical
shapes. Distributed adaptation of this sparse path is a separate step.
"""

from dataclasses import dataclass
from typing import Any, Literal

import spmd_types as spmd

import torch
import torch_remat as remat

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.flops import quadratic_attention_flops_per_token
from torchtitan.models.common.linear import maybe_gather_tp_input
from torchtitan.models.common.moe import MoE
from torchtitan.models.deepseek_v3.attention import DeepSeekMLAAttention
from torchtitan.models.deepseek_v3.model import (
    DeepSeekV3Model,
    DeepSeekV3TransformerBlock,
)
from torchtitan.models.deepseek_v3.mtp import roll_mtp_sequence

from .dsa import DsaMetadata, Glm5DsaIndexer, Glm5SparseAttention
from .state_dict_adapter import Glm5StateDictAdapter

# Tensor dimensions used in this file:
# T/Q/K: token, query-token, and key-token dimensions.
# D/N/H/R/P/V: model, head-count, head, RoPE, pass-through, and value dimensions.


class Glm5Attention(DeepSeekMLAAttention):
    """Multi-head latent attention (MLA) gated by DSA top-k indices.

    Q follows ``D -> q_lora_rank -> N * (nope + rope)``. KV follows
    ``D -> (kv_lora_rank + rope)`` and remains a single-head latent pool.
    The indexer consumes the same token states and returns ``[T, topk]`` key
    indices. ``inner_attention`` applies token-selected sparse attention to
    absorbed K/query contraction and the post-attention V projection.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekMLAAttention.Config):
        indexer: Glm5DsaIndexer.Config | None
        index_topk: int

        def flops_per_token(self, seq_len: int) -> int:
            """Useful absorbed attention FLOPs, excluding remat and kernel overhead."""
            leaf = self.inner_attention.inner_attention
            assert isinstance(leaf, Glm5SparseAttention.Config)
            attended_keys = (
                seq_len
                if leaf.training_stage == "warmup"
                else min(seq_len, self.index_topk)
            )
            return quadratic_attention_flops_per_token(
                num_heads=self.n_heads,
                qk_head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
                v_head_dim=self.kv_lora_rank,
                seq_len=attended_keys,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.indexer = config.indexer.build() if config.indexer is not None else None

    # DSA specializes the parent's attention metadata contract.
    def forward(  # pyrefly: ignore[bad-override]
        self,
        x_TD: torch.Tensor,
        attention_masks: DsaMetadata,
        positions: torch.Tensor | None = None,
        *,
        topk_indices_TS: torch.Tensor | None = None,
        prepared_query: tuple[torch.Tensor, torch.Tensor] | None = None,
        indexer_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_TD, q_resid_TR = (
            self.prepare_query(x_TD) if prepared_query is None else prepared_query
        )
        T = x_TD.shape[0]
        if positions is None:
            positions = torch.arange(T, device=x_TD.device)

        q_TNH = self.wq_b(q_resid_TR)
        # The head view/split reads the saved projection output with bare ops.
        remat.recompute_needs_tensor(q_TNH)
        with spmd.local():
            q_TNH = q_TNH.view(T, -1, self.qk_head_dim)
        # KV path: keep content in a compressed latent representation and keep
        # one shared rotary key. The backend absorbs K into Q and projects V
        # after attention, without expanding the key/value sequence per head.
        compressed_kv_TC = self.wkv_a(x_TD)
        # The latent split and RMSNorm read this projection with bare ops.
        remat.recompute_needs_tensor(compressed_kv_TC)
        kv_TR, k_rope_TR = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_TR = self.kv_norm(kv_TR)
        q_TNH, k_rope_T1R = self._apply_rope_to_q_and_k_pe(q_TNH, k_rope_TR, positions)
        with spmd.local():
            k_shared_TR = k_rope_T1R.squeeze(1)
        # Selection and value computation are separate on purpose: the indexer
        # is discrete; the optional KL objective trains its feature projections.
        if topk_indices_TS is None:
            if self.indexer is None:
                raise ValueError("shared-indexer attention requires top-k indices")
            selection = self.indexer(x_TD, q_resid_TR, positions, attention_masks)
            if isinstance(selection, tuple):
                topk_indices_TS, indexer_features = selection
            else:
                topk_indices_TS = selection
        output_TNV = self.inner_attention(
            q_TNH,
            kv_TR,
            k_shared_TR,
            attention_metadata=attention_masks,
            topk_indices_QS=topk_indices_TS,
            scale=self.softmax_scale,
            indexer_features=indexer_features,
            aux_loss_denominator=aux_loss_denominator,
        )
        # The contiguous view reads the attention output with bare ops.
        remat.recompute_needs_tensor(output_TNV)
        output_TD: torch.Tensor = self.wo(output_TNV.contiguous().view(T, -1))
        return output_TD

    def prepare_query(self, x_TD):
        """Share the gathered input and normalized query with the indexer."""
        gathered_TD = maybe_gather_tp_input(self, x_TD)
        q_resid_TR = self.wq_a(gathered_TD)
        remat.recompute_needs_tensor(q_resid_TR)
        return gathered_TD, self.q_norm(q_resid_TR)


class Glm5TransformerBlock(DeepSeekV3TransformerBlock):
    """GLM-5 decoder block with either a dense FFN or common MoE."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3TransformerBlock.Config):
        pass

    # Index reuse is internal to GLM blocks, not a shared Module tuple hook.
    def forward(  # pyrefly: ignore[bad-override]
        self,
        x_TD: torch.Tensor,
        attention_masks: DsaMetadata,
        positions: torch.Tensor | None = None,
        topk_indices_TS: torch.Tensor | None = None,
        *,
        return_indices: bool = False,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        normalized_TD = self.attention_norm(x_TD)
        assert isinstance(self.attention, Glm5Attention)
        prepared = self.attention.prepare_query(normalized_TD)
        indexer_features = None
        if topk_indices_TS is None:
            if self.attention.indexer is None:
                raise ValueError("shared-indexer block requires producer indices")
            selection = self.attention.indexer(*prepared, positions, attention_masks)
            if isinstance(selection, tuple):
                topk_indices_TS, indexer_features = selection
            else:
                topk_indices_TS = selection
        attention_TD = self.attention(
            normalized_TD,
            attention_masks,
            positions,
            topk_indices_TS=topk_indices_TS,
            prepared_query=prepared,
            indexer_features=indexer_features,
            aux_loss_denominator=aux_loss_denominator,
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
        # Trailing add, always saved: it saves nothing for backward, so replay skips
        # it and its inputs need no persisting, matching checkpoint early stop.
        output_TD: torch.Tensor = remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(x_TD, ffn_output_TD)
        if return_indices:
            assert topk_indices_TS is not None
            return output_TD, topk_indices_TS
        return output_TD


class Glm5Model(DeepSeekV3Model):
    """GLM-5 decoder with gathered DSA attention.

    The inherited decoder accepts token ids on the first PP stage and hidden
    states on later stages. A PP stage may own zero decoder layers, which is why
    mask construction and forward preserve embedding-only/output-only stages.
    """

    state_dict_adapter_cls = Glm5StateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Model.Config):
        dim: int = 6144
        vocab_size: int = 154880
        index_sources: tuple[int, ...] = ()
        indexer_training_stage: Literal["frozen", "warmup", "sparse"] = "frozen"

        def __post_init__(self) -> None:
            super(Glm5Model.Config, self).__post_init__()
            self.validate_indexer_training()
            if self.index_sources:
                if len(self.index_sources) != len(self.layers):
                    raise ValueError("index_sources must contain one source per layer")
                for layer, source in enumerate(self.index_sources):
                    if not 0 <= source <= layer or self.index_sources[source] != source:
                        raise ValueError(
                            "index source must be a current/earlier producer layer"
                        )
                    # PP forwards one live producer's indices across each boundary.
                    if layer and source < self.index_sources[layer - 1]:
                        raise ValueError("index sharing groups must be contiguous")

        def validate_indexer_training(self) -> None:
            stage = self.indexer_training_stage
            if stage not in ("frozen", "warmup", "sparse"):
                raise ValueError("Unknown GLM indexer training stage")
            for layer in self.layers + list(self.mtp_layers or ()):
                attention = layer.attention
                leaf = attention.inner_attention.inner_attention
                if leaf.training_stage != stage:
                    raise ValueError("GLM model and attention indexer stages disagree")
                indexer = attention.indexer
                if indexer is not None and indexer.training_stage != stage:
                    raise ValueError("GLM model and indexer stages disagree")
                needs_loss = indexer is not None and stage != "frozen"
                if (leaf.indexer_loss is not None) != needs_loss:
                    raise ValueError("GLM indexer stage and auxiliary loss disagree")

        def build_flops_estimator(self, model, *, seq_len, excluded_submodules=()):
            from torchtitan.models.common.flops import active_parameter_flops_per_token

            excluded_submodules = tuple(excluded_submodules)
            excluded_ids = {
                id(child)
                for parent in excluded_submodules
                if parent is not None
                for child in parent.modules()
            }
            indexers = [
                module
                for module in model.modules()
                if isinstance(module, Glm5DsaIndexer) and id(module) not in excluded_ids
            ]
            excluded = (*excluded_submodules, *indexers)
            base = super(Glm5Model.Config, self).build_flops_estimator(
                model, seq_len=seq_len, excluded_submodules=excluded
            )
            extra = 0
            for module in indexers:
                projection_flops = active_parameter_flops_per_token(
                    module, excluded_submodules=excluded_submodules
                )
                extra += (
                    projection_flops // 3
                    if module.training_stage == "frozen"
                    else projection_flops
                )
                # Discrete selection is forward-only even in trainable stages.
                # Count QK and head mixing over the full document, not top-k.
                if module.training_stage != "warmup":
                    extra += 2 * module.n_heads * (module.head_dim + 1) * seq_len
            for attention in model.modules():
                if (
                    isinstance(attention, Glm5Attention)
                    and attention.indexer in indexers
                    and self.indexer_training_stage != "frozen"
                ):
                    assert attention.indexer is not None
                    idx = attention.indexer
                    selected = (
                        seq_len
                        if self.indexer_training_stage == "warmup"
                        else min(seq_len, idx.topk.index_topk)
                    )
                    extra += 6 * idx.n_heads * (idx.head_dim + 1) * selected
                    extra += (
                        2 * attention.n_heads
                        * (attention.kv_lora_rank + attention.qk_rope_head_dim)
                        * selected
                    )
            if self.indexer_training_stage == "warmup":
                # Frozen backbone projections still propagate carrier/input gradients.
                extra -= (
                    active_parameter_flops_per_token(
                        model, excluded_submodules=excluded
                    )
                    // 3
                )
                if model.lm_head is not None:
                    extra -= (
                        2
                        * len(self.mtp_layers or ())
                        * sum(p.numel() for p in model.lm_head.parameters())
                    )
            return lambda batch: base(batch) + extra * batch["input"].numel()

    def get_attention_masks(
        self,
        positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> DsaMetadata:
        # Like Decoder's CP lifecycle, empty PP stages still prepare metadata
        # from the global input and retained full-model config.
        if positions.ndim != 1:
            raise ValueError("GLM-5 positions must have shape [T].")
        return DsaMetadata.from_positions(positions, padding_mask)

    def __init__(self, config):
        # Overrides mutate nested configs after their dataclass construction.
        # Reject inconsistent stages before freezing any parameters.
        config.validate_indexer_training()
        super().__init__(config)
        if config.indexer_training_stage == "warmup":
            # AuxLoss uses the shared backward carrier contract. Freezing weights
            # does not remove activation backward; a KL-only training driver is a
            # separate optimization (PR5578 discussion_r4545630656).
            self.requires_grad_(False)
            for module in self.modules():
                if isinstance(module, Glm5DsaIndexer):
                    module.requires_grad_(True)
        self.index_sources = config.index_sources

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context,
        parallelism,
        global_routing_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        """Build compact document metadata and prepare rank-local CP queries."""
        del kwargs
        loss_kwargs = input_dict.pop("loss_kwargs", {})
        positions = input_dict.get("positions")
        if positions is not None:
            input_dict["attention_masks"] = self.get_attention_masks(
                positions,
                input_dict.get("padding_mask"),
            )

        input_shardings = decoder_input_sharding()
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
        input_shardings.update(dict.fromkeys(loss_kwargs, input_shardings["labels"]))
        input_dict.update(loss_kwargs)
        if parallelism_context.cp_enabled:
            backend = self.config.layers[0].attention.inner_attention._owner
            input_dict["attention_metadata"] = {
                backend: input_dict.pop("attention_masks")
            }
            input_dict = self._cp_shard(
                input_dict,
                input_shardings=input_shardings,
                parallelism_context=parallelism_context,
                parallelism=parallelism,
            )
            metadata = input_dict.pop("attention_metadata")[backend]
            input_dict["attention_masks"] = metadata
            for depth in depths:
                key = f"mtp_attention_masks_{depth}"
                input_dict[key] = backend.prepare_cp_metadata(
                    input_dict[key], permutation=None
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
        attention_masks: DsaMetadata | None = None,
        mtp_input_valid_masks: tuple[torch.Tensor, ...] | None = None,
        mtp_attention_masks: tuple[DsaMetadata, ...] | None = None,
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
            if len(external_sources) != 1:
                raise ValueError(
                    "GLM pipeline boundary requires exactly one external index source"
                )
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
