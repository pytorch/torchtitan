# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, cast

import spmd_types as spmd
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import DataParallelMeshDims

from torchtitan.components.loss import CrossEntropyLoss, IGNORE_INDEX
from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import FSDPSymmMemScope, ParallelismConfig
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.parallelism_context import MeshAxisName, ParallelismContext
from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
    PipelineResult,
    PipelineSharedParameter,
    SharedParameterPipelineRuntime,
)
from torchtitan.distributed.spmd_types import (
    annotate_input_spmd_types,
    spmd_dense_sp_enabled,
    spmd_mesh_group,
)
from torchtitan.models.common.attention import (
    AttentionMetadataMap,
    FlexAttentionMetadata,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.module import ModuleList

# Shape suffix legend for MTP decoder tensors:
# T = flattened tokens, D = model dimension, V = vocabulary,
# M = MTP layers, O = objectives (main objective plus MTP objectives).


def roll_mtp_sequence(
    sequence: torch.Tensor,
    *,
    shift: int,
    fill_value: int | bool,
    positions: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    return_valid_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Left-roll an MTP sequence without crossing documents or padding."""
    seq_len = sequence.shape[0]
    if shift <= 0 or shift > seq_len:
        raise ValueError(f"MTP roll shift must be in [1, {seq_len}], got {shift}.")

    shifted = torch.full_like(sequence, fill_value)
    valid_mask = torch.zeros_like(sequence, dtype=torch.bool)
    source = sequence[shift:]
    valid_tokens = torch.ones_like(source, dtype=torch.bool)
    if positions is not None:
        valid_tokens &= positions[shift:seq_len] == positions[: seq_len - shift] + shift
    if padding_mask is not None:
        valid_tokens &= ~padding_mask[: seq_len - shift] & ~padding_mask[shift:seq_len]
    with spmd.no_typecheck():
        valid_mask[: seq_len - shift] = valid_tokens
    shifted[: seq_len - shift] = torch.where(
        valid_mask[: seq_len - shift], source, shifted[: seq_len - shift]
    )

    if return_valid_mask:
        return shifted, valid_mask
    return shifted


def get_mtp_token_counts(
    *,
    target_mask: torch.Tensor,
    positions: torch.Tensor,
    padding_mask: torch.Tensor,
    num_mtp_layers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return loss-token and routed-token counts for the main and MTP depths."""
    loss_token_counts = [target_mask.sum()]
    routing_token_counts = [(~padding_mask).sum()]
    for depth in range(1, num_mtp_layers + 1):
        shifted_target_mask = roll_mtp_sequence(
            target_mask,
            shift=depth,
            positions=positions,
            padding_mask=padding_mask,
            fill_value=False,
        )
        assert isinstance(shifted_target_mask, torch.Tensor)
        _, routing_mask = roll_mtp_sequence(
            padding_mask,
            shift=depth,
            positions=positions,
            padding_mask=padding_mask,
            fill_value=True,
            return_valid_mask=True,
        )
        loss_token_counts.append(shifted_target_mask.sum())
        routing_token_counts.append(routing_mask.sum())
    return torch.stack(loss_token_counts), torch.stack(routing_token_counts)


class MTPTransformerBlock(TransformerBlock):
    """Generic MTP block for decoder-only transformer models.

    The block implements the DeepSeek-V3 style fusion:

    ``eh_proj(cat(enorm(shifted_embedding), hnorm(previous_hidden)))``

    followed by one regular transformer block.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        enorm: RMSNorm.Config
        hnorm: RMSNorm.Config
        eh_proj: Linear.Config
        mtp_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.enorm = config.enorm.build()
        self.hnorm = config.hnorm.build()
        self.eh_proj = config.eh_proj.build()
        self.mtp_norm = config.mtp_norm.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        mtp_input_embed: torch.Tensor,
        prev_embed: torch.Tensor,
        mtp_input_valid_mask: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ):
        mtp_padding_mask_T = ~mtp_input_valid_mask
        if padding_mask is not None:
            mtp_padding_mask_T = mtp_padding_mask_T | padding_mask
        # Under SP, prev_embed already arrives Shard(0) from the preceding
        # decoder or MTP block, while the validity mask arrives replicated.
        # The old module boundary implicitly sharded only this mask; do that
        # explicitly here. The MoE owns padding-mask sharding for its branch.
        local_mtp_input_valid_mask_T = self._maybe_shard_mtp_valid_mask_across_tp(
            mtp_input_valid_mask
        )
        prev_embed = prev_embed * local_mtp_input_valid_mask_T.unsqueeze(-1).to(
            dtype=prev_embed.dtype
        )
        h = self.eh_proj(
            torch.cat([self.enorm(mtp_input_embed), self.hnorm(prev_embed)], dim=-1)
        )
        h = h + self.attention(self.attention_norm(h), attention_metadata, positions)
        if self.moe_enabled:
            h = h + self.moe(
                self.ffn_norm(h),
                padding_mask_T=mtp_padding_mask_T,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            h = h + self.feed_forward(self.ffn_norm(h))
        return self.mtp_norm(h)

    def _maybe_shard_mtp_valid_mask_across_tp(
        self, mtp_input_valid_mask_T: torch.Tensor
    ) -> torch.Tensor:
        """Shard the validity mask to match sequence-parallel MTP activations."""
        if not spmd_dense_sp_enabled():
            return mtp_input_valid_mask_T
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is None:
            return mtp_input_valid_mask_T
        return spmd.redistribute(
            mtp_input_valid_mask_T,
            tp_group,
            src=spmd.R,
            dst=spmd.S(0),
            backward_options={"op_dtype": mtp_input_valid_mask_T.dtype},
        )


class MTPDecoder(Decoder):
    """Decoder variant that owns MTP layers.

    MTP is kept as model behavior: the main decoder consumes the normal input
    sequence, and each MTP layer predicts one extra depth from preprocessed
    shifted token embeddings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        mtp_layers: list = field(default_factory=list)

    def __init__(self, config: Config):
        super().__init__(config)
        self._num_mtp_layers = len(config.mtp_layers)
        if not config.mtp_layers:
            self.mtp_layers = None
            return

        self.mtp_layers = ModuleList()
        for layer_config in config.mtp_layers:
            if not isinstance(layer_config, MTPTransformerBlock.Config):
                raise ValueError(
                    "MTPDecoder requires Config.mtp_layers to contain "
                    "MTPTransformerBlock.Config instances."
                )
            self.mtp_layers.append(layer_config.build())

    @property
    def num_mtp_layers(self) -> int:
        """Return configured MTP depth, including on pruned pipeline stages."""
        configured = getattr(self, "_num_mtp_layers", None)
        if configured is not None:
            return configured
        return 0 if self.mtp_layers is None else len(self.mtp_layers)

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torchtitan.distributed.fsdp import (
            resolve_fsdp_mesh,
            resolve_sparse_fsdp_mesh,
        )

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)
        apply_fsdp_to_mtp_decoder(
            self,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallelism_context.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallelism_context.ep,
            edp_mesh=edp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh_dims=edp_mesh_dims,
            symm_mem_scope=parallelism.fsdp_symm_mem_scope,
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
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...],
        torch.Tensor | tuple[torch.Tensor, ...],
        dict[str, Any],
    ]:
        """Prepare aligned pairs before applying CP sharding and annotations."""
        del kwargs
        tokens_T = input_dict["input"]
        labels_T = input_dict["labels"]
        positions_T = input_dict.get("positions")
        padding_mask_T = input_dict.get("padding_mask")
        if self.num_mtp_layers > 0 and positions_T is None:
            raise ValueError("MTP input preprocessing requires positions.")

        depths = range(1, self.num_mtp_layers + 1)
        input_shardings = decoder_input_sharding()
        for depth in depths:
            mtp_input_tokens_T, mtp_input_valid_mask_T = roll_mtp_sequence(
                tokens_T,
                shift=depth,
                positions=positions_T,
                padding_mask=padding_mask_T,
                fill_value=0,
                return_valid_mask=True,
            )
            mtp_labels_T = roll_mtp_sequence(
                labels_T,
                shift=depth,
                positions=positions_T,
                padding_mask=padding_mask_T,
                fill_value=IGNORE_INDEX,
                return_valid_mask=False,
            )
            input_dict[f"mtp_input_tokens_{depth}"] = mtp_input_tokens_T
            input_dict[f"mtp_labels_{depth}"] = mtp_labels_T
            input_dict[f"mtp_input_valid_mask_{depth}"] = mtp_input_valid_mask_T
            input_shardings[f"mtp_input_tokens_{depth}"] = input_shardings["input"]
            input_shardings[f"mtp_labels_{depth}"] = input_shardings["labels"]
            input_shardings[f"mtp_input_valid_mask_{depth}"] = input_shardings["input"]

        if positions_T is not None:
            attention_metadata = self._get_attention_metadata(
                positions=positions_T,
                padding_mask=padding_mask_T,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            input_dict["attention_metadata"] = attention_metadata

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

        main_tokens_T = input_dict.pop("input")
        main_labels_T = input_dict.pop("labels")
        input_dict["aux_loss_denominators"] = None
        if self.num_mtp_layers == 0:
            return main_tokens_T, main_labels_T, input_dict

        mtp_input_tokens_MT = tuple(
            input_dict.pop(f"mtp_input_tokens_{depth}") for depth in depths
        )
        loss_labels_OT = (
            main_labels_T,
            *(input_dict.pop(f"mtp_labels_{depth}") for depth in depths),
        )
        mtp_input_valid_masks_MT = tuple(
            input_dict.pop(f"mtp_input_valid_mask_{depth}") for depth in depths
        )
        if getattr(parallelism_context, "pp_enabled", False):
            # Pipeline metadata inference accepts tensor leaves. Keep MTP depth
            # on a leading dimension and restore tuples in the final stage.
            input_dict["mtp_input_tokens"] = torch.stack(mtp_input_tokens_MT)
            input_dict["mtp_input_valid_masks"] = torch.stack(mtp_input_valid_masks_MT)
            return main_tokens_T, torch.stack(loss_labels_OT), input_dict
        input_dict["mtp_input_valid_masks"] = mtp_input_valid_masks_MT
        return (main_tokens_T, *mtp_input_tokens_MT), loss_labels_OT, input_dict

    def forward(
        self,
        tokens: torch.Tensor | tuple[torch.Tensor, ...],
        positions: torch.Tensor | None = None,
        attention_metadata: AttentionMetadataMap | None = None,
        mtp_input_tokens: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        mtp_input_valid_masks: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ):
        if self.num_mtp_layers == 0:
            if not isinstance(tokens, torch.Tensor):
                raise ValueError("A decoder without MTP expects one token tensor.")
            return super().forward(
                tokens,
                positions,
                attention_metadata,
                padding_mask=padding_mask,
                aux_loss_denominators=aux_loss_denominators,
            )
        if isinstance(tokens, tuple):
            if len(tokens) != self.num_mtp_layers + 1:
                raise ValueError(
                    "MTP decoder requires one main token tensor and one prepared "
                    "token tensor per MTP layer."
                )
            main_tokens_T, *local_mtp_input_tokens_T = tokens
            mtp_input_tokens = tuple(local_mtp_input_tokens_T)
            if self.tok_embeddings is None:
                raise ValueError("MTP decoder forward requires token embeddings.")
            hidden_TD = self.tok_embeddings(main_tokens_T)
        elif tokens.ndim == 1:
            if self.tok_embeddings is None:
                raise ValueError("The first MTP pipeline stage requires embeddings.")
            hidden_TD = self.tok_embeddings(tokens)
        else:
            hidden_TD = tokens

        if (
            aux_loss_denominators is not None
            and aux_loss_denominators.numel() != self.num_mtp_layers + 1
        ):
            raise ValueError("Expected one aux-loss denominator per MTP objective.")
        with spmd.no_typecheck():
            main_aux_loss_denominator = (
                None if aux_loss_denominators is None else aux_loss_denominators[0]
            )
        for layer in self.layers.values():
            layer_attention_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.get(
                    cast(TransformerBlock, layer).attention.attention_metadata_key
                )
            )
            hidden_TD = layer(
                hidden_TD,
                layer_attention_metadata,
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=main_aux_loss_denominator,
            )

        if not self.mtp_layers:
            hidden_TD = self.norm(hidden_TD) if self.norm is not None else hidden_TD
            if self._skip_lm_head:
                return hidden_TD
            return self.lm_head(hidden_TD) if self.lm_head is not None else hidden_TD

        if self.tok_embeddings is None:
            raise ValueError("The final MTP pipeline stage requires embeddings.")
        if isinstance(mtp_input_tokens, torch.Tensor):
            mtp_input_tokens = tuple(mtp_input_tokens.unbind(0))
        if mtp_input_tokens is None or len(mtp_input_tokens) != len(self.mtp_layers):
            raise ValueError("MTP decoder requires one token tensor per MTP layer.")
        if isinstance(mtp_input_valid_masks, torch.Tensor):
            mtp_input_valid_masks = tuple(mtp_input_valid_masks.unbind(0))
        if mtp_input_valid_masks is None or len(mtp_input_valid_masks) != len(
            self.mtp_layers
        ):
            raise ValueError("MTP decoder requires one validity mask per MTP layer.")

        prev_depth_hidden_TD = hidden_TD
        hidden_TD = self.norm(hidden_TD) if self.norm is not None else hidden_TD

        mtp_outputs_TD = []
        for depth, (layer, depth_tokens_T, mtp_input_valid_mask_T) in enumerate(
            zip(
                self.mtp_layers,
                mtp_input_tokens,
                mtp_input_valid_masks,
                strict=True,
            ),
            1,
        ):
            mtp_input_embed_TD = self.tok_embeddings(depth_tokens_T)
            layer_attention_metadata = (
                None
                if attention_metadata is None
                else attention_metadata.get(
                    cast(TransformerBlock, layer).attention.attention_metadata_key
                )
            )
            with spmd.no_typecheck():
                aux_loss_denominator = (
                    None
                    if aux_loss_denominators is None
                    else aux_loss_denominators[depth]
                )
            prev_depth_hidden_TD = layer(
                mtp_input_embed_TD,
                prev_depth_hidden_TD,
                mtp_input_valid_mask_T,
                layer_attention_metadata,
                positions,
                padding_mask=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
            mtp_outputs_TD.append(prev_depth_hidden_TD)

        outputs_TD = (hidden_TD, *mtp_outputs_TD)
        if self._skip_lm_head:
            predictions_TD = outputs_TD
        else:
            predictions_TV = tuple(
                self.lm_head(item) if self.lm_head is not None else item
                for item in outputs_TD
            )
            return predictions_TV
        return predictions_TD


def _mtp_pipeline_parallelism(
    model: MTPDecoder,
    *,
    parallelism_context: ParallelismContext,
    parallelism: ParallelismConfig,
    model_config: MTPDecoder.Config,
    **kwargs: Any,
) -> tuple[ParallelismConfig, int]:
    """Return an MTP-valid pipeline layout without mutating user config."""
    del kwargs
    configured_layout = parallelism.pipeline_parallel_module_fqns_per_model_part
    if configured_layout is None:
        num_stages, num_layers, input_weight, output_weight = _get_pipeline_metadata(
            parallelism_context,
            parallelism,
            model_config,
        )
        layout = _generate_llm_fqn_per_model_part(
            num_stages,
            num_layers,
            input_weight,
            output_weight,
        )
        layout[-1].extend(
            [
                *(f"mtp_layers.{index}" for index in range(model.num_mtp_layers)),
                "tok_embeddings",
            ]
        )
    else:
        layout = copy.deepcopy(configured_layout)
        num_stages = len(layout)

    embedding_stages = [
        stage_index
        for stage_index, module_fqns in enumerate(layout)
        if "tok_embeddings" in module_fqns
    ]
    expected_embedding_stages = [0, num_stages - 1]
    if embedding_stages != expected_embedding_stages:
        raise ValueError(
            "MTP PP requires tok_embeddings on exactly the first and last "
            f"virtual stages, got {embedding_stages}."
        )
    for index in range(model.num_mtp_layers):
        fqn = f"mtp_layers.{index}"
        owner_stages = [
            stage_index
            for stage_index, module_fqns in enumerate(layout)
            if fqn in module_fqns
        ]
        if owner_stages != [num_stages - 1]:
            raise ValueError(
                f"MTP PP requires {fqn} on exactly the final virtual stage, "
                f"got {owner_stages}."
            )
    for fqn in ("norm", "lm_head"):
        owner_stages = [
            stage_index
            for stage_index, module_fqns in enumerate(layout)
            if fqn in module_fqns
        ]
        if owner_stages != [num_stages - 1]:
            raise ValueError(
                f"MTP PP requires {fqn} on exactly the final virtual stage, "
                f"got {owner_stages}."
            )

    parallelism = copy.copy(parallelism)
    parallelism.pipeline_parallel_module_fqns_per_model_part = layout
    return parallelism, num_stages


def _with_mtp_pipeline_runtime(
    pipeline: PipelineResult,
    *,
    num_stages: int,
    parallelism_context: ParallelismContext,
    parallelism: ParallelismConfig,
) -> PipelineResult:
    """Return pipeline artifacts with the MTP shared-embedding lifecycle."""
    pp_mesh = parallelism_context.get_mesh("pp")
    runtime = SharedParameterPipelineRuntime(
        model_parts=pipeline.model_parts,
        stage_indices=pipeline.stage_indices,
        pp_mesh=pp_mesh,
        pp_schedule=parallelism.pipeline_parallel_schedule,
        num_stages=num_stages,
        shared_parameters=(
            PipelineSharedParameter(
                fqn="tok_embeddings.weight",
                stage_indices=(0, num_stages - 1),
            ),
        ),
    )
    return replace(pipeline, runtime=runtime)


def apply_fsdp_to_mtp_decoder(
    model: MTPDecoder,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    dp_mesh_dims: DataParallelMeshDims | None = None,
    edp_mesh_dims: DataParallelMeshDims | None = None,
    symm_mem_scope: FSDPSymmMemScope = None,
) -> None:
    mtp_layer_keys = []
    try:
        if model.mtp_layers is not None:
            first_mtp_layer_id = len(model.layers)
            for i, layer in enumerate(model.mtp_layers):
                key = str(first_mtp_layer_id + i)
                model.layers[key] = layer
                mtp_layer_keys.append(key)

        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            pp_enabled=pp_enabled,
            cpu_offload=cpu_offload,
            reshard_after_forward_policy=reshard_after_forward_policy,
            ep_degree=ep_degree,
            edp_mesh=edp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh_dims=edp_mesh_dims,
            symm_mem_scope=symm_mem_scope,
        )
    finally:
        for key in mtp_layer_keys:
            del model.layers[key]


class MTPLoss(CrossEntropyLoss):
    """DeepSeek-V3 weighted multi-term cross-entropy objective."""

    @dataclass(kw_only=True, slots=True)
    class Config(CrossEntropyLoss.Config):
        mtp_scale: float = 0.3

    def __init__(self, config: Config):
        super().__init__(config)
        self.mtp_scale = config.mtp_scale

    def __call__(
        self,
        pred: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor | tuple[torch.Tensor, ...],
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted objective from aligned prediction/label pairs."""
        del loss_inputs
        if not isinstance(pred, tuple):
            raise ValueError("MTPLoss expects a prediction tuple.")
        label_tensors_T: tuple[torch.Tensor, ...]
        if isinstance(labels, torch.Tensor):
            if labels.ndim == 0 or labels.shape[0] != len(pred):
                raise ValueError(
                    "MTPLoss requires the leading label dimension to match "
                    f"the {len(pred)} predictions, got shape {labels.shape}."
                )
            label_tensors_T = tuple(labels.unbind(0))
        else:
            label_tensors_T = labels
        if len(pred) != len(label_tensors_T):
            raise ValueError(
                "MTPLoss requires one labels tensor per prediction, "
                f"got {len(pred)} predictions and {len(label_tensors_T)} labels."
            )
        num_mtp_layers = len(pred) - 1
        if num_mtp_layers <= 0:
            raise ValueError(
                "MTPLoss expects a main prediction and at least one auxiliary "
                "prediction."
            )
        mtp_weight = self.mtp_scale / num_mtp_layers
        if global_loss_token_counts is not None and global_loss_token_counts.ndim != 1:
            raise ValueError(
                "MTPLoss requires a per-objective global_loss_token_counts vector."
            )
        if (
            global_loss_token_counts is not None
            and global_loss_token_counts.numel() != len(pred)
        ):
            raise ValueError(
                "MTPLoss requires one denominator per prediction, "
                f"got {global_loss_token_counts.numel()} for {len(pred)} predictions."
            )
        if global_loss_token_counts is None:
            loss_token_counts: tuple[torch.Tensor | None, ...] = (None,) * len(pred)
        else:
            with spmd.no_typecheck():
                loss_token_counts = tuple(
                    count.clamp_min(1) for count in global_loss_token_counts.unbind()
                )
        main_loss, _ = super().__call__(
            pred[0], label_tensors_T[0], loss_token_counts[0]
        )
        mtp_loss = pred[0].new_zeros((), dtype=torch.float32)
        if spmd.is_type_checking():
            mtp_loss = spmd.mutate_type(
                mtp_loss,
                src=spmd.R,
                dst={"dp": spmd.P, "cp": spmd.P, "tp": spmd.I},
            )
        for depth, (mtp_pred, mtp_labels) in enumerate(
            zip(pred[1:], label_tensors_T[1:], strict=True), 1
        ):
            depth_loss, _ = super().__call__(
                mtp_pred,
                mtp_labels,
                loss_token_counts[depth],
            )
            mtp_loss = mtp_loss + depth_loss * mtp_weight
        loss = main_loss + mtp_loss
        return loss, {}
