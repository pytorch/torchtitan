# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import spmd_types as spmd
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import DataParallelMeshDims

from torchtitan.components.loss import (
    BaseLoss,
    ChunkedLossWrapper,
    cross_entropy_loss,
    IGNORE_INDEX,
    LossFunction,
    LossTerm,
)
from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
    pipeline_llm,
    PipelineResult,
    PipelineSharedParameterSpec,
    SharedParameterPipelineRuntime,
)
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ParallelizeFunction
from torchtitan.protocols.module import ModuleList


def roll_mtp_sequence(
    sequence: torch.Tensor,
    *,
    shift: int,
    fill_value: int,
    positions: torch.Tensor | None = None,
    return_valid_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Left-roll an MTP sequence while preserving packed-document boundaries.

    MTP depth ``k`` needs the token or label at ``i + k`` for each position
    ``i``. This helper builds that shifted view along the token axis
    (dimension 0). Tail positions, and positions that would cross a packed
    document boundary, are filled instead of wrapped around.

    Args:
        sequence: Tensor to shift, with shape ``[T, ...]``. Any trailing
            dimensions are carried along unchanged.
        shift: Future-token offset to use. ``shift=1`` maps each position to the
            next token, ``shift=2`` maps to the token after next, and so on.
            Must be positive and no larger than ``seq_len``.
        positions: Optional reset-style position IDs with shape ``[T]``. When
            present, a shifted source position is valid only if
            ``positions[i + shift] == positions[i] + shift``.
            This prevents MTP inputs or labels from crossing packed-document
            boundaries.
        fill_value: Value used for invalid positions. Use token id ``0`` for
            shifted input tokens and ``IGNORE_INDEX`` for shifted labels.
        return_valid_mask: If true, also return a boolean mask marking positions
            where the shifted value came from a valid source position.

    Returns:
        The shifted tensor. If ``return_valid_mask`` is true, returns
        ``(shifted, valid_mask)``.

    Example:
        ``sequence=[A0, A1, A2, B0, B1]`` and
        ``positions=[0, 1, 2, 0, 1]`` with ``shift=1`` returns
        ``[A1, A2, fill, B1, fill]``.
    """
    seq_len = sequence.shape[0]
    if shift <= 0 or shift > seq_len:
        raise ValueError(f"MTP roll shift must be in [1, {seq_len}], got {shift}.")

    rolled = torch.full_like(sequence, fill_value)
    valid_mask = torch.zeros_like(sequence, dtype=torch.bool)

    source = sequence[shift:]
    if positions is None:
        rolled[: seq_len - shift] = source
        valid_mask[: seq_len - shift] = True
        if return_valid_mask:
            return rolled, valid_mask
        return rolled

    if positions.shape[0] < seq_len:
        raise ValueError(
            f"MTP positions need at least {seq_len} tokens, got {positions.shape[0]}."
        )
    valid_tokens = positions[shift:seq_len] == positions[: seq_len - shift] + shift
    # valid_tokens follows positions placement, while valid_mask intentionally
    # follows sequence placement for the following where.
    with spmd.no_typecheck():
        valid_mask[: seq_len - shift] = valid_tokens
    rolled[: seq_len - shift] = torch.where(
        valid_mask[: seq_len - shift],
        source,
        rolled[: seq_len - shift],
    )
    if return_valid_mask:
        return rolled, valid_mask
    return rolled


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
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        mtp_input_valid_mask = mtp_input_valid_mask.unsqueeze(-1).to(
            dtype=prev_embed.dtype
        )
        prev_embed = prev_embed * mtp_input_valid_mask
        h = self.eh_proj(
            torch.cat([self.enorm(mtp_input_embed), self.hnorm(prev_embed)], dim=-1)
        )
        h = h + self.attention(self.attention_norm(h), attention_masks, positions)
        if self.moe_enabled:
            h = h + self.moe(self.ffn_norm(h))
        else:
            h = h + self.feed_forward(self.ffn_norm(h))
        return self.mtp_norm(h)


class MTPDecoder(Decoder):
    """Decoder variant that owns MTP layers.

    MTP is kept as model behavior: the main decoder consumes the normal input
    sequence, and each MTP layer predicts one extra depth from internally shifted
    token embeddings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        mtp_layers: list = field(default_factory=list)

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            if len(self.mtp_layers) <= 0:
                return Decoder.Config.update_from_config(
                    self,
                    config=config,
                    **kwargs,
                )

            num_main_layers = len(self.layers)
            self.layers.extend(self.mtp_layers)
            try:
                Decoder.Config.update_from_config(self, config=config, **kwargs)
            finally:
                del self.layers[num_main_layers:]

            # TODO: Add Context Parallel support for MTP.
            if config.parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "MTP does not support context parallelism yet."
                )

    def __init__(self, config: Config):
        super().__init__(config)
        # num_mtp_layers records the MTP depth configured for the full model.
        self.num_mtp_layers = len(config.mtp_layers)
        # self.mtp_layers holds all MTP layers before PP splitting and only the
        # layers assigned to this model chunk's virtual stage afterward.
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

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        *,
        mtp_source_tokens: torch.Tensor | None = None,
    ):
        """Run an unsplit decoder or one stage of an MTP pipeline.

        Args:
            tokens: Token IDs for an unsplit/first stage, or hidden activations
                for a later pipeline stage.
            positions: Optional token positions.
            attention_masks: Optional attention metadata.
            mtp_source_tokens: Stage-local source token IDs used to construct
                shifted MTP inputs. Pipeline ranks read the same microbatch, so
                this tensor is not transmitted over pipeline edges.

        Returns:
            Hidden activations for an intermediate stage, or the flat MTP
            prediction-and-mask tuple for the final stage.
        """
        if self.num_mtp_layers == 0:
            return super().forward(tokens, positions, attention_masks)

        # The unsplit model and first PP stage receive token IDs [T]; later
        # stages receive hidden activations [T, D] from the preceding stage.
        if tokens.ndim == 1:
            if self.tok_embeddings is None:
                raise ValueError("The first MTP stage requires token embeddings.")
            mtp_source_tokens = tokens
            h = self.tok_embeddings(tokens)
        else:
            h = tokens

        for layer in self.layers.values():
            h = layer(h, attention_masks, positions)

        if not self.mtp_layers:
            h = self.norm(h) if self.norm is not None else h
            if self._skip_lm_head:
                return h
            return self.lm_head(h) if self.lm_head is not None else h

        if mtp_source_tokens is None:
            raise ValueError("The final MTP stage requires original token IDs.")
        if self.tok_embeddings is None:
            raise ValueError("The final MTP stage requires an embedding replica.")

        prev_depth_hidden = h
        h = self.norm(h) if self.norm is not None else h

        mtp_outputs = []
        mtp_valid_masks = []
        for depth, layer in enumerate(self.mtp_layers, 1):
            # NOTE: Without SP, the local main embedding output has shape
            # [tokens, hidden_dim] and could be shifted and reused. Under SP,
            # its token dimension is sharded, so a local shift
            # would be incorrect at shard boundaries. Reuse in that case
            # would require a cross-shard shift or redistribution.
            mtp_input_tokens, mtp_input_valid_mask = roll_mtp_sequence(
                mtp_source_tokens,
                shift=depth,
                positions=positions,
                fill_value=0,
                return_valid_mask=True,
            )
            mtp_input_embed = self.tok_embeddings(mtp_input_tokens)
            prev_depth_hidden = layer(
                mtp_input_embed,
                prev_depth_hidden,
                mtp_input_valid_mask,
                attention_masks,
                positions,
            )
            mtp_outputs.append(prev_depth_hidden)
            mtp_valid_masks.append(mtp_input_valid_mask)

        outputs = (h, *mtp_outputs)
        if self._skip_lm_head:
            predictions = outputs
        else:
            predictions = tuple(
                self.lm_head(item) if self.lm_head is not None else item
                for item in outputs
            )
        return (*predictions, *mtp_valid_masks)


def _generate_mtp_fqn_per_model_part(
    model_config: MTPDecoder.Config,
    num_stages: int,
    num_layers: int,
    input_weight: int,
    output_weight: int,
) -> list[list[str]]:
    """Generate a pipeline layout with MTP modules on the final stage."""
    stages = _generate_llm_fqn_per_model_part(
        num_stages,
        num_layers,
        input_weight,
        output_weight,
    )
    mtp_fqns = tuple(
        f"mtp_layers.{index}" for index in range(len(model_config.mtp_layers))
    )
    stages[-1].extend(mtp_fqns)
    stages[-1].append("tok_embeddings")
    return stages


def _validate_mtp_fqn_per_model_part(
    model_config: MTPDecoder.Config,
    module_fqns_per_stage: list[list[str]],
) -> None:
    """Validate module ownership in a user-defined MTP pipeline layout."""
    final_stage = len(module_fqns_per_stage) - 1
    mtp_fqns = tuple(
        f"mtp_layers.{index}" for index in range(len(model_config.mtp_layers))
    )
    final_only = (*mtp_fqns, "norm", "lm_head")
    for fqn in final_only:
        owners = [
            index for index, stage in enumerate(module_fqns_per_stage) if fqn in stage
        ]
        if owners != [final_stage]:
            raise ValueError(
                f"MTP pipeline module {fqn} must belong only to final stage "
                f"{final_stage}, got owners {owners}."
            )
    embedding_owners = [
        index
        for index, stage in enumerate(module_fqns_per_stage)
        if "tok_embeddings" in stage
    ]
    if embedding_owners != [0, final_stage]:
        raise ValueError(
            "MTP pipeline layouts must place tok_embeddings only on the first "
            f"and final stages, got owners {embedding_owners}."
        )


def _build_mtp_stage_metadata(
    stage_idx: int,
    num_stages: int,
    *,
    training: TrainingConfig,
    model_config: MTPDecoder.Config,
    loss_fn: LossFunction,
) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, ...]]:
    """Build static input and output metadata for one MTP virtual stage."""
    num_tokens = training.num_tokens_per_microbatch_per_dp_rank
    hidden_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    input_args = torch.empty(
        (num_tokens,) if stage_idx == 0 else (num_tokens, model_config.dim),
        dtype=torch.int64 if stage_idx == 0 else hidden_dtype,
        device="meta",
        requires_grad=stage_idx != 0,
    )
    if stage_idx != num_stages - 1:
        output_args = torch.empty(
            (num_tokens, model_config.dim),
            dtype=hidden_dtype,
            device="meta",
            requires_grad=True,
        )
        return input_args, output_args

    # The final stage returns main and MTP predictions, followed by one
    # validity mask per MTP depth. Chunked loss consumes hidden predictions.
    output_dim = (
        model_config.dim
        if isinstance(loss_fn, ChunkedLossWrapper)
        else model_config.vocab_size
    )
    predictions = tuple(
        torch.empty(
            (num_tokens, output_dim),
            dtype=hidden_dtype,
            device="meta",
            requires_grad=True,
        )
        for _ in range(len(model_config.mtp_layers) + 1)
    )
    masks = tuple(
        torch.empty((num_tokens,), dtype=torch.bool, device="meta")
        for _ in model_config.mtp_layers
    )
    return input_args, (*predictions, *masks)


class _MTPPipelineRuntime(SharedParameterPipelineRuntime):
    """Provide final-stage tokens and shared-embedding lifecycle hooks."""

    def prepare_microbatch(
        self,
        inputs: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Make locally loaded token IDs available to the final MTP stage."""
        kwargs["mtp_source_tokens"] = inputs
        return kwargs


def pipeline_deepseek_v3(
    model: torch.nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    device: torch.device,
    model_config: BaseModel.Config,
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> PipelineResult:
    """Build an eager DeepSeek-V3 pipeline with MTP ownership.

    Args:
        model: Complete model before pipeline splitting.
        parallel_dims: Distributed mesh dimensions.
        training: Training shape and dtype configuration.
        parallelism: Parallelism and pipeline schedule configuration.
        compile_config: Model compilation configuration.
        ac_config: Activation-checkpointing configuration.
        dump_folder: Output directory used by parallelization helpers.
        device: Device used to construct pipeline stages.
        model_config: DeepSeek-V3 MTP decoder configuration.
        parallelize_fn: Function applying stage-local parallelisms.
        loss_fn: Loss used by the pipeline schedule.

    Returns:
        Pipeline artifacts with MTP stage ownership and runtime hooks.

    Raises:
        TypeError: If ``model_config`` is not an MTP decoder configuration.
    """
    if not isinstance(model_config, MTPDecoder.Config):
        raise TypeError(
            "pipeline_deepseek_v3 requires MTPDecoder.Config, got "
            f"{type(model_config).__qualname__}."
        )

    num_stages, num_layers, input_weight, output_weight = _get_pipeline_metadata(
        parallel_dims,
        parallelism,
        model_config,
    )
    module_fqns_per_stage = parallelism.module_fqns_per_model_part
    if module_fqns_per_stage is None:
        module_fqns_per_stage = _generate_mtp_fqn_per_model_part(
            model_config,
            num_stages,
            num_layers,
            input_weight,
            output_weight,
        )
    else:
        _validate_mtp_fqn_per_model_part(model_config, module_fqns_per_stage)
    # Pass the MTP layout to the generic builder without mutating user config.
    mtp_parallelism = replace(
        parallelism,
        module_fqns_per_model_part=module_fqns_per_stage,
    )
    result = pipeline_llm(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=mtp_parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
        device=device,
        model_config=model_config,
        parallelize_fn=parallelize_fn,
        loss_fn=loss_fn,
        stage_metadata_fn=partial(
            _build_mtp_stage_metadata,
            training=training,
            model_config=model_config,
            loss_fn=loss_fn,
        ),
    )
    runtime = _MTPPipelineRuntime(
        model_parts=result.model_parts,
        stage_indices=result.stage_indices,
        pp_mesh=parallel_dims.get_mesh("pp"),
        pp_schedule=parallelism.pipeline_parallel_schedule,
        num_stages=num_stages,
        shared_parameter_specs=(
            PipelineSharedParameterSpec(
                fqn="tok_embeddings.weight",
                stage_indices=(0, num_stages - 1),
            ),
        ),
    )
    return replace(result, runtime=runtime)


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
    enable_symm_mem: bool = False,
) -> None:
    mtp_layer_keys = []
    try:
        if model.mtp_layers is not None:
            for i, layer in enumerate(model.mtp_layers):
                key = f"_mtp_{i}"
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
            enable_symm_mem=enable_symm_mem,
        )
    finally:
        for key in mtp_layer_keys:
            del model.layers[key]


def _unpack_mtp_output(
    output: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Split ``(main_pred, *mtp_preds, *valid_masks)`` into two tuples.

    For two MTP layers, returns ``(main_pred, mtp_pred_1, mtp_pred_2)``
    and ``(valid_mask_1, valid_mask_2)``.
    """
    if len(output) < 3 or len(output) % 2 == 0:
        raise ValueError(
            "MTP output must contain one main prediction and matching "
            "auxiliary prediction/mask pairs."
        )
    num_mtp_layers = (len(output) - 1) // 2
    return output[: num_mtp_layers + 1], output[num_mtp_layers + 1 :]


class MTPLoss(BaseLoss):
    """DeepSeek-V3 multi-token prediction loss."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        mtp_scale: float = 0.3
        global_vocab_size: int | None = None
        """Full vocabulary size, needed for spmd_types loss-parallel CE."""

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.fn: LossFunction = cross_entropy_loss
        self._maybe_compile(compile_config)
        self.mtp_scale = config.mtp_scale
        self.global_vocab_size = config.global_vocab_size

    def __call__(
        self,
        pred: tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted main and auxiliary MTP objectives."""
        loss_terms = self._build_loss_terms(pred, labels, **loss_inputs)
        loss: torch.Tensor | None = None
        for loss_term in loss_terms:
            term_loss, _ = self._compute_loss_term(
                loss_term.pred,
                loss_term.labels,
                **loss_term.inputs,
            )
            weighted_loss = term_loss * loss_term.weight
            loss = weighted_loss if loss is None else loss + weighted_loss
        assert loss is not None
        with spmd.no_typecheck():
            if global_valid_tokens is not None:
                loss = loss / global_valid_tokens
        return loss, {}

    def _build_loss_terms(
        self,
        pred: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        **loss_inputs: Any,
    ) -> tuple[LossTerm, ...]:
        """Build main and depth-aligned MTP loss terms before chunking."""
        del loss_inputs
        if not isinstance(pred, tuple):
            raise ValueError(
                "MTPLoss expects MTPDecoder's flat tuple of predictions and "
                "validity masks."
            )
        pred, valid_masks = _unpack_mtp_output(pred)
        mtp_weight = self.mtp_scale / len(valid_masks)
        loss_terms = [
            LossTerm(
                pred[0],
                labels[: pred[0].shape[0]],
            )
        ]
        for label_offset, (mtp_pred, valid_mask) in enumerate(
            zip(pred[1:], valid_masks, strict=True),
            1,
        ):
            mtp_seq_len = mtp_pred.shape[0]
            if labels.shape[0] < mtp_seq_len:
                raise ValueError(
                    f"MTP labels need at least {mtp_seq_len} tokens for depth "
                    f"{label_offset}, got {labels.shape[0]}."
                )
            mtp_labels = roll_mtp_sequence(
                labels[:mtp_seq_len],
                shift=label_offset,
                fill_value=IGNORE_INDEX,
            )
            assert isinstance(mtp_labels, torch.Tensor)
            loss_terms.append(
                LossTerm(
                    mtp_pred,
                    torch.where(
                        valid_mask[:mtp_seq_len],
                        mtp_labels,
                        IGNORE_INDEX,
                    ),
                    weight=mtp_weight,
                )
            )
        return tuple(loss_terms)

    def _compute_loss_term(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute one unnormalized MTP cross-entropy term."""
        del loss_inputs
        return (
            self.fn(
                pred,
                labels,
                global_vocab_size=self.global_vocab_size,
            ),
            {},
        )
