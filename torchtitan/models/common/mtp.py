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

from torchtitan.components.loss import CrossEntropyLoss, IGNORE_INDEX, Loss
from torchtitan.config import PrecisionConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.base.model import FlopsEstimator
from torchtitan.models.base.module import Module, ModuleList
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.decoder_sharding import decoder_input_sharding


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


class MTPDecoder(Decoder):
    """Decoder variant that owns MTP layers.

    MTP is kept as model behavior: the main decoder consumes the normal input
    sequence, and each MTP layer predicts one extra depth from preprocessed
    shifted token embeddings.

    Without MTP layers this is a plain ``Decoder``: every override below first
    returns the ``Decoder`` implementation, so the rest of each method only
    handles the MTP case. Subclasses own the MTP blocks and the MTP forward.
    """

    mtp_layers: ModuleList[TransformerBlock]

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        mtp_layers: list[TransformerBlock.Config] = field(default_factory=list)

        def build_flops_estimator(
            self,
            model: Module,
            *,
            seq_len: int,
            excluded_submodules: tuple[Module | None, ...] = (),
        ) -> FlopsEstimator:
            decoder_estimate = super(MTPDecoder.Config, self).build_flops_estimator(
                model, seq_len=seq_len, excluded_submodules=excluded_submodules
            )
            if not self.mtp_layers:
                return decoder_estimate
            model = cast(MTPDecoder, model)
            repeated_head_flops_per_token = (
                6
                * len(self.mtp_layers)
                * sum(param.numel() for param in model.lm_head.parameters())
            )
            return lambda input_dict: (
                decoder_estimate(input_dict)
                + repeated_head_flops_per_token
                * int((~input_dict["padding_mask"]).sum())
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.mtp_layers = ModuleList(
            layer_config.build() for layer_config in config.mtp_layers
        )

    def _remat_block_container_fqns(self) -> list[str]:
        if not self.mtp_layers:
            return super()._remat_block_container_fqns()
        return [*super()._remat_block_container_fqns(), "mtp_layers"]

    def _apply_fsdp(
        self,
        *,
        parallelism_context: ParallelismContext,
        precision: PrecisionConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        if not self.mtp_layers:
            return super()._apply_fsdp(
                parallelism_context=parallelism_context,
                precision=precision,
                parallelism=parallelism,
            )
        if parallelism.fsdp_unshard_in_current_stream:
            # tok_embeddings runs once per MTP depth, which the explicit
            # prefetching chain in apply_fsdp_to_decoder does not model.
            raise ValueError(
                "parallelism.fsdp_unshard_in_current_stream is not supported for models "
                "with MTP layers."
            )
        # Wrap the MTP layers like decoder layers by appending them to
        # self.layers for the duration of the Decoder FSDP pass.
        mtp_layer_keys = []
        try:
            first_mtp_layer_id = len(self.layers)
            for i, layer in enumerate(self.mtp_layers):
                key = str(first_mtp_layer_id + i)
                self.layers[key] = layer
                mtp_layer_keys.append(key)
            super()._apply_fsdp(
                parallelism_context=parallelism_context,
                precision=precision,
                parallelism=parallelism,
            )
        finally:
            for key in mtp_layer_keys:
                del self.layers[key]

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context: ParallelismContext,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        global_routing_token_counts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...],
        torch.Tensor | tuple[torch.Tensor, ...],
        dict[str, Any],
        dict[str, Any],
    ]:
        """Prepare aligned pairs before applying CP sharding and annotations."""
        if not self.mtp_layers:
            return super().preprocess_inputs(
                input_dict,
                parallelism_context=parallelism_context,
                parallelism=parallelism,
                max_num_documents=max_num_documents,
                global_routing_token_counts=global_routing_token_counts,
                **kwargs,
            )
        del kwargs
        loss_kwargs = input_dict.pop("loss_kwargs", {})
        tokens = input_dict["input"]
        labels = input_dict["labels"]
        positions = input_dict.get("positions")
        padding_mask = input_dict.get("padding_mask")
        if positions is None:
            raise ValueError("MTP input preprocessing requires positions.")

        depths = range(1, len(self.mtp_layers) + 1)
        input_shardings = decoder_input_sharding()
        for depth in depths:
            mtp_input_tokens, mtp_input_valid_mask = roll_mtp_sequence(
                tokens,
                shift=depth,
                positions=positions,
                padding_mask=padding_mask,
                fill_value=0,
                return_valid_mask=True,
            )
            mtp_labels = roll_mtp_sequence(
                labels,
                shift=depth,
                positions=positions,
                padding_mask=padding_mask,
                fill_value=IGNORE_INDEX,
                return_valid_mask=False,
            )
            input_dict[f"mtp_input_tokens_{depth}"] = mtp_input_tokens
            input_dict[f"mtp_labels_{depth}"] = mtp_labels
            input_dict[f"mtp_input_valid_mask_{depth}"] = mtp_input_valid_mask
            input_shardings[f"mtp_input_tokens_{depth}"] = input_shardings["input"]
            input_shardings[f"mtp_labels_{depth}"] = input_shardings["labels"]
            input_shardings[f"mtp_input_valid_mask_{depth}"] = input_shardings["input"]
        input_shardings.update(dict.fromkeys(loss_kwargs, input_shardings["labels"]))
        input_dict.update(loss_kwargs)

        if positions is not None:
            attention_metadata = self._get_attention_metadata(
                positions=positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
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

        loss_kwargs = {name: input_dict.pop(name) for name in loss_kwargs}
        main_tokens = input_dict.pop("input")
        main_labels = input_dict.pop("labels")
        input_dict["aux_loss_denominators"] = global_routing_token_counts
        input_tokens = (
            main_tokens,
            *(input_dict.pop(f"mtp_input_tokens_{depth}") for depth in depths),
        )
        loss_labels = (
            main_labels,
            *(input_dict.pop(f"mtp_labels_{depth}") for depth in depths),
        )
        input_dict["mtp_input_valid_masks"] = tuple(
            input_dict.pop(f"mtp_input_valid_mask_{depth}") for depth in depths
        )
        return input_tokens, loss_labels, input_dict, loss_kwargs


class MTPCrossEntropyLoss(Loss[tuple[torch.Tensor, ...]]):
    """Weighted cross-entropy over the main prediction and each MTP depth.

    The main term has weight 1; the MTP terms share ``mtp_scale`` equally.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Loss.Config):
        mtp_scale: float = 0.3
        global_vocab_size: int | None = None

    def __init__(self, config: Config):
        self.cross_entropy_loss = CrossEntropyLoss(
            CrossEntropyLoss.Config(global_vocab_size=config.global_vocab_size)
        )
        self.mtp_scale = config.mtp_scale

    def __call__(
        self,
        pred: tuple[torch.Tensor, ...],
        labels: tuple[torch.Tensor, ...],
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the weighted objective from aligned prediction/label pairs."""
        del loss_inputs
        if not isinstance(pred, tuple) or not isinstance(labels, tuple):
            raise ValueError(
                "MTPCrossEntropyLoss expects prediction and labels tuples."
            )
        if len(pred) != len(labels):
            raise ValueError(
                "MTPCrossEntropyLoss requires one labels tensor per prediction, "
                f"got {len(pred)} predictions and {len(labels)} labels."
            )
        num_mtp_layers = len(pred) - 1
        if num_mtp_layers <= 0:
            raise ValueError(
                "MTPCrossEntropyLoss expects a main prediction and at least one auxiliary "
                "prediction."
            )
        mtp_weight = self.mtp_scale / num_mtp_layers
        if global_loss_token_counts is not None and global_loss_token_counts.ndim != 1:
            raise ValueError(
                "MTPCrossEntropyLoss requires a per-objective "
                "global_loss_token_counts vector."
            )
        if (
            global_loss_token_counts is not None
            and global_loss_token_counts.numel() != len(pred)
        ):
            raise ValueError(
                "MTPCrossEntropyLoss requires one denominator per prediction, "
                f"got {global_loss_token_counts.numel()} for {len(pred)} predictions."
            )
        if global_loss_token_counts is None:
            loss_token_counts: tuple[torch.Tensor | None, ...] = (None,) * len(pred)
        else:
            with spmd.no_typecheck():
                loss_token_counts = tuple(
                    count.clamp_min(1) for count in global_loss_token_counts.unbind()
                )
        main_loss, _ = self.cross_entropy_loss(pred[0], labels[0], loss_token_counts[0])
        mtp_loss = pred[0].new_zeros((), dtype=torch.float32)
        if spmd.is_type_checking():
            mtp_loss = spmd.mutate_type(
                mtp_loss,
                src=spmd.R,
                dst={"dp": spmd.P, "cp": spmd.P, "tp": spmd.I},
            )
        for depth, (mtp_pred, mtp_labels) in enumerate(
            zip(pred[1:], labels[1:], strict=True), 1
        ):
            depth_loss, _ = self.cross_entropy_loss(
                mtp_pred,
                mtp_labels,
                loss_token_counts[depth],
            )
            mtp_loss = mtp_loss + depth_loss * mtp_weight
        loss = main_loss + mtp_loss
        return loss, {}
