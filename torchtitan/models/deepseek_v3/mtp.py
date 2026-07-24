# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import spmd_types as spmd
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import DataParallelMeshDims

from torchtitan.components.loss import (
    BaseLoss,
    cross_entropy_loss,
    IGNORE_INDEX,
    LossFunction,
)
from torchtitan.config import CompileConfig
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.spmd_types import (
    current_spmd_mesh,
    spmd_redistribute_per_axis,
)
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.protocols.module import ModuleList


def _redistribute_valid_mask_to_sp(valid_mask: torch.Tensor) -> torch.Tensor:
    return spmd_redistribute_per_axis(
        valid_mask,
        current_spmd_mesh(),
        dense_activation_placement(tp=spmd.R, cp=spmd.R).per_axis_spmd_types(),
        dense_activation_placement(tp=spmd.S(1), cp=spmd.R).per_axis_spmd_types(),
    )


def _maybe_redistribute_valid_mask_to_target(
    valid_mask: torch.Tensor,
    target: torch.Tensor | None,
) -> torch.Tensor:
    if target is None or valid_mask.shape[1] == target.shape[1]:
        return valid_mask
    return _redistribute_valid_mask_to_sp(valid_mask)


def roll_mtp_sequence(
    sequence: torch.Tensor,
    *,
    shift: int,
    positions: torch.Tensor | None = None,
    fill_value: int = 0,
    return_valid_mask: bool = False,
    valid_mask_target: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Left-roll an MTP sequence while preserving packed-document boundaries.

    Without ``positions``, this is a regular left shift with the tail filled.
    With reset-style positions, each document slice is shifted independently:
    ``[A0, A1, A2, B0, B1]`` becomes ``[A1, A2, fill, B1, fill]``.
    """
    if shift < 0:
        raise ValueError(f"MTP roll shift must be non-negative, got {shift}.")
    if shift == 0:
        if return_valid_mask:
            valid_mask = torch.ones_like(sequence, dtype=torch.bool)
            return sequence, _maybe_redistribute_valid_mask_to_target(
                valid_mask,
                valid_mask_target,
            )
        return sequence

    seq_len = sequence.shape[1]
    rolled = torch.full_like(sequence, fill_value)
    valid_mask = torch.zeros_like(sequence, dtype=torch.bool)
    if shift >= seq_len:
        if return_valid_mask:
            return rolled, _maybe_redistribute_valid_mask_to_target(
                valid_mask,
                valid_mask_target,
            )
        return rolled

    source = sequence[:, shift:]
    if positions is None:
        rolled[:, : seq_len - shift] = source
        valid_mask[:, : seq_len - shift] = True
        if return_valid_mask:
            return rolled, _maybe_redistribute_valid_mask_to_target(
                valid_mask,
                valid_mask_target,
            )
        return rolled

    if positions.shape[1] < seq_len:
        raise ValueError(
            f"MTP positions need at least {seq_len} tokens, got {positions.shape[1]}."
        )
    valid_tokens = (
        positions[:, shift:seq_len]
        == positions[:, : seq_len - shift] + shift
    )
    # valid_tokens follows positions placement, while valid_mask intentionally
    # follows sequence placement for the following where.
    with spmd.no_typecheck():
        valid_mask[:, : seq_len - shift] = valid_tokens
    rolled[:, : seq_len - shift] = torch.where(
        valid_mask[:, : seq_len - shift],
        source,
        rolled[:, : seq_len - shift],
    )
    if return_valid_mask:
        return rolled, _maybe_redistribute_valid_mask_to_target(
            valid_mask,
            valid_mask_target,
        )
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
        input_offset: torch.Tensor,
        prev_embed: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        h = self.eh_proj(
            torch.cat([self.enorm(input_offset), self.hnorm(prev_embed)], dim=-1)
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
        num_mtp_layers: int = 0
        mtp_layers: list = field(default_factory=list)

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            self.num_mtp_layers = len(self.mtp_layers)
            if self.num_mtp_layers <= 0:
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

            parallelism = config.parallelism
            if parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "MTP does not support pipeline parallelism yet."
                )
            if parallelism.context_parallel_degree > 1:
                raise NotImplementedError(
                    "MTP does not support context parallelism yet."
                )

    def __init__(self, config: Config):
        super().__init__(config)
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
    ):
        if self.mtp_layers is None:
            return super().forward(tokens, positions, attention_masks)
        if self.tok_embeddings is None:
            raise ValueError("MTP decoder forward requires token embeddings.")

        # Keep this aligned with Decoder.forward(), but preserve the pre-norm
        # hidden state because MTP consumes the last decoder-layer output.
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h, attention_masks, positions)

        mtp_prev_embed = h
        h = self.norm(h) if self.norm is not None else h

        mtp_outputs = []
        for depth, layer in enumerate(self.mtp_layers, 1):
            offset_tokens, valid_offset_tokens = roll_mtp_sequence(
                tokens,
                shift=depth,
                positions=positions,
                fill_value=0,
                return_valid_mask=True,
                valid_mask_target=mtp_prev_embed,
            )
            input_offset = self.tok_embeddings(offset_tokens)
            valid_offset_tokens = valid_offset_tokens.unsqueeze(-1).to(
                dtype=mtp_prev_embed.dtype
            )
            mtp_prev_embed = mtp_prev_embed * valid_offset_tokens
            mtp_prev_embed = layer(
                input_offset,
                mtp_prev_embed,
                attention_masks,
                positions,
            )
            mtp_outputs.append(mtp_prev_embed)

        outputs = [h] + mtp_outputs
        if self._skip_lm_head:
            return outputs
        return [
            self.lm_head(item) if self.lm_head is not None else item
            for item in outputs
        ]


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
            enable_symm_mem=enable_symm_mem,
        )
    finally:
        for key in mtp_layer_keys:
            del model.layers[key]


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
        pred: torch.Tensor | list[torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        positions = kwargs.pop("positions", None)
        del kwargs

        if isinstance(pred, (list, tuple)):
            if positions is None:
                raise ValueError("MTPLoss requires positions for MTP predictions.")
            num_mtp_layers = len(pred) - 1
            if num_mtp_layers <= 0:
                raise ValueError(
                    "MTPLoss expects main prediction plus at least one MTP "
                    f"prediction, got {len(pred)} predictions."
                )

            main_loss = self.fn(
                pred[0],
                labels[:, : pred[0].shape[1]],
                global_vocab_size=self.global_vocab_size,
            )
            mtp_loss = None

            for label_offset, mtp_pred in enumerate(pred[1:], 1):
                mtp_seq_len = mtp_pred.shape[1]
                if labels.shape[1] < mtp_seq_len:
                    raise ValueError(
                        f"MTP labels need at least {mtp_seq_len} "
                        f"tokens for depth {label_offset}, got {labels.shape[1]}."
                    )
                if positions.shape[1] < mtp_seq_len:
                    raise ValueError(
                        f"MTP positions need at least {mtp_seq_len} tokens "
                        f"for depth {label_offset}, got {positions.shape[1]}."
                    )
                mtp_labels = roll_mtp_sequence(
                    labels[:, :mtp_seq_len],
                    shift=label_offset,
                    positions=positions[:, :mtp_seq_len],
                    fill_value=IGNORE_INDEX,
                )
                depth_loss = self.fn(
                    mtp_pred,
                    mtp_labels,
                    global_vocab_size=self.global_vocab_size,
                )
                mtp_loss = depth_loss if mtp_loss is None else mtp_loss + depth_loss
            assert mtp_loss is not None
            if num_mtp_layers > 1:
                # TODO: Teach spmd_types that V / scalar preserves the scalar
                # loss placement. This mirrors the base loss normalization.
                with spmd.no_typecheck():
                    mtp_loss = mtp_loss / num_mtp_layers
            # TODO: Teach spmd_types that scalar loss composition preserves
            # the loss placement across auxiliary weighted losses.
            with spmd.no_typecheck():
                loss = main_loss + mtp_loss * self.mtp_scale
        else:
            loss = self.fn(
                pred,
                labels,
                global_vocab_size=self.global_vocab_size,
            )
        if global_valid_tokens is not None:
            # TODO: Teach spmd_types that scalar loss normalization preserves
            # the loss placement.
            with spmd.no_typecheck():
                loss = loss / global_valid_tokens
        return loss, {}
