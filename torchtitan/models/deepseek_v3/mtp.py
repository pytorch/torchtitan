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
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.observability import tensor_logging
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
    ``i``. This helper builds that shifted view along the sequence axis
    (dimension 1). Tail positions, and positions that would cross a packed
    document boundary, are filled instead of wrapped around.

    Args:
        sequence: Tensor to shift, with shape ``[batch, seq_len, ...]``. The
            first two dimensions are batch and sequence; any trailing dimensions
            are carried along unchanged.
        shift: Future-token offset to use. ``shift=1`` maps each position to the
            next token, ``shift=2`` maps to the token after next, and so on.
            Must be positive and no larger than ``seq_len``.
        positions: Optional reset-style position IDs with shape
            ``[batch, seq_len]``. When present, a shifted source position is
            valid only if ``positions[:, i + shift] == positions[:, i] + shift``.
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
    seq_len = sequence.shape[1]
    if shift <= 0 or shift > seq_len:
        raise ValueError(f"MTP roll shift must be in [1, {seq_len}], got {shift}.")

    rolled = torch.full_like(sequence, fill_value)
    valid_mask = torch.zeros_like(sequence, dtype=torch.bool)

    source = sequence[:, shift:]
    if positions is None:
        rolled[:, : seq_len - shift] = source
        valid_mask[:, : seq_len - shift] = True
        if return_valid_mask:
            return rolled, valid_mask
        return rolled

    if positions.shape[1] < seq_len:
        raise ValueError(
            f"MTP positions need at least {seq_len} tokens, got {positions.shape[1]}."
        )
    valid_tokens = (
        positions[:, shift:seq_len] == positions[:, : seq_len - shift] + shift
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

            parallelism = config.parallelism
            # TODO: Add Pipeline Parallel support for MTP.
            if parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "MTP does not support pipeline parallelism yet."
                )
            # TODO: Add Context Parallel support for MTP.
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
        tensor_logging.log_fwd_bwd_stats(self, input=h)
        for layer in self.layers.values():
            h = layer(h, attention_masks, positions)

        prev_depth_hidden = h
        h = self.norm(h) if self.norm is not None else h

        mtp_outputs = []
        for depth, layer in enumerate(self.mtp_layers, 1):
            # NOTE: Without SP, the local main embedding output has shape
            # [batch, seq_len, hidden_dim] and could be shifted and reused.
            # Under SP, its sequence dimension is sharded, so a local shift
            # would be incorrect at shard boundaries. Reuse in that case
            # would require a cross-shard shift or redistribution.
            mtp_input_tokens, mtp_input_valid_mask = roll_mtp_sequence(
                tokens,
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

        outputs = [h] + mtp_outputs
        if self._skip_lm_head:
            raise ValueError(
                "skip_lm_head is not supported with MTP decoder until "
                "ChunkedLoss supports MTP outputs."
            )
        if self.lm_head is None:
            return outputs
        predictions = [self.lm_head(item) for item in outputs]
        for output in predictions:
            tensor_logging.log_fwd_bwd_stats(self.lm_head, output=output)
        return predictions


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


# TODO: Add ChunkedLoss support for the main and per-depth MTP outputs.
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
        pred: list[torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        positions = kwargs.pop("positions", None)
        del kwargs

        if not isinstance(pred, list):
            raise ValueError(
                "MTPLoss expects a list of predictions: main logits followed "
                "by one tensor per MTP layer."
            )
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
        mtp_loss: torch.Tensor | None = None

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
                fill_value=IGNORE_INDEX,
                positions=positions[:, :mtp_seq_len],
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
        if global_valid_tokens is not None:
            # TODO: Teach spmd_types that scalar loss normalization preserves
            # the loss placement.
            with spmd.no_typecheck():
                loss = loss / global_valid_tokens
        return loss, {}
