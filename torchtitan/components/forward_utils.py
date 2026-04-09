# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast

import torch
import torch.nn as nn

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.decoder import Decoder
from torchtitan.protocols import BaseModel


def build_forward_extra_kwargs(
    model_config: BaseModel.Config | None,
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    tokenizer: BaseTokenizer | None = None,
    parallel_dims: ParallelDims | None = None,
    extra_inputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Construct the extra keyword arguments for model.forward().

    This is the single source of truth for determining which extra kwargs
    (positions, attention_masks) are needed based on model config and
    parallelism settings. Used by Trainer, Validator, and precompile.

    The insertion order must be stable: positions first, then
    attention_masks. Pytree-flattened input order depends on this.

    Args:
        model_config: Model configuration (e.g. Decoder.Config).
        model: The model instance (used to call get_attention_masks).
        inputs: Input tensor with shape (batch_size, seq_len).
        positions: Pre-computed positions from the dataloader (block_causal).
            If None and positions are needed, synthetic ones are created.
        tokenizer: Tokenizer for attention mask construction. Required when
            flex/varlen attention is configured.
        parallel_dims: Parallelism dimensions (used to check CP).
        extra_inputs: Auxiliary inputs dict passed to get_attention_masks.
    """
    extra_kwargs: dict[str, Any] = {}

    if not isinstance(model_config, Decoder.Config) or not model_config.layers:
        return extra_kwargs

    attn_config = model_config.layers[0].attention
    mask_type = getattr(attn_config, "mask_type", "causal")

    if mask_type == "block_causal":
        if positions is not None:
            extra_kwargs["positions"] = positions
        else:
            extra_kwargs["positions"] = torch.arange(
                0, inputs.shape[1], dtype=torch.int32, device=inputs.device
            ).expand(inputs.shape)
    elif parallel_dims is not None and parallel_dims.cp_enabled:
        extra_kwargs["positions"] = torch.arange(
            0, inputs.shape[1], dtype=torch.int32, device=inputs.device
        ).expand(inputs.shape)

    inner_attention = getattr(attn_config, "inner_attention", None)
    if inner_attention is not None:
        from torchtitan.models.common.attention import FlexAttention, VarlenAttention

        if isinstance(inner_attention, (FlexAttention.Config, VarlenAttention.Config)):
            assert (
                tokenizer is not None
            ), "tokenizer is required for flex/varlen attention"
            extra_kwargs["attention_masks"] = cast(Decoder, model).get_attention_masks(
                input_batch=inputs,
                tokenizer=tokenizer,
                extra_inputs=extra_inputs or {},
            )

    return extra_kwargs
