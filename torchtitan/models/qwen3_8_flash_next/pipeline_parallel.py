# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline parallelism for Flash-Next: stages that pass all residual streams."""

from dataclasses import replace
from typing import Any, cast, TYPE_CHECKING

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.distributed.pipeline_parallel import build_decoder_stage_io
from torchtitan.models.base import BaseModel
from torchtitan.models.common.decoder import Decoder

from .embedding import NGramEmbedding

if TYPE_CHECKING:
    from .model import Qwen38FlashNextModel


def pipeline_flash_next(
    model: "Qwen38FlashNextModel", **kwargs: Any
) -> tuple[tuple[Any, list[BaseModel], bool, bool], NGramEmbedding | None]:
    """Partition Flash-Next into pipeline stages.

    Returns ``BaseModel.pipeline``'s result and the local PLE embedding, if
    any part holds it, for the first part's preprocessing.
    """
    # Declared stage boundaries skip PipelineStage's metadata inference,
    # which runs one forward per stage on uninitialized hidden states:
    # their routing can collapse onto a few experts and overflow
    # Dist-MoE's scratch. Core's decoder boundaries fit, except that
    # non-final stages send all residual streams, [T, num_streams * D].
    stage_io = build_decoder_stage_io(
        parallelism_context=kwargs["parallelism_context"],
        parallelism=kwargs["parallelism"],
        training=kwargs["training"],
        precision=kwargs["precision"],
        model_config=cast(Decoder.Config, kwargs["model_config"]),
        lm_head_in_loss=isinstance(kwargs["loss_fn"], ChunkedLossWrapper),
    )
    num_tokens, dim = stage_io.hidden.shape
    hidden_TC = stage_io.hidden.new_empty(num_tokens, model.num_streams * dim)
    result = BaseModel.pipeline(
        model,
        **kwargs,
        stage_io=replace(stage_io, hidden=hidden_TC.requires_grad_()),
    )
    model_parts = result[1]
    for part in model_parts:
        part = cast("Qwen38FlashNextModel", part)
        # A stage without the final mixer passes all residual streams on, so
        # the mixer and the final norm must sit on the stage that holds the
        # LM head.
        final_modules = (part.hyper_connection_mixer, part.norm, part.lm_head)
        if len({module is None for module in final_modules}) > 1:
            raise ValueError(
                "Flash-Next pipeline stages must keep hyper_connection_mixer, "
                "norm and lm_head together"
            )
    # Config validation allows exactly one PLE layer.
    embedding = next(
        (
            module
            for part in model_parts
            for module in part.modules()
            if isinstance(module, NGramEmbedding)
        ),
        None,
    )
    return result, embedding
