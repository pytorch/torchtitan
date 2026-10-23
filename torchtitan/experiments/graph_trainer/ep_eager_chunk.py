# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from graph_trainer.ep_eager_chunk import (
    maybe_apply_ep_overlap_eager_chunking as _maybe_apply_ep_overlap_eager_chunking,
    populate_eager_chunk_metadata_pass,
)

from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.moe import MoE


def maybe_apply_ep_overlap_eager_chunking(
    model: nn.Module,
    compile_config: GraphTrainerCompileConfig,
) -> None:
    """Apply standalone eager chunking with TorchTitan model type checks."""
    _maybe_apply_ep_overlap_eager_chunking(
        model,
        compile_config,
        transformer_block_type=TransformerBlock,
        moe_type=MoE,
    )


__all__ = [
    "maybe_apply_ep_overlap_eager_chunking",
    "populate_eager_chunk_metadata_pass",
]
