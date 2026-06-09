# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torchtitan.models.qwen3.sharding import set_qwen3_sharding_config

if TYPE_CHECKING:
    from torchtitan.models.qwen3_vl.model import Qwen3VLModel


def set_qwen3_vl_sharding_config(
    config: "Qwen3VLModel.Config",
    *,
    loss_parallel: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all Qwen3-VL sub-configs.

    Delegates to ``set_qwen3_sharding_config`` with ``enable_sp=False``
    because Qwen3-VL keeps hidden states as ``Replicate`` (not
    ``Shard(1)``) — no SequenceParallel due to vision scatter and
    DeepStack needing full-sequence access.
    """
    set_qwen3_sharding_config(
        config,
        loss_parallel=loss_parallel,
        enable_sp=False,
        enable_ep=enable_ep,
    )
