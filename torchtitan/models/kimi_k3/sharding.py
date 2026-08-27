# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Kimi K3. Same pattern as ``qwen3_5/sharding.py``.

Declarations only: functions here set ``ShardingConfig`` on sub-configs of an
already-built config tree, and ``model.parallelize()`` applies them through the
Module protocol. Nothing here touches a mesh or a device, which is what lets
the expert-parallel declaration be unit-tested on CPU.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.decoder_sharding import set_decoder_sharding_config
from torchtitan.models.common.moe_sharding import set_moe_sharding_config

if TYPE_CHECKING:
    from torchtitan.models.kimi_k3.model import KimiK3Model


def set_expert_parallel_sharding_config(config: "KimiK3Model.Config") -> None:
    """Declare the sharding expert parallel acts on.

    * The routed experts shard on the expert axis; ``set_moe_sharding_config``
      declares that layout.
    * The decoder-level distribution makes the activations reaching the MoE
      boundary DTensors it can redistribute onto the expert mesh.
    """
    set_decoder_sharding_config(config, enable_sp=False)
    for layer in config.layers:
        if layer.moe is not None:
            set_moe_sharding_config(
                layer.moe,
                enable_ep=True,
                enable_sp=False,
                expert_param_layout={
                    "w1_EFD": spmd.S(1),
                    "w2_EDF": spmd.S(2),
                    "w3_EFD": spmd.S(1),
                },
            )
