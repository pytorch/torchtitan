# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sharding configs for Kimi K3. Same pattern as ``qwen3_5/sharding.py``.

Declarations only: functions here set ``ShardingConfig`` on sub-configs of an
already-built config tree, and ``model.parallelize()`` applies them through the
Module protocol. Nothing here touches a mesh or a device.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd

from torchtitan.models.common.moe_sharding import set_moe_sharding_config

if TYPE_CHECKING:
    from torchtitan.models.kimi_k3.model import KimiK3Model


def set_kimi_k3_sharding_config(
    config: "KimiK3Model.Config", *, enable_ep: bool, enable_sp: bool = False
) -> None:
    """Declare the sharding expert parallel acts on.

    The routed experts shard on the expert axis; ``set_moe_sharding_config``
    declares that layout, and its input boundary lifts the plain incoming
    activations itself, so no decoder-level declaration is needed.
    """
    for layer in config.layers:
        if layer.moe is not None:
            set_moe_sharding_config(
                layer.moe,
                enable_ep=enable_ep,
                # TODO: flip to True from the caller once the
                # tensor-parallel PR lands; with EP alone the internals run
                # without sequence parallel.
                enable_sp=enable_sp,
                expert_param_layout={
                    "w1_EFD": spmd.S(1),
                    "w2_EDF": spmd.S(2),
                    "w3_EFD": spmd.S(1),
                },
            )
