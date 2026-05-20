# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative sharding specs for HF MoE sub-modules.

Mirrors ``torchtitan/models/common/moe_sharding.py`` but works with
HF module attribute names (``gate_up_proj``/``down_proj`` vs ``w1``/``w2``/``w3``,
``gate``/``router`` vs ``router.gate``, etc.).

These specs are applied by ``shard_module_states`` in ``parallelize.py``
rather than by the ``Module.parallelize`` protocol, since HF modules don't
implement titan's ``Module`` protocol.
"""

from torch.distributed.tensor import Partial, Placement, Replicate, Shard

from torchtitan.protocols.sharding import NamedPlacement
from torchtitan.protocols.types import MeshAxisName

TP = MeshAxisName.TP
EP = MeshAxisName.EP


def expert_ep_shardings() -> dict[str, NamedPlacement]:
    """Expert params sharded on dim 0 across the EP axis."""
    return {
        "gate_up_proj": {EP: Shard(0)},
        "down_proj": {EP: Shard(0)},
    }


def expert_tp_shardings() -> dict[str, NamedPlacement]:
    """Expert params TP-sharded (colwise gate_up, rowwise down)."""
    return {
        "gate_up_proj": {TP: Shard(1)},
        "down_proj": {TP: Shard(2)},
    }


def gate_shardings() -> dict[str, NamedPlacement]:
    """Replicate gate (router) params on TP mesh."""
    return {
        "weight": {TP: Replicate()},
    }


def shared_expert_shardings() -> dict[str, NamedPlacement]:
    """TP-shard shared expert MLP (colwise gate/up, rowwise down)."""
    shardings: dict[str, NamedPlacement] = {}
    for name in ("gate_proj", "up_proj"):
        shardings[f"{name}.weight"] = {TP: Shard(0)}
    shardings["down_proj.weight"] = {TP: Shard(1)}
    return shardings
