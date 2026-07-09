# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: set the HybridEP MoE capacity factor per actor.

The RL trainer and generator share one ``model_spec`` but need opposite HybridEP
dispatch modes: the trainer runs eagerly and backprops, so it needs the blocking,
dropless path (capacity factor ``None``); the generator captures a CUDA graph and
needs the static, host-sync-free non-blocking path (a float capacity factor).

Rather than hardcode that split in the trainer, an actor activates this one module
and passes the value it wants as a per-entry ``capacity_factor`` kwarg on its own
``override``::

    # generator.override -- non-blocking / cudagraph-safe dispatch:
    OverrideConfig(imports=[(
        "torchtitan.distributed.deepep.hybridep_override",
        {"capacity_factor": 0.0325},
    )])

The ``capacity_factor`` kwarg sets ``HybridEPTokenDispatcher.Config``'s
``non_blocking_capacity_factor`` (``None`` = blocking; a float in (0, 1] = the
non-blocking capacity factor; see that config for exact semantics). The blocking
path is the config default, so the trainer needs no override; only the actor that
wants the non-blocking path activates this. ``capacity_factor`` is required (this
override exists to set it).
"""

from __future__ import annotations

import dataclasses

from torchtitan.config import override
from torchtitan.models.common.token_dispatcher import HybridEPTokenDispatcher


@override(
    "hybridep_override",
    target=HybridEPTokenDispatcher.Config,
    description="Set the HybridEP non-blocking capacity factor per actor.",
)
def hybridep_override(
    cfg: HybridEPTokenDispatcher.Config,
    *,
    capacity_factor: float | None,
) -> HybridEPTokenDispatcher.Config:
    return dataclasses.replace(cfg, non_blocking_capacity_factor=capacity_factor)
