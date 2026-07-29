# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Overrides: cudagraph-capturable EP MoE token dispatch for the RL generator.

The RL trainer and generator share one ``model_spec``, but the generator captures a CUDA
graph and needs a static, host-sync-free MoE expert-parallel dispatch path that the eager,
backward-able trainer does not. Each EP comm backend has its own override here; the
generator activates the one matching its backend (passing its kwarg), while the trainer
keeps the shared spec's default. Activate per-actor via the ``module.function`` target::

    # generator.override -- DeepEP cudagraph-able EXPAND dispatch:
    OverrideConfig(imports=[(
        "torchtitan.overrides.moe_token_dispatcher.deepep_override",
        {"cudagraphable": True},
    )])
    # ...or HybridEP static non-blocking dispatch:
    OverrideConfig(imports=[(
        "torchtitan.overrides.moe_token_dispatcher.hybridep_override",
        {"capacity_factor": 0.0325},
    )])

``deepep_override`` (``DeepEPTokenDispatcher.Config``): ``cudagraphable=True`` flips DeepEP
to the static, host-sync-free EXPAND layout a CUDA graph can capture (the compact,
host-synced, backward-able path is the default). ``deepep.dispatch_tokens`` also gates the
expand path to ``not torch.is_grad_enabled()``, so it only takes effect for the no-grad
inference forward. The per-rank EXPAND capacity ``num_max_tokens_per_rank`` is
inference-only and REQUIRED for this path (the recipe sets it on the ``model_spec``
dispatchers; ``DeepEPTokenDispatcher.wire_meshes`` raises if it is still None when the
expand buffer is built). Set it to the largest per-rank token count for droplessness
(= max_num_batched_tokens / sp), or lower to save memory. ``cudagraphable`` is required.

``hybridep_override`` (``HybridEPTokenDispatcher.Config``): ``capacity_factor`` sets
``non_blocking_capacity_factor`` (``None`` = blocking, dropless; a float in (0, 1] = the
non-blocking capacity factor -- see that config for exact semantics). The trainer runs
eagerly and backprops, so it needs the blocking path (the default); the generator captures
a CUDA graph, so it needs the static non-blocking path. ``capacity_factor`` is required.
"""

from __future__ import annotations

import dataclasses

from torchtitan.config import override
from torchtitan.models.common.token_dispatcher import (
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
)


@override(
    target=DeepEPTokenDispatcher.Config,
    description="DeepEP cudagraph-able expand dispatch for inference.",
)
def deepep_override(
    cfg: DeepEPTokenDispatcher.Config,
    *,
    cudagraphable: bool,
) -> DeepEPTokenDispatcher.Config:
    # cudagraphable=True flips the DeepEP dispatchers to the static, cudagraph-able EXPAND
    # layout (False keeps the compact host-synced default). The expand path also needs
    # num_max_tokens_per_rank set on the dispatcher (the per-rank capacity); that is
    # validated where the buffer is built (DeepEPTokenDispatcher.wire_meshes), not here,
    # since the value is set on the dispatcher by the inference recipe rather than to us.
    return dataclasses.replace(cfg, cudagraphable=cudagraphable)


@override(
    target=HybridEPTokenDispatcher.Config,
    description="Set the HybridEP non-blocking capacity factor per actor.",
)
def hybridep_override(
    cfg: HybridEPTokenDispatcher.Config,
    *,
    capacity_factor: float | None,
) -> HybridEPTokenDispatcher.Config:
    return dataclasses.replace(cfg, non_blocking_capacity_factor=capacity_factor)
