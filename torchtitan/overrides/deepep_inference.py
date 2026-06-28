# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: switch DeepEP MoE dispatch to the cudagraph-able EXPAND layout for inference.

Imported per-actor via ``--override.imports torchtitan.overrides.deepep_inference`` (e.g. on
the RL generator's ``override``, separate from the trainer's). The shared ``model_spec`` keeps
its default compact, host-synced, backward-able DeepEP path for the trainer; on the generator
this override flips the DeepEP dispatchers to the static, host-sync-free EXPAND layout
(``cudagraphable=True``) that a CUDA graph can capture. ``deepep.dispatch_tokens`` also gates
the expand path to ``not torch.is_grad_enabled()``, so it only takes effect for the no-grad
inference forward.

The static per-rank EXPAND capacity (``num_max_tokens_per_rank``) is left at the dispatcher
default; sizing it for the worst-case forward (max_num_batched_tokens / sp_size) is tracked
separately (see ``DeepEPTokenDispatcher``).
"""

from __future__ import annotations

import dataclasses

from torchtitan.config import override
from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher


@override(
    "deepep_inference",
    target=DeepEPTokenDispatcher.Config,
    description="DeepEP cudagraph-able expand dispatch for inference.",
)
def deepep_inference(
    cfg: DeepEPTokenDispatcher.Config,
) -> DeepEPTokenDispatcher.Config:
    return dataclasses.replace(cfg, cudagraphable=True)
