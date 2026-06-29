# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: switch DeepEP MoE dispatch to the cudagraph-able EXPAND layout for inference.

Imported per-actor via ``--override.imports torchtitan.distributed.deepep.inference_override``
(e.g. on the RL generator's ``override``, separate from the trainer's). The shared
``model_spec`` keeps its default compact, host-synced, backward-able DeepEP path for the
trainer; on the generator this override flips the DeepEP dispatchers to the static,
host-sync-free EXPAND layout (``cudagraphable=True``) that a CUDA graph can capture.
``deepep.dispatch_tokens`` also gates the expand path to ``not torch.is_grad_enabled()``, so it
only takes effect for the no-grad inference forward.

The per-rank EXPAND capacity ``num_max_tokens_per_rank`` is inference-only (the trainer's
compact path auto-sizes and never reads it). The inference recipe sets it on the DeepEP
dispatchers of the ``model_spec``; the trainer ignores that value (it infers its own at
dispatch), so training and inference effectively differ. It is REQUIRED for this path and
defaults to None -- ``DeepEPTokenDispatcher.wire_meshes`` raises if it is still None when the
expand buffer is built. Set it to the largest per-rank token count for droplessness
(= max_num_batched_tokens / sp), or lower to save memory.
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
    # Flip the DeepEP dispatchers to the static, cudagraph-able EXPAND layout. The expand path
    # also needs num_max_tokens_per_rank set on the dispatcher (the per-rank capacity); that is
    # validated where the buffer is built (DeepEPTokenDispatcher.wire_meshes), not here, since
    # the value is set on the dispatcher by the inference recipe rather than passed to us.
    return dataclasses.replace(cfg, cudagraphable=True)
