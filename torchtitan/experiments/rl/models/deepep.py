# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator-side ModelConfigConverter that switches DeepEP MoE dispatchers to the
cudagraph-able inference layout, keeping a single shared ``model_spec`` for trainer and
generator."""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher
from torchtitan.protocols.model import ModelConfigConverter


class DeepEPInferenceConverter(ModelConfigConverter):
    """Set the DeepEP v2 inference-only dispatch knobs on every MoE layer.

    ``model_spec`` is shared by the trainer and the generator, so the trainer keeps the
    default compact, host-synced, backward-able DeepEP path. Applying this converter on the
    generator only (via ``VLLMGenerator.Config.converters``, on the generator's process-local
    copy of the spec) flips its DeepEP dispatchers to the static, host-sync-free EXPAND layout
    (``cudagraphable=True``) with a fixed per-rank capacity, which a CUDA graph can capture.
    (``deepep.dispatch_tokens`` additionally gates expand to ``not is_grad_enabled()``, so it
    only takes effect for the no-grad inference forward.)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        num_max_tokens_per_rank: int = 128
        """Static per-rank EXPAND dispatch capacity. Must be >= the max per-rank token count
        of any forward (= max_num_batched_tokens / sp_size); tokens beyond it are dropped."""
        # TODO(deepep-autosize): derive from the generator's max_num_batched_tokens / sp_size
        # instead of being set on the converter.

    def __init__(self, config: Config):
        self._num_max_tokens_per_rank = config.num_max_tokens_per_rank

    def convert(self, model_config):
        for _fqn, dispatcher, _parent, _attr in model_config.traverse(
            DeepEPTokenDispatcher.Config
        ):
            dispatcher.cudagraphable = True
            dispatcher.num_max_tokens_per_rank = self._num_max_tokens_per_rank
        return model_config
