# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.trainer import Trainer


class HFTransformerTrainer(Trainer):
    """Trainer for the HF transformers backend.

    Adds one behavior over the core ``Trainer``: under context parallelism with
    flex attention, it builds the full document-causal ``BlockMask`` from the
    unsharded positions *before* the base class shards inputs. The base
    ``post_dataloading_process`` then routes the mask through
    ``prepare_context_parallel_input``, which shards the mask's Q axis (and the
    inputs/positions) with the configured load balancer -- yielding a
    local-Q x full-KV mask that matches the k/v all-gathered inside the flex
    kernel (see ``_wrap_flex_kernel_cp`` in parallelize.py).

    The core ``Trainer`` only builds masks for ``Decoder.Config`` models; the HF
    backend is not one, so without this the mask would be built inside
    ``model.forward`` from already-sharded positions (a square local x local
    mask, wrong once k/v span the full sequence). Outside CP, mask building
    stays in the model -- this hook is a no-op there.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.parallel_dims.cp_enabled and "attention_masks" not in input_dict:
            positions = input_dict.get("positions")
            if positions is not None:
                masks = self.model_parts[0].get_attention_masks(positions=positions)
                if masks is not None:
                    input_dict = {**input_dict, "attention_masks": masks}
        return super().post_dataloading_process(input_dict, labels)
