# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transform that keeps MoE routing as loaded."""

from torchtitan.models.base.module import Module
from torchtitan.models.common.moe import MoE

from .base import ModelConfigTransform

__all__ = ["FreezeMoERoutingTransform"]


class FreezeMoERoutingTransform(ModelConfigTransform):
    """Keep every MoE router as loaded, so expert choices change only with the hidden states.

    Example, for every ``MoE.Config`` in the tree:
        freeze_expert_bias   False -> True   no balancing hook moves ``expert_bias_E``
        router.freeze_gate   False -> True   no gradient, no optimizer state for the gate
        router.aux_loss      any -> None     no load-balancing loss
    Dense models are unchanged.
    """

    def transform(self, model: Module.Config) -> Module.Config:
        for _fqn, moe_config, _parent, _attr in model.traverse(MoE.Config):
            moe_config.freeze_expert_bias = True
            moe_config.router.freeze_gate = True
            moe_config.router.aux_loss = None
        return model
