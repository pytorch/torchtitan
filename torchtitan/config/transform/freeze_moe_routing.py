# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transform that turns off MoE load balancing."""

from dataclasses import dataclass

from torchtitan.models.base.module import Module
from torchtitan.models.common.moe import MoE

from .base import ModelConfigTransform

__all__ = ["FreezeMoERoutingTransform"]


@dataclass(kw_only=True, slots=True, frozen=True)
class FreezeMoERoutingTransform(ModelConfigTransform):
    """Turn off MoE load balancing, so only the training loss changes expert choices.

    Each knob, when True, sets a field on every ``MoE.Config`` in the tree:
        freeze_expert_bias   freeze_expert_bias = True   no balancing hook moves ``expert_bias_E``
        freeze_gate          router.freeze_gate = True   no gradient, no optimizer state for the gate
        drop_aux_loss        router.aux_loss = None      no load-balancing loss
    A False knob leaves its field as built.
    Dense models are unchanged.
    """

    freeze_expert_bias: bool = True
    freeze_gate: bool = False
    drop_aux_loss: bool = True

    def transform(self, model: Module.Config) -> Module.Config:
        for _fqn, moe_config, _parent, _attr in model.traverse(MoE.Config):
            if self.freeze_expert_bias:
                moe_config.freeze_expert_bias = True
            if self.freeze_gate:
                moe_config.router.freeze_gate = True
            if self.drop_aux_loss:
                moe_config.router.aux_loss = None
        return model
