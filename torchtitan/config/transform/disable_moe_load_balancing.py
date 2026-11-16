# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transform that turns off MoE load balancing."""

from torchtitan.models.base.module import Module
from torchtitan.models.common.moe import MoE

from .base import ModelConfigTransform

__all__ = ["DisableMoELoadBalancingTransform"]


class DisableMoELoadBalancingTransform(ModelConfigTransform):
    """Turn off MoE load balancing: no hook or loss pushes expert loads toward even.

    Example, for every ``MoE.Config`` in the tree:
        freeze_expert_bias   False -> True   no balancing hook moves ``expert_bias_E``
        router.aux_loss      any -> None     no load-balancing loss
    Dense models are unchanged.
    """

    def transform(self, model: Module.Config) -> Module.Config:
        for _fqn, moe_config, _parent, _attr in model.traverse(MoE.Config):
            moe_config.freeze_expert_bias = True
            moe_config.router.aux_loss = None
        return model
