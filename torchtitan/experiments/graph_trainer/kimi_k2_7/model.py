# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe import register_moe_load_balancing_hook
from torchtitan.models.kimi_k2_7 import KimiK2Model
from torchtitan.models.kimi_k2_7.qk_clip import register_qk_clip_hook

from ..model import GraphTrainerModel


def _sharded_parameter(module: nn.Module, name: str) -> DTensor:
    param = dict(module.named_parameters(recurse=False))[name]
    if isinstance(param, DTensor):
        return param
    return param._graph_trainer_dtensor_param


class GraphTrainerKimiK2Model(GraphTrainerModel, KimiK2Model):
    @dataclass(kw_only=True, slots=True)
    class Config(KimiK2Model.Config):
        pass

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_qk_clip_hook(
            optimizers,
            model_parts,
            parallelism_context,
            parameter_resolver=_sharded_parameter,
        )
