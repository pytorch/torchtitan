# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
from torchtitan.models.kimi_k3.model import KimiK3Model

from .state_dict_adapter import KimiK31StateDictAdapter

__all__ = ["KimiK31Model"]


class KimiK31Model(KimiK3Model):
    """Kimi K3 with Gated DSA in place of the full-attention MLA layers."""

    state_dict_adapter_cls = KimiK31StateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(KimiK3Model.Config):
        pass

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        super()._register_optimizer_hooks(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)
