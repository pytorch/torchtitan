# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.llama3 import Llama3Model
from torchtitan.protocols.module import NamedParamInitializer

from ..simple_fsdp import disable_active_parametrization


class GraphTrainerLlama3Model(Llama3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def init_states(
        self,
        *,
        param_init: NamedParamInitializer | None = None,
        param_prefix: str = "",
        **kwargs,
    ) -> None:
        with disable_active_parametrization():
            super().init_states(
                param_init=param_init, param_prefix=param_prefix, **kwargs
            )
