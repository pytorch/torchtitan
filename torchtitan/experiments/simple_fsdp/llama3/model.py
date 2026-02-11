# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.llama3 import Transformer

from ..simple_fsdp import disable_active_parametrization


class SimpleFSDPTransformer(Transformer):
    @dataclass(kw_only=True, slots=True)
    class Config(Transformer.Config):
        pass

    def __init__(self, config: "SimpleFSDPTransformer.Config"):
        super().__init__(config)

    def init_weights(self, *args, **kwargs):
        with disable_active_parametrization():
            super().init_weights(*args, **kwargs)
