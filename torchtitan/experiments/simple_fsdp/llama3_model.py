# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.llama3 import Transformer, TransformerModelArgs
from .simple_fsdp import disable_data_parallel


class SimpleFSDPTransformer(Transformer):
    def __init__(self, model_args: TransformerModelArgs):
        super().__init__(model_args)
        self.init_weights()

    def init_weights(self, *args, **kwargs):
        with disable_data_parallel():
            super().init_weights(*args, **kwargs)
