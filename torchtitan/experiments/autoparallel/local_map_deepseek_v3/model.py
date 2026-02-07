# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from autoparallel._testing.models.dsv3 import DeepSeekV3Model as _DeepSeekV3Model
from torchtitan.protocols.train_spec import ModelProtocol

from .args import DeepSeekV3ModelArgs


# Need to share same base class with torchtitan models
class DeepSeekV3Model(_DeepSeekV3Model, ModelProtocol):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__(model_args)
