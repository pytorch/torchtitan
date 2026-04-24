# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from autoparallel._testing.models.dsv3 import DeepSeekV3Model as _DeepSeekV3Model
from torchtitan.protocols.model import BaseModel

from .args import DeepSeekV3ModelArgs


# Need to share same base class with torchtitan models
class DeepSeekV3Model(_DeepSeekV3Model, BaseModel):
    def __init__(self, config: DeepSeekV3ModelArgs):
        _DeepSeekV3Model.__init__(self, config)

    def verify_module_protocol(self) -> None:
        # Autoparallel submodules are standard nn.Modules,
        # not torchtitan Module instances — skip the check.
        pass


# Wire Configurable pattern: build() calls DeepSeekV3Model(config=...)
DeepSeekV3ModelArgs._owner = DeepSeekV3Model
