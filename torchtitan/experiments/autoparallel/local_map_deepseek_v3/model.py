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
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        # Call _DeepSeekV3Model.__init__ which calls nn.Module.__init__
        # Note: We don't call BaseModel.__init__ separately because:
        # 1. nn.Module.__init__() is already called by _DeepSeekV3Model.__init__
        # 2. Calling BaseModel.__init__ after would reset all module state
        #    (nn.Module.__init__ clears _modules, _parameters, etc.)
        _DeepSeekV3Model.__init__(self, model_args)
