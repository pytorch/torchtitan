# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 3 is licensed under the LLAMA 3 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.models.llama_multimodal.model import (
    ModelArgs,
    MultimodalDecoder,
    VisionEncoder,
)

__all__ = ["VisionEncoder", "ModelArgs", "MultimodalDecoder"]

llama3_2_configs = {
    # TODO: add configs for llama3.2
}
