# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import pytest

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.kimi_k3 import build_model_config


def test_multimodal_config_requires_an_encoder() -> None:
    """Kimi K3's vision_encoder is its only encoder, so MultimodalModel.Config
    is the only check that rejects setting it to None."""
    config = build_model_config("debugmodel_mm", parallelism=ParallelismConfig())
    with pytest.raises(ValueError, match="needs at least one of"):
        dataclasses.replace(config, vision_encoder=None)
