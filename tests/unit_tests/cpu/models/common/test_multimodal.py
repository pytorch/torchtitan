# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import pytest

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.muse_spark import build_model_config


def test_multimodal_config_requires_an_encoder() -> None:
    """Muse Spark's encoders are each optional, so MultimodalModel.Config is
    what rejects a multimodal config with none."""
    config = build_model_config("debugmodel_mm", parallelism=ParallelismConfig())
    dataclasses.replace(config, audio_encoder=None, dataset_modalities=("vision",))
    with pytest.raises(ValueError, match="needs at least one of"):
        dataclasses.replace(
            config, audio_encoder=None, vision_encoder=None, dataset_modalities=()
        )
