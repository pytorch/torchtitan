# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.muse_spark.audio_encoder import _AudioPatchifier
from torchtitan.models.muse_spark.flavors import muse_spark_audio_encoder_config


def _patchifier_config(patch_size: int) -> _AudioPatchifier.Config:
    return _AudioPatchifier.Config(
        patch_size=patch_size,
        linear=Linear.Config(in_features=1, out_features=8, bias=True),
        norm=RMSNorm.Config(normalized_shape=8),
    )


def test_audio_patch_size_must_be_positive() -> None:
    for patch_size in (0, -1):
        with pytest.raises(
            ValueError, match=rf"audio patch size must be positive, got {patch_size}"
        ):
            _patchifier_config(patch_size)

    config = _patchifier_config(2)
    assert config.patch_size == 2


def test_audio_encoder_config_rejects_zero_patch_size() -> None:
    with pytest.raises(ValueError, match=r"audio patch size must be positive, got 0"):
        muse_spark_audio_encoder_config(
            output_size=4,
            patch_size=0,
            patchify_sizes=(0,),
            hidden_sizes=(8,),
        )
