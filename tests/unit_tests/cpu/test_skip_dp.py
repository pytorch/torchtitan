# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM inference calls model.parallelize(..., skip_dp=True)."""

import inspect

import pytest

from torchtitan.models.deepseek_v3 import DeepSeekV3Model
from torchtitan.models.gpt_oss import GptOssModel
from torchtitan.models.kimi_k2_7 import KimiK25Model
from torchtitan.models.kimi_k3 import KimiK3Model
from torchtitan.models.llama3 import Llama3Model
from torchtitan.models.muse_glimmer import MuseGlimmerModel
from torchtitan.models.qwen3 import Qwen3Model
from torchtitan.models.qwen3_5 import Qwen35Model


@pytest.mark.parametrize(
    "model_cls",
    [
        Llama3Model,
        DeepSeekV3Model,
        KimiK25Model,
        KimiK3Model,
        Qwen3Model,
        Qwen35Model,
        GptOssModel,
        MuseGlimmerModel,
    ],
)
def test_parallelize_accepts_skip_dp(model_cls) -> None:
    parameter = inspect.signature(model_cls.parallelize).parameters["skip_dp"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False
