# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.trainer import Trainer

from tests.integration_tests.base_config import apply_base_settings

default_config = Trainer.Config()
apply_base_settings(default_config)
default_config.model_converters.converters = [
    Float8LinearConverter.Config(
        enable_fsdp_float8_all_gather=True,
        precompute_float8_dynamic_scale_for_fsdp=True,
        emulate=True,
    ),
]
