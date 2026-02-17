# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torchtitan.config.configs import JobConfig
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class TransformersBackendJobConfig(JobConfig):
    hf_model: str = ""
    """HuggingFace model ID (e.g., 'Qwen/Qwen2.5-7B')"""


@dataclass(kw_only=True, slots=True)
class TransformersBackendConfig(Trainer.Config):
    job: TransformersBackendJobConfig = field(
        default_factory=TransformersBackendJobConfig
    )
