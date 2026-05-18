# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, fields, replace

from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class TransformersBackendConfig(Trainer.Config):
    hf_model: str = ""
    """HuggingFace model ID (e.g., 'Qwen/Qwen2.5-7B')"""

    def build(self, **kwargs):
        from torchtitan.experiments.transformers_modeling_backend.trainer import (
            SFTTrainer,
        )

        if not kwargs:
            return SFTTrainer(config=replace(self))

        config_fields = {f.name for f in fields(self)}
        overlap = config_fields & kwargs.keys()
        if overlap:
            raise ValueError(
                f"build() kwargs {overlap} overlap with config fields. "
                "Put these values in the Config, not in build() kwargs."
            )
        return SFTTrainer(config=replace(self), **kwargs)
