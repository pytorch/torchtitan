# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-config transform for an fp32-output lm_head."""

from dataclasses import dataclass
from typing import cast, Literal

from torchtitan.models.base.model import BaseModel
from torchtitan.models.base.module import Module
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.linear import Linear

from .base import convert_config_type, ModelConfigTransform
from .utils import matches_fqn

__all__ = ["LMHeadFP32OutputTransform"]


@dataclass(kw_only=True, slots=True)
class LMHeadFP32OutputTransform(ModelConfigTransform):
    """Swap the decoder lm_head's ``Linear.Config`` to ``HiMidLoLinear.Config``.

    Matches ``lm_head`` by exact FQN or dot-delimited suffix. Run before LoRA
    so adapters and freezing wrap the fp32-output implementation. Specialized
    head configs that HiMidLoLinear.Config does not inherit are rejected rather
    than losing their computation or wrappers.
    """

    backward_mode: Literal["hi_mid", "hi_mid_lo"] = "hi_mid"
    """Split the fp32 grad_output into two or three bf16 pieces for backward.

    ``hi_mid_lo`` is exact but slower. For an LM head, accumulation over the
    vocabulary usually hides the gain from a third piece.
    """

    def transform(self, model: Module.Config) -> Module.Config:
        assert isinstance(model, BaseModel.Config)
        found = False
        for fqn, linear_config, parent, attr in model.traverse(Linear.Config):
            if not matches_fqn(fqn, "lm_head"):
                continue
            found = True
            # Already converted, e.g. by a recipe: keep its backward_mode.
            if isinstance(linear_config, HiMidLoLinear.Config):
                continue
            new_config = cast(
                HiMidLoLinear.Config, convert_config_type(linear_config, HiMidLoLinear)
            )
            new_config.backward_mode = self.backward_mode
            assert parent is not None and isinstance(attr, str)
            setattr(parent, attr, new_config)
        if not found:
            raise ValueError(
                "LMHeadFP32OutputTransform found no Linear named 'lm_head' in "
                "the model config. The torchtitan decoder names its output projection "
                "'lm_head' (see torchtitan/models/common/decoder.py)."
            )
        if "fp32_to_bf16_split" not in model.local_compile_regions:
            model.local_compile_regions.append("fp32_to_bf16_split")
        return model
