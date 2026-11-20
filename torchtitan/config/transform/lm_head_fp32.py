# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transform for an fp32-output lm_head."""

from dataclasses import dataclass, fields
from typing import Literal

from torchtitan.models.common.linear import Linear

from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear

from .base import ModelConfigTransform

__all__ = ["LMHeadFP32OutputTransform"]


@dataclass(kw_only=True, slots=True)
class LMHeadFP32OutputTransform(ModelConfigTransform):
    """Swap the decoder lm_head's ``Linear.Config`` to ``HiMidLoLinear.Config``.

    Only the lm_head changes. The same model config backs the trainer and the vLLM
    generator, so both compute fp32 logits with the same op.
    """

    _TARGET = "lm_head"

    backward_mode: Literal["hi_mid", "hi_mid_lo"] = "hi_mid"
    """How many bf16 pieces the fp32 grad_output is split into for the backward GEMMs.
    "hi_mid_lo" is exact, but slower (1.1-1.5x backward), and the gain may be too small to
    notice. It matters most for small out_features, e.g. a router, not an LM head: the GEMM's
    own rounding grows with out_features and hides the gain."""

    def transform(self, model):
        found = False
        for fqn, linear_config, parent, attr in model.traverse(Linear.Config):
            if fqn.rsplit(".", 1)[-1] != self._TARGET:
                continue
            found = True
            # Already converted, e.g. by a recipe: keep its backward_mode.
            if isinstance(linear_config, HiMidLoLinear.Config):
                continue
            if type(linear_config) is not Linear.Config:
                raise ValueError(
                    f"LMHeadFP32OutputTransform only converts a plain Linear.Config, but "
                    f"{fqn!r} is a {type(linear_config).__qualname__}, whose extra fields "
                    "or forward (e.g. quantization or a logit soft cap) HiMidLoLinear "
                    "would drop. If a quantization transform converted it, exclude "
                    f"{self._TARGET!r} with that transform's fqns."
                )
            kwargs = {
                f.name: getattr(linear_config, f.name) for f in fields(linear_config)
            }
            kwargs["backward_mode"] = self.backward_mode
            new_config = HiMidLoLinear.Config(**kwargs)
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)
        if not found:
            raise ValueError(
                f"LMHeadFP32OutputTransform found no Linear named {self._TARGET!r} in "
                "the model config. The torchtitan decoder names its output projection "
                f"{self._TARGET!r} (see torchtitan/models/common/decoder.py)."
            )
        if "fp32_to_bf16_split" not in model.local_compile.regions:
            model.local_compile.regions.append("fp32_to_bf16_split")
        return model
