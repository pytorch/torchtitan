# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-config converter for an fp32-output lm_head."""

from dataclasses import dataclass, fields

from torchtitan.models.common.fp32_output_linear import FP32OutputLinear
from torchtitan.models.common.linear import Linear

from .converter import ModelConfigConverter

__all__ = ["LMHeadFP32OutputConverter"]


class LMHeadFP32OutputConverter(ModelConfigConverter):
    """Swap the decoder lm_head's ``Linear.Config`` to ``FP32OutputLinear.Config``.

    Only the lm_head changes. The same model config backs the trainer and the vLLM
    generator, so both compute fp32 logits with the same op.
    """

    _TARGET = "lm_head"

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        higher_precision_bwd: bool = False
        """Split grad_output into 3 bf16 pieces (exact) instead of 2. Slower (1.1-1.5x backward),
        and the gain may be too small to notice. It matters most for small out_features, e.g. a
        router, not an LM head: the GEMM's own rounding grows with out_features and hides the
        gain."""

    def __init__(self, config: Config):
        self.config = config

    def convert(self, model_config):
        found = False
        for fqn, linear_config, parent, attr in model_config.traverse(Linear.Config):
            if fqn.rsplit(".", 1)[-1] != self._TARGET:
                continue
            found = True
            kwargs = {
                f.name: getattr(linear_config, f.name) for f in fields(linear_config)
            }
            kwargs["higher_precision_bwd"] = self.config.higher_precision_bwd
            new_config = FP32OutputLinear.Config(**kwargs)
            if isinstance(parent, list):
                parent[attr] = new_config
            else:
                setattr(parent, attr, new_config)
        if not found:
            raise ValueError(
                f"LMHeadFP32OutputConverter found no Linear named {self._TARGET!r} in "
                "the model config. The torchtitan decoder names its output projection "
                f"{self._TARGET!r} (see torchtitan/models/common/decoder.py)."
            )
        if "fp32_output_split" not in model_config.local_compile_regions:
            model_config.local_compile_regions.append("fp32_output_split")
        return model_config
