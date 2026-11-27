# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transform that swaps HiMidLoLinear for FullPrecisionLinear."""

from dataclasses import fields

from torchtitan.models.base.module import Module
from torchtitan.models.common.linear.full_precision_linear import FullPrecisionLinear
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear

from .base import ModelConfigTransform

__all__ = ["FullPrecisionLinearTransform"]


class FullPrecisionLinearTransform(ModelConfigTransform):
    """Swap every ``HiMidLoLinear`` (router gates, an fp32-output LM head) for a
    ``FullPrecisionLinear``.

    ``HiMidLoLinear`` runs bf16 GEMMs on CUDA and raises where they do not apply:
    batch-invariant mode (cuBLAS's bf16-input, fp32-output GEMM is not batch-invariant), CPU,
    or fp32 weights (no bf16 mixed precision). ``FullPrecisionLinear`` runs fp32 matmuls there,
    with its weight kept in fp32.
    """

    def transform(self, model: Module.Config) -> Module.Config:
        for fqn, config, parent, attr in list(model.traverse(HiMidLoLinear.Config)):
            if type(config) is not HiMidLoLinear.Config:
                raise ValueError(
                    f"FullPrecisionLinearTransform only converts a plain "
                    f"HiMidLoLinear.Config, but {fqn!r} is a {type(config).__qualname__}, "
                    "whose extra fields or forward FullPrecisionLinear would drop."
                )
            kwargs = {
                f.name: getattr(config, f.name)
                for f in fields(config)
                if f.name != "backward_mode"
            }
            kwargs["preserve_dtype_params"] = tuple(
                dict.fromkeys((*config.preserve_dtype_params, "weight"))
            )
            new_config = FullPrecisionLinear.Config(**kwargs)
            if parent is None:
                model = new_config
            elif isinstance(parent, list):
                assert isinstance(attr, int)
                parent[attr] = new_config
            else:
                assert isinstance(attr, str)
                setattr(parent, attr, new_config)
        return model
