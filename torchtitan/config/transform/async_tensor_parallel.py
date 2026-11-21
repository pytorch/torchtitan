# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Asynchronous tensor-parallel model transform."""

from dataclasses import dataclass, fields
from typing import cast

from torchtitan.models.base.module import Module
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)

from torchtitan.models.common.linear.async_linear import (
    AsyncColumnParallelLinear,
    AsyncRowParallelLinear,
)
from torchtitan.quantization.mxfp8 import MXFP8Linear
from torchtitan.quantization.utils import get_quantized_linear

from .base import ModelConfigTransform

__all__ = ["AsyncTensorParallelTransform"]


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Replace supported synchronous TP projections with async versions.

    Matching is by exact owner type. A projection subclass may change collective
    semantics, which would be lost if its config were replaced by an async base
    type. The input config is already sharded; each replacement keeps the
    projection's ``sharding_config``.

    MXFP8 column- and row-parallel projections are replaced by their async MXFP8
    versions.

    The BF16 async projections compute WGRAD inside their fused collectives, so
    they cannot add it into ``weight.grad`` in place (``inplace_wgrad_accum``).
    The transform raises for a projection that has the option on unless
    ``disable_inplace_wgrad_accum`` acknowledges turning it off, so switching to
    async TP never silently drops it. The MXFP8 ones compute WGRAD separately
    and keep the option.
    """

    enable_sequence_parallel: bool
    disable_inplace_wgrad_accum: bool = False
    """Turn off ``inplace_wgrad_accum`` on the converted projections instead of
    raising for them."""

    def transform(self, model: Module.Config) -> Module.Config:
        if not self.enable_sequence_parallel:
            raise ValueError("Async tensor parallelism requires sequence parallelism.")

        replacements: dict[type, type] = {
            ColumnParallelLinear: AsyncColumnParallelLinear,
            RowParallelLinear: AsyncRowParallelLinear,
        }
        # Replacements that compute WGRAD outside the fused collectives.
        keeps_inplace_wgrad_accum: set[type] = set()
        if MXFP8Linear is not None:
            from torchtitan.quantization.mxfp8.async_linear import (
                MXFP8AsyncColumnParallelLinear,
                MXFP8AsyncRowParallelLinear,
            )

            replacements[
                get_quantized_linear(MXFP8Linear, ColumnParallelLinear)
            ] = MXFP8AsyncColumnParallelLinear
            replacements[
                get_quantized_linear(MXFP8Linear, RowParallelLinear)
            ] = MXFP8AsyncRowParallelLinear
            keeps_inplace_wgrad_accum = {
                MXFP8AsyncColumnParallelLinear,
                MXFP8AsyncRowParallelLinear,
            }

        conversions = []
        for fqn, config, parent, attr in list(model.traverse(Linear.Config)):
            owner = config._owner
            assert owner is not None
            if owner in replacements:
                conversions.append((fqn, config, parent, attr, replacements[owner]))
        if not self.disable_inplace_wgrad_accum:
            enabled = [
                fqn
                for fqn, config, _, _, replacement in conversions
                if config.inplace_wgrad_accum
                and replacement not in keeps_inplace_wgrad_accum
            ]
            if enabled:
                raise ValueError(
                    "Async tensor parallelism computes WGRAD inside its fused "
                    "collectives and cannot add it into weight.grad in place. Pass "
                    "AsyncTensorParallelTransform(disable_inplace_wgrad_accum=True) "
                    f"or set inplace_wgrad_accum=False on {enabled}."
                )

        for _fqn, config, parent, attr, replacement in conversions:
            # Copy every field, including sharding_config, except
            # inplace_wgrad_accum on BF16 projections, which
            # convert_config_type() cannot drop.
            keep_inplace = replacement in keeps_inplace_wgrad_accum
            converted = cast(
                ColumnParallelLinear.Config | RowParallelLinear.Config,
                replacement.Config(
                    **{
                        f.name: getattr(config, f.name)
                        for f in fields(config)
                        # Checked above; the BF16 async projections reject it.
                        if keep_inplace or f.name != "inplace_wgrad_accum"
                    }
                ),
            )
            if parent is None:
                model = cast(Module.Config, converted)
            elif isinstance(parent, list):
                assert isinstance(attr, int)
                parent[attr] = converted
            else:
                assert isinstance(attr, str)
                setattr(parent, attr, converted)
        return model
