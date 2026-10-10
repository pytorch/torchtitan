# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Asynchronous tensor-parallel model transform."""

from dataclasses import dataclass, fields
from typing import cast

from torchtitan.models.common.async_linear import (
    AsyncColumnParallelLinear,
    AsyncRowParallelLinear,
)
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
    SharedExpertRowParallelLinear,
)
from torchtitan.protocols.module import Module

from .base import ModelConfigTransform, ModelConfigTransformContext

__all__ = ["AsyncTensorParallelTransform"]


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Replace supported synchronous TP projections with async versions.

    Matching is by exact owner type. A projection subclass may change collective
    semantics, which would be lost if its config were replaced by an async base
    type. ``SharedExpertRowParallelLinear`` is handled explicitly: async TP
    requires sequence parallelism, where its reduction is identical to
    ``RowParallelLinear`` and conversion to ``AsyncRowParallelLinear`` is safe.

    The async projections compute WGRAD inside their fused collectives, so they
    cannot add it into ``weight.grad`` in place (``inplace_wgrad_accum``). The
    transform raises for a projection that has the option on unless
    ``disable_inplace_wgrad_accum`` acknowledges turning it off, so switching to
    async TP never silently drops it.
    """

    enable_sequence_parallel: bool
    disable_inplace_wgrad_accum: bool = False
    """Turn off ``inplace_wgrad_accum`` on the converted projections instead of
    raising for them."""

    def transform(
        self,
        model: Module.Config,
        *,
        context: ModelConfigTransformContext | None = None,
    ) -> Module.Config:
        del context
        if not self.enable_sequence_parallel:
            raise ValueError("Async tensor parallelism requires sequence parallelism.")

        conversions = []
        for fqn, config, parent, attr in list(model.traverse(Linear.Config)):
            owner = config._owner
            assert owner is not None
            if owner is ColumnParallelLinear:
                conversions.append(
                    (fqn, config, parent, attr, AsyncColumnParallelLinear)
                )
            elif owner in (RowParallelLinear, SharedExpertRowParallelLinear):
                conversions.append((fqn, config, parent, attr, AsyncRowParallelLinear))
        if not self.disable_inplace_wgrad_accum:
            enabled = [
                fqn
                for fqn, config, _, _, _ in conversions
                if config.inplace_wgrad_accum
            ]
            if enabled:
                raise ValueError(
                    "Async tensor parallelism computes WGRAD inside its fused "
                    "collectives and cannot add it into weight.grad in place. Pass "
                    "AsyncTensorParallelTransform(disable_inplace_wgrad_accum=True) "
                    f"or set inplace_wgrad_accum=False on {enabled}."
                )

        for _fqn, config, parent, attr, replacement in conversions:
            # Do not use convert_config_type() here. It requires the replacement
            # config to inherit the source config, but the explicitly supported
            # SharedExpertRowParallelLinear -> AsyncRowParallelLinear conversion
            # intentionally drops the shared expert's unused non-SP behavior.
            converted = cast(
                ColumnParallelLinear.Config | RowParallelLinear.Config,
                replacement.Config(
                    **{
                        f.name: getattr(config, f.name)
                        for f in fields(config)
                        # Checked above; the async projections reject it.
                        if f.name != "inplace_wgrad_accum"
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
