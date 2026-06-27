# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

from torchtitan.components.optimizer import ParamGroupConfig


@dataclass(frozen=True)
class MuPSpec:
    base_width: int
    num_layers: int
    hidden_pattern: str


def hidden_std(fan_in: int, spec: MuPSpec, *, mup: bool) -> float:
    return fan_in**-0.5 if mup else spec.base_width**-0.5


def residual_std(fan_in: int, spec: MuPSpec, *, mup: bool) -> float:
    return hidden_std(fan_in, spec, mup=mup) / math.sqrt(2 * spec.num_layers)


def output_mult(width: int, spec: MuPSpec, *, mup: bool) -> float:
    return (spec.base_width / width) if mup else 1.0


def param_groups(
    width: int,
    spec: MuPSpec,
    *,
    mup: bool,
    optimizer_name: str,
    optimizer_kwargs: dict,
) -> list[ParamGroupConfig]:
    groups = [
        ParamGroupConfig(
            pattern=r".*",
            optimizer_name=optimizer_name,
            optimizer_kwargs=dict(optimizer_kwargs),
        )
    ]
    if mup:
        # first-match-wins: scale hidden matmuls by lr_mult = 1/m before the catch-all
        m = width / spec.base_width
        groups.insert(
            0,
            ParamGroupConfig(
                pattern=spec.hidden_pattern,
                optimizer_name=optimizer_name,
                lr_mult=1.0 / m,
                optimizer_kwargs=dict(optimizer_kwargs),
            ),
        )
    return groups
