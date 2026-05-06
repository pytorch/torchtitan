# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure helpers used by the RL controller's metric construction.

No torch.distributed, Monarch, vLLM, or W&B imports. Tests import this
module directly without the actor stack.
"""

from __future__ import annotations

import math
from collections import defaultdict

from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Step


def _population_std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mean = sum(values) / len(values)
    var = sum((v - mean) * (v - mean) for v in values) / len(values)
    return math.sqrt(max(0.0, var))


def build_reward_component_metrics(
    prefix: str,
    steps: list[Step],
) -> list[m.Metric]:
    """Per-component reward `Mean` metric entries, grouped over observed names only.

    Emits one ``Metric(f"{prefix}/{name}", Mean.from_list(values))`` per
    observed component name. A component appearing in some steps but
    not others gets a Mean over the steps where it appears. We do not
    invent zeros for missing components.

    Args:
        prefix: Full metric prefix (callers pass e.g. ``"reward/component"``
            or ``"validation/reward/component"``).
        steps: Env step results to aggregate.
    """
    component_values: dict[str, list[float]] = defaultdict(list)
    for step in steps:
        for name, value in step.rewards.items():
            component_values[name].append(float(value))
    return [
        m.Metric(f"{prefix}/{name}", m.Mean.from_list(values))
        for name, values in sorted(component_values.items())
    ]
