# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The shared-expert override preserves ordinary and latent model ownership."""

from dataclasses import replace

import pytest

from torchtitan.models.common.moe import MoE
from torchtitan.models.kimi_k3.moe import KimiLatentMoE
from torchtitan_recipes.overrides.shared_expert_overlap import shared_expert_overlap
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel
from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


@pytest.mark.parametrize("latent", [False, True])
def test_shared_overlap_override_preserves_layout(latent: bool) -> None:
    """The opt-in keeps router/expert configs shared, including Kimi's latent path."""
    model = (kimi_k3_debugmodel() if latent else deepseek_v3_debugmodel()).model
    config = next(
        moe
        for _, moe, _, _ in model.traverse(MoE.Config)
        if isinstance(moe, KimiLatentMoE.Config) == latent
        and moe.shared_experts is not None
    )
    actual = shared_expert_overlap(config)
    assert type(actual) is type(config)
    assert actual.overlap_shared_experts
    assert not config.overlap_shared_experts
    assert actual.routed_experts is config.routed_experts
    assert actual.shared_experts is config.shared_experts
    assert actual.router is config.router
    unshared = replace(config, shared_experts=None)
    assert shared_expert_overlap(unshared) is unshared
