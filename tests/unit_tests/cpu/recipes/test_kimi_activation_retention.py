# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi activation-retention selection preserves the training workload."""

from dataclasses import replace

import pytest

from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.common.attention.kda import KDA

from torchtitan_recipes.overrides.kimi_k3_activation_retention import (
    retain_kda,
    retain_last_kda,
)
from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


def test_last_kda_retention_does_not_change_workload() -> None:
    """Select the last original KDA FQN without mutating the policy or workload."""
    config = kimi_k3_debugmodel(seq_len=256)
    config = replace(
        config,
        activation_remat=ActivationRematConfig(
            save_regions=["attention.inner_attention"],
            recompute_regions=["*routed_experts*"],
            block_save_regions={"layers.0": ["feed_forward.w2.*"]},
        ),
    )
    actual = retain_last_kda(config)
    kda_ids = [
        layer_id
        for layer_id, layer in enumerate(config.model.layers)
        if isinstance(layer.attention, KDA.Config)
    ]
    assert actual.model is config.model
    assert actual.parallelism is config.parallelism
    assert actual.training is config.training
    assert actual.dataloader is config.dataloader
    assert actual.dist_moe_runtime is config.dist_moe_runtime
    assert actual.activation_remat.save_regions == ["attention.inner_attention"]
    assert actual.activation_remat.recompute_regions == ["*routed_experts*"]
    assert actual.activation_remat.block_save_regions == {
        "layers.0": ["feed_forward.w2.*"],
        f"layers.{kda_ids[-1]}": ["attention.inner_attention"],
    }
    assert config.activation_remat.block_save_regions == {
        "layers.0": ["feed_forward.w2.*"]
    }


def test_retention_override_is_idempotent() -> None:
    """Repeated selection neither duplicates the region nor mutates the input."""
    baseline = FullActivationRematConfig()
    once = retain_kda(baseline, block_fqns=["layers.91"])
    twice = retain_kda(once, block_fqns=["layers.91"])
    assert once == twice
    assert baseline.block_save_regions == {}
    with pytest.raises(ValueError, match="at least one block FQN"):
        retain_kda(baseline, block_fqns=[])


def test_last_kda_retention_rejects_a_model_without_kda() -> None:
    """An invalid KDA selection raises instead of silently retaining nothing."""
    config = kimi_k3_debugmodel(seq_len=256)
    non_kda_layers = [
        layer
        for layer in config.model.layers
        if not isinstance(layer.attention, KDA.Config)
    ]
    config = replace(config, model=replace(config.model, layers=non_kda_layers))
    with pytest.raises(ValueError, match="requires a KDA layer"):
        retain_last_kda(config)
