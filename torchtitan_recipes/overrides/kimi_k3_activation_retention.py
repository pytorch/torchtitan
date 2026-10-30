# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recipe-time block activation-retention policies for Kimi K3."""

from dataclasses import replace

from torchtitan.distributed.activation_remat import ActivationRematConfig
from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.trainer import Trainer


def retain_kda(
    config: ActivationRematConfig, *, block_fqns: list[str]
) -> ActivationRematConfig:
    """Retain KDA cores only in explicitly selected original layer FQNs.

    Args:
        config: Existing global checkpoint policy.
        block_fqns: Block-FQN globs, such as ["layers.91"], not PP-local IDs.

    Returns:
        An additive remat policy. Existing recompute rules still win over
        saves. This recipe-time remat policy composes with model overrides.
    """
    if not block_fqns:
        raise ValueError("retain_kda requires at least one block FQN")
    block_regions = {
        name: list(regions) for name, regions in config.block_save_regions.items()
    }
    for block_fqn in block_fqns:
        regions = block_regions.setdefault(block_fqn, [])
        if "attention.inner_attention" not in regions:
            regions.append("attention.inner_attention")
    return ActivationRematConfig(
        save_regions=list(config.save_regions),
        recompute_regions=list(config.recompute_regions),
        block_save_regions=block_regions,
        determinism_check=config.determinism_check,
    )


def retain_last_kda(config: Trainer.Config) -> Trainer.Config:
    """Retain one late KDA core while leaving expert recomputation unchanged.

    The existing model/data/parallelism and global checkpoint policy are
    preserved. Additional KDA saves increase retained activation memory;
    the saved-tensor hooks determine placement. Account for that memory
    with FakePG before a production submission.

    Args:
        config: Kimi K3 training config with an activation remat policy.

    Returns:
        A training config sharing the original workload and using an additive
        save policy for its last original KDA block.

    Raises:
        ValueError: If the model is not Kimi K3, the remat policy is absent,
            or the model has no KDA block.
    """
    if not isinstance(config.model, KimiK3Model.Config):
        raise ValueError("retain_last_kda requires a Kimi K3 model")
    current = config.activation_remat
    if not isinstance(current, ActivationRematConfig):
        raise ValueError("retain_last_kda requires an activation remat policy")
    layer_ids = [
        layer_id
        for layer_id, layer in enumerate(config.model.layers)
        if isinstance(layer.attention, KDA.Config)
    ]
    if not layer_ids:
        raise ValueError("retain_last_kda requires a KDA layer")
    checkpoint = retain_kda(current, block_fqns=[f"layers.{layer_ids[-1]}"])
    return replace(config, activation_remat=checkpoint)
