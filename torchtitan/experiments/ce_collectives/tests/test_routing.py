# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.experiments.ce_collectives.collectives import domain_rank_groups
from torchtitan.experiments.ce_collectives.configs import CECollectivesConfig


@pytest.mark.parametrize(
    "num_domains,ranks_per_domain", [(1, 4), (2, 2), (2, 64), (3, 4)]
)
def test_hierarchical_shards(num_domains, ranks_per_domain):
    world_size = num_domains * ranks_per_domain
    domains, lanes = domain_rank_groups(world_size, ranks_per_domain)
    torch.manual_seed(42)
    gradients = torch.randn(world_size, world_size, 7, dtype=torch.float64)
    partials = torch.empty(world_size, num_domains, 7, dtype=torch.float64)
    for domain_ranks in domains:
        for lane, rank in enumerate(domain_ranks):
            destinations = lanes[lane]
            received = gradients[domain_ranks][:, destinations]
            partials[rank] = received.sum(dim=0)
    actual = torch.empty(world_size, 7, dtype=torch.float64)
    for lane_ranks in lanes:
        for domain, rank in enumerate(lane_ranks):
            actual[rank] = partials[lane_ranks, domain].sum(dim=0)
    torch.testing.assert_close(actual, gradients.sum(dim=0))


@pytest.mark.parametrize(
    "world_size,ranks_per_domain", [(4, 0), (4, 1), (4, 3), (128, 72), (128, 128)]
)
def test_invalid_domains(world_size, ranks_per_domain):
    with pytest.raises(ValueError, match="ranks_per_domain"):
        domain_rank_groups(world_size, ranks_per_domain)


def test_invalid_collective_options():
    with pytest.raises(ValueError, match="at least one"):
        CECollectivesConfig(all_gather=False, reduce_scatter=False)
    with pytest.raises(ValueError, match="ranks_per_domain"):
        CECollectivesConfig(ranks_per_domain=128)
    with pytest.raises(ValueError, match="scope"):
        CECollectivesConfig(scope="experts")


def test_recipe_conversion():
    from torchtitan.experiments.ce_collectives.trainer import CETrainer
    from torchtitan_recipes.ce_collectives import ce_debug

    config = ce_debug()
    assert config._owner is CETrainer
    assert config.training.steps >= 10
    assert config.ce_collectives.all_gather and config.ce_collectives.reduce_scatter
    assert config.debug.seed == 42 and config.debug.deterministic
