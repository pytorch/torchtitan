# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.rl.controller.controller import _generator_max_num_seqs
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.routing.intra_generator import IntraGeneratorRouter
from torchtitan.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    StickySessionRoutingStrategy,
)


def _max_num_seqs(
    *,
    rollout_concurrency: int = 80,
    num_generators: int = 4,
    generator_dp_degree: int = 4,
    strategy=None,
) -> tuple[int, float]:
    routers = {}
    if strategy is not None:
        routers = {
            "generator_router": InterGeneratorRouter.Config(strategy=strategy),
            "intra_generator_router": IntraGeneratorRouter.Config(strategy=strategy),
        }
    return _generator_max_num_seqs(
        rollout_concurrency=rollout_concurrency,
        num_generators=num_generators,
        generator_dp_degree=generator_dp_degree,
        generator_router=routers.get("generator_router", InterGeneratorRouter.Config()),
        intra_generator_router=routers.get(
            "intra_generator_router", IntraGeneratorRouter.Config()
        ),
    )


def test_sticky_routing_sizes_ranks_for_the_load_it_tolerates_before_repinning() -> None:
    # 80 sequences over 4 generators x 4 DP ranks: an even share is 5. Both sticky
    # layers repin above 1.5x the mean, so one rank can reach 1.5 x 1.5 x 5 = 11.25.
    assert _max_num_seqs() == (12, 2.25)


def test_only_routing_layers_with_several_candidates_count() -> None:
    assert _max_num_seqs(num_generators=1) == (30, 1.5)
    assert _max_num_seqs(generator_dp_degree=1) == (30, 1.5)
    assert _max_num_seqs(num_generators=1, generator_dp_degree=1) == (80, 1.0)


def test_inter_and_intra_router_strategies_are_sized_independently() -> None:
    assert _generator_max_num_seqs(
        rollout_concurrency=80,
        num_generators=4,
        generator_dp_degree=4,
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(repin_load_ratio=2.0)
        ),
        intra_generator_router=IntraGeneratorRouter.Config(
            strategy=LeastLoadedRoutingStrategy.Config()
        ),
    ) == (10, 2.0)
    assert _generator_max_num_seqs(
        rollout_concurrency=80,
        num_generators=4,
        generator_dp_degree=4,
        generator_router=InterGeneratorRouter.Config(
            strategy=LeastLoadedRoutingStrategy.Config()
        ),
        intra_generator_router=IntraGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(repin_load_ratio=3.0)
        ),
    ) == (15, 3.0)


def test_routing_without_a_repin_bound_keeps_an_even_share() -> None:
    assert _max_num_seqs(strategy=LeastLoadedRoutingStrategy.Config()) == (5, 1.0)
    never_repins = StickySessionRoutingStrategy.Config(repin_load_ratio=None)
    assert _max_num_seqs(strategy=never_repins) == (5, 1.0)


def test_max_num_seqs_stays_capped() -> None:
    assert _max_num_seqs(rollout_concurrency=10_000)[0] == 512
