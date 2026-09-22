# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""update_ep_token_dispatcher_config fills every static-capacity EP backend."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import pytest

from torchtitan.models.common.token_dispatcher import (
    BaseEPTokenDispatcher,
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
    update_ep_token_dispatcher_config,
)


@dataclass(kw_only=True, slots=True)
class _Plain(BaseEPTokenDispatcher.Config):
    num_max_tokens_per_rank: int | None = None


@dataclass(kw_only=True, slots=True)
class _Static(_Plain):
    static_token_capacity: ClassVar[bool] = True


@dataclass(kw_only=True, slots=True)
class _StaticWithFallback(_Static):
    requires_ep: ClassVar[bool] = False


def _run(cfg, *, ep, cp=1, tp=1, tokens=256):
    moe = SimpleNamespace(routed_experts=SimpleNamespace(token_dispatcher=cfg))
    model_config = SimpleNamespace(traverse=lambda _cls: [(None, moe, None, None)])
    config = SimpleNamespace(
        parallelism=SimpleNamespace(
            expert_parallel_degree=ep,
            context_parallel_degree=cp,
            tensor_parallel_degree=tp,
        ),
        training=SimpleNamespace(num_tokens_per_microbatch_per_dp_rank=tokens),
    )
    update_ep_token_dispatcher_config(model_config, config)
    return cfg


def test_core_persistent_backends_declare_a_static_capacity():
    assert DeepEPTokenDispatcher.Config.static_token_capacity
    assert HybridEPTokenDispatcher.Config.static_token_capacity
    assert DeepEPTokenDispatcher.Config.requires_ep
    assert HybridEPTokenDispatcher.Config.requires_ep


def test_static_capacity_is_the_token_count_after_cp_and_tp():
    cfg = _run(_Static(num_experts=4, top_k=2), ep=2, cp=2, tp=2, tokens=256)
    assert cfg.num_max_tokens_per_rank == 64


def test_static_capacity_without_ep_is_refused():
    with pytest.raises(ValueError, match="requires expert parallelism"):
        _run(_Static(num_experts=4, top_k=2), ep=1)


def test_indivisible_token_count_is_refused():
    with pytest.raises(ValueError, match="divisible"):
        _run(_Static(num_experts=4, top_k=2), ep=2, tp=2, tokens=255)


def test_local_fallback_leaves_the_capacity_unset_without_ep():
    cfg = _run(_StaticWithFallback(num_experts=4, top_k=2), ep=1, tp=2, tokens=255)
    assert cfg.num_max_tokens_per_rank is None


def test_local_fallback_is_filled_under_ep():
    cfg = _run(_StaticWithFallback(num_experts=4, top_k=2), ep=2, tokens=256)
    assert cfg.num_max_tokens_per_rank == 256


def test_backends_without_a_static_capacity_are_left_alone():
    cfg = _run(_Plain(num_experts=4, top_k=2), ep=1)
    assert cfg.num_max_tokens_per_rank is None
