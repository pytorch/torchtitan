# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import torchtitan.distributed.flex_ep.flex_ep as flex_ep_mod
from torchtitan.models.common.config_utils import make_experts_config
from torchtitan.models.common.moe import FlexGroupedExperts, _pack_flex_ep_w13
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel_flex_ep,
)


@pytest.fixture(autouse=True)
def reset_flex_ep_workspace_cache():
    flex_ep_mod._clear_flex_ep_workspace_cache()
    yield
    flex_ep_mod._clear_flex_ep_workspace_cache()


def test_make_experts_config_flex_ep():
    experts_config = make_experts_config(
        dim=16,
        hidden_dim=32,
        num_experts=8,
        top_k=2,
        param_init={
            "w1": torch.nn.init.zeros_,
            "w2": torch.nn.init.zeros_,
            "w3": torch.nn.init.zeros_,
        },
        score_before_experts=False,
        comm_backend="flex_ep",
    )

    assert isinstance(experts_config, FlexGroupedExperts.Config)
    assert experts_config.num_experts == 8
    assert experts_config.top_k == 2
    assert not experts_config.score_before_experts


def test_make_experts_config_flex_ep_rejects_score_before_experts():
    with pytest.raises(ValueError, match="score_before_experts=False"):
        make_experts_config(
            dim=16,
            hidden_dim=32,
            num_experts=8,
            top_k=2,
            param_init={
                "w1": torch.nn.init.zeros_,
                "w2": torch.nn.init.zeros_,
                "w3": torch.nn.init.zeros_,
            },
            score_before_experts=True,
            comm_backend="flex_ep",
        )


def test_flex_grouped_experts_builds_without_token_dispatcher():
    config = FlexGroupedExperts.Config(
        dim=16,
        hidden_dim=32,
        num_experts=8,
        top_k=2,
        param_init={
            "w1": torch.nn.init.zeros_,
            "w2": torch.nn.init.zeros_,
            "w3": torch.nn.init.zeros_,
        },
    )
    with torch.device("meta"):
        experts = config.build()

    assert isinstance(experts, FlexGroupedExperts)
    assert experts.num_experts == 8
    assert not hasattr(experts, "token_dispatcher")
    assert experts.ep_mesh is None


def test_deepseek_v3_flex_ep_config_uses_flex_grouped_experts():
    config = deepseek_v3_debugmodel_flex_ep()
    moe_config = next(
        layer.moe for layer in config.model_spec.model.layers if layer.moe is not None
    )

    assert isinstance(moe_config.experts, FlexGroupedExperts.Config)


@pytest.mark.parametrize(
    "parallelism_field",
    (
        "tensor_parallel_degree",
        "context_parallel_degree",
        "pipeline_parallel_degree",
    ),
)
def test_deepseek_v3_flex_ep_config_rejects_tp_cp_pp(parallelism_field):
    config = deepseek_v3_debugmodel_flex_ep()
    setattr(config.parallelism, parallelism_field, 2)

    with pytest.raises(ValueError, match="DP\\+EP"):
        config.model_spec.model.update_from_config(trainer_config=config)


def test_flex_w13_pack_helper():
    w1 = torch.randn(4, 8, 16)
    w3 = torch.randn(4, 8, 16)

    w13 = _pack_flex_ep_w13(w1, w3)

    assert w13.shape == (4, 16, 16)
    torch.testing.assert_close(w13[:, :8], w1)
    torch.testing.assert_close(w13[:, 8:], w3)


def test_flex_ep_routers_reuse_ep1_workspace(monkeypatch):
    monkeypatch.setattr(flex_ep_mod, "_ensure_flex_ep_imported", lambda: None)
    router1 = flex_ep_mod.FlexEPRouter.create(
        max_tokens=16,
        dim=8,
        num_experts=4,
        top_k=2,
        device=torch.device("cpu"),
        ep_mesh=None,
    )
    router2 = flex_ep_mod.FlexEPRouter.create(
        max_tokens=16,
        dim=8,
        num_experts=4,
        top_k=2,
        device=torch.device("cpu"),
        ep_mesh=None,
    )

    assert router1 is not router2
    assert router1.workspace is router2.workspace
    assert router1.raw.data_ptr() == router2.raw.data_ptr()


def test_flex_ep_balanced_capacity_is_smaller_than_fanout_capacity():
    max_tokens = 8 * 2048
    ep_size = 2
    num_experts = 8
    top_k = 3
    local_experts = num_experts // ep_size

    balanced_capacity = flex_ep_mod._compute_max_tokens_recv(
        max_tokens=max_tokens,
        ep_size=ep_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    old_fanout_capacity = (
        flex_ep_mod._align_up(
            max_tokens * ep_size * min(local_experts, top_k),
            flex_ep_mod.TOKEN_ALIGNMENT,
        )
        + flex_ep_mod.TOKEN_ALIGNMENT * local_experts
    )

    assert balanced_capacity == 49152
    assert balanced_capacity < old_fanout_capacity
    assert (
        flex_ep_mod._compute_dispatch_recv_weights_numel(
            max_tokens,
            balanced_capacity,
            top_k,
        )
        == 98304
    )


def test_nvl_shared_buffer_accepts_larger_buffer_and_rejects_too_small():
    kwargs = {
        "max_tokens": 16,
        "dim": 8,
        "ep_size": 1,
        "num_experts": 4,
        "top_k": 2,
    }
    exact_size = flex_ep_mod.NvlSharedBuffer.get_buffer_size_bytes(**kwargs)
    larger_raw = torch.empty(exact_size + 1024, dtype=torch.uint8)

    view = flex_ep_mod.NvlSharedBuffer.view_from_buffer(larger_raw, **kwargs)

    assert view.raw.data_ptr() == larger_raw.data_ptr()
    with pytest.raises(ValueError, match="raw buffer is too small"):
        flex_ep_mod.NvlSharedBuffer.view_from_buffer(
            torch.empty(exact_size - 1, dtype=torch.uint8),
            **kwargs,
        )


def test_flex_ep_balanced_capacity_guard_mentions_balanced_routing():
    local_experts_start = torch.tensor([0, 128, 384], dtype=torch.int64)

    with pytest.raises(ValueError, match="balanced routing"):
        flex_ep_mod._validate_balanced_routing_capacity(
            local_experts_start,
            max_recv_tokens=256,
        )
