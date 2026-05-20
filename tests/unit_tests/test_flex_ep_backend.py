# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import torchtitan.distributed.flex_ep.flex_ep as flex_ep_mod
from torchtitan.models.common.config_utils import make_experts_config
from torchtitan.models.common.moe import _pack_flex_ep_w13, FlexGroupedExperts
from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel_flex_ep


@pytest.fixture(autouse=True)
def reset_flex_ep_workspace_cache():
    flex_ep_mod._clear_flex_ep_workspace_cache()
    yield
    flex_ep_mod._clear_flex_ep_workspace_cache()


class _FakeEPMesh:
    def __init__(self, *, ep_size: int = 2, ep_rank: int = 0) -> None:
        self._ep_size = ep_size
        self._ep_rank = ep_rank
        self._group = object()

    def size(self) -> int:
        return self._ep_size

    def get_local_rank(self) -> int:
        return self._ep_rank

    def get_group(self) -> object:
        return self._group


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
    assert experts_config.flex_ep_capacity_factor == 1.0


def test_make_experts_config_flex_ep_threads_capacity_factor():
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
        non_blocking_capacity_factor=0.5,
    )

    assert isinstance(experts_config, FlexGroupedExperts.Config)
    assert experts_config.flex_ep_capacity_factor == 0.5


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


@pytest.mark.parametrize("capacity_factor", (0, -0.5, 1.1))
def test_make_experts_config_flex_ep_rejects_invalid_capacity_factor(
    capacity_factor,
):
    with pytest.raises(ValueError, match="0 < non_blocking_capacity_factor <= 1.0"):
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
            score_before_experts=False,
            comm_backend="flex_ep",
            non_blocking_capacity_factor=capacity_factor,
        )


@pytest.mark.parametrize("capacity_factor", (0, -0.5, 1.1))
def test_flex_grouped_experts_rejects_invalid_capacity_factor(capacity_factor):
    config = FlexGroupedExperts.Config(
        dim=16,
        hidden_dim=32,
        num_experts=8,
        top_k=2,
        flex_ep_capacity_factor=capacity_factor,
        param_init={
            "w1": torch.nn.init.zeros_,
            "w2": torch.nn.init.zeros_,
            "w3": torch.nn.init.zeros_,
        },
    )

    with pytest.raises(ValueError, match="0 < flex_ep_capacity_factor <= 1.0"):
        with torch.device("meta"):
            config.build()


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
    assert experts.flex_ep_capacity_factor == 1.0


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
    config.parallelism.expert_parallel_degree = 2
    setattr(config.parallelism, parallelism_field, 2)

    with pytest.raises(ValueError, match="DP\\+EP"):
        config.model_spec.model.update_from_config(trainer_config=config)


def test_deepseek_v3_flex_ep_config_requires_expert_parallelism():
    config = deepseek_v3_debugmodel_flex_ep()

    with pytest.raises(ValueError, match="expert_parallel_degree > 1"):
        config.model_spec.model.update_from_config(trainer_config=config)


@pytest.mark.parametrize("ep_mesh", (None, _FakeEPMesh(ep_size=1)))
def test_flex_ep_router_requires_multi_rank_ep_mesh(ep_mesh):
    with pytest.raises(ValueError, match="expert_parallel_degree > 1"):
        flex_ep_mod.FlexEPRouter.create(
            max_tokens=16,
            dim=8,
            num_experts=4,
            top_k=2,
            ep_mesh=ep_mesh,
        )


def test_deepseek_v3_flex_ep_config_sets_router_force_load_balance_only():
    config = deepseek_v3_debugmodel_flex_ep()
    config.parallelism.expert_parallel_degree = 2
    config.debug.moe_force_load_balance = True

    config.model_spec.model.update_from_config(trainer_config=config)
    moe_config = next(
        layer.moe for layer in config.model_spec.model.layers if layer.moe is not None
    )

    assert moe_config.router._debug_force_load_balance
    assert isinstance(moe_config.experts, FlexGroupedExperts.Config)
    assert not hasattr(moe_config.experts, "_debug_force_load_balance")


def test_flex_w13_pack_helper():
    w1 = torch.randn(4, 8, 16)
    w3 = torch.randn(4, 8, 16)

    w13 = _pack_flex_ep_w13(w1, w3)

    assert w13.shape == (4, 16, 16)
    torch.testing.assert_close(w13[:, :8], w1)
    torch.testing.assert_close(w13[:, 8:], w3)


def test_flex_ep_routing_uses_factor_capacity():
    max_tokens = 8 * 2048
    ep_size = 2
    num_experts = 8
    top_k = 3
    local_experts = num_experts // ep_size

    capacity = flex_ep_mod._compute_max_tokens_recv(
        max_tokens=max_tokens,
        ep_size=ep_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    half_capacity = flex_ep_mod._compute_max_tokens_recv(
        max_tokens=max_tokens,
        ep_size=ep_size,
        num_experts=num_experts,
        top_k=top_k,
        capacity_factor=0.5,
    )
    expected_capacity = (
        flex_ep_mod._align_up(
            max_tokens * ep_size * min(local_experts, top_k),
            flex_ep_mod.TOKEN_ALIGNMENT,
        )
        + flex_ep_mod.TOKEN_ALIGNMENT * local_experts
    )
    expected_half_factor_capacity = (
        flex_ep_mod._align_up(
            max_tokens * ep_size * min(local_experts, top_k) // 2,
            flex_ep_mod.TOKEN_ALIGNMENT,
        )
        + flex_ep_mod.TOKEN_ALIGNMENT * local_experts
    )

    assert capacity == expected_capacity
    assert half_capacity == expected_half_factor_capacity
    assert (
        flex_ep_mod._compute_dispatch_recv_weights_numel(
            max_tokens,
            capacity,
            top_k,
        )
        == capacity
    )


def test_nvl_shared_buffer_accepts_larger_buffer_and_rejects_too_small():
    kwargs = {
        "max_tokens": 16,
        "dim": 8,
        "ep_size": 2,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flex_ep_weighted_sum_matches_reference_on_cuda():
    try:
        flex_ep_mod._register_ep_backend_ops()
    except (ImportError, RuntimeError) as exc:
        pytest.skip(str(exc))

    device = torch.device("cuda")
    y_partial = torch.randn(
        3,
        2,
        4,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    top_scores = torch.randn(
        3,
        2,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    y_partial_ref = y_partial.detach().clone().requires_grad_()
    top_scores_ref = top_scores.detach().clone().requires_grad_()
    grad = torch.randn(3, 4, device=device, dtype=torch.bfloat16)

    out = flex_ep_mod.flex_ep_weighted_sum(y_partial, top_scores)
    ref = (
        (
            y_partial_ref.to(torch.float32)
            * top_scores_ref.to(torch.float32).unsqueeze(-1)
        )
        .sum(dim=1)
        .to(torch.bfloat16)
    )
    out.backward(grad)
    ref.backward(grad)

    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(y_partial.grad, y_partial_ref.grad)
    torch.testing.assert_close(top_scores.grad, top_scores_ref.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flex_ep_registered_offset_kernels_match_reference_prefix():
    flex_ep_mod._ensure_flex_ep_imported()
    try:
        flex_ep_mod._register_ep_backend_ops()
    except (ImportError, RuntimeError) as exc:
        pytest.skip(str(exc))

    device = torch.device("cuda")
    tokens = 9
    hidden = 16
    token_end_value = 5
    generator = torch.Generator(device=device).manual_seed(42)
    token_end = torch.tensor([token_end_value], device=device, dtype=torch.int64)
    y1 = torch.randn(
        tokens,
        2 * hidden,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    dy2 = torch.randn(
        tokens,
        hidden,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )

    y2 = torch.ops.inductor.flex_ep_swiglu_forward_with_offsets(y1, token_end)
    gate, up = y1[:token_end_value].chunk(2, dim=-1)
    ref_y2 = (
        torch.nn.functional.silu(gate.to(torch.float32)) * up.to(torch.float32)
    ).to(torch.bfloat16)
    torch.testing.assert_close(y2[:token_end_value], ref_y2)

    dy1 = torch.ops.inductor.flex_ep_swiglu_backward_with_offsets(
        dy2,
        y1,
        token_end,
    )
    dy2_prefix = dy2[:token_end_value].to(torch.float32)
    gate_fp32 = gate.to(torch.float32)
    up_fp32 = up.to(torch.float32)
    sig = torch.sigmoid(gate_fp32)
    ref_dgate = dy2_prefix * up_fp32 * sig * (1 + gate_fp32 * (1 - sig))
    ref_dup = dy2_prefix * torch.nn.functional.silu(gate_fp32)
    ref_dy1 = torch.cat((ref_dgate, ref_dup), dim=-1).to(torch.bfloat16)
    torch.testing.assert_close(dy1[:token_end_value], ref_dy1)

    input = (
        torch.arange(
            tokens * 7,
            device=device,
            dtype=torch.float32,
        )
        .view(tokens, 7)
        .to(torch.bfloat16)
    )
    cloned = torch.ops.inductor.flex_ep_clone_valid_prefix(input, token_end)
    torch.testing.assert_close(cloned[:token_end_value], input[:token_end_value])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("top_k", (6, 8))
def test_flex_ep_registered_router_metadata_kernels_match_reference(top_k):
    flex_ep_mod._ensure_flex_ep_imported()
    try:
        flex_ep_mod._register_ep_backend_ops()
    except (ImportError, RuntimeError) as exc:
        pytest.skip(str(exc))

    ep_rank = 3
    ep_size = 8
    local_experts = 8
    num_experts = ep_size * local_experts
    batch = 257
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(42 + top_k)

    all_expert_counts = torch.randint(
        0,
        32,
        (ep_size, num_experts),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    (
        ref_offsets,
        ref_total,
        ref_starts,
    ) = flex_ep_mod._compute_all_expert_offsets_reference(
        all_expert_counts,
        ep_rank,
        local_experts,
        flex_ep_mod.TOKEN_ALIGNMENT,
    )
    (
        actual_offsets,
        actual_total,
        actual_starts,
    ) = torch.ops.inductor.flex_ep_router_compute_all_expert_offsets(
        all_expert_counts,
        ep_rank,
        local_experts,
        flex_ep_mod.TOKEN_ALIGNMENT,
    )

    torch.testing.assert_close(actual_offsets, ref_offsets)
    torch.testing.assert_close(actual_total, ref_total)
    torch.testing.assert_close(actual_starts, ref_starts)

    scores = torch.rand(
        (batch, num_experts),
        device=device,
        generator=generator,
    )
    topk_idx = torch.topk(scores, top_k, dim=1).indices.to(torch.int32)
    recv_ofs = ref_offsets[:, :, ep_rank].reshape(-1)
    ref_ranks, ref_dest_offsets = flex_ep_mod._compute_dest_offsets_reference(
        topk_idx,
        recv_ofs,
        ep_size,
    )
    (
        actual_ranks,
        actual_dest_offsets,
    ) = torch.ops.inductor.flex_ep_router_compute_dest_offsets(
        topk_idx,
        recv_ofs,
        ep_size,
    )

    torch.testing.assert_close(actual_ranks, ref_ranks)
    torch.testing.assert_close(actual_dest_offsets, ref_dest_offsets)
