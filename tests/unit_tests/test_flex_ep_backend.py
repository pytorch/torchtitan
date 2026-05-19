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
    setattr(config.parallelism, parallelism_field, 2)

    with pytest.raises(ValueError, match="DP\\+EP"):
        config.model_spec.model.update_from_config(trainer_config=config)


def test_deepseek_v3_flex_ep_config_sets_router_force_load_balance_only():
    config = deepseek_v3_debugmodel_flex_ep()
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


def test_flex_ep_capacity_guard_reports_capacity_without_balanced_requirement():
    local_experts_start = torch.tensor([0, 128, 384], dtype=torch.int64)

    with pytest.raises(ValueError) as exc_info:
        flex_ep_mod._validate_flex_ep_capacity(
            local_experts_start,
            max_recv_tokens=256,
            capacity_factor=0.5,
        )
    message = str(exc_info.value)
    assert "Received 384 local expert tokens" in message
    assert "Receive workspace capacity is 256" in message
    assert "increase the FlexEP capacity factor" in message
    assert "requires balanced routing" not in message


def test_flex_ep_workspace_cache_and_views_include_capacity_factor(monkeypatch):
    monkeypatch.setattr(flex_ep_mod, "_ensure_flex_ep_imported", lambda: None)
    kwargs = {
        "max_tokens": 256,
        "dim": 8,
        "num_experts": 4,
        "top_k": 2,
        "device": torch.device("cpu"),
        "ep_mesh": None,
    }

    router1 = flex_ep_mod.FlexEPRouter.create(**kwargs, capacity_factor=1.0)
    router2 = flex_ep_mod.FlexEPRouter.create(**kwargs, capacity_factor=0.5)
    router3 = flex_ep_mod.FlexEPRouter.create(**kwargs, capacity_factor=1.0)

    assert router1.workspace is router3.workspace
    assert router1.workspace is not router2.workspace

    view1 = router1.workspace.view(
        max_tokens=kwargs["max_tokens"],
        dim=kwargs["dim"],
        num_experts=kwargs["num_experts"],
        top_k=kwargs["top_k"],
        capacity_factor=1.0,
    )
    view2 = router1.workspace.view(
        max_tokens=kwargs["max_tokens"],
        dim=kwargs["dim"],
        num_experts=kwargs["num_experts"],
        top_k=kwargs["top_k"],
        capacity_factor=0.5,
    )

    assert view1 is not view2
    assert view1.dispatch_recv_buffer.shape[0] != view2.dispatch_recv_buffer.shape[0]


def test_flex_ep_zfill_offsets_are_contiguous(monkeypatch):
    import torch._higher_order_ops.flex_ep  # noqa: F401

    zfill_calls = []

    def fake_zfill(input, begin_ofs, end_ofs, max_values_per_batch):
        zfill_calls.append((begin_ofs.is_contiguous(), end_ofs.is_contiguous()))
        del begin_ofs, end_ofs, max_values_per_batch
        return input

    monkeypatch.setattr(torch.ops._flex_ep, "zfill_ranges_inplace", fake_zfill)
    router = flex_ep_mod.FlexEPRouter.create(
        max_tokens=5,
        dim=8,
        num_experts=4,
        top_k=2,
        device=torch.device("cpu"),
        ep_mesh=None,
    )
    build_dispatch_plan_fn, dispatch_fn = router.router_fns[:2]
    x_expanded = torch.randn(5, 2, 8, dtype=torch.bfloat16)
    topk_idx = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [1, 3],
        ],
        dtype=torch.int64,
    )

    plan = build_dispatch_plan_fn(topk_idx, router.router_operands)
    dispatch_fn(x_expanded, plan, router.router_operands)

    assert zfill_calls
    assert all(begin and end for begin, end in zfill_calls)


def test_flex_ep_barrier_wait_no_clone_returns_alias_on_cpu_fallback():
    flex_ep_mod._ensure_flex_ep_imported()
    input_tensor = torch.ones(4, dtype=torch.float32)
    cuda_ptrs = torch.empty(1, dtype=torch.int64)
    expected = torch.zeros(1, dtype=torch.int32)

    out = torch.ops._flex_ep.barrier_wait_no_clone(
        input_tensor,
        cuda_ptrs,
        0,
        expected,
    )

    assert out.data_ptr() == input_tensor.data_ptr()


def test_flex_ep_weighted_sum_matches_reference_on_cpu():
    y_partial = torch.randn(3, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    top_scores = torch.randn(3, 2, dtype=torch.float32, requires_grad=True)
    y_partial_ref = y_partial.detach().clone().requires_grad_()
    top_scores_ref = top_scores.detach().clone().requires_grad_()
    grad = torch.randn(3, 4, dtype=torch.bfloat16)

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


def test_flex_ep_offset_ops_cpu_fallbacks_match_full_reference():
    flex_ep_mod._ensure_flex_ep_imported()
    y1 = torch.randn(5, 8, dtype=torch.bfloat16)
    dy2 = torch.randn(5, 4, dtype=torch.bfloat16)
    token_end = torch.tensor([2], dtype=torch.int64)

    out = torch.ops._flex_ep.swiglu_forward_with_offsets(y1, token_end)
    gate, up = y1.chunk(2, dim=-1)
    ref = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(out, ref)

    dy1 = torch.ops._flex_ep.swiglu_backward_with_offsets(dy2, y1, token_end)
    sig = torch.sigmoid(gate)
    ref_dgate = dy2 * up * sig * (1 + gate * (1 - sig))
    ref_dup = dy2 * torch.nn.functional.silu(gate)
    torch.testing.assert_close(dy1, torch.cat((ref_dgate, ref_dup), dim=-1))

    cloned = torch.ops._flex_ep.clone_valid_prefix(y1, token_end)
    torch.testing.assert_close(cloned, y1)


def test_flex_ep_combine_does_not_zero_capacity_tail(monkeypatch):
    flex_ep_mod._ensure_flex_ep_imported()
    router = flex_ep_mod.FlexEPRouter.create(
        max_tokens=5,
        dim=8,
        num_experts=4,
        top_k=2,
        device=torch.device("cpu"),
        ep_mesh=None,
    )
    build_dispatch_plan_fn, dispatch_fn, combine_fn, _, dispatch_bwd_fn = (
        router.router_fns
    )
    x_expanded = torch.randn(5, 2, 8, dtype=torch.bfloat16)
    topk_idx = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [1, 3],
        ],
        dtype=torch.int64,
    )
    plan = build_dispatch_plan_fn(topk_idx, router.router_operands)
    recv_x = dispatch_fn(x_expanded, plan, router.router_operands)
    local_experts_start = plan.local_experts_start
    token_end = int(local_experts_start[-1].item())
    assert token_end < recv_x.shape[0]

    sentinel = torch.tensor(-7.0, dtype=recv_x.dtype)
    calls = []

    def fake_router_combine(
        send_tokens,
        send_scale_factors,
        send_weights,
        expert_begin_offset_per_ep,
        token_send_end,
        send_origin_global_token_id,
        buffers_cuda_ptrs,
        combine_recv_buffer,
        combine_recv_scale_factors,
        combine_recv_weights,
        *args,
    ):
        del (
            send_scale_factors,
            send_weights,
            expert_begin_offset_per_ep,
            send_origin_global_token_id,
            buffers_cuda_ptrs,
            args,
        )
        assert int(token_send_end.item()) == token_end
        assert torch.all(send_tokens[token_end:] == sentinel).item()
        calls.append(send_tokens.data_ptr())
        return combine_recv_buffer, combine_recv_scale_factors, combine_recv_weights

    monkeypatch.setattr(torch.ops._flex_ep, "router_combine", fake_router_combine)

    y3 = torch.full_like(recv_x, -7.0)
    combine_fn(y3, plan, router.router_operands)

    dx_recv = torch.full_like(recv_x, -7.0)
    dispatch_bwd_fn(dx_recv, plan, router.router_operands)

    assert len(calls) == 2
    assert torch.all(y3[token_end:] == sentinel).item()
    assert torch.all(dx_recv[token_end:] == sentinel).item()


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
