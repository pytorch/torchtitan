# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

# TODO: TokenSwitch is not in PyTorch nightly yet; drop the skip once it lands.
_NCCLEPObject = pytest.importorskip("torch.distributed._token_switch")._NCCLEPObject

from torchtitan.distributed import nccl_ep  # noqa: E402
from torchtitan.distributed.nccl_ep_capture import (  # noqa: E402
    NCCLEPRoutingCapturePlan,
)
from torchtitan.models.common.moe.token_dispatcher import (  # noqa: E402
    NCCLEPTokenDispatcher,
)


def _routing(ids, layout, handle):
    return SimpleNamespace(
        topk_idx=ids, layout=layout, handle=handle, _opaque_handle=_NCCLEPObject(handle)
    )


class _Switch:
    def create_routing(self, ids, per_expert_token_counts=None, *, layout, alignment):
        assert alignment == 32
        return _routing(ids, layout, object())

    def update_routing(self, routing, ids, per_expert_token_counts=None):
        # Like TokenSwitchNCCL, each update returns a new opaque wrapper.
        return _routing(ids, routing.layout, routing.handle)


def test_routing_lease_and_generation():
    pool = nccl_ep.RoutingSlots(
        _Switch(), torch.zeros(4, 1, dtype=torch.int64), 1, group=None
    )
    opaque = _NCCLEPObject(pool)
    native = nccl_ep._prepare_routing(torch.zeros(4, 1, dtype=torch.int64), opaque, 1)
    assert pool.slots[0][1]
    previous_generation = pool.generations[0]
    del native
    gc.collect()
    assert not pool.slots[0][1]
    index, _, _ = pool.acquire(torch.zeros(4, 1, dtype=torch.int64), "expert_major")
    pool.release(index, previous_generation)
    assert pool.slots[0][1]
    pool.release(index, pool.generations[index])
    assert not pool.slots[0][1]


def test_each_prepare_runs_one_collective_update():
    # Creating a handle already runs the collective routing update. A rank that
    # must create a slot (its earlier lease is still alive) and a rank that
    # reuses one must issue the same number of collectives, or they hang.
    calls = []

    class _CountingSwitch(_Switch):
        def create_routing(self, ids, *args, **kwargs):
            calls.append("create")
            return super().create_routing(ids, *args, **kwargs)

        def update_routing(self, routing, ids, per_expert_token_counts=None):
            calls.append("update")
            return super().update_routing(routing, ids)

    ids = torch.zeros(4, 1, dtype=torch.int64)
    pool = nccl_ep.RoutingSlots(_CountingSwitch(), ids, 1, group=None)
    opaque = _NCCLEPObject(pool)
    calls.clear()
    first = nccl_ep._prepare_routing(ids, opaque, 1)
    assert calls == ["update"]
    calls.clear()
    second = nccl_ep._prepare_routing(ids, opaque, 1)  # first is still leased
    assert calls == ["create"]
    del first, second


def test_capture_reserves_total_demand_not_only_eager_peak():
    ids = torch.zeros(4, 1, dtype=torch.int64)
    pool = nccl_ep.RoutingSlots(_Switch(), ids, 1, group=None)
    plan = NCCLEPRoutingCapturePlan()
    with plan.observe():
        for _ in range(3):
            index, _, _ = pool.acquire(ids, "expert_major")
            pool.release(index)
    assert len(pool.slots) == 1
    plan.prepare()
    assert len(pool.slots) == 3
    plan.release()


def test_shared_buffer_closes_after_final_user():
    group = object()
    with patch.object(nccl_ep, "TokenSwitchNCCL") as constructor:
        first = nccl_ep.acquire_buffer(group, 4, 16, 128, 512, 2)
        second = nccl_ep.acquire_buffer(group, 4, 16, 128, 512, 2)
        assert first is second
        constructor.assert_called_once()
        nccl_ep.release_buffer(first)
        constructor.return_value.close.assert_not_called()
        nccl_ep.release_buffer(second)
        constructor.return_value.close.assert_called_once()
        assert not nccl_ep._buffers


def test_canonical_order_sorts_sources_and_marks_padding():
    source = torch.zeros(6, 32, dtype=torch.uint8)
    source[[0, 1, 3, 4], -16] = torch.tensor([4, 1, 3, 2], dtype=torch.uint8)
    order, inverse, valid = nccl_ep.canonical_expert_order(
        source, torch.tensor([3, 3], dtype=torch.int32), 4
    )
    assert order is not None and inverse is not None
    assert order.tolist() == [1, 0, 2, 4, 3, 5]
    torch.testing.assert_close(order[inverse], torch.arange(6), rtol=0, atol=0)
    assert valid.tolist() == [True, True, False, True, True, False]


@pytest.mark.parametrize("kwargs", ({"num_max_tokens_per_rank": 0}, {"top_k": 5}))
def test_invalid_dispatcher_config(kwargs):
    values = dict(num_experts=4, top_k=2, hidden_dim=512, num_max_tokens_per_rank=16)
    values.update(kwargs)
    with pytest.raises(ValueError):
        NCCLEPTokenDispatcher.Config(**values)


def test_dispatch_returns_waited_identity_and_counts(monkeypatch):
    tokens = torch.zeros(2, 256, dtype=torch.bfloat16)
    scores = torch.ones(2, 1)
    raw_identity = torch.zeros(2, 16, dtype=torch.uint8)
    waited_identity = torch.full_like(raw_identity, 1)
    waited_counts = torch.tensor([2], dtype=torch.int32)
    counts = torch.tensor([2], dtype=torch.int32)
    native = object()
    monkeypatch.setattr(
        nccl_ep, "_prepare_scanned_routing", lambda *args: (native, counts)
    )
    monkeypatch.setattr(
        nccl_ep,
        "_dispatch_quantized_async_op",
        lambda *args, **kwargs: (tokens, scores[:, 0], None, raw_identity),
    )
    monkeypatch.setattr(
        nccl_ep,
        "wait_tensor",
        lambda value: (
            waited_identity
            if value is raw_identity
            else (waited_counts if value is counts else value)
        ),
    )
    ctx = SimpleNamespace(mark_non_differentiable=lambda *args: None)
    outputs = nccl_ep._Dispatch.forward(
        ctx, tokens, scores, torch.zeros(2, 1, dtype=torch.int64), None, 2, 0, 2, 1
    )
    assert outputs[4] is waited_identity
    assert outputs[2] is waited_counts


@pytest.mark.parametrize("hidden_dim", [0, 7, 257])
def test_bf16_dispatch_requires_native_row_alignment(hidden_dim):
    with pytest.raises(ValueError, match="divisible by 8"):
        NCCLEPTokenDispatcher.Config(
            num_experts=4, top_k=2, hidden_dim=hidden_dim, num_max_tokens_per_rank=16
        )


def test_bf16_dispatch_preserves_values_and_forwards_identity(monkeypatch):
    tokens = torch.tensor([1.0078125, -1.0078125], dtype=torch.bfloat16)
    tokens = tokens[:, None].expand(2, 256).contiguous()
    scores = torch.tensor([[0.25], [0.75]], dtype=torch.float32)
    counts = torch.tensor([2], dtype=torch.int32)
    native = object()
    monkeypatch.setattr(nccl_ep, "_prepare_routing", lambda *args: native)
    monkeypatch.setattr(nccl_ep, "wait_tensor", lambda value: value)

    def forward_payload(
        handle, payload, scales, weights, capacity, layout, num_local_experts
    ):
        assert handle is native and capacity == 2 and layout == "expert_major"
        assert payload.dtype == torch.bfloat16
        assert scales.dtype == torch.uint8 and scales.shape == (2, 16)
        assert torch.equal(payload.view(torch.int16), tokens.view(torch.int16))
        expected = torch.zeros_like(scales)
        expected[:, -16] = torch.tensor([3, 4], dtype=torch.uint8)
        assert torch.equal(scales, expected)
        return payload.clone(), weights[:, 0].clone(), None, scales.clone(), counts

    monkeypatch.setattr(
        nccl_ep, "_prepare_scanned_routing", lambda *args: (native, counts)
    )

    def scanned_forward_payload(
        handle, payload, scales, weights, capacity, layout, *, initialize_tokens=True
    ):
        assert initialize_tokens is False
        return forward_payload(handle, payload, scales, weights, capacity, layout, 1)[
            :4
        ]

    monkeypatch.setattr(
        nccl_ep, "_dispatch_quantized_async_op", scanned_forward_payload
    )
    ctx = SimpleNamespace(
        mark_non_differentiable=lambda *args: None, save_for_backward=lambda *args: None
    )
    output = nccl_ep._Dispatch.forward(
        ctx,
        tokens,
        scores,
        torch.ones(2, 1, dtype=torch.int64),
        None,
        2,
        1,
        2,
        1,
    )
    assert torch.equal(output[0].view(torch.int16), tokens.view(torch.int16))


@pytest.mark.parametrize("permuted", [False, True])
def test_bf16_combine_derivative_zeroes_group_padding(monkeypatch, permuted):
    # Combine's derivative feeds W2's backward, whose weight gradient reads
    # every row of each expert group. Native leaves unmapped rows
    # uninitialized (NaN here); padding rows inside the groups must be zero.
    # Alignment 4: group rows 0-3 (2, 3 padding) and 4-7 (7 padding); rows 8-9
    # are unused capacity, which the grouped GEMMs skip.
    valid_R = torch.tensor([1, 1, 0, 0, 1, 1, 1, 0, 0, 0], dtype=torch.bool)
    order_R = torch.tensor([3, 9, 0, 6, 1, 8, 2, 5, 4, 7]) if permuted else None
    native = _NCCLEPObject(object())
    monkeypatch.setattr(nccl_ep, "wait_tensor", lambda value: value)
    monkeypatch.setattr(nccl_ep, "_combine_async_op", lambda *args: torch.zeros(3, 8))

    def dispatch_backward(handle, grad_TD, capacity, *, initialize_tokens):
        assert handle is native and not initialize_tokens
        grad_RD = torch.full((capacity, 8), float("nan"))
        # Canonical row i is native row order_R[i].
        grad_RD[valid_R if order_R is None else order_R[valid_R]] = 1.0
        return grad_RD

    monkeypatch.setattr(nccl_ep, "_dispatch_backward_async_op", dispatch_backward)
    metadata = nccl_ep.DispatchMetadata(
        native=native,
        scores_R=torch.ones(10),
        order_R=order_R,
        inverse_R=None if order_R is None else order_R.argsort(),
        valid_R=valid_R,
        counts_e=torch.tensor([4, 4], dtype=torch.int32),
        alignment=4,
    )
    x_RD = torch.zeros(10, 8, requires_grad=True)
    nccl_ep.combine(x_RD, metadata, 3).sum().backward()
    assert x_RD.grad is not None
    assert torch.equal(x_RD.grad[:8], valid_R[:8, None].float().expand(8, 8))


@pytest.mark.parametrize("pad_multiple", [0, -1, 3])
def test_invalid_native_padding(pad_multiple):
    with pytest.raises(ValueError, match="positive power of two"):
        NCCLEPTokenDispatcher.Config(
            num_experts=4,
            top_k=2,
            hidden_dim=512,
            num_max_tokens_per_rank=16,
            pad_multiple=pad_multiple,
        )


@pytest.mark.parametrize("ht_flow", ["push_pull", "pull_push"])
def test_bf16_dispatcher_accepts_explicit_native_flow(ht_flow):
    config = NCCLEPTokenDispatcher.Config(
        num_experts=4,
        top_k=2,
        hidden_dim=512,
        num_max_tokens_per_rank=16,
        ht_flow=ht_flow,
    )
    assert config.ht_flow == ht_flow


@pytest.mark.parametrize("ht_flow", ["auto", "invalid"])
def test_bf16_dispatcher_rejects_unqualified_flow(ht_flow):
    with pytest.raises(ValueError, match="ht_flow"):
        NCCLEPTokenDispatcher.Config(
            num_experts=4,
            top_k=2,
            hidden_dim=512,
            num_max_tokens_per_rank=16,
            ht_flow=ht_flow,
        )


def test_buffer_cache_keeps_native_flows_separate():
    group = object()
    with patch.object(nccl_ep, "TokenSwitchNCCL") as constructor:
        first = nccl_ep.acquire_buffer(group, 4, 16, 128, 512, 2)
        second = nccl_ep.acquire_buffer(group, 4, 16, 128, 512, 2, ht_flow="pull_push")
        try:
            assert first is not second
            assert constructor.call_count == 2
            assert [call.kwargs["ht_flow"] for call in constructor.call_args_list] == [
                "push_pull",
                "pull_push",
            ]
        finally:
            nccl_ep.release_buffer(first)
            nccl_ep.release_buffer(second)
        assert not nccl_ep._buffers


@pytest.mark.parametrize("ordered", [False, True])
def test_native_scan_order_is_verified_before_sort_bypass(ordered):
    sources_R = torch.tensor([1, 4, 0, 2, 3, 0] if ordered else [4, 1, 0, 2, 3, 0])
    metadata_RU = nccl_ep.row_identity_bytes(sources_R)
    counts_e = torch.tensor([3, 3], dtype=torch.int32)
    if not ordered:
        with pytest.raises(RuntimeError, match="Native scan must return"):
            nccl_ep.canonical_expert_order(
                metadata_RU, counts_e, 4, native_source_order=True
            )
        return
    order_R, inverse_R, valid_R = nccl_ep.canonical_expert_order(
        metadata_RU, counts_e, 4, native_source_order=True
    )
    assert order_R is None and inverse_R is None
    assert valid_R.tolist() == [True, True, False, True, True, False]


def test_routing_slots_keep_layouts_separate():
    ids = torch.zeros(4, 1, dtype=torch.int64)
    pool = nccl_ep.RoutingSlots(_Switch(), ids, 1, group=None)
    index, routing, _ = pool.acquire(ids, "flat")
    assert index == 1
    assert routing.layout == "flat"
    pool.release(index)
    index, routing, _ = pool.acquire(ids, "expert_major")
    assert index == 0
    assert routing.layout == "expert_major"
    pool.release(index)


def test_raw_capture_rejected_before_routing_slot_acquisition(monkeypatch):
    ids = SimpleNamespace(shape=torch.Size([4, 1]), is_cuda=True)
    pool = nccl_ep.RoutingSlots(_Switch(), ids, 1, group=None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="routing capture plan owner"):
        pool.acquire(ids, "expert_major")
    assert pool.slots[0][1:] == [False, False]
    assert pool.generations == [0]


def test_native_buffer_uses_configured_top_k(monkeypatch):
    from unittest.mock import MagicMock

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    dispatcher = NCCLEPTokenDispatcher.Config(
        num_experts=4,
        top_k=2,
        hidden_dim=256,
        num_max_tokens_per_rank=16,
        pad_multiple=32,
    ).build()
    ep_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    monkeypatch.setattr(
        "torchtitan.models.common.moe.token_dispatcher._ep_process_group",
        lambda: ep_group,
    )
    owner = SimpleNamespace(
        acquire_buffer=MagicMock(return_value=SimpleNamespace(switch=_Switch())),
        release_buffer=MagicMock(),
        RoutingSlots=nccl_ep.RoutingSlots,
    )
    dispatcher._nccl_ep = owner
    from torchtitan.distributed.nccl_ep_capture import NCCLEPBufferOwner

    buffer_owner = NCCLEPBufferOwner()
    with buffer_owner.initialize():
        dispatcher.init_buffer()
    call = owner.acquire_buffer.call_args
    assert call.args[2] == 16
    assert call.args[5] == 2
    assert call.args[0] is ep_group
    monkeypatch.setattr(
        "torchtitan.models.common.moe.token_dispatcher._ep_process_group",
        lambda: object(),
    )
    with pytest.raises(ValueError, match="ambient process group changed"):
        dispatcher.dispatch(
            torch.zeros(2, 256, dtype=torch.bfloat16),
            torch.ones(2, 2),
            torch.tensor([[0, 1], [0, 1]]),
            torch.empty(0),
        )
    buffer_owner.close()
    assert dispatcher._ep_group is None


@pytest.mark.parametrize(
    "scan_mode,ht_flow,bypass",
    [
        (None, "pull_push", False),
        ("0", "pull_push", False),
        ("1", "push_pull", False),
        ("1", "pull_push", True),
    ],
)
def test_bypass_requires_scan_mode_and_pull_push(
    monkeypatch, scan_mode, ht_flow, bypass
):
    if scan_mode is None:
        monkeypatch.delenv("NCCL_EP_HT_AG_SCAN_MODE", raising=False)
    else:
        monkeypatch.setenv("NCCL_EP_HT_AG_SCAN_MODE", scan_mode)
    tokens = torch.ones(2, 8, dtype=torch.bfloat16)
    scores = torch.ones(2)
    counts = torch.tensor([2], dtype=torch.int32)
    monkeypatch.setattr(
        nccl_ep._Dispatch,
        "apply",
        lambda *args: (tokens, scores, counts, object(), torch.empty(2, 16)),
    )
    observed = []

    def canonical(*args, native_source_order):
        observed.append(native_source_order)
        order = torch.arange(2)
        return order, order, torch.ones(2, dtype=torch.bool)

    monkeypatch.setattr(nccl_ep, "canonical_expert_order", canonical)
    nccl_ep.dispatch(
        tokens,
        scores[:, None],
        torch.zeros(2, 1, dtype=torch.int64),
        None,
        2,
        0,
        2,
        1,
        1,
        alignment=1,
        ht_flow=ht_flow,
    )
    assert observed == [bypass]


def test_unowned_dispatcher_rejects_before_native_allocation(monkeypatch):
    from unittest.mock import MagicMock

    dispatcher = NCCLEPTokenDispatcher.Config(
        num_experts=4, top_k=2, hidden_dim=512, num_max_tokens_per_rank=16
    ).build()
    acquire = MagicMock()
    monkeypatch.setattr(dispatcher._nccl_ep, "acquire_buffer", acquire)
    with pytest.raises(ValueError, match="explicit native-buffer owner"):
        dispatcher.init_buffer()
    acquire.assert_not_called()
