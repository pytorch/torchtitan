# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from dataclasses import replace
from unittest import mock

import pytest
import spmd_types as spmd
import torch
import torch.utils._pytree as pytree
import torch_remat as remat
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_utils import DeterministicGuard

from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import GQAttention, QKVLinear
from torchtitan.models.common.attention.cp_attention import (
    SelectiveGatherKVSlidingWindowVarlenInnerAttention,
)
from torchtitan.models.common.linear import Linear


def _config(**kwargs):
    return SelectiveGatherKVSlidingWindowVarlenInnerAttention.Config(
        sliding_window_size=4, gather_block_size=2, **kwargs
    )


def _prepare(config, positions, *, head_tail=True):
    metadata = config.build_attention_metadata(
        positions, max_num_documents=4, max_context_length=16
    )
    permutation = (
        _HeadTailLoadBalancer(16, 2, "cpu")._generate_indices() if head_tail else None
    )
    return SelectiveGatherKVSlidingWindowVarlenInnerAttention.prepare_cp_metadata(
        metadata, permutation=permutation
    )


def test_selective_varlen_rejects_full_causal_metadata() -> None:
    metadata = replace(
        _config().build_attention_metadata(
            torch.arange(16), max_num_documents=4, max_context_length=16
        ),
        window_size=(-1, 0),
    )
    permutation = _HeadTailLoadBalancer(16, 2, "cpu")._generate_indices()
    with pytest.raises(ValueError, match="sliding window"):
        SelectiveGatherKVSlidingWindowVarlenInnerAttention.prepare_cp_metadata(
            metadata, permutation=permutation
        )


@pytest.mark.parametrize("head_tail", [False, True])
def test_document_packing_keeps_fixed_graph_inputs(head_tail) -> None:
    prepared = []
    for positions in (
        torch.arange(16),
        torch.tensor([0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7]),
    ):
        prepared.append(_prepare(_config(), positions, head_tail=head_tail))
    assert not torch.equal(prepared[0].cu_seq_q, prepared[1].cu_seq_q)
    first, spec = pytree.tree_flatten(prepared[0])
    second, other_spec = pytree.tree_flatten(prepared[1])
    assert spec == other_spec
    assert [leaf.shape for leaf in first if isinstance(leaf, torch.Tensor)] == [
        leaf.shape for leaf in second if isinstance(leaf, torch.Tensor)
    ]


@pytest.mark.parametrize("head_tail", [False, True])
def test_static_exchange_is_shared_and_rejects_layout_changes(head_tail):
    kernel = SelectiveGatherKVSlidingWindowVarlenInnerAttention(
        SelectiveGatherKVSlidingWindowVarlenInnerAttention.Config(
            sliding_window_size=8, gather_block_size=4
        )
    )
    group = mock.Mock()
    group.size.return_value = 4
    group.rank.return_value = 1
    k, v = torch.randn(16, 2, 3), torch.randn(16, 2, 5)
    with mock.patch(
        "attn_gym.distributed.selective_gather.CopyEngineBlockExchange"
    ) as factory:
        first = kernel._get_exchange(k, v, cp_group=group, head_tail=head_tail)
        assert (
            kernel._get_exchange(
                k.clone(), v.clone(), cp_group=group, head_tail=head_tail
            )
            is first
        )
        other = SelectiveGatherKVSlidingWindowVarlenInnerAttention.Config(
            sliding_window_size=8, gather_block_size=4
        ).build()
        assert other._get_exchange(k, v, cp_group=group, head_tail=head_tail) is first
        factory.assert_called_once()
        assert factory.call_args.kwargs["k_token_shape"] == (2, 3)
        assert factory.call_args.kwargs["v_token_shape"] == (2, 5)
        different_window = SelectiveGatherKVSlidingWindowVarlenInnerAttention.Config(
            sliding_window_size=4, gather_block_size=4
        ).build()
        different_window._get_exchange(k, v, cp_group=group, head_tail=head_tail)
        assert factory.call_count == 2
        for changed_k, changed_v in (
            (torch.randn(8, 2, 3), torch.randn(8, 2, 5)),
            (k, torch.randn(16, 2, 6)),
            (k.double(), v.double()),
        ):
            with pytest.raises(ValueError, match="fixed"):
                kernel._get_exchange(
                    changed_k, changed_v, cp_group=group, head_tail=head_tail
                )
        with pytest.raises(ValueError, match="fixed"):
            kernel._get_exchange(k, v, cp_group=group, head_tail=not head_tail)
        with pytest.raises(ValueError, match="CP group"):
            kernel._get_exchange(
                k,
                v,
                cp_group=mock.Mock(size=lambda: 4, rank=lambda: 1),
                head_tail=head_tail,
            )


def test_exchange_config_rejects_nonpositive_block_size():
    with pytest.raises(ValueError, match="gather_block_size"):
        SelectiveGatherKVSlidingWindowVarlenInnerAttention.Config(
            sliding_window_size=8, gather_block_size=0
        )


class _CountingExchange:
    def __init__(self):
        self.calls = Counter()

    def exchange_and_assemble(self, k, v):
        self.calls["fused"] += 1
        return k + k, v + 2 * v

    def exchange(self, k, v):
        self.calls["receive"] += 1
        return k.clone(), v.clone()

    def assemble(self, k, v, remote_k, remote_v):
        self.calls["assemble"] += 1
        return k + remote_k, v + 2 * remote_v


@pytest.mark.parametrize(
    "policy, expected",
    [
        (None, {"fused": 1}),
        (FullActivationRematConfig(), {"receive": 1, "assemble": 2}),
        (
            ActivationRematConfig(save_regions=["attention.inner_attention*"]),
            {"fused": 1},
        ),
        (
            ActivationRematConfig(save_regions=["attention.inner_attention"]),
            None,
        ),
    ],
)
def test_checkpoint_keeps_received_kv_and_repeats_only_assembly(policy, expected):
    torch.manual_seed(42)
    attention = GQAttention.Config(
        n_heads=2,
        n_kv_heads=2,
        head_dim=4,
        dim=8,
        qkv_linear=QKVLinear.Config(
            head_dim=4,
            n_heads=2,
            n_kv_heads=2,
            wqkv=Linear.Config(in_features=8, out_features=24),
        ),
        wo=Linear.Config(in_features=8, out_features=8),
        inner_attention=SelectiveGatherKVSlidingWindowVarlenInnerAttention.Config(
            sliding_window_size=8,
        ),
        rope=None,
    ).build()
    block = Module()
    block.attention = attention
    kernel = attention.inner_attention
    exchange = _CountingExchange()

    def inner_forward(q, k, v, **kwargs):
        compact_k, compact_v = kernel._exchange_kv(exchange, k, v)
        remat.recompute_needs_tensor(compact_k, compact_v)
        return q + compact_k + compact_v

    inputs = torch.randn(8, 8)
    positions = torch.arange(8)
    with mock.patch.object(
        kernel, "forward", side_effect=inner_forward
    ), spmd.no_typecheck(), DeterministicGuard(True):

        def run(checkpoint):
            attention.zero_grad(set_to_none=True)
            x = inputs.detach().clone().requires_grad_()
            forward = remat.checkpoint()(attention) if checkpoint else attention
            output = forward(x, None, positions)
            output.square().sum().backward()
            grads = []
            for tensor in (x, *attention.parameters()):
                assert tensor.grad is not None
                grads.append(tensor.grad.detach().clone())
            return output.detach(), grads

        reference = run(False)
        exchange.calls.clear()
        if policy is not None:
            block.configure_remat_regions(policy.save_regions, policy.recompute_regions)
        if expected is None:
            with pytest.raises(RuntimeError, match="[Rr]ecompute"):
                run(True)
            return
        actual = run(policy is not None)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert exchange.calls == expected


class _MemoryExchange:
    def __init__(self):
        self.remote = []
        self.compact = []

    def exchange(self, k, v):
        remote = (k[:2].detach().clone(), v[:2].detach().clone())
        self.remote = [
            (
                StorageWeakRef(tensor.untyped_storage()),
                tensor.untyped_storage().nbytes(),
            )
            for tensor in remote
        ]
        return remote

    def assemble(self, k, v, remote_k, remote_v):
        compact = (torch.cat((remote_k, k)), torch.cat((remote_v, v)))
        self.compact = [
            (
                StorageWeakRef(tensor.untyped_storage()),
                tensor.untyped_storage().nbytes(),
            )
            for tensor in compact
        ]
        return compact

    def exchange_and_assemble(self, k, v):
        return self.assemble(k, v, *self.exchange(k, v))


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_checkpoint_retains_only_halos_not_assembled_kv(use_checkpoint):
    module = SelectiveGatherKVSlidingWindowVarlenInnerAttention.Config(
        sliding_window_size=8,
    ).build()
    if use_checkpoint:
        block = Module()
        block.add_module("inner_attention", module)
        block.configure_remat_regions([])
    exchange = _MemoryExchange()
    k = torch.randn(8, 2, 3, requires_grad=True)
    v = torch.randn(8, 2, 5, requires_grad=True)

    def forward(k, v):
        compact_k, compact_v = module._exchange_kv(exchange, k, v)
        remat.recompute_needs_tensor(compact_k, compact_v)
        return compact_k.square().sum() + compact_v.square().sum()

    train = (
        remat.checkpoint(region_name="layer")(forward) if use_checkpoint else forward
    )
    loss = train(k, v)
    kept = exchange.remote if use_checkpoint else exchange.compact
    freed = exchange.compact if use_checkpoint else exchange.remote
    assert all(ref.expired() for ref, _ in freed)
    assert all(not ref.expired() for ref, _ in kept)
    saved_bytes = sum(nbytes for _, nbytes in kept)
    assert saved_bytes == (2 if use_checkpoint else 10) * 2 * (3 + 5) * 4
    loss.backward()
