# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear-attention metadata ownership and CUDA-graph input contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from attn_gym.linear.context_parallel import ContextParallelRouting
from torch.nn.attention.flex_attention import create_block_mask
from torch.utils import _pytree as pytree
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.accelerator_graph import AcceleratorGraphInputSpec
from torchtitan.distributed.context_parallel import (
    get_token_fragments,
    PTRRFlexAttentionCPLoadBalancer,
)

from torchtitan.models.common.attention import (
    LinearAttentionMetadata,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.attention.cp_gdn import UlyssesInnerGDN
from torchtitan.models.common.attention.cp_kda import (
    ContextParallelInnerKDA,
    UlyssesInnerKDA,
)
from torchtitan.models.common.attention.cp_linear_attention import (
    ContextParallelLinearAttentionMetadata,
)


def _linear_metadata(layout: list[int]) -> LinearAttentionMetadata:
    offsets = torch.tensor(layout, dtype=torch.int32)
    return LinearAttentionMetadata(
        varlen=VarlenAttentionMetadata(
            cu_seq_q=offsets, cu_seq_k=offsets, max_q=8, max_k=8
        ),
        num_conv_history_tokens=3,
    )


class TestLinearAttentionMetadataPytree(unittest.TestCase):
    def test_graph_inputs_refresh_every_metadata_tensor(self):
        """Replaying with a new batch's metadata must copy all of its tensors.

        An unregistered metadata type is one opaque graph input, so its fresh
        offsets, token orders and routing would never reach the replayed graph.
        """
        layouts = ([0, 4, 8, 8], [0, 2, 6, 8])
        permutations = ([[0, 1, 6, 7, 2, 3, 4, 5]], [[0, 2, 4, 6, 1, 3, 5, 7]])
        cases = {
            "base": [_linear_metadata(layout) for layout in layouts],
            "context_parallel": [
                ContextParallelLinearAttentionMetadata(
                    varlen=_linear_metadata(layout).varlen,
                    num_conv_history_tokens=3,
                    cp_routing=ContextParallelRouting.from_fragments(
                        cu_seqlens_global=layout,
                        fragments=[[(0, 4)], [(4, 8)]],
                        cp_rank=0,
                        device="cpu",
                        slots=1,
                        max_subsequences=3,
                        conv_history=3,
                    ),
                )
                for layout in layouts
            ],
        }
        for backend in (UlyssesInnerKDA, UlyssesInnerGDN):
            cases[f"ulysses_{backend.__name__}"] = [
                backend.prepare_cp_metadata(
                    _linear_metadata(layout),
                    permutation=torch.tensor(permutation),
                    max_num_fragments=1,
                )
                for layout, permutation in zip(layouts, permutations, strict=True)
            ]
        for name, (captured, current) in cases.items():
            with self.subTest(metadata=name):
                spec = AcceleratorGraphInputSpec(captured)
                saved, fresh = (spec.flatten(value) for value in (captured, current))
                for destination, source in zip(saved, fresh, strict=True):
                    if isinstance(destination, torch.Tensor):
                        destination.copy_(source)
                    else:
                        self.assertEqual(destination, source)
                refreshed = spec.unflatten(saved)
                self.assertIs(type(refreshed), type(current))
                for actual, expected in zip(
                    pytree.tree_leaves(refreshed),
                    pytree.tree_leaves(current),
                    strict=True,
                ):
                    if isinstance(expected, torch.Tensor):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _assert_active_routing(routing, compact):
    """Padding must retain active offsets, state indices and convolution sources."""
    for name, value in vars(compact).items():
        actual = getattr(routing, name)
        if isinstance(value, torch.Tensor):
            active = tuple(slice(0, length) for length in value.shape)
            torch.testing.assert_close(actual[active], value, rtol=0, atol=0)
        else:
            assert actual == value
    count = compact.cu_seqlens.numel() - 1
    assert torch.all(routing.cu_seqlens[count + 1 :] == routing.tokens)
    assert torch.all(routing.entry_sources[count:] == routing.slots)
    assert torch.all(routing.exit_sources[count:] == routing.slots)
    assert not torch.any(routing.terminal[count:])
    assert torch.all(routing.conv_sources[count:] == routing.world * routing.slots * 3)


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("permuted", [False, True])
def test_routing_preserves_global_capacity_and_active_entries(cp_rank, permuted):
    """Padded document capacity keeps changing local document cuts replayable."""
    permutation = torch.tensor([[0, 1, 6, 7, 2, 3, 4, 5]]) if permuted else None
    max_num_fragments = 2 if permuted else 1
    fragments = get_token_fragments(8, cp_size=2, permutation=permutation)
    shapes = []
    group = SimpleNamespace(size=lambda: 2)
    with patch(
        "torchtitan.models.common.attention.cp_linear_attention.spmd_mesh_group",
        return_value=group,
    ), patch(
        "torchtitan.models.common.attention.cp_linear_attention.dist.get_rank",
        return_value=cp_rank,
    ):
        for layout in ([0, 4, 8, 8, 8], [0, 2, 6, 8, 8]):
            offsets = torch.tensor(layout, dtype=torch.int32)
            metadata = LinearAttentionMetadata(
                varlen=VarlenAttentionMetadata(
                    cu_seq_q=offsets, cu_seq_k=offsets, max_q=8, max_k=8
                ),
                num_conv_history_tokens=3,
            )
            result = ContextParallelInnerKDA.prepare_cp_metadata(
                metadata,
                permutation=permutation,
                max_num_fragments=max_num_fragments,
            )
            routing = result.cp_routing
            compact = ContextParallelRouting.from_fragments(
                cu_seqlens_global=layout,
                fragments=fragments,
                cp_rank=cp_rank,
                device="cpu",
                slots=max_num_fragments,
                conv_history=3,
            )
            assert routing.cu_seqlens.shape == ((6,) if permuted else (5,))
            assert result.varlen is metadata.varlen
            _assert_active_routing(routing, compact)
            shapes.append(
                {
                    name: value.shape
                    for name, value in vars(routing).items()
                    if isinstance(value, torch.Tensor)
                }
            )
    assert shapes[0] == shapes[1]


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_ptrr_routing_capacity_is_independent_of_batch_fragmentation(cp_rank):
    """Rebuilding real PTRR for a new packed batch must not resize graph inputs."""
    tokens = 1024
    parallelism = ParallelismConfig(
        context_parallel_degree=2,
        context_parallel_load_balancer=PTRRFlexAttentionCPLoadBalancer.Config(),
    )
    group = SimpleNamespace(size=lambda: 2)
    routings, fragment_counts = [], []
    with patch(
        "torchtitan.distributed.context_parallel.spmd_mesh_group", return_value=group
    ), patch(
        "torchtitan.models.common.attention.cp_linear_attention.spmd_mesh_group",
        return_value=group,
    ), patch(
        "torchtitan.models.common.attention.cp_linear_attention.dist.get_rank",
        return_value=cp_rank,
    ):
        for layout in ([0, 1024, 1024, 1024, 1024], [0, 512, 1024, 1024, 1024]):
            document_ids = torch.bucketize(
                torch.arange(tokens), torch.tensor(layout[1:-1]), right=True
            )

            def document_causal(batch, head, query, key):
                return (query >= key) & (document_ids[query] == document_ids[key])

            block_mask = create_block_mask(
                document_causal,
                B=1,
                H=1,
                Q_LEN=tokens,
                KV_LEN=tokens,
                device="cpu",
                BLOCK_SIZE=128,
            )
            load_balancer = parallelism.context_parallel_load_balancer.build(
                seq_len=tokens, attention_metadata=block_mask
            )
            permutation = load_balancer.generate_permutation()
            assert load_balancer.max_num_fragments == 4
            fragments = get_token_fragments(tokens, cp_size=2, permutation=permutation)
            fragment_counts.append(max(len(owned) for owned in fragments))
            offsets = torch.tensor(layout, dtype=torch.int32)
            metadata = LinearAttentionMetadata(
                varlen=VarlenAttentionMetadata(
                    cu_seq_q=offsets, cu_seq_k=offsets, max_q=tokens, max_k=tokens
                ),
                num_conv_history_tokens=3,
            )
            routing = ContextParallelInnerKDA.prepare_cp_metadata(
                metadata,
                permutation=permutation,
                max_num_fragments=load_balancer.max_num_fragments,
            ).cp_routing
            compact = ContextParallelRouting.from_fragments(
                cu_seqlens_global=layout,
                fragments=fragments,
                cp_rank=cp_rank,
                device="cpu",
                slots=4,
                conv_history=3,
            )
            assert routing.slots == 4
            assert routing.cu_seqlens.shape == (8,)
            _assert_active_routing(routing, compact)
            routings.append(routing)
    assert fragment_counts == [3, 4]
    spec = AcceleratorGraphInputSpec(routings[0])
    saved, current = (spec.flatten(routing) for routing in routings)
    for destination, source in zip(saved, current, strict=True):
        if isinstance(destination, torch.Tensor):
            assert destination.shape == source.shape
            destination.copy_(source)
        else:
            assert destination == source
    refreshed = spec.unflatten(saved)
    for name, value in vars(routings[1]).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(getattr(refreshed, name), value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
