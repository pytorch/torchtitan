# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear-attention metadata ownership and CUDA-graph input contracts."""

import unittest
from types import SimpleNamespace
from typing import cast
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
    create_varlen_metadata_for_document,
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
    UlyssesLinearAttentionMetadata,
)


class TestLinearAttentionMetadataPytree(unittest.TestCase):
    def test_ulysses_metadata_exposes_and_refreshes_token_orders(self):
        """Graph input copying refreshes the permutation as well as the offsets."""
        for backend in (UlyssesInnerKDA, UlyssesInnerGDN):
            with self.subTest(backend=backend):
                layouts = (
                    ([0, 4, 8, 8], [[0, 1, 6, 7, 2, 3, 4, 5]]),
                    ([0, 2, 6, 8], [[0, 2, 4, 6, 1, 3, 5, 7]]),
                )
                metadata = []
                for layout, permutation in layouts:
                    offsets = torch.tensor(layout, dtype=torch.int32)
                    metadata.append(
                        backend.prepare_cp_metadata(
                            LinearAttentionMetadata(
                                varlen=VarlenAttentionMetadata(offsets, offsets, 8, 8),
                                num_conv_history_tokens=3,
                            ),
                            permutation=torch.tensor(permutation),
                        )
                    )
                spec = AcceleratorGraphInputSpec(metadata[0])
                saved, current = (spec.flatten(value) for value in metadata)
                self.assertTrue(
                    all(isinstance(value, (torch.Tensor, int)) for value in saved)
                )
                for destination, source in zip(saved, current, strict=True):
                    if isinstance(destination, torch.Tensor):
                        destination.copy_(source)
                    else:
                        self.assertEqual(destination, source)
                refreshed = spec.unflatten(saved)
                self.assertIsInstance(refreshed, UlyssesLinearAttentionMetadata)
                torch.testing.assert_close(
                    refreshed.varlen.cu_seq_q,
                    metadata[1].varlen.cu_seq_q,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    refreshed.permutation, metadata[1].permutation, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    refreshed.inverse_permutation,
                    metadata[1].inverse_permutation,
                    rtol=0,
                    atol=0,
                )

    def test_kda_metadata_exposes_sequence_offsets_as_pytree_leaves(self):
        """Keep sequence tensors visible to graph input copying and replay."""
        offsets = torch.tensor([0, 4], dtype=torch.int32)
        metadata = LinearAttentionMetadata(
            varlen=VarlenAttentionMetadata(offsets, offsets, 4, 4),
            num_conv_history_tokens=3,
        )

        leaves, tree_spec = pytree.tree_flatten(metadata)
        self.assertIs(leaves[0], offsets)
        self.assertIs(leaves[1], offsets)
        self.assertEqual(leaves[2:], [4, 4, -1, 0, 3])
        rebuilt = pytree.tree_unflatten(leaves, tree_spec)
        self.assertIsInstance(rebuilt, LinearAttentionMetadata)
        self.assertIs(rebuilt.varlen.cu_seq_q, offsets)
        self.assertIs(rebuilt.varlen.cu_seq_k, offsets)
        self.assertEqual(rebuilt.varlen.window_size, (-1, 0))
        self.assertEqual(rebuilt.num_conv_history_tokens, 3)

    def test_cp_kda_metadata_exposes_routing_tensors_as_pytree_leaves(self):
        """Keep every CP routing tensor visible without changing its ownership."""
        offsets = torch.tensor([0, 4], dtype=torch.int32)
        routing = ContextParallelRouting.from_fragments(
            cu_seqlens_global=[0, 4],
            fragments=[[(0, 2)], [(2, 4)]],
            cp_rank=0,
            device="cpu",
            conv_history=3,
        )
        metadata = ContextParallelLinearAttentionMetadata(
            varlen=VarlenAttentionMetadata(offsets, offsets, 4, 4),
            num_conv_history_tokens=3,
            cp_routing=routing,
        )

        leaves, tree_spec = pytree.tree_flatten(metadata)

        self.assertNotIn(metadata, leaves)
        self.assertNotIn(routing, leaves)
        for value in vars(routing).values():
            if isinstance(value, torch.Tensor):
                self.assertTrue(any(value is leaf for leaf in leaves))

        rebuilt = pytree.tree_unflatten(leaves, tree_spec)
        self.assertIsInstance(rebuilt, ContextParallelLinearAttentionMetadata)
        self.assertIsInstance(rebuilt.cp_routing, ContextParallelRouting)
        self.assertIs(rebuilt.varlen.cu_seq_q, offsets)
        for name, value in vars(routing).items():
            rebuilt_value = getattr(rebuilt.cp_routing, name)
            if isinstance(value, torch.Tensor):
                self.assertIs(rebuilt_value, value)
            else:
                self.assertEqual(rebuilt_value, value)


class TestKDAContextParallelMetadata(unittest.TestCase):
    def test_fragments_follow_the_explicit_permutation(self):
        permutation = torch.tensor([[0, 1, 6, 7, 2, 3, 4, 5]])
        self.assertEqual(
            get_token_fragments(8, cp_size=2, permutation=permutation),
            [[(0, 2), (6, 8)], [(2, 6)]],
        )

    def test_backend_builds_routing_from_global_varlen_metadata(self):
        varlen = create_varlen_metadata_for_document(torch.tensor([0, 1, 0, 1]))
        attention_metadata = LinearAttentionMetadata(
            varlen=varlen,
            num_conv_history_tokens=3,
        )
        group = SimpleNamespace(size=lambda: 2)
        permutation = torch.tensor([[0, 3, 1, 2]])
        routing = cast(ContextParallelRouting, object())
        with patch(
            "torchtitan.models.common.attention.cp_linear_attention.ContextParallelRouting.from_fragments",
            return_value=routing,
        ) as build_routing, patch(
            "torchtitan.models.common.attention.cp_linear_attention.spmd_mesh_group",
            return_value=group,
        ), patch(
            "torchtitan.models.common.attention.cp_linear_attention.dist.get_rank",
            return_value=0,
        ):
            kda_metadata = ContextParallelInnerKDA.prepare_cp_metadata(
                attention_metadata,
                permutation=permutation,
                max_num_fragments=2,
            )
        self.assertIs(kda_metadata.cp_routing, routing)
        build_routing.assert_called_once_with(
            cu_seqlens_global=[0, 2, 4],
            fragments=[[(0, 1), (3, 4)], [(1, 3)]],
            cp_rank=0,
            device=varlen.cu_seq_q.device,
            slots=2,
            max_subsequences=3,
            conv_history=3,
        )

    def test_metadata_annotates_cp_routing_tensors(self):
        varlen = create_varlen_metadata_for_document(torch.tensor([0, 1, 0, 1]))
        routing = ContextParallelRouting.from_fragments(
            cu_seqlens_global=[0, 2, 4],
            fragments=[[(0, 2)], [(2, 4)]],
            cp_rank=0,
            device="cpu",
            conv_history=3,
        )
        metadata = ContextParallelLinearAttentionMetadata(
            varlen=varlen,
            num_conv_history_tokens=3,
            cp_routing=routing,
        )
        with patch(
            "torchtitan.models.common.attention.cp_linear_attention.spmd.assert_type"
        ) as annotate:
            metadata.annotate_spmd_types()
        annotated_tensors = [call.args[0] for call in annotate.call_args_list]
        for value in vars(routing).values():
            if isinstance(value, torch.Tensor):
                self.assertTrue(
                    any(value is annotated for annotated in annotated_tensors)
                )


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
                varlen=VarlenAttentionMetadata(offsets, offsets, 8, 8),
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
                varlen=VarlenAttentionMetadata(offsets, offsets, tokens, tokens),
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
