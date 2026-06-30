# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import unittest.mock

import spmd_types as spmd
import torch
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.distributed.utils import set_spmd_backend
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)


class TestPermute(unittest.TestCase):
    """Test AllToAllTokenDispatcher._permute which reorders tokens from rank-major to expert-major layout.

    Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
    Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
    """

    def _make_dispatcher(self, num_ranks: int) -> AllToAllTokenDispatcher:
        """Create a minimal AllToAllTokenDispatcher for testing _permute."""
        cfg = AllToAllTokenDispatcher.Config(num_experts=1, top_k=1)
        dispatcher = AllToAllTokenDispatcher(cfg)
        # Mock ep_mesh with a simple object that has .size() returning num_ranks
        mock_mesh = unittest.mock.MagicMock()
        mock_mesh.size.return_value = num_ranks
        dispatcher.ep_mesh = mock_mesh
        return dispatcher

    def _permute(self, tokens_per_expert_group, experts_per_rank, num_ranks):
        """Helper that calls _permute and returns (permuted_indices, num_tokens_per_expert)."""
        dispatcher = self._make_dispatcher(num_ranks)
        total = tokens_per_expert_group.sum().item()
        dummy_input = torch.zeros(total, 1)
        _, _, permuted_indices, num_tokens_per_expert = dispatcher._permute(
            dummy_input, tokens_per_expert_group
        )
        return permuted_indices, num_tokens_per_expert

    def test_basic_2ranks_2experts(self):
        # 2 ranks, 2 experts per rank
        # tokens_per_expert_group: [r0e0, r0e1, r1e0, r1e1] = [2, 3, 1, 4]
        tokens_per_expert_group = torch.tensor([2, 3, 1, 4])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        # Expert-major layout: e0r0(2), e0r1(1), e1r0(3), e1r1(4)
        # Input positions:
        #   r0e0: [0, 1], r0e1: [2, 3, 4], r1e0: [5], r1e1: [6, 7, 8, 9]
        # Output order:
        #   e0r0: [0, 1], e0r1: [5], e1r0: [2, 3, 4], e1r1: [6, 7, 8, 9]
        expected_indices = torch.tensor([0, 1, 5, 2, 3, 4, 6, 7, 8, 9])
        torch.testing.assert_close(permuted_indices, expected_indices)

        # num_tokens_per_expert: sum across ranks for each expert
        # e0: r0e0 + r1e0 = 2 + 1 = 3, e1: r0e1 + r1e1 = 3 + 4 = 7
        expected_num_tokens = torch.tensor([3, 7])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_single_rank(self):
        # 1 rank, 3 experts: no reordering needed
        tokens_per_expert_group = torch.tensor([4, 2, 5])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=3, num_ranks=1
        )

        expected_indices = torch.arange(11)
        torch.testing.assert_close(permuted_indices, expected_indices)
        torch.testing.assert_close(num_tokens_per_expert, tokens_per_expert_group)

    def test_single_expert(self):
        # 3 ranks, 1 expert per rank: no reordering needed
        tokens_per_expert_group = torch.tensor([3, 5, 2])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=1, num_ranks=3
        )

        expected_indices = torch.arange(10)
        torch.testing.assert_close(permuted_indices, expected_indices)

        # Single expert gets all tokens
        expected_num_tokens = torch.tensor([10])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_zero_tokens_for_some_experts(self):
        # 2 ranks, 2 experts, some with zero tokens
        # [r0e0, r0e1, r1e0, r1e1] = [0, 3, 2, 0]
        tokens_per_expert_group = torch.tensor([0, 3, 2, 0])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        # Expert-major: e0r0(0), e0r1(2), e1r0(3), e1r1(0)
        # Input positions: r0e0: [], r0e1: [0, 1, 2], r1e0: [3, 4], r1e1: []
        # Output order: e0r0: [], e0r1: [3, 4], e1r0: [0, 1, 2], e1r1: []
        expected_indices = torch.tensor([3, 4, 0, 1, 2])
        torch.testing.assert_close(permuted_indices, expected_indices)

        expected_num_tokens = torch.tensor([2, 3])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_all_zero_tokens(self):
        tokens_per_expert_group = torch.tensor([0, 0, 0, 0])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        self.assertEqual(permuted_indices.numel(), 0)
        expected_num_tokens = torch.tensor([0, 0])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_uniform_distribution(self):
        # 3 ranks, 2 experts, uniform token counts
        # [r0e0, r0e1, r1e0, r1e1, r2e0, r2e1] = [2, 2, 2, 2, 2, 2]
        tokens_per_expert_group = torch.tensor([2, 2, 2, 2, 2, 2])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=3
        )

        # Expert-major: e0r0(2), e0r1(2), e0r2(2), e1r0(2), e1r1(2), e1r2(2)
        # Input positions:
        #   r0e0: [0,1], r0e1: [2,3], r1e0: [4,5], r1e1: [6,7], r2e0: [8,9], r2e1: [10,11]
        # Output: e0r0[0,1], e0r1[4,5], e0r2[8,9], e1r0[2,3], e1r1[6,7], e1r2[10,11]
        expected_indices = torch.tensor([0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11])
        torch.testing.assert_close(permuted_indices, expected_indices)

        expected_num_tokens = torch.tensor([6, 6])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_permutation_is_valid(self):
        # The output should be a permutation of [0, total)
        tokens_per_expert_group = torch.tensor([3, 1, 4, 1, 5, 9])
        permuted_indices, _ = self._permute(
            tokens_per_expert_group, experts_per_rank=3, num_ranks=2
        )

        total = tokens_per_expert_group.sum().item()
        self.assertEqual(permuted_indices.numel(), total)
        self.assertEqual(
            set(permuted_indices.tolist()),
            set(range(total)),
        )


class TestAllToAllTokenDispatcherSpmdTypes(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    def test_dispatch_combine_typechecked_matches_local_dispatcher(self):
        ep_mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("ep",),
        )
        dense_mesh = init_device_mesh(
            self.device_type,
            (1, 1, self.world_size),
            mesh_dim_names=("dp", "cp", "tp"),
        )
        dispatcher = AllToAllTokenDispatcher(
            AllToAllTokenDispatcher.Config(num_experts=2, top_k=1)
        )
        dispatcher.wire_meshes(ep_mesh=ep_mesh, tp_mesh=None)

        x_TD = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            device=self.device_type,
        )
        topk_scores_TK = torch.tensor(
            [[0.5], [1.5], [2.0], [0.25]],
            device=self.device_type,
        )
        topk_expert_ids_TK = torch.tensor(
            [[0], [1], [0], [1]],
            device=self.device_type,
            dtype=torch.long,
        )
        num_local_tokens_per_expert_E = torch.bincount(
            topk_expert_ids_TK.flatten(),
            minlength=2,
        )

        local_dispatcher = LocalTokenDispatcher(
            LocalTokenDispatcher.Config(num_experts=2, top_k=1)
        )
        local_routed_input, _, local_metadata = local_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )
        local_output = local_dispatcher.combine(
            local_routed_input,
            local_metadata,
            x_TD,
            num_local_tokens_after_padding=x_TD.shape[0],
            local_seq_len_after_padding=x_TD.shape[0],
        )

        set_spmd_backend("spmd_types")
        try:
            set_spmd_meshes(dense_mesh=dense_mesh, sparse_mesh=ep_mesh)
            typed_x_TD = x_TD.detach().clone()
            token_type = {"dp": spmd.V, "cp": spmd.V, "tp": spmd.V}
            count_type = {"dp": spmd.P, "cp": spmd.P, "tp": spmd.P}
            with set_current_spmd_mesh(dense_mesh):
                with typecheck(strict_mode="strict", local=True):
                    typed_x_TD = spmd.assert_type(typed_x_TD, token_type)
                    typed_topk_scores_TK = spmd.assert_type(
                        topk_scores_TK, token_type
                    )
                    typed_topk_expert_ids_TK = spmd.assert_type(
                        topk_expert_ids_TK, token_type
                    )
                    typed_num_tokens_E = spmd.assert_type(
                        num_local_tokens_per_expert_E, count_type
                    )
                    routed_input, _, metadata = dispatcher.dispatch(
                        typed_x_TD,
                        typed_topk_scores_TK,
                        typed_topk_expert_ids_TK,
                        typed_num_tokens_E,
                    )
                    output = dispatcher.combine(
                        routed_input,
                        metadata,
                        typed_x_TD,
                        num_local_tokens_after_padding=x_TD.shape[0],
                        local_seq_len_after_padding=x_TD.shape[0],
                    )
                    spmd.assert_type(output, token_type)
        finally:
            set_spmd_backend("default")

        self.assertTrue(torch.equal(output, local_output))


if __name__ == "__main__":
    unittest.main()
