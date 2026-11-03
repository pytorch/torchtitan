# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import spmd_types as spmd
import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import DataDependentOutputException, FakeTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.moe.sharding import expert_param_placement_sparse
from torchtitan.models.common.moe.token_dispatcher import (
    AllToAllCapacityDispatchMetadata,
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)
from torchtitan.models.qwen3 import build_model_config
from torchtitan.trainer import Trainer


class TestTokenDispatcherModule(unittest.TestCase):
    def test_hybridep_backward_is_fake_tensor_traceable(self):
        # Setup contexts must not read the CPU handle ID during AOT tracing.
        from torchtitan.distributed.deepep import hybridep  # noqa: F401

        with FakeTensorMode():
            x_TD = torch.empty(64, 256, requires_grad=True)
            expert_ids_TK = torch.empty(64, 2, dtype=torch.int64)
            scores_TK = torch.empty(64, 2, requires_grad=True)
            hidden_RD, scores_R, _, handle_id = torch.ops.hybridep.dispatch(
                x_TD, expert_ids_TK, scores_TK, 8, 2, "ep", True, 1.0, None
            )
            out_TD = torch.ops.hybridep.combine(
                hidden_RD * scores_R[:, None], handle_id, 64, None
            )
            out_TD.sum().backward()
            self.assertEqual(x_TD.grad.shape, x_TD.shape)
            self.assertEqual(scores_TK.grad.shape, scores_TK.shape)

    def test_dispatcher_has_no_checkpoint_state(self):
        dispatcher = AllToAllTokenDispatcher.Config(
            num_experts=2,
            top_k=1,
        ).build()

        self.assertIsInstance(dispatcher, torch.nn.Module)
        self.assertEqual(list(dispatcher.state_dict()), [])


class TestExpertParallelConfigValidation(unittest.TestCase):
    @staticmethod
    def _config(ep: int, tp: int = 1):
        parallelism = ParallelismConfig(
            expert_parallel_degree=ep,
            tensor_parallel_degree=tp,
        )
        model_config = build_model_config("debugmodel_moe", parallelism=parallelism)
        runtime_config = Trainer.Config(
            model=model_config,
            training=TrainingConfig(
                disable_accelerator_graphs=True,
            ),
            parallelism=parallelism,
        )
        return model_config, runtime_config

    def test_all_moe_layers_divisible(self):
        self._config(ep=8)

    def test_later_moe_layer_not_divisible(self):
        model_config = build_model_config(
            "debugmodel_moe", parallelism=ParallelismConfig()
        )
        moe = model_config.layers[1].feed_forward
        assert moe is not None
        moe.num_experts = 63

        with self.assertRaisesRegex(
            ValueError,
            r"layers\.1\.feed_forward\.num_experts \(63\).*expert_parallel_degree \(8\)",
        ):
            Trainer.Config(
                model=model_config,
                training=TrainingConfig(
                    disable_accelerator_graphs=True,
                ),
                parallelism=ParallelismConfig(expert_parallel_degree=8),
            )

    def test_tensor_parallel_requires_expert_parallel(self):
        with self.assertRaisesRegex(
            ValueError,
            r"expert_parallel_degree \(1\).*tensor_parallel_degree \(2\)",
        ):
            self._config(ep=1, tp=2)

    def test_tensor_parallel_preserves_explicit_expert_parallel(self):
        _, runtime_config = self._config(ep=4, tp=2)

        self.assertEqual(runtime_config.parallelism.expert_parallel_degree, 4)

    def test_moe_without_tensor_parallel_preserves_sequence_parallel_config(self):
        _, runtime_config = self._config(ep=1, tp=1)

        self.assertTrue(runtime_config.parallelism.enable_sequence_parallel)

    def test_dense_tensor_parallel_does_not_require_expert_parallel(self):
        model_config = build_model_config("debugmodel", parallelism=ParallelismConfig())
        runtime_config = Trainer.Config(
            model=model_config,
            training=TrainingConfig(
                disable_accelerator_graphs=True,
            ),
            parallelism=ParallelismConfig(tensor_parallel_degree=2),
        )
        self.assertEqual(runtime_config.parallelism.expert_parallel_degree, 1)
        self.assertTrue(runtime_config.parallelism.enable_sequence_parallel)

    def test_model_sharding_follows_build_parallelism(self):
        trainer_config = build_model_config(
            "debugmodel_moe",
            parallelism=ParallelismConfig(
                expert_parallel_degree=2,
                enable_sequence_parallel=False,
            ),
        )
        generator_config = build_model_config(
            "debugmodel_moe",
            parallelism=ParallelismConfig(
                expert_parallel_degree=1,
                enable_sequence_parallel=True,
            ),
        )

        trainer_moe = trainer_config.layers[0].feed_forward
        generator_moe = generator_config.layers[0].feed_forward
        assert trainer_moe is not None
        assert generator_moe is not None
        self.assertEqual(
            trainer_moe.routed_experts.w13.sharding_config.state_shardings["weight"],
            expert_param_placement_sparse(),
        )
        self.assertEqual(
            generator_moe.routed_experts.w13.sharding_config.state_shardings["weight"],
            dense_param_placement(tp=spmd.R),
        )
        self.assertNotEqual(
            trainer_config.tok_embeddings.sharding_config,
            generator_config.tok_embeddings.sharding_config,
        )


class TestPermute(unittest.TestCase):
    """Test AllToAllTokenDispatcher._permute which reorders tokens from rank-major to expert-major layout.

    Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
    Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
    """

    def _make_dispatcher(self) -> AllToAllTokenDispatcher:
        """Create a minimal AllToAllTokenDispatcher for testing _permute."""
        cfg = AllToAllTokenDispatcher.Config(num_experts=1, top_k=1)
        return AllToAllTokenDispatcher(cfg)

    def _permute(self, tokens_per_expert_group, experts_per_rank, num_ranks):
        """Helper that calls _permute and returns (permuted_indices, num_tokens_per_expert)."""
        dispatcher = self._make_dispatcher()
        total = tokens_per_expert_group.sum().item()
        dummy_input = torch.zeros(total, 1)
        _, permuted_indices, num_tokens_per_expert = dispatcher._permute(
            dummy_input, tokens_per_expert_group, ep_size=num_ranks
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


_NUM_EXPERTS = 8
_TOP_K = 3
_NUM_TOKENS = 16
_DIM = 4


def _scale_by_expert(
    routed_RD: torch.Tensor, num_tokens_per_expert_e: torch.Tensor, first_expert: int
) -> torch.Tensor:
    """Scale each row by its global expert ID + 1, and set the padding rows to NaN."""
    offsets_e = num_tokens_per_expert_e.cumsum(0)
    rows_R = torch.arange(routed_RD.shape[0])
    expert_R = torch.searchsorted(offsets_e, rows_R, right=True)
    out_RD = routed_RD * (first_expert + expert_R + 1).to(routed_RD.dtype)[:, None]
    return torch.where((rows_R < offsets_e[-1])[:, None], out_RD, float("nan"))


def _forward_backward(dispatcher, x_TD, scores_TK, ids_TK, first_expert, grad_TD):
    x_TD = x_TD.detach().requires_grad_()
    scores_TK = scores_TK.detach().requires_grad_()
    routing_map_TE = torch.zeros(
        x_TD.shape[0], _NUM_EXPERTS, dtype=torch.bool
    ).scatter_(1, ids_TK, True)
    routed_RD, num_tokens_per_expert_e, metadata = dispatcher.dispatch(
        x_TD, scores_TK, ids_TK, routing_map_TE
    )
    out_TD = dispatcher.combine(
        _scale_by_expert(routed_RD, num_tokens_per_expert_e, first_expert),
        metadata,
        x_TD,
    )
    out_TD.backward(grad_TD)
    return out_TD.detach(), x_TD.grad, scores_TK.grad, metadata, num_tokens_per_expert_e


def _kept_routes(ids_TK: torch.Tensor, ep_size: int, capacity: int) -> torch.Tensor:
    """The routes that fit: the first ``capacity`` routes to each EP rank, by expert."""
    order_N = torch.argsort(ids_TK.view(-1), stable=True)
    rank_N = ids_TK.view(-1)[order_N] // (_NUM_EXPERTS // ep_size)
    kept_N = torch.zeros(order_N.numel(), dtype=torch.bool)
    for rank in range(ep_size):
        kept_N[order_N[rank_N == rank][:capacity]] = True
    return kept_N.view_as(ids_TK)


class TestAllToAllCapacityDispatch(DTensorTestBase):
    """AllToAllTokenDispatcher with ``non_blocking_capacity_factor``, on 4 CPU ranks.

    Stand-in experts scale each row by its global expert ID + 1, so every rank can
    compute the expected result with LocalTokenDispatcher over all experts.
    """

    @property
    def world_size(self) -> int:
        return 4

    def _routing(self):
        """Per-rank tokens and top-k routing that favors experts 0-2."""
        generator = torch.Generator().manual_seed(self.rank)
        x_TD = torch.randn(_NUM_TOKENS, _DIM, generator=generator)
        weights_E = torch.ones(_NUM_EXPERTS)
        weights_E[:3] = 100.0
        ids_TK = torch.stack(
            [
                torch.multinomial(weights_E, _TOP_K, generator=generator)
                for _ in range(_NUM_TOKENS)
            ]
        )
        scores_TK = torch.rand(_NUM_TOKENS, _TOP_K, generator=generator)
        grad_TD = torch.randn(_NUM_TOKENS, _DIM, generator=generator)
        return x_TD, scores_TK, ids_TK, grad_TD

    def _dispatch_and_reference(self, capacity_factor: float, dtype: torch.dtype):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("ep",))
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh, dense_sp_enabled=False)
        x_TD, scores_TK, ids_TK, grad_TD = self._routing()
        x_TD, grad_TD = x_TD.to(dtype), grad_TD.to(dtype)
        num_local_experts = _NUM_EXPERTS // self.world_size
        dispatcher = AllToAllTokenDispatcher.Config(
            num_experts=_NUM_EXPERTS,
            top_k=_TOP_K,
            non_blocking_capacity_factor=capacity_factor,
        ).build()
        with set_current_spmd_mesh(mesh):
            actual = _forward_backward(
                dispatcher,
                x_TD,
                scores_TK,
                ids_TK,
                self.rank * num_local_experts,
                grad_TD,
            )
        capacity = math.ceil(
            _NUM_TOKENS * min(num_local_experts, _TOP_K) * capacity_factor
        )
        kept_TK = _kept_routes(ids_TK, self.world_size, capacity)
        local = LocalTokenDispatcher.Config(
            num_experts=_NUM_EXPERTS, top_k=_TOP_K
        ).build()
        expected = _forward_backward(
            local,
            x_TD,
            torch.where(kept_TK, scores_TK, 0),
            ids_TK,
            0,
            grad_TD,
        )
        return actual, expected, kept_TK

    @with_comms
    def test_full_capacity_matches_local_dispatch(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                actual, expected, kept_TK = self._dispatch_and_reference(1.0, dtype)
                self.assertTrue(kept_TK.all())
                self.assertIsInstance(actual[3], AllToAllCapacityDispatchMetadata)
                for got, want in zip(actual[:3], expected[:3]):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)

    @with_comms
    def test_drops_routes_past_capacity(self):
        actual, expected, kept_TK = self._dispatch_and_reference(0.25, torch.float32)
        # The routing favors EP ranks 0 and 1, so every rank drops some routes.
        self.assertFalse(kept_TK.all())
        out_TD, x_grad_TD, scores_grad_TK, _, num_tokens_per_expert_e = actual
        # The grouped GEMM must not read past the received rows.
        num_received_rows = self.world_size * math.ceil(
            _NUM_TOKENS * min(_NUM_EXPERTS // self.world_size, _TOP_K) * 0.25
        )
        self.assertLessEqual(int(num_tokens_per_expert_e.sum()), num_received_rows)
        torch.testing.assert_close(out_TD, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(x_grad_TD, expected[1], rtol=0, atol=0)
        # A dropped route gets no score gradient.
        torch.testing.assert_close(
            scores_grad_TK, torch.where(kept_TK, expected[2], 0), rtol=0, atol=0
        )


class TestAllToAllCapacityNoHostSync(unittest.TestCase):
    """Fake tensors hold no data, so a host read of a tensor value raises."""

    def setUp(self):
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
        self.addCleanup(dist.destroy_process_group)

    def _dispatch_fake(self, capacity_factor: float | None) -> torch.Size:
        # Only set the current mesh: a registered mesh would outlive the fake
        # process group and break later tests in this process.
        mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("ep",))
        dispatcher = AllToAllTokenDispatcher.Config(
            num_experts=_NUM_EXPERTS,
            top_k=_TOP_K,
            non_blocking_capacity_factor=capacity_factor,
        ).build()
        with FakeTensorMode(), set_current_spmd_mesh(mesh):
            x_TD = torch.randn(_NUM_TOKENS, _DIM, requires_grad=True)
            scores_TK = torch.rand(_NUM_TOKENS, _TOP_K, requires_grad=True)
            ids_TK = torch.randint(0, _NUM_EXPERTS, (_NUM_TOKENS, _TOP_K))
            routing_map_TE = torch.zeros(
                _NUM_TOKENS, _NUM_EXPERTS, dtype=torch.bool
            ).scatter_(1, ids_TK, True)
            routed_RD, _, metadata = dispatcher.dispatch(
                x_TD, scores_TK, ids_TK, routing_map_TE
            )
            dispatcher.combine(routed_RD * 2, metadata, x_TD).sum().backward()
        return routed_RD.shape

    def test_only_capacity_dispatch_needs_no_host_sync(self):
        # 4 ranks, each with C = 16 tokens * min(2 local experts, top_k 3) * factor.
        self.assertEqual(self._dispatch_fake(1.0), (4 * 32, _DIM))
        self.assertEqual(self._dispatch_fake(0.25), (4 * 8, _DIM))
        # The dynamic path reads the token counts on the host.
        with self.assertRaises(DataDependentOutputException):
            self._dispatch_fake(None)


if __name__ == "__main__":
    unittest.main()
