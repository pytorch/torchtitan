# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive test suite for PrimusTurboFlexTokenDispatcher.

Includes both unit tests (with mocking) and distributed tests (multi-GPU).

Run unit tests:
    python -m unittest tests.unit_tests.deepep.test_primus_turbo_flex_token_dispatcher.TestUnit -v

Run distributed tests (requires 2+ GPUs):
    torchrun --nproc_per_node=2 -m tests.unit_tests.deepep.test_primus_turbo_flex_token_dispatcher

Or use the convenience script:
    ./tests/unit_tests/deepep/run_tests.sh
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist

# Mock deep_ep module if not available
try:
    from deep_ep import Config
except ImportError:
    mock_deep_ep = MagicMock()
    mock_deep_ep.Config = MagicMock
    mock_deep_ep_utils = MagicMock()
    mock_deep_ep_utils.EventOverlap = MagicMock
    mock_deep_ep_utils.EventHandle = MagicMock
    sys.modules["deep_ep"] = mock_deep_ep
    sys.modules["deep_ep.utils"] = mock_deep_ep_utils

from torchtitan.distributed.deepep.utils import PrimusTurboFlexTokenDispatcher


# ============================================================================
# Unit Tests (with mocking - no GPU required)
# ============================================================================


class TestUnit(unittest.TestCase):
    """Unit tests for PrimusTurboFlexTokenDispatcher using mocking."""

    def setUp(self):
        """Set up common test parameters."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        self.moe_router_topk = 2
        self.num_moe_experts = 8
        self.batch_size = 4
        self.seq_len = 16
        self.hidden_dim = 128

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_initialization_default_params(self, mock_set_deepep):
        """Test initialization with default parameters."""
        dispatcher = PrimusTurboFlexTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        self.assertIsNotNone(dispatcher._comm_manager)
        self.assertEqual(dispatcher.tp_size, 1)
        self.assertIsNone(dispatcher.shared_experts)
        mock_set_deepep.assert_called_once_with(
            PrimusTurboFlexTokenDispatcher.turbo_deepep_num_cus
        )

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_initialization_with_class_attributes(self, mock_set_deepep):
        """Test initialization respects class-level configuration."""
        PrimusTurboFlexTokenDispatcher.turbo_deepep_backend = "mori"
        PrimusTurboFlexTokenDispatcher.turbo_deepep_num_cus = 64

        try:
            dispatcher = PrimusTurboFlexTokenDispatcher(
                moe_router_topk=self.moe_router_topk,
                num_moe_experts=self.num_moe_experts,
            )
            self.assertEqual(dispatcher._comm_manager.backend_type, "mori")
            mock_set_deepep.assert_called_with(64)
        finally:
            PrimusTurboFlexTokenDispatcher.turbo_deepep_backend = "deepep"
            PrimusTurboFlexTokenDispatcher.turbo_deepep_num_cus = 32

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_dispatch_preprocess(self, mock_set_deepep):
        """Test dispatch_preprocess method."""
        dispatcher = PrimusTurboFlexTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        top_scores = torch.rand(num_tokens, self.moe_router_topk, device=self.device)
        selected_indices = torch.randint(
            0, self.num_moe_experts, (num_tokens, self.moe_router_topk), device=self.device
        )
        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        self.assertIsNotNone(dispatcher._comm_manager.token_probs)
        self.assertIsNotNone(dispatcher._comm_manager.token_indices)
        self.assertEqual(
            dispatcher._comm_manager.token_probs.shape,
            (num_tokens, self.moe_router_topk),
        )

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_token_dispatch(self, mock_set_deepep):
        """Test token_dispatch method."""
        dispatcher = PrimusTurboFlexTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        top_scores = torch.rand(num_tokens, self.moe_router_topk, device=self.device)
        selected_indices = torch.randint(
            0, self.num_moe_experts, (num_tokens, self.moe_router_topk), device=self.device
        )
        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        hidden_states = torch.randn(num_tokens, self.hidden_dim, device=self.device)
        mock_dispatched_hidden = torch.randn(num_tokens * 2, self.hidden_dim, device=self.device)
        mock_dispatched_probs = torch.rand(num_tokens * 2, device=self.device)

        with patch.object(dispatcher._comm_manager, "dispatch") as mock_dispatch:
            mock_dispatch.return_value = mock_dispatched_hidden
            dispatcher._comm_manager.dispatched_probs = mock_dispatched_probs
            mock_group = MagicMock()
            result_hidden, result_probs = dispatcher.token_dispatch(
                hidden_states, probs=None, group=mock_group, async_finish=True, allocate_on_comm_stream=True
            )
            mock_dispatch.assert_called_once_with(hidden_states, mock_group, True, True)
            self.assertTrue(torch.equal(result_hidden, mock_dispatched_hidden))
            self.assertTrue(torch.equal(result_probs, mock_dispatched_probs))

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    @patch("torchtitan.distributed.deepep.utils.fused_indices_to_multihot")
    @patch("torchtitan.distributed.deepep.utils.permute")
    def test_dispatch_postprocess(self, mock_permute, mock_fused_indices_to_multihot, mock_set_deepep):
        """Test dispatch_postprocess method."""
        dispatcher = PrimusTurboFlexTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_dispatched_tokens = self.batch_size * self.seq_len * 2
        dispatcher._comm_manager.dispatched_indices = torch.randint(
            0, num_dispatched_tokens, (num_dispatched_tokens,), device=self.device
        )
        dispatcher._comm_manager.dispatched_probs = torch.rand(
            num_dispatched_tokens, device=self.device, dtype=torch.float32
        )
        dispatcher._comm_manager.num_local_experts = self.num_moe_experts
        dispatcher._comm_manager.tokens_per_expert = torch.tensor(
            [16] * self.num_moe_experts, device=self.device
        )

        mock_routing_map = torch.randint(0, 2, (num_dispatched_tokens, self.num_moe_experts), device=self.device)
        mock_probs = torch.rand(num_dispatched_tokens, device=self.device, dtype=torch.float32)
        mock_fused_indices_to_multihot.return_value = (mock_routing_map, mock_probs)

        hidden_states = torch.randn(num_dispatched_tokens, self.hidden_dim, device=self.device)
        mock_permuted_hidden = torch.randn(num_dispatched_tokens, self.hidden_dim, device=self.device)
        mock_permuted_probs = torch.rand(num_dispatched_tokens, device=self.device, dtype=torch.float32)
        mock_sorted_indices = torch.randint(0, num_dispatched_tokens, (num_dispatched_tokens,), device=self.device)
        mock_permute.return_value = (mock_permuted_hidden, mock_permuted_probs, mock_sorted_indices)

        probs = torch.rand(num_dispatched_tokens, device=self.device)
        result_hidden, result_tokens_per_expert, result_probs = dispatcher.dispatch_postprocess(hidden_states, probs)

        mock_fused_indices_to_multihot.assert_called_once()
        mock_permute.assert_called_once()
        self.assertTrue(torch.equal(result_hidden, mock_permuted_hidden))
        self.assertTrue(torch.equal(result_probs, mock_permuted_probs))

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    @patch("torchtitan.distributed.deepep.utils.unpermute")
    def test_combine_preprocess(self, mock_unpermute, mock_set_deepep):
        """Test combine_preprocess method."""
        dispatcher = PrimusTurboFlexTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len * 2
        hidden_states = torch.randn(num_tokens, self.hidden_dim, device=self.device)
        dispatcher._comm_manager.reversed_mapping_for_combine = torch.randint(0, num_tokens, (num_tokens,), device=self.device)
        dispatcher._comm_manager.hidden_shape_before_permute = torch.Size([num_tokens, self.hidden_dim])

        mock_unpermuted = torch.randn(num_tokens, self.hidden_dim, device=self.device)
        mock_unpermute.return_value = mock_unpermuted
        result = dispatcher.combine_preprocess(hidden_states)

        mock_unpermute.assert_called_once_with(
            hidden_states,
            dispatcher._comm_manager.reversed_mapping_for_combine,
            restore_shape=dispatcher._comm_manager.hidden_shape_before_permute,
        )
        self.assertTrue(torch.equal(result, mock_unpermuted))

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    @patch("torchtitan.distributed.deepep.utils.fused_combine")
    def test_token_combine(self, mock_fused_combine, mock_set_deepep):
        """Test token_combine method."""
        dispatcher = PrimusTurboFlexTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        hidden_states = torch.randn(num_tokens, self.hidden_dim, device=self.device)
        mock_handle = MagicMock()
        dispatcher._comm_manager.handle = mock_handle

        mock_combined_hidden = torch.randn(num_tokens, self.hidden_dim, device=self.device)
        mock_event = MagicMock()
        mock_fused_combine.return_value = (mock_combined_hidden, mock_event)
        mock_group = MagicMock()

        result = dispatcher.token_combine(hidden_states, group=mock_group, async_finish=True, allocate_on_comm_stream=True)

        mock_fused_combine.assert_called_once_with(hidden_states, mock_group, mock_handle, async_finish=True, allocate_on_comm_stream=True)
        self.assertTrue(torch.equal(result, mock_combined_hidden))
        self.assertIsNone(dispatcher._comm_manager.handle)

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_combine_postprocess(self, mock_set_deepep):
        """Test combine_postprocess method."""
        dispatcher = PrimusTurboFlexTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        dispatcher.hidden_shape = torch.Size([self.batch_size, self.seq_len, self.hidden_dim])
        hidden_states = torch.randn(num_tokens, self.hidden_dim, device=self.device)
        result = dispatcher.combine_postprocess(hidden_states)
        self.assertEqual(result.shape, dispatcher.hidden_shape)

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_dispatch_preprocess_with_zero_capacity(self, mock_set_deepep):
        """Test dispatch_preprocess with capacity_factor causing zero probs."""
        dispatcher = PrimusTurboFlexTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        dispatcher._comm_manager.capacity_factor = 1.0
        num_tokens = self.batch_size * self.seq_len
        top_scores = torch.rand(num_tokens, self.moe_router_topk, device=self.device)
        top_scores[0, 0] = 0.0
        top_scores[1, 1] = 0.0
        selected_indices = torch.randint(0, self.num_moe_experts, (num_tokens, self.moe_router_topk), device=self.device)
        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        self.assertEqual(dispatcher._comm_manager.token_indices[0, 0].item(), -1)
        self.assertEqual(dispatcher._comm_manager.token_indices[1, 1].item(), -1)


# ============================================================================
# Distributed Tests (requires multi-GPU)
# ============================================================================


def _is_distributed():
    """Check if running in distributed environment."""
    return dist.is_available() and dist.is_initialized()


def _get_world_size():
    """Get world size or 1 if not distributed."""
    return dist.get_world_size() if _is_distributed() else 1


def _get_rank():
    """Get rank or 0 if not distributed."""
    return dist.get_rank() if _is_distributed() else 0


class TestDistributed(unittest.TestCase):
    """Distributed tests requiring actual multi-GPU setup."""

    @classmethod
    def setUpClass(cls):
        """Set up distributed environment."""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest(f"Need 2+ GPUs, found {torch.cuda.device_count()}")
        if not dist.is_initialized():
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                dist.init_process_group(backend="nccl")
                cls.distributed_initialized_here = True
            else:
                raise unittest.SkipTest("Run with: torchrun --nproc_per_node=2 ...")
        else:
            cls.distributed_initialized_here = False

        cls.rank = _get_rank()
        cls.world_size = _get_world_size()
        cls.device = torch.device(f"cuda:{cls.rank}")
        torch.cuda.set_device(cls.device)

        if cls.rank == 0:
            print(f"\n=== Distributed tests with {cls.world_size} GPUs ===")

    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment."""
        if cls.distributed_initialized_here and dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        """Set up test parameters."""
        torch.manual_seed(42 + self.rank)
        self.moe_router_topk = 2
        self.num_moe_experts = self.world_size * 2
        self.batch_size = 4
        self.seq_len = 16
        self.hidden_dim = 128

    def test_dispatcher_initialization(self):
        """Test dispatcher initialization in distributed environment."""
        dispatcher = PrimusTurboFlexTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        self.assertIsNotNone(dispatcher._comm_manager)
        dist.barrier()
        if self.rank == 0:
            print(f"✓ Dispatcher initialized on {self.world_size} ranks")

    def test_dispatch_preprocess(self):
        """Test dispatch_preprocess across ranks."""
        dispatcher = PrimusTurboFlexTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        top_scores = torch.rand(num_tokens, self.moe_router_topk, device=self.device)
        selected_indices = torch.randint(0, self.num_moe_experts, (num_tokens, self.moe_router_topk), device=self.device)
        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        self.assertIsNotNone(dispatcher._comm_manager.token_probs)
        dist.barrier()
        if self.rank == 0:
            print(f"✓ Dispatch preprocess on {self.world_size} ranks")

    def test_token_shapes_consistency(self):
        """Test token shapes match across ranks."""
        dispatcher = PrimusTurboFlexTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        torch.manual_seed(42)  # Same seed for all ranks
        top_scores = torch.rand(num_tokens, self.moe_router_topk, device=self.device)
        selected_indices = torch.randint(0, self.num_moe_experts, (num_tokens, self.moe_router_topk), device=self.device)
        dispatcher.dispatch_preprocess(top_scores, selected_indices)

        local_shape = torch.tensor(list(dispatcher._comm_manager.token_probs.shape), device=self.device)
        shape_list = [torch.zeros_like(local_shape) for _ in range(self.world_size)]
        dist.all_gather(shape_list, local_shape)

        for i, shape in enumerate(shape_list):
            self.assertTrue(torch.equal(shape, local_shape), f"Shape mismatch at rank {i}")

        dist.barrier()
        if self.rank == 0:
            print(f"✓ Token shapes consistent: {local_shape.tolist()}")

    def test_expert_distribution(self):
        """Test experts correctly distributed across ranks."""
        dispatcher = PrimusTurboFlexTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        expected_experts_per_rank = self.num_moe_experts // self.world_size
        pg = dist.group.WORLD
        dispatcher._comm_manager.num_local_experts = self.num_moe_experts // dist.get_world_size(pg)
        self.assertEqual(dispatcher._comm_manager.num_local_experts, expected_experts_per_rank)
        dist.barrier()
        if self.rank == 0:
            print(f"✓ {expected_experts_per_rank} experts per rank")

    def test_cross_rank_communication(self):
        """Test cross-rank communication works."""
        test_tensor = torch.ones(10, device=self.device) * (self.rank + 1)
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        expected_sum = sum(range(1, self.world_size + 1))
        self.assertAlmostEqual(test_tensor[0].item(), expected_sum, places=5)
        dist.barrier()
        if self.rank == 0:
            print(f"✓ Cross-rank communication verified")


# ============================================================================
# Main Entry Point
# ============================================================================


def run_tests():
    """Run appropriate tests based on environment."""
    loader = unittest.TestLoader()

    # Check if running in distributed mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Run distributed tests
        suite = loader.loadTestsFromTestCase(TestDistributed)
        runner = unittest.TextTestRunner(verbosity=2 if _get_rank() == 0 else 0)
        result = runner.run(suite)
        if dist.is_initialized():
            dist.barrier()
        return 0 if result.wasSuccessful() else 1
    else:
        # Run unit tests
        suite = loader.loadTestsFromTestCase(TestUnit)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
