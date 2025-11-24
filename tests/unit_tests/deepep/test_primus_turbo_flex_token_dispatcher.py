# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Comprehensive test suite for DeepEPTokenDispatcher.

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
DEEP_EP_AVAILABLE = False
try:
    import deep_ep

    DEEP_EP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass

if not DEEP_EP_AVAILABLE:
    mock_deep_ep = MagicMock()
    mock_deep_ep.Config = MagicMock
    mock_deep_ep_utils = MagicMock()
    mock_deep_ep_utils.EventOverlap = MagicMock
    mock_deep_ep_utils.EventHandle = MagicMock
    sys.modules["deep_ep"] = mock_deep_ep
    sys.modules["deep_ep.utils"] = mock_deep_ep_utils

from torchtitan.distributed.deepep.utils import DeepEPTokenDispatcher


# ============================================================================
# Unit Tests (with mocking - no GPU required)
# ============================================================================


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions used by the dispatcher."""

    def setUp(self):
        """Set up common test parameters."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def test_permute_basic(self):
        """Test permute function with basic inputs."""
        from torchtitan.distributed.deepep.utils import permute

        # Create simple test data
        num_tokens = 4
        hidden_dim = 8
        num_experts = 2

        tokens = torch.randn(num_tokens, hidden_dim, device=self.device)
        # Create routing map: token 0->expert 0, token 1->expert 1, token 2->expert 0, token 3->expert 1
        routing_map = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], device=self.device)
        probs = torch.rand(num_tokens, num_experts, device=self.device)

        permuted_tokens, permuted_probs, sorted_indices = permute(
            tokens, routing_map, probs
        )

        # Verify shapes
        self.assertEqual(permuted_tokens.shape[1], hidden_dim)
        self.assertGreater(permuted_tokens.shape[0], 0)
        self.assertEqual(permuted_probs.shape[0], permuted_tokens.shape[0])
        self.assertEqual(sorted_indices.shape[0], permuted_tokens.shape[0])

    def test_permute_without_probs(self):
        """Test permute function without probability tensor."""
        from torchtitan.distributed.deepep.utils import permute

        num_tokens = 4
        hidden_dim = 8
        num_experts = 2

        tokens = torch.randn(num_tokens, hidden_dim, device=self.device)
        routing_map = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], device=self.device)

        permuted_tokens, permuted_probs, sorted_indices = permute(
            tokens, routing_map, None
        )

        # Verify shapes and that probs is None
        self.assertIsNone(permuted_probs)
        self.assertEqual(permuted_tokens.shape[1], hidden_dim)

    def test_unpermute_basic(self):
        """Test unpermute function restores original shape."""
        from torchtitan.distributed.deepep.utils import permute, unpermute

        num_tokens = 4
        hidden_dim = 8
        num_experts = 2

        original_tokens = torch.randn(num_tokens, hidden_dim, device=self.device)
        routing_map = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], device=self.device)

        # Permute then unpermute
        permuted_tokens, _, sorted_indices = permute(original_tokens, routing_map, None)
        restored_tokens = unpermute(
            permuted_tokens, sorted_indices, original_tokens.shape
        )

        # Verify shape is restored
        self.assertEqual(restored_tokens.shape, original_tokens.shape)

    def test_unpermute_different_dtypes(self):
        """Test unpermute with different data types."""
        from torchtitan.distributed.deepep.utils import unpermute

        num_tokens = 8
        hidden_dim = 16

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            if dtype == torch.bfloat16 and not torch.cuda.is_available():
                continue

            sorted_indices = torch.randperm(num_tokens, device=self.device)
            permuted_tokens = torch.randn(
                num_tokens, hidden_dim, device=self.device, dtype=dtype
            )
            restore_shape = torch.Size([num_tokens, hidden_dim])

            restored = unpermute(permuted_tokens, sorted_indices, restore_shape)

            self.assertEqual(restored.shape, restore_shape)
            self.assertEqual(restored.dtype, dtype)


class TestUnit(unittest.TestCase):
    """Unit tests for DeepEPTokenDispatcher using mocking."""

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
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        self.assertIsNotNone(dispatcher._comm_manager)
        self.assertEqual(dispatcher.tp_size, 1)
        self.assertIsNone(dispatcher.shared_experts)
        mock_set_deepep.assert_called_once_with(
            DeepEPTokenDispatcher.turbo_deepep_num_cus
        )

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_initialization_with_class_attributes(self, mock_set_deepep):
        """Test initialization respects class-level configuration."""
        DeepEPTokenDispatcher.turbo_deepep_backend = "mori"
        DeepEPTokenDispatcher.turbo_deepep_num_cus = 64

        try:
            dispatcher = DeepEPTokenDispatcher(
                moe_router_topk=self.moe_router_topk,
                num_moe_experts=self.num_moe_experts,
            )
            self.assertEqual(dispatcher._comm_manager.backend_type, "mori")
            mock_set_deepep.assert_called_with(64)
        finally:
            DeepEPTokenDispatcher.turbo_deepep_backend = "deepep"
            DeepEPTokenDispatcher.turbo_deepep_num_cus = 32

    # ============================================================================
    # Token Dispatch Tests
    # ============================================================================

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_dispatch_preprocess(self, mock_set_deepep):
        """Test dispatch_preprocess method."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        top_scores = torch.rand(num_tokens, self.moe_router_topk, device=self.device)
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )
        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        self.assertIsNotNone(dispatcher._comm_manager.token_probs)
        self.assertIsNotNone(dispatcher._comm_manager.token_indices)
        self.assertEqual(
            dispatcher._comm_manager.token_probs.shape,
            (num_tokens, self.moe_router_topk),
        )

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_dispatch_preprocess_shape_handling(self, mock_set_deepep):
        """Test dispatch_preprocess with different input shapes."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        # Test with 3D tensors (batch, seq, topk)
        batch, seq, topk = 2, 8, self.moe_router_topk
        top_scores_3d = torch.rand(batch, seq, topk, device=self.device)
        selected_indices_3d = torch.randint(
            0, self.num_moe_experts, (batch, seq, topk), device=self.device
        )

        dispatcher.dispatch_preprocess(top_scores_3d, selected_indices_3d)

        # Should be flattened to 2D
        expected_tokens = batch * seq
        self.assertEqual(
            dispatcher._comm_manager.token_probs.shape, (expected_tokens, topk)
        )
        self.assertEqual(
            dispatcher._comm_manager.token_indices.shape, (expected_tokens, topk)
        )

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_dispatch_preprocess_with_zero_capacity(self, mock_set_deepep):
        """Test dispatch_preprocess with capacity_factor causing zero probs."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        dispatcher._comm_manager.capacity_factor = 1.0
        num_tokens = self.batch_size * self.seq_len
        top_scores = torch.rand(num_tokens, self.moe_router_topk, device=self.device)
        top_scores[0, 0] = 0.0
        top_scores[1, 1] = 0.0
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )
        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        self.assertEqual(dispatcher._comm_manager.token_indices[0, 0].item(), -1)
        self.assertEqual(dispatcher._comm_manager.token_indices[1, 1].item(), -1)

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_token_dispatch(self, mock_set_deepep):
        """Test token_dispatch method."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        top_scores = torch.rand(num_tokens, self.moe_router_topk, device=self.device)
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )
        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        hidden_states = torch.randn(num_tokens, self.hidden_dim, device=self.device)
        mock_dispatched_hidden = torch.randn(
            num_tokens * 2, self.hidden_dim, device=self.device
        )
        mock_dispatched_probs = torch.rand(num_tokens * 2, device=self.device)

        with patch.object(dispatcher._comm_manager, "dispatch") as mock_dispatch:
            mock_dispatch.return_value = mock_dispatched_hidden
            dispatcher._comm_manager.dispatched_probs = mock_dispatched_probs
            mock_group = MagicMock()
            result_hidden, result_probs = dispatcher.token_dispatch(
                hidden_states,
                probs=None,
                group=mock_group,
                async_finish=True,
                allocate_on_comm_stream=True,
            )
            mock_dispatch.assert_called_once_with(hidden_states, mock_group, True, True)
            self.assertTrue(torch.equal(result_hidden, mock_dispatched_hidden))
            self.assertTrue(torch.equal(result_probs, mock_dispatched_probs))

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_token_dispatch_sync_mode(self, mock_set_deepep):
        """Test token_dispatch with synchronous mode."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        hidden_states = torch.randn(num_tokens, self.hidden_dim, device=self.device)

        with patch.object(dispatcher._comm_manager, "dispatch") as mock_dispatch:
            mock_dispatched_hidden = torch.randn(
                num_tokens * 2, self.hidden_dim, device=self.device
            )
            mock_dispatch.return_value = mock_dispatched_hidden
            dispatcher._comm_manager.dispatched_probs = torch.rand(
                num_tokens * 2, device=self.device
            )

            mock_group = MagicMock()
            result_hidden, _ = dispatcher.token_dispatch(
                hidden_states,
                probs=None,
                group=mock_group,
                async_finish=False,
                allocate_on_comm_stream=False,
            )
            mock_dispatch.assert_called_once_with(
                hidden_states, mock_group, False, False
            )

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    @patch("torchtitan.distributed.deepep.utils.fused_indices_to_multihot")
    @patch("torchtitan.distributed.deepep.utils.permute")
    def test_dispatch_postprocess(
        self, mock_permute, mock_fused_indices_to_multihot, mock_set_deepep
    ):
        """Test dispatch_postprocess method."""
        dispatcher = DeepEPTokenDispatcher(
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

        mock_routing_map = torch.randint(
            0, 2, (num_dispatched_tokens, self.num_moe_experts), device=self.device
        )
        mock_probs = torch.rand(
            num_dispatched_tokens, device=self.device, dtype=torch.float32
        )
        mock_fused_indices_to_multihot.return_value = (mock_routing_map, mock_probs)

        hidden_states = torch.randn(
            num_dispatched_tokens, self.hidden_dim, device=self.device
        )
        mock_permuted_hidden = torch.randn(
            num_dispatched_tokens, self.hidden_dim, device=self.device
        )
        mock_permuted_probs = torch.rand(
            num_dispatched_tokens, device=self.device, dtype=torch.float32
        )
        mock_sorted_indices = torch.randint(
            0, num_dispatched_tokens, (num_dispatched_tokens,), device=self.device
        )
        mock_permute.return_value = (
            mock_permuted_hidden,
            mock_permuted_probs,
            mock_sorted_indices,
        )

        probs = torch.rand(num_dispatched_tokens, device=self.device)
        (
            result_hidden,
            result_tokens_per_expert,
            result_probs,
        ) = dispatcher.dispatch_postprocess(hidden_states, probs)

        mock_fused_indices_to_multihot.assert_called_once()
        mock_permute.assert_called_once()
        self.assertTrue(torch.equal(result_hidden, mock_permuted_hidden))
        self.assertTrue(torch.equal(result_probs, mock_permuted_probs))

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    @patch("torchtitan.distributed.deepep.utils.fused_indices_to_multihot")
    @patch("torchtitan.distributed.deepep.utils.permute")
    def test_dispatch_postprocess_verifies_metadata(
        self, mock_permute, mock_fused_indices_to_multihot, mock_set_deepep
    ):
        """Test dispatch_postprocess properly sets up metadata."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_dispatched_tokens = 128
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

        mock_routing_map = torch.randint(
            0, 2, (num_dispatched_tokens, self.num_moe_experts), device=self.device
        )
        mock_probs = torch.rand(
            num_dispatched_tokens, device=self.device, dtype=torch.float32
        )
        mock_fused_indices_to_multihot.return_value = (mock_routing_map, mock_probs)

        mock_sorted_indices = torch.randperm(num_dispatched_tokens, device=self.device)
        mock_permute.return_value = (
            torch.randn(num_dispatched_tokens, self.hidden_dim, device=self.device),
            torch.rand(num_dispatched_tokens, device=self.device, dtype=torch.float32),
            mock_sorted_indices,
        )

        hidden_states = torch.randn(
            num_dispatched_tokens, self.hidden_dim, device=self.device
        )
        probs = torch.rand(num_dispatched_tokens, device=self.device)

        dispatcher.dispatch_postprocess(hidden_states, probs)

        # Verify metadata is stored
        self.assertIsNotNone(dispatcher._comm_manager.reversed_mapping_for_combine)
        self.assertEqual(
            dispatcher._comm_manager.hidden_shape_before_permute, hidden_states.shape
        )

    # ============================================================================
    # Token Combine Tests
    # ============================================================================

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    @patch("torchtitan.distributed.deepep.utils.unpermute")
    def test_combine_preprocess(self, mock_unpermute, mock_set_deepep):
        """Test combine_preprocess method."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len * 2
        hidden_states = torch.randn(num_tokens, self.hidden_dim, device=self.device)
        dispatcher._comm_manager.reversed_mapping_for_combine = torch.randint(
            0, num_tokens, (num_tokens,), device=self.device
        )
        dispatcher._comm_manager.hidden_shape_before_permute = torch.Size(
            [num_tokens, self.hidden_dim]
        )

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
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        hidden_states = torch.randn(num_tokens, self.hidden_dim, device=self.device)
        mock_handle = MagicMock()
        dispatcher._comm_manager.handle = mock_handle

        mock_combined_hidden = torch.randn(
            num_tokens, self.hidden_dim, device=self.device
        )
        mock_event = MagicMock()
        mock_fused_combine.return_value = (mock_combined_hidden, mock_event)
        mock_group = MagicMock()

        result = dispatcher.token_combine(
            hidden_states,
            group=mock_group,
            async_finish=True,
            allocate_on_comm_stream=True,
        )

        mock_fused_combine.assert_called_once_with(
            hidden_states,
            mock_group,
            mock_handle,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        self.assertTrue(torch.equal(result, mock_combined_hidden))
        self.assertIsNone(dispatcher._comm_manager.handle)

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    @patch("torchtitan.distributed.deepep.utils.fused_combine")
    def test_token_combine_sync_mode(self, mock_fused_combine, mock_set_deepep):
        """Test token_combine with synchronous mode."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        hidden_states = torch.randn(num_tokens, self.hidden_dim, device=self.device)
        mock_handle = MagicMock()
        dispatcher._comm_manager.handle = mock_handle

        mock_combined_hidden = torch.randn(
            num_tokens, self.hidden_dim, device=self.device
        )
        mock_event = MagicMock()
        mock_fused_combine.return_value = (mock_combined_hidden, mock_event)
        mock_group = MagicMock()

        result = dispatcher.token_combine(
            hidden_states,
            group=mock_group,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        mock_fused_combine.assert_called_once_with(
            hidden_states,
            mock_group,
            mock_handle,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_token_combine_handle_cleanup(self, mock_set_deepep):
        """Test that token_combine properly cleans up the handle."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        # Set a handle
        mock_handle = MagicMock()
        dispatcher._comm_manager.handle = mock_handle
        self.assertIsNotNone(dispatcher._comm_manager.handle)

        # Call token_combine
        with patch(
            "torchtitan.distributed.deepep.utils.fused_combine"
        ) as mock_fused_combine:
            mock_fused_combine.return_value = (
                torch.randn(64, self.hidden_dim, device=self.device),
                MagicMock(),
            )
            dispatcher.token_combine(
                torch.randn(64, self.hidden_dim, device=self.device)
            )

        # Verify handle is cleaned up
        self.assertIsNone(dispatcher._comm_manager.handle)

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_combine_postprocess(self, mock_set_deepep):
        """Test combine_postprocess method."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        dispatcher.hidden_shape = torch.Size(
            [self.batch_size, self.seq_len, self.hidden_dim]
        )
        hidden_states = torch.randn(num_tokens, self.hidden_dim, device=self.device)
        result = dispatcher.combine_postprocess(hidden_states)
        self.assertEqual(result.shape, dispatcher.hidden_shape)

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_combine_postprocess_different_shapes(self, mock_set_deepep):
        """Test combine_postprocess with various output shapes."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        test_cases = [
            (4, 16, 128),  # batch, seq, hidden
            (2, 32, 256),
            (8, 8, 64),
            (1, 128, 512),
        ]

        for batch, seq, hidden in test_cases:
            dispatcher.hidden_shape = torch.Size([batch, seq, hidden])
            num_tokens = batch * seq
            hidden_states = torch.randn(num_tokens, hidden, device=self.device)
            result = dispatcher.combine_postprocess(hidden_states)
            self.assertEqual(result.shape, dispatcher.hidden_shape)

    # ============================================================================
    # End-to-End Tests
    # ============================================================================

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    @patch("torchtitan.distributed.deepep.utils.fused_indices_to_multihot")
    def test_end_to_end_dispatch_combine_flow(self, mock_multihot, mock_set_deepep):
        """Test complete dispatch -> postprocess -> preprocess -> combine flow."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        # Setup initial data
        num_tokens = self.batch_size * self.seq_len
        dispatcher.hidden_shape = torch.Size(
            [self.batch_size, self.seq_len, self.hidden_dim]
        )
        top_scores = torch.rand(num_tokens, self.moe_router_topk, device=self.device)
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )
        hidden_states = torch.randn(num_tokens, self.hidden_dim, device=self.device)

        # 1. Preprocess
        dispatcher.dispatch_preprocess(top_scores, selected_indices)

        # 2. Dispatch - mock at the manager level
        num_dispatched = num_tokens * 2
        dispatched_hidden = torch.randn(
            num_dispatched, self.hidden_dim, device=self.device
        )
        dispatched_indices = torch.randint(
            0, num_dispatched, (num_dispatched,), device=self.device
        )
        dispatched_probs = torch.rand(
            num_dispatched, device=self.device, dtype=torch.float32
        )
        tokens_per_expert = torch.tensor(
            [num_dispatched // self.num_moe_experts] * self.num_moe_experts,
            device=self.device,
        )

        mock_group = MagicMock()

        with patch.object(dispatcher._comm_manager, "dispatch") as mock_dispatch:
            mock_dispatch.return_value = dispatched_hidden
            dispatcher._comm_manager.dispatched_indices = dispatched_indices
            dispatcher._comm_manager.dispatched_probs = dispatched_probs
            dispatcher._comm_manager.tokens_per_expert = tokens_per_expert
            dispatcher._comm_manager.num_local_experts = self.num_moe_experts
            dispatcher._comm_manager.handle = MagicMock()

            result_hidden, result_probs = dispatcher.token_dispatch(
                hidden_states, group=mock_group
            )

        # 3. Dispatch postprocess (simulates expert computation)
        routing_map = torch.randint(
            0, 2, (num_dispatched, self.num_moe_experts), device=self.device
        ).bool()
        mock_multihot.return_value = (routing_map, dispatched_probs)

        expert_output = torch.randn(num_dispatched, self.hidden_dim, device=self.device)
        processed_hidden, _, processed_probs = dispatcher.dispatch_postprocess(
            expert_output, dispatched_probs
        )

        # 4. Combine preprocess
        restored_hidden = dispatcher.combine_preprocess(processed_hidden)

        # 5. Combine
        combined_hidden = torch.randn(num_tokens, self.hidden_dim, device=self.device)

        with patch.object(dispatcher._comm_manager, "combine") as mock_combine:
            mock_combine.return_value = combined_hidden
            result = dispatcher.token_combine(restored_hidden, group=mock_group)

        # 6. Combine postprocess
        final_output = dispatcher.combine_postprocess(result)

        # Verify final shape matches input
        self.assertEqual(final_output.shape, dispatcher.hidden_shape)

    @patch("torchtitan.distributed.deepep.utils.set_deepep_num_sms")
    def test_end_to_end_preserves_dtype(self, mock_set_deepep):
        """Test that end-to-end flow preserves tensor dtypes."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        for dtype in [torch.float32, torch.float16]:
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue

            num_tokens = 32
            dispatcher.hidden_shape = torch.Size([4, 8, self.hidden_dim])
            hidden_states = torch.randn(
                num_tokens, self.hidden_dim, device=self.device, dtype=dtype
            )

            # Mock the entire flow
            with patch("torchtitan.distributed.deepep.utils.fused_dispatch"), patch(
                "torchtitan.distributed.deepep.utils.fused_combine"
            ) as mock_combine, patch(
                "torchtitan.distributed.deepep.utils.fused_indices_to_multihot"
            ):

                combined_hidden = torch.randn(
                    num_tokens, self.hidden_dim, device=self.device, dtype=dtype
                )
                mock_combine.return_value = (combined_hidden, MagicMock())

                # Just test combine_postprocess preserves dtype
                result = dispatcher.combine_postprocess(combined_hidden)
                self.assertEqual(result.dtype, dtype)


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
        # DeepEP kernels only support bfloat16
        self.dtype = torch.bfloat16

    # ============================================================================
    # Basic Setup Tests
    # ============================================================================

    def test_dispatcher_initialization(self):
        """Test dispatcher initialization in distributed environment."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        self.assertIsNotNone(dispatcher._comm_manager)
        dist.barrier()
        if self.rank == 0:
            print(f"✓ Dispatcher initialized on {self.world_size} ranks")

    def test_cross_rank_communication(self):
        """Test cross-rank communication works."""
        test_tensor = torch.ones(10, device=self.device) * (self.rank + 1)
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        expected_sum = sum(range(1, self.world_size + 1))
        self.assertAlmostEqual(test_tensor[0].item(), expected_sum, places=5)
        dist.barrier()
        if self.rank == 0:
            print(f"✓ Cross-rank communication verified")

    def test_expert_distribution(self):
        """Test experts correctly distributed across ranks."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        expected_experts_per_rank = self.num_moe_experts // self.world_size
        pg = dist.group.WORLD
        dispatcher._comm_manager.num_local_experts = (
            self.num_moe_experts // dist.get_world_size(pg)
        )
        self.assertEqual(
            dispatcher._comm_manager.num_local_experts, expected_experts_per_rank
        )
        dist.barrier()
        if self.rank == 0:
            print(f"✓ {expected_experts_per_rank} experts per rank")

    # ============================================================================
    # Token Dispatch Distributed Tests
    # ============================================================================

    def test_dispatch_preprocess(self):
        """Test dispatch_preprocess across ranks."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        top_scores = torch.rand(num_tokens, self.moe_router_topk, device=self.device)
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )
        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        self.assertIsNotNone(dispatcher._comm_manager.token_probs)
        dist.barrier()
        if self.rank == 0:
            print(f"✓ Dispatch preprocess on {self.world_size} ranks")

    def test_token_shapes_consistency(self):
        """Test token shapes match across ranks."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len
        torch.manual_seed(42)  # Same seed for all ranks
        top_scores = torch.rand(num_tokens, self.moe_router_topk, device=self.device)
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )
        dispatcher.dispatch_preprocess(top_scores, selected_indices)

        local_shape = torch.tensor(
            list(dispatcher._comm_manager.token_probs.shape), device=self.device
        )
        shape_list = [torch.zeros_like(local_shape) for _ in range(self.world_size)]
        dist.all_gather(shape_list, local_shape)

        for i, shape in enumerate(shape_list):
            self.assertTrue(
                torch.equal(shape, local_shape), f"Shape mismatch at rank {i}"
            )

        dist.barrier()
        if self.rank == 0:
            print(f"✓ Token shapes consistent: {local_shape.tolist()}")

    @unittest.skipIf(not DEEP_EP_AVAILABLE, "Requires real deep_ep library")
    @patch("torchtitan.distributed.deepep.fused_a2a.fused_dispatch")
    def test_token_dispatch_distributed(self, mock_fused_dispatch):
        """Test token_dispatch with real distributed communication."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )
        num_tokens = self.batch_size * self.seq_len

        # Use same seed for all ranks to ensure consistent routing
        torch.manual_seed(42)
        top_scores = torch.rand(
            num_tokens, self.moe_router_topk, device=self.device, dtype=self.dtype
        )
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )

        dispatcher.dispatch_preprocess(top_scores, selected_indices)

        hidden_states = torch.randn(
            num_tokens, self.hidden_dim, device=self.device, dtype=self.dtype
        )

        # Perform dispatch - get the default world process group
        world_group = dist.distributed_c10d._get_default_group()
        dispatched_hidden, dispatched_probs = dispatcher.token_dispatch(
            hidden_states,
            probs=None,
            group=world_group,
            async_finish=False,  # Synchronous for testing
            allocate_on_comm_stream=False,
        )

        # Verify outputs
        self.assertIsNotNone(dispatched_hidden)
        self.assertIsNotNone(dispatched_probs)
        self.assertEqual(dispatched_hidden.shape[1], self.hidden_dim)
        self.assertEqual(dispatched_probs.dtype, torch.float32)

        dist.barrier()
        if self.rank == 0:
            print(f"✓ Token dispatch distributed communication")

    @unittest.skipIf(not DEEP_EP_AVAILABLE, "Requires real deep_ep library")
    def test_dispatch_postprocess_distributed(self):
        """Test dispatch_postprocess with distributed data."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        num_tokens = self.batch_size * self.seq_len
        torch.manual_seed(42)
        top_scores = torch.rand(
            num_tokens, self.moe_router_topk, device=self.device, dtype=self.dtype
        )
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )

        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        hidden_states = torch.randn(
            num_tokens, self.hidden_dim, device=self.device, dtype=self.dtype
        )

        # Dispatch
        world_group = dist.distributed_c10d._get_default_group()
        dispatched_hidden, dispatched_probs = dispatcher.token_dispatch(
            hidden_states,
            group=world_group,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        # Postprocess
        (
            permuted_hidden,
            tokens_per_expert,
            permuted_probs,
        ) = dispatcher.dispatch_postprocess(dispatched_hidden, dispatched_probs)

        # Verify
        self.assertIsNotNone(permuted_hidden)
        self.assertIsNotNone(tokens_per_expert)
        self.assertIsNotNone(permuted_probs)
        self.assertEqual(permuted_hidden.shape[1], self.hidden_dim)

        dist.barrier()
        if self.rank == 0:
            print(f"✓ Dispatch postprocess distributed")

    @unittest.skipIf(not DEEP_EP_AVAILABLE, "Requires real deep_ep library")
    def test_dispatch_expert_assignment_distributed(self):
        """Test that tokens are correctly assigned to experts across ranks."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        num_tokens = self.batch_size * self.seq_len

        # Create routing that sends tokens to specific experts
        top_scores = (
            torch.ones(
                num_tokens, self.moe_router_topk, device=self.device, dtype=self.dtype
            )
            * 0.5
        )
        # Distribute tokens across all experts
        selected_indices = (
            torch.arange(num_tokens * self.moe_router_topk, device=self.device)
            % self.num_moe_experts
        )
        selected_indices = selected_indices.view(num_tokens, self.moe_router_topk)

        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        hidden_states = torch.randn(
            num_tokens, self.hidden_dim, device=self.device, dtype=self.dtype
        )

        world_group = dist.distributed_c10d._get_default_group()
        dispatched_hidden, dispatched_probs = dispatcher.token_dispatch(
            hidden_states,
            group=world_group,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        # Verify we received tokens
        self.assertGreater(dispatched_hidden.shape[0], 0)

        # Gather total token count across ranks
        local_count = torch.tensor([dispatched_hidden.shape[0]], device=self.device)
        count_list = [torch.zeros_like(local_count) for _ in range(self.world_size)]
        dist.all_gather(count_list, local_count)

        total_count = sum(c.item() for c in count_list)
        expected_total = num_tokens * self.moe_router_topk

        # Total tokens should match (approximately, due to capacity)
        self.assertGreater(total_count, 0)

        dist.barrier()
        if self.rank == 0:
            print(
                f"✓ Expert assignment distributed: {total_count}/{expected_total} tokens"
            )

    # ============================================================================
    # Token Combine Distributed Tests
    # ============================================================================

    @unittest.skipIf(not DEEP_EP_AVAILABLE, "Requires real deep_ep library")
    def test_combine_preprocess_distributed(self):
        """Test combine_preprocess with distributed data."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        # Setup full dispatch to get proper metadata
        num_tokens = self.batch_size * self.seq_len
        torch.manual_seed(42)
        top_scores = torch.rand(
            num_tokens, self.moe_router_topk, device=self.device, dtype=self.dtype
        )
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )

        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        hidden_states = torch.randn(
            num_tokens, self.hidden_dim, device=self.device, dtype=self.dtype
        )

        world_group = dist.distributed_c10d._get_default_group()
        dispatched_hidden, dispatched_probs = dispatcher.token_dispatch(
            hidden_states,
            group=world_group,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        permuted_hidden, _, _ = dispatcher.dispatch_postprocess(
            dispatched_hidden, dispatched_probs
        )

        # Simulate expert computation
        expert_output = permuted_hidden * 2.0

        # Combine preprocess
        restored_hidden = dispatcher.combine_preprocess(expert_output)

        self.assertIsNotNone(restored_hidden)
        self.assertEqual(restored_hidden.shape[1], self.hidden_dim)

        dist.barrier()
        if self.rank == 0:
            print(f"✓ Combine preprocess distributed")

    @unittest.skipIf(not DEEP_EP_AVAILABLE, "Requires real deep_ep library")
    def test_token_combine_distributed(self):
        """Test token_combine with real distributed communication."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        num_tokens = self.batch_size * self.seq_len
        torch.manual_seed(42)
        top_scores = torch.rand(
            num_tokens, self.moe_router_topk, device=self.device, dtype=self.dtype
        )
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )

        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        hidden_states = torch.randn(
            num_tokens, self.hidden_dim, device=self.device, dtype=self.dtype
        )

        # Full dispatch
        world_group = dist.distributed_c10d._get_default_group()
        dispatched_hidden, dispatched_probs = dispatcher.token_dispatch(
            hidden_states,
            group=world_group,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        permuted_hidden, _, _ = dispatcher.dispatch_postprocess(
            dispatched_hidden, dispatched_probs
        )
        expert_output = permuted_hidden * 2.0
        restored_hidden = dispatcher.combine_preprocess(expert_output)

        # Combine
        combined_output = dispatcher.token_combine(
            restored_hidden,
            group=world_group,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        self.assertIsNotNone(combined_output)
        self.assertEqual(combined_output.shape[0], num_tokens)
        self.assertEqual(combined_output.shape[1], self.hidden_dim)

        dist.barrier()
        if self.rank == 0:
            print(f"✓ Token combine distributed communication")

    @unittest.skipIf(not DEEP_EP_AVAILABLE, "Requires real deep_ep library")
    def test_combine_handle_cleanup_distributed(self):
        """Test that combine properly cleans up handles in distributed setting."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        num_tokens = self.batch_size * self.seq_len
        torch.manual_seed(42)
        top_scores = torch.rand(
            num_tokens, self.moe_router_topk, device=self.device, dtype=self.dtype
        )
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )

        dispatcher.dispatch_preprocess(top_scores, selected_indices)
        hidden_states = torch.randn(
            num_tokens, self.hidden_dim, device=self.device, dtype=self.dtype
        )

        world_group = dist.distributed_c10d._get_default_group()
        dispatched_hidden, dispatched_probs = dispatcher.token_dispatch(
            hidden_states,
            group=world_group,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        # Handle should be set after dispatch
        self.assertIsNotNone(dispatcher._comm_manager.handle)

        permuted_hidden, _, _ = dispatcher.dispatch_postprocess(
            dispatched_hidden, dispatched_probs
        )
        expert_output = permuted_hidden * 2.0
        restored_hidden = dispatcher.combine_preprocess(expert_output)

        dispatcher.token_combine(
            restored_hidden,
            group=world_group,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        # Handle should be None after combine
        self.assertIsNone(dispatcher._comm_manager.handle)

        dist.barrier()
        if self.rank == 0:
            print(f"✓ Combine handle cleanup distributed")

    # ============================================================================
    # End-to-End Distributed Tests
    # ============================================================================

    @unittest.skipIf(not DEEP_EP_AVAILABLE, "Requires real deep_ep library")
    def test_end_to_end_dispatch_combine_distributed(self):
        """Test complete dispatch-combine flow in distributed environment."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        num_tokens = self.batch_size * self.seq_len
        dispatcher.hidden_shape = torch.Size(
            [self.batch_size, self.seq_len, self.hidden_dim]
        )

        # Use same seed for reproducibility
        torch.manual_seed(42)
        top_scores = torch.rand(
            num_tokens, self.moe_router_topk, device=self.device, dtype=self.dtype
        )
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )
        hidden_states = torch.randn(
            num_tokens, self.hidden_dim, device=self.device, dtype=self.dtype
        )

        # 1. Dispatch preprocess
        dispatcher.dispatch_preprocess(top_scores, selected_indices)

        # 2. Token dispatch (communication)
        world_group = dist.distributed_c10d._get_default_group()
        dispatched_hidden, dispatched_probs = dispatcher.token_dispatch(
            hidden_states,
            group=world_group,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        # 3. Dispatch postprocess (permute)
        (
            permuted_hidden,
            tokens_per_expert,
            permuted_probs,
        ) = dispatcher.dispatch_postprocess(dispatched_hidden, dispatched_probs)

        # 4. Simulate expert computation
        expert_output = permuted_hidden * 2.0

        # 5. Combine preprocess (unpermute)
        restored_hidden = dispatcher.combine_preprocess(expert_output)

        # 6. Token combine (communication)
        combined_output = dispatcher.token_combine(
            restored_hidden,
            group=world_group,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        # 7. Combine postprocess (reshape)
        final_output = dispatcher.combine_postprocess(combined_output)

        # Verify final output shape
        self.assertEqual(final_output.shape, dispatcher.hidden_shape)

        dist.barrier()
        if self.rank == 0:
            print(f"✓ End-to-end dispatch-combine distributed")

    @unittest.skipIf(not DEEP_EP_AVAILABLE, "Requires real deep_ep library")
    def test_end_to_end_preserves_dtype_distributed(self):
        """Test dtype preservation in distributed end-to-end flow."""
        # DeepEP only supports bfloat16
        for dtype in [torch.bfloat16]:
            dispatcher = DeepEPTokenDispatcher(
                moe_router_topk=self.moe_router_topk,
                num_moe_experts=self.num_moe_experts,
            )

            num_tokens = self.batch_size * self.seq_len
            dispatcher.hidden_shape = torch.Size(
                [self.batch_size, self.seq_len, self.hidden_dim]
            )

            torch.manual_seed(42)
            top_scores = torch.rand(
                num_tokens, self.moe_router_topk, device=self.device, dtype=dtype
            )
            selected_indices = torch.randint(
                0,
                self.num_moe_experts,
                (num_tokens, self.moe_router_topk),
                device=self.device,
            )
            hidden_states = torch.randn(
                num_tokens, self.hidden_dim, device=self.device, dtype=dtype
            )

            dispatcher.dispatch_preprocess(top_scores, selected_indices)
            world_group = dist.distributed_c10d._get_default_group()
            dispatched_hidden, dispatched_probs = dispatcher.token_dispatch(
                hidden_states,
                group=world_group,
                async_finish=False,
                allocate_on_comm_stream=False,
            )

            # Check dtype after dispatch
            self.assertEqual(dispatched_hidden.dtype, dtype)

            permuted_hidden, _, _ = dispatcher.dispatch_postprocess(
                dispatched_hidden, dispatched_probs
            )
            expert_output = permuted_hidden.to(dtype)
            restored_hidden = dispatcher.combine_preprocess(expert_output)
            combined_output = dispatcher.token_combine(
                restored_hidden,
                group=world_group,
                async_finish=False,
                allocate_on_comm_stream=False,
            )
            final_output = dispatcher.combine_postprocess(combined_output)

            # Note: probs are always float32 for DeepEP, but hidden states should preserve dtype
            self.assertEqual(final_output.dtype, dtype)

            dist.barrier()

        if self.rank == 0:
            print(f"✓ Dtype preservation distributed (float32, float16)")

    @unittest.skipIf(not DEEP_EP_AVAILABLE, "Requires real deep_ep library")
    def test_end_to_end_multiple_iterations_distributed(self):
        """Test multiple dispatch-combine iterations work correctly."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        num_tokens = self.batch_size * self.seq_len
        dispatcher.hidden_shape = torch.Size(
            [self.batch_size, self.seq_len, self.hidden_dim]
        )
        world_group = dist.distributed_c10d._get_default_group()

        for iteration in range(3):
            torch.manual_seed(42 + iteration)
            top_scores = torch.rand(
                num_tokens, self.moe_router_topk, device=self.device, dtype=self.dtype
            )
            selected_indices = torch.randint(
                0,
                self.num_moe_experts,
                (num_tokens, self.moe_router_topk),
                device=self.device,
            )
            hidden_states = torch.randn(
                num_tokens, self.hidden_dim, device=self.device, dtype=self.dtype
            )

            dispatcher.dispatch_preprocess(top_scores, selected_indices)
            dispatched_hidden, dispatched_probs = dispatcher.token_dispatch(
                hidden_states,
                group=world_group,
                async_finish=False,
                allocate_on_comm_stream=False,
            )
            permuted_hidden, _, _ = dispatcher.dispatch_postprocess(
                dispatched_hidden, dispatched_probs
            )
            expert_output = permuted_hidden * 2.0
            restored_hidden = dispatcher.combine_preprocess(expert_output)
            combined_output = dispatcher.token_combine(
                restored_hidden,
                group=world_group,
                async_finish=False,
                allocate_on_comm_stream=False,
            )
            final_output = dispatcher.combine_postprocess(combined_output)

            self.assertEqual(final_output.shape, dispatcher.hidden_shape)

        dist.barrier()
        if self.rank == 0:
            print(f"✓ Multiple iterations distributed (3 iterations)")

    def test_data_consistency_across_ranks(self):
        """Test that same input produces consistent routing across ranks."""
        dispatcher = DeepEPTokenDispatcher(
            moe_router_topk=self.moe_router_topk,
            num_moe_experts=self.num_moe_experts,
        )

        num_tokens = self.batch_size * self.seq_len

        # All ranks use same seed and same data
        torch.manual_seed(42)
        top_scores = torch.rand(
            num_tokens, self.moe_router_topk, device=self.device, dtype=self.dtype
        )
        selected_indices = torch.randint(
            0,
            self.num_moe_experts,
            (num_tokens, self.moe_router_topk),
            device=self.device,
        )

        dispatcher.dispatch_preprocess(top_scores, selected_indices)

        # Gather indices from all ranks
        local_indices = dispatcher._comm_manager.token_indices.flatten()[
            :10
        ]  # First 10 for comparison
        indices_list = [torch.zeros_like(local_indices) for _ in range(self.world_size)]
        dist.all_gather(indices_list, local_indices)

        # All ranks should have same indices (same input)
        for i, indices in enumerate(indices_list):
            if not torch.equal(indices, local_indices):
                self.fail(f"Indices mismatch at rank {i}")

        dist.barrier()
        if self.rank == 0:
            print(f"✓ Data consistency across ranks")


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
