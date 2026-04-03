# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for MoE parallelism in the HF Transformers modeling backend.

These tests verify that Expert Parallelism (EP), Tensor Parallelism (TP),
and their combination work correctly with HF MoE models.

Run with:
    python -m pytest torchtitan/experiments/transformers_modeling_backend/tests/test_moe_parallelism.py -x -v
"""

import unittest

import torch

from torchtitan.experiments.transformers_modeling_backend.parallelize import (
    apply_moe_ep_tp,
)

try:
    from torch.testing._internal.distributed._tensor.common_dtensor import (
        DTensorTestBase,
        with_comms,
    )

    _HAS_DTENSOR_TEST = True
except ImportError:
    DTensorTestBase = unittest.TestCase  # fallback base class
    _HAS_DTENSOR_TEST = False

    def with_comms(fn):
        return fn


def _create_tiny_qwen3moe_model(
    num_experts=4,
    num_experts_per_tok=2,
    hidden_size=64,
    moe_intermediate_size=32,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    decoder_sparse_step=1,
    vocab_size=256,
    max_position_embeddings=64,
):
    """Create a minimal Qwen3-MoE model for testing."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "qwen3_moe",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        num_local_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        decoder_sparse_step=decoder_sparse_step,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        attn_implementation="sdpa",
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model


class TestMoeBlockForward(unittest.TestCase):
    """Test MoE block forwards on a real HF MoE model (single GPU)."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")

    def test_hf_moe_block_forward_plain_tensor(self):
        """Test original HF MoE block forward with plain tensor input."""
        model = _create_tiny_qwen3moe_model(
            num_experts=4, num_experts_per_tok=2, num_hidden_layers=1
        ).to(self.device)

        layer = model.model.layers[0]
        moe_block = layer.mlp

        x = torch.randn(2, 16, 64, device=self.device)
        output = moe_block(x)

        self.assertEqual(output.shape, (2, 16, 64))
        self.assertFalse(torch.isnan(output).any())

    def test_hf_moe_block_backward(self):
        """Test backward pass through original HF MoE block forward."""
        model = _create_tiny_qwen3moe_model(
            num_experts=4, num_experts_per_tok=2, num_hidden_layers=1
        ).to(self.device)

        layer = model.model.layers[0]
        moe_block = layer.mlp

        x = torch.randn(2, 16, 64, device=self.device, requires_grad=True)
        output = moe_block(x)
        output.sum().backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        # Expert weights should have gradients
        self.assertIsNotNone(moe_block.experts.gate_up_proj.grad)
        self.assertIsNotNone(moe_block.experts.down_proj.grad)


class TestMoeDetection(unittest.TestCase):
    """Test that MoE layers are correctly detected."""

    def test_moe_enabled_all_moe(self):
        """All layers should be MoE with decoder_sparse_step=1."""
        model = _create_tiny_qwen3moe_model(num_hidden_layers=4, decoder_sparse_step=1)
        for layer in model.model.layers:
            has_gate = hasattr(layer.mlp, "gate")
            has_experts = hasattr(layer.mlp, "experts")
            self.assertTrue(has_gate and has_experts)

    def test_moe_enabled_mixed(self):
        """With decoder_sparse_step=2, alternating MoE and dense layers."""
        model = _create_tiny_qwen3moe_model(num_hidden_layers=4, decoder_sparse_step=2)
        for i, layer in enumerate(model.model.layers):
            has_gate = hasattr(layer.mlp, "gate")
            has_experts = hasattr(layer.mlp, "experts")
            expected_moe = (i + 1) % 2 == 0
            self.assertEqual(
                has_gate and has_experts,
                expected_moe,
                f"Layer {i}: expected moe_enabled={expected_moe}",
            )


@unittest.skipUnless(_HAS_DTENSOR_TEST, "DTensor test utilities not available")
class TestEPMoeForwardBackward(DTensorTestBase):
    """Test EP MoE with real distributed setup."""

    @property
    def world_size(self):
        return 4

    @with_comms
    def test_ep_only_forward_backward(self):
        """Test EP-only MoE forward/backward (no TP)."""
        model = _create_tiny_qwen3moe_model(
            num_experts=4,
            num_experts_per_tok=2,
            num_hidden_layers=1,
            hidden_size=64,
            moe_intermediate_size=32,
        ).cuda()

        # Set moe_enabled on layers (matching model.py behavior)
        from torchtitan.experiments.transformers_modeling_backend.model import (
            SliceableModuleDict,
        )

        model.model.layers = SliceableModuleDict(
            {str(i): layer for i, layer in enumerate(model.model.layers)}
        )
        for layer in model.model.layers.values():
            layer.moe_enabled = hasattr(layer.mlp, "gate") and hasattr(
                layer.mlp, "experts"
            )

        # Create EP mesh (4 ranks, ep=4)
        ep_mesh = self.build_device_mesh()

        from torchtitan.experiments.transformers_modeling_backend.parallelize import (
            _experts_restore_post_hook,
            _experts_to_local_pre_hook,
        )

        # Apply EP
        apply_moe_ep_tp(model.model, ep_mesh=ep_mesh)

        # Register experts to_local hooks (normally done in
        # parallelize_hf_transformers after apply_fsdp)
        for layer in model.model.layers.values():
            if layer.moe_enabled:
                layer.mlp.experts.register_forward_pre_hook(_experts_to_local_pre_hook)
                layer.mlp.experts.register_forward_hook(
                    _experts_restore_post_hook, prepend=True
                )

        # Forward
        torch.manual_seed(42)
        x = torch.randn(2, 16, 64, device="cuda")
        layer = list(model.model.layers.values())[0]
        output = layer.mlp(x)

        self.assertEqual(output.shape, (2, 16, 64))
        self.assertFalse(torch.isnan(output).any())

        # Backward
        output.sum().backward()
        experts = layer.mlp.experts
        # Expert params are plain tensors (manually sliced, no FSDP in this test)
        self.assertIsNotNone(experts.gate_up_proj.grad)
        self.assertIsNotNone(experts.down_proj.grad)

    @with_comms
    def test_ep_tp_forward_backward(self):
        """Test EP+TP MoE forward/backward with 2D mesh."""
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard

        model = _create_tiny_qwen3moe_model(
            num_experts=4,
            num_experts_per_tok=2,
            num_hidden_layers=1,
            hidden_size=64,
            moe_intermediate_size=32,
        ).cuda()

        from torchtitan.experiments.transformers_modeling_backend.model import (
            SliceableModuleDict,
        )

        model.model.layers = SliceableModuleDict(
            {str(i): layer for i, layer in enumerate(model.model.layers)}
        )
        for layer in model.model.layers.values():
            layer.moe_enabled = hasattr(layer.mlp, "gate") and hasattr(
                layer.mlp, "experts"
            )

        # Build 2D mesh: ep=2, tp=2
        mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=("ep", "tp"))
        ep_mesh = mesh_2d["ep"]
        tp_mesh = mesh_2d["tp"]

        from torchtitan.experiments.transformers_modeling_backend.parallelize import (
            _experts_restore_post_hook,
            _experts_to_local_pre_hook,
        )

        # Apply EP+TP (shards params, registers all hooks)
        apply_moe_ep_tp(model.model, tp_mesh=tp_mesh, ep_mesh=ep_mesh)

        # Register experts to_local hooks (normally done in
        # parallelize_hf_transformers after apply_fsdp)
        for layer_val in model.model.layers.values():
            if layer_val.moe_enabled:
                layer_val.mlp.experts.register_forward_pre_hook(
                    _experts_to_local_pre_hook
                )
                layer_val.mlp.experts.register_forward_hook(
                    _experts_restore_post_hook, prepend=True
                )

        # Create Shard(1) DTensor input (simulating SP output)
        torch.manual_seed(42)
        x_local = torch.randn(2, 8, 64, device="cuda")  # slen/tp = 16/2 = 8
        x_dt = DTensor.from_local(x_local, tp_mesh, [Shard(1)])

        layer = list(model.model.layers.values())[0]
        output = layer.mlp(x_dt)

        # Output should be plain tensor (to_local inside EP dispatch pre-hook)
        self.assertIsInstance(output, torch.Tensor)
        self.assertNotIsInstance(output, DTensor)
        # Local shape: each TP rank has slen/tp tokens
        self.assertEqual(output.shape, (2, 8, 64))
        self.assertFalse(torch.isnan(output).any())

        # Backward
        output.sum().backward()
        experts = layer.mlp.experts
        # Expert params are TP-sharded DTensors (via distribute_tensor in apply_moe_ep_tp)
        # but with hook-based EP, they may be plain tensors after manual slicing
        self.assertIsNotNone(experts.gate_up_proj.grad)
        self.assertIsNotNone(experts.down_proj.grad)


@unittest.skipUnless(_HAS_DTENSOR_TEST, "DTensor test utilities not available")
class TestTPOnlyMoeForwardBackward(DTensorTestBase):
    """Test TP-only MoE (no EP) with real distributed setup."""

    @property
    def world_size(self):
        return 4

    @with_comms
    def test_tp_only_forward_backward(self):
        """Test TP-only MoE forward/backward."""
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard

        model = _create_tiny_qwen3moe_model(
            num_experts=4,
            num_experts_per_tok=2,
            num_hidden_layers=1,
            hidden_size=64,
            moe_intermediate_size=32,
        ).cuda()

        from torchtitan.experiments.transformers_modeling_backend.model import (
            SliceableModuleDict,
        )

        model.model.layers = SliceableModuleDict(
            {str(i): layer for i, layer in enumerate(model.model.layers)}
        )
        for layer in model.model.layers.values():
            layer.moe_enabled = hasattr(layer.mlp, "gate") and hasattr(
                layer.mlp, "experts"
            )

        # Build 1D TP mesh
        tp_mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("tp",))

        from torchtitan.experiments.transformers_modeling_backend.parallelize import (
            _experts_restore_post_hook,
            _experts_to_local_pre_hook,
        )

        # Apply TP-only MoE (shards expert weights, registers hooks)
        apply_moe_ep_tp(model.model, tp_mesh=tp_mesh)

        # Register experts to_local hooks (normally done in
        # parallelize_hf_transformers after apply_fsdp)
        for layer_val in model.model.layers.values():
            if layer_val.moe_enabled:
                layer_val.mlp.experts.register_forward_pre_hook(
                    _experts_to_local_pre_hook
                )
                layer_val.mlp.experts.register_forward_hook(
                    _experts_restore_post_hook, prepend=True
                )

        # Forward with Shard(1) DTensor input (simulating SP output).
        # PrepareModuleInputOutput all-gathers to Replicate, then
        # _moe_to_local_pre_hook converts to local for the MoE forward.
        # Output is reduce-scattered back to Shard(1) by PrepareModuleInputOutput.
        torch.manual_seed(42)
        local_seq = 16 // self.world_size  # seq_len / tp
        x_local = torch.randn(2, local_seq, 64, device="cuda")
        x_dt = DTensor.from_local(x_local, tp_mesh, [Shard(1)])

        layer = list(model.model.layers.values())[0]
        output = layer.mlp(x_dt)

        # Output shape should match input local shape
        self.assertEqual(output.shape, (2, local_seq, 64))
        self.assertFalse(torch.isnan(output).any())

        # Backward
        output.sum().backward()
        experts = layer.mlp.experts
        # Expert params should be DTensors (TP-sharded)
        self.assertIsInstance(experts.gate_up_proj, DTensor)
        self.assertIsInstance(experts.down_proj, DTensor)
        # Check sharding placements
        self.assertEqual(experts.gate_up_proj.placements, (Shard(1),))
        self.assertEqual(experts.down_proj.placements, (Shard(2),))
        # Gradients should exist
        self.assertIsNotNone(experts.gate_up_proj.grad)
        self.assertIsNotNone(experts.down_proj.grad)


@unittest.skipUnless(_HAS_DTENSOR_TEST, "DTensor test utilities not available")
class TestMixedMoeDenseLayers(DTensorTestBase):
    """Test model with interleaved MoE and dense layers under EP."""

    @property
    def world_size(self):
        return 4

    @with_comms
    def test_mixed_layers_ep(self):
        """EP on MoE layers, dense layers untouched."""
        model = _create_tiny_qwen3moe_model(
            num_experts=4,
            num_experts_per_tok=2,
            num_hidden_layers=4,
            hidden_size=64,
            moe_intermediate_size=32,
            decoder_sparse_step=2,  # layers 1,3 are MoE; layers 0,2 are dense
        ).cuda()

        from torchtitan.experiments.transformers_modeling_backend.model import (
            SliceableModuleDict,
        )

        model.model.layers = SliceableModuleDict(
            {str(i): layer for i, layer in enumerate(model.model.layers)}
        )
        for layer in model.model.layers.values():
            layer.moe_enabled = hasattr(layer.mlp, "gate") and hasattr(
                layer.mlp, "experts"
            )

        from torchtitan.experiments.transformers_modeling_backend.parallelize import (
            _experts_restore_post_hook,
            _experts_to_local_pre_hook,
        )

        # Apply EP (only affects MoE layers)
        ep_mesh = self.build_device_mesh()
        apply_moe_ep_tp(model.model, ep_mesh=ep_mesh)

        # Register experts to_local hooks (normally done in
        # parallelize_hf_transformers after apply_fsdp)
        for layer_val in model.model.layers.values():
            if layer_val.moe_enabled:
                layer_val.mlp.experts.register_forward_pre_hook(
                    _experts_to_local_pre_hook
                )
                layer_val.mlp.experts.register_forward_hook(
                    _experts_restore_post_hook, prepend=True
                )

        # Forward through all layers sequentially
        torch.manual_seed(42)
        hidden = torch.randn(2, 16, 64, device="cuda")
        for layer in model.model.layers.values():
            # Simplified forward: layernorm → attention-free → mlp
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        self.assertEqual(hidden.shape, (2, 16, 64))
        self.assertFalse(torch.isnan(hidden).any())

        # Backward
        hidden.sum().backward()


if __name__ == "__main__":
    unittest.main()
