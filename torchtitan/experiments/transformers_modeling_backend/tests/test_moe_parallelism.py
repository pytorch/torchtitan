# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Titan MoE replacement in the HF Transformers modeling backend.

These tests verify that the two-phase MoE replacement (probe → build → swap)
produces correct Titan MoE modules and that they work under EP, TP, and EP+TP.

Run with:
    python -m pytest torchtitan/experiments/transformers_modeling_backend/tests/test_moe_parallelism.py -x -v
"""

import unittest

import torch

from torchtitan.components.optimizer import (
    default_adamw,
    register_moe_load_balancing_hook,
)
from torchtitan.models.common.moe import MoE


def _expert_weights(experts):
    """Return (w1, w2, w3) expert params by their dynamically-discovered names.

    GroupedExperts param names carry dimension suffixes (e.g. ``w1_EFD``), so
    resolve the canonical (gate, down, up) roles via the same helper the
    production state-dict adapter uses instead of hardcoding ``w1``/``w2``/``w3``.
    """
    from torchtitan.experiments.transformers_modeling_backend.state_dict_adapter import (
        _expert_names,
    )

    gate_name, down_name, up_name = _expert_names()
    return (
        getattr(experts, gate_name),
        getattr(experts, down_name),
        getattr(experts, up_name),
    )


def _moe_buffer(moe, prefix):
    """Return an MoE buffer by canonical role, tolerating dimension suffixes.

    MoE load-balancing buffers carry a dimension suffix (e.g.
    ``tokens_per_expert_E``, ``expert_bias_E``), so match by prefix instead of
    hardcoding the exact name.
    """
    for name, buf in moe.named_buffers(recurse=False):
        if name == prefix or name.startswith(prefix + "_"):
            return buf
    raise AttributeError(f"{type(moe).__name__} has no buffer matching '{prefix}*'")


try:
    from torch.testing._internal.distributed._tensor.common_dtensor import (
        DTensorTestBase,
        with_comms,
    )

    _HAS_DTENSOR_TEST = True
except ImportError:
    DTensorTestBase = unittest.TestCase
    _HAS_DTENSOR_TEST = False

    def with_comms(fn):
        return fn


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


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
    return model, config


def _create_tiny_deepseek_v3_model(
    num_experts=4,
    num_experts_per_tok=2,
    hidden_size=64,
    moe_intermediate_size=32,
    intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=4,
    vocab_size=256,
    max_position_embeddings=64,
):
    """Create a minimal DeepSeek-V3 MoE model for testing."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "deepseek_v3",
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        first_k_dense_replace=0,
        n_routed_experts=num_experts,
        num_local_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        n_group=2,
        topk_group=1,
        n_shared_experts=1,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        attn_implementation="sdpa",
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_config(config)
    return model, config


def _prepare_layers(model):
    """Convert layers to SliceableModuleDict and set moe_enabled flags."""
    from torchtitan.experiments.transformers_modeling_backend.model import (
        SliceableModuleDict,
    )

    model.model.layers = SliceableModuleDict(
        {str(i): layer for i, layer in enumerate(model.model.layers)}
    )
    for layer in model.model.layers.values():
        has_gate = hasattr(layer.mlp, "gate") or hasattr(layer.mlp, "router")
        layer.moe_enabled = has_gate and hasattr(layer.mlp, "experts")


class _FakeParallelDims:
    """Minimal ParallelDims stub for tests that don't use full distributed setup."""

    full_dtensor = False
    spmd_backend = "default"
    tp_enabled = False
    ep_enabled = False
    tp = 1
    ep = 1

    def __init__(self, meshes=None, tp_enabled=False, ep_enabled=False):
        self._meshes = meshes or {}
        self.tp_enabled = tp_enabled
        self.ep_enabled = ep_enabled

    def get_optional_mesh(self, _mesh_axis_names):
        if isinstance(_mesh_axis_names, str):
            return self._meshes.get(_mesh_axis_names)
        for name in _mesh_axis_names:
            if name in self._meshes:
                return self._meshes[name]
        return None

    def get_mesh(self, mesh_axis_names):
        mesh = self.get_optional_mesh(mesh_axis_names)
        if mesh is None:
            raise KeyError(f"No mesh for {mesh_axis_names}")
        return mesh

    def resolve_mesh(self, axes):
        for axis in axes:
            key = axis.value if hasattr(axis, "value") else axis
            if key in self._meshes:
                return self._meshes[key]
        return None

    @property
    def seq_len_divisor(self):
        return self.tp * 2 if self.tp > 1 else 1


# ---------------------------------------------------------------------------
# Phase 1 tests: config probing
# ---------------------------------------------------------------------------


class TestPrepareNativeMoeConfigs(unittest.TestCase):
    """Test Phase 1: probing HF MoE blocks and building MoE.Config."""

    def test_qwen3_moe_config_probing(self):
        """Qwen3 MoE block is correctly probed into a MoE.Config."""
        model, config = _create_tiny_qwen3moe_model(num_hidden_layers=1)
        _prepare_layers(model)

        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            _probe_hf_moe_block,
        )

        layer = model.model.layers["0"]
        params = _probe_hf_moe_block(layer.mlp, config)

        self.assertEqual(params["num_experts"], 4)
        self.assertEqual(params["top_k"], 2)
        self.assertEqual(params["dim"], 64)
        self.assertEqual(params["moe_intermediate_size"], 32)
        self.assertEqual(params["score_func"], "softmax")
        self.assertIsNone(params["shared_expert_info"])

    def test_deepseek_v3_config_probing(self):
        """DeepSeek V3 MoE block is correctly probed (sigmoid, shared experts, groups)."""
        model, config = _create_tiny_deepseek_v3_model()
        _prepare_layers(model)

        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            _probe_hf_moe_block,
        )

        layer = model.model.layers["0"]
        params = _probe_hf_moe_block(layer.mlp, config)

        self.assertEqual(params["num_experts"], 4)
        self.assertEqual(params["score_func"], "sigmoid")
        self.assertEqual(params["num_expert_groups"], 2)
        self.assertEqual(params["num_limited_groups"], 1)
        self.assertIsNotNone(params["shared_expert_info"])
        self.assertFalse(params["shared_expert_info"]["has_sigmoid_gate"])

    def test_moe_config_build(self):
        """MoE.Config is built correctly from probed params."""
        model, config = _create_tiny_qwen3moe_model(num_hidden_layers=1)
        _prepare_layers(model)

        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            _build_moe_config,
            _probe_hf_moe_block,
        )

        layer = model.model.layers["0"]
        params = _probe_hf_moe_block(layer.mlp, config)
        moe_config = _build_moe_config(params, config)

        self.assertIsInstance(moe_config, MoE.Config)
        self.assertEqual(moe_config.num_experts, 4)
        self.assertIsNone(moe_config.shared_experts)

    def test_prepare_stores_config_on_layers(self):
        """prepare_native_moe_configs stores _native_moe_config on MoE layers."""
        model, config = _create_tiny_qwen3moe_model(
            num_hidden_layers=2, decoder_sparse_step=1
        )
        _prepare_layers(model)

        # Need config attributes for prepare
        config.load_balance_coeff = 1e-3
        config.comm_backend = "standard"

        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            prepare_native_moe_configs,
        )

        # Mock the model interface expected by prepare_native_moe_configs
        class _ModelProxy:
            layers = model.model.layers

        prepare_native_moe_configs(_ModelProxy(), config)

        for layer in model.model.layers.values():
            if layer.moe_enabled:
                self.assertTrue(hasattr(layer, "_native_moe_config"))
                self.assertIsInstance(layer._native_moe_config, MoE.Config)


# ---------------------------------------------------------------------------
# Phase 2 tests: build and swap (single GPU)
# ---------------------------------------------------------------------------


class TestNativeMoeBuildAndSwap(unittest.TestCase):
    """Test building and swapping Titan MoE modules (single device, no parallelism)."""

    def test_build_produces_native_moe(self):
        """Building from MoE.Config produces a Titan MoE with correct shapes."""
        model, config = _create_tiny_qwen3moe_model(num_hidden_layers=1)
        _prepare_layers(model)

        config.load_balance_coeff = 1e-3
        config.comm_backend = "standard"

        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            _build_moe_config,
            _probe_hf_moe_block,
        )

        layer = model.model.layers["0"]
        params = _probe_hf_moe_block(layer.mlp, config)
        moe_config = _build_moe_config(params, config)

        with torch.device("meta"):
            native_moe = moe_config.build()

        self.assertIsInstance(native_moe, MoE)
        w1, w2, w3 = _expert_weights(native_moe.experts)
        self.assertEqual(w1.shape, (4, 32, 64))
        self.assertEqual(w2.shape, (4, 64, 32))
        self.assertEqual(w3.shape, (4, 32, 64))
        self.assertEqual(native_moe.router.gate.weight.shape, (4, 64))

    def test_init_states_materializes_params(self):
        """init_states materializes parameters from meta device."""
        model, config = _create_tiny_qwen3moe_model(num_hidden_layers=1)
        _prepare_layers(model)

        config.load_balance_coeff = 1e-3
        config.comm_backend = "standard"

        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            _build_moe_config,
            _probe_hf_moe_block,
        )

        layer = model.model.layers["0"]
        params = _probe_hf_moe_block(layer.mlp, config)
        moe_config = _build_moe_config(params, config)

        with torch.device("meta"):
            native_moe = moe_config.build()

        self.assertTrue(_expert_weights(native_moe.experts)[0].device.type == "meta")

        native_moe.to_empty(device=torch.device("cpu"))
        native_moe.init_states(buffer_device=torch.device("cpu"))

        self.assertTrue(_expert_weights(native_moe.experts)[0].device.type == "cpu")
        self.assertTrue(native_moe.router.gate.weight.device.type == "cpu")
        self.assertTrue(
            _moe_buffer(native_moe, "tokens_per_expert").device.type == "cpu"
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for MoE forward")
    def test_native_moe_forward(self):
        """Titan MoE forward produces correct output shape."""
        model, config = _create_tiny_qwen3moe_model(num_hidden_layers=1)
        _prepare_layers(model)

        config.load_balance_coeff = 1e-3
        config.comm_backend = "standard"

        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            _build_moe_config,
            _probe_hf_moe_block,
        )

        layer = model.model.layers["0"]
        params = _probe_hf_moe_block(layer.mlp, config)
        moe_config = _build_moe_config(params, config)

        with torch.device("meta"):
            native_moe = moe_config.build()
        native_moe.to_empty(device=torch.device("cuda"))
        native_moe.init_states(buffer_device=torch.device("cuda"))

        x = torch.randn(2, 16, 64, device="cuda")
        output = native_moe(x)

        self.assertEqual(output.shape, (2, 16, 64))
        self.assertFalse(torch.isnan(output).any())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for MoE backward")
    def test_native_moe_backward(self):
        """Titan MoE backward pass produces gradients."""
        model, config = _create_tiny_qwen3moe_model(num_hidden_layers=1)
        _prepare_layers(model)

        config.load_balance_coeff = 1e-3
        config.comm_backend = "standard"

        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            _build_moe_config,
            _probe_hf_moe_block,
        )

        layer = model.model.layers["0"]
        params = _probe_hf_moe_block(layer.mlp, config)
        moe_config = _build_moe_config(params, config)

        with torch.device("meta"):
            native_moe = moe_config.build()
        native_moe.to_empty(device=torch.device("cuda"))
        native_moe.init_states(buffer_device=torch.device("cuda"))

        x = torch.randn(2, 16, 64, device="cuda", requires_grad=True)
        output = native_moe(x)
        output.sum().backward()

        self.assertIsNotNone(x.grad)
        w1, w2, w3 = _expert_weights(native_moe.experts)
        self.assertIsNotNone(w1.grad)
        self.assertIsNotNone(w2.grad)
        self.assertIsNotNone(w3.grad)


# ---------------------------------------------------------------------------
# Load balancing contract
# ---------------------------------------------------------------------------


class TestNativeMoeLoadBalancing(unittest.TestCase):
    """Test that the Titan MoE exposes the load-balancing contract."""

    def test_native_moe_exposes_load_balance_attrs(self):
        """Titan MoE has tokens_per_expert, expert_bias, load_balance_coeff."""
        model, config = _create_tiny_qwen3moe_model(num_hidden_layers=1)
        _prepare_layers(model)

        config.load_balance_coeff = 1e-3
        config.comm_backend = "standard"

        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            _build_moe_config,
            _probe_hf_moe_block,
        )

        layer = model.model.layers["0"]
        params = _probe_hf_moe_block(layer.mlp, config)
        moe_config = _build_moe_config(params, config)

        with torch.device("meta"):
            native_moe = moe_config.build()
        native_moe.to_empty(device=torch.device("cpu"))
        native_moe.init_states(buffer_device=torch.device("cpu"))

        self.assertEqual(native_moe.load_balance_coeff, 1e-3)
        self.assertEqual(_moe_buffer(native_moe, "tokens_per_expert").shape, (4,))
        self.assertEqual(_moe_buffer(native_moe, "expert_bias").shape, (4,))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for MoE forward")
    def test_forward_accumulates_tokens_per_expert(self):
        """Forward pass accumulates per-expert token counts."""
        model, config = _create_tiny_qwen3moe_model(num_hidden_layers=1)
        _prepare_layers(model)

        config.load_balance_coeff = 1e-3
        config.comm_backend = "standard"

        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            _build_moe_config,
            _probe_hf_moe_block,
        )

        layer = model.model.layers["0"]
        params = _probe_hf_moe_block(layer.mlp, config)
        moe_config = _build_moe_config(params, config)

        with torch.device("meta"):
            native_moe = moe_config.build()
        native_moe.to_empty(device=torch.device("cuda"))
        native_moe.init_states(buffer_device=torch.device("cuda"))

        x = torch.randn(2, 8, 64, device="cuda")
        native_moe(x)

        # 2*8 tokens, top_k=2 → 32 total expert assignments
        self.assertEqual(
            _moe_buffer(native_moe, "tokens_per_expert").sum().item(), 2 * 8 * 2
        )

    def test_optimizer_hook_updates_expert_bias(self):
        """register_moe_load_balancing_hook works with the Titan MoE."""
        model, config = _create_tiny_qwen3moe_model(num_hidden_layers=1)
        _prepare_layers(model)

        config.load_balance_coeff = 1e-3
        config.comm_backend = "standard"

        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            _build_moe_config,
            _probe_hf_moe_block,
        )

        layer = model.model.layers["0"]
        params = _probe_hf_moe_block(layer.mlp, config)
        moe_config = _build_moe_config(params, config)

        with torch.device("meta"):
            native_moe = moe_config.build()
        native_moe.to_empty(device=torch.device("cpu"))
        native_moe.init_states(buffer_device=torch.device("cpu"))

        # Simulate the swap
        layer.mlp = native_moe
        object.__setattr__(layer, "moe", native_moe)

        # Set unbalanced token counts
        with torch.no_grad():
            _moe_buffer(native_moe, "tokens_per_expert").copy_(
                torch.tensor([10.0, 0.0, 0.0, 0.0])
            )

        # Build optimizer and register hook
        opt_config = default_adamw(lr=1e-3)
        opt_config.implementation = "for-loop"
        optimizers = opt_config.build(model_parts=[model.model])
        register_moe_load_balancing_hook(
            optimizers,
            [model.model],
            _FakeParallelDims(),
        )

        optimizers.step()

        # Counts should be reset
        self.assertTrue(
            torch.equal(_moe_buffer(native_moe, "tokens_per_expert"), torch.zeros(4))
        )
        # Bias should be updated (non-zero for the imbalanced expert)
        self.assertFalse(
            torch.equal(_moe_buffer(native_moe, "expert_bias"), torch.zeros(4))
        )


# ---------------------------------------------------------------------------
# MoE detection
# ---------------------------------------------------------------------------


class TestMoeDetection(unittest.TestCase):
    """Test that MoE layers are correctly detected."""

    def test_moe_enabled_all_moe(self):
        model, _ = _create_tiny_qwen3moe_model(
            num_hidden_layers=4, decoder_sparse_step=1
        )
        for layer in model.model.layers:
            has_gate = hasattr(layer.mlp, "gate")
            has_experts = hasattr(layer.mlp, "experts")
            self.assertTrue(has_gate and has_experts)

    def test_moe_enabled_mixed(self):
        model, _ = _create_tiny_qwen3moe_model(
            num_hidden_layers=4, decoder_sparse_step=2
        )
        for i, layer in enumerate(model.model.layers):
            has_gate = hasattr(layer.mlp, "gate")
            has_experts = hasattr(layer.mlp, "experts")
            expected_moe = (i + 1) % 2 == 0
            self.assertEqual(
                has_gate and has_experts,
                expected_moe,
                f"Layer {i}: expected moe_enabled={expected_moe}",
            )


if __name__ == "__main__":
    unittest.main()
