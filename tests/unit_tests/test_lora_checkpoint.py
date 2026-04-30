# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for LoRA checkpoint save/load and PEFT format round-trip.

Exercises:
1. Trainable vs frozen parameter filtering with LoRA
2. LoRAStateDictAdapter round-trips native ↔ PEFT key naming
3. Save adapter in PEFT safetensors format, reload via 2-phase
4. Verify loaded model matches original
"""

import os
import tempfile
import unittest

import torch

from torchtitan.components.lora import (
    remap_lora_keys_from_hf,
    remap_lora_keys_to_hf,
)
from torchtitan.models.llama3.config_registry import llama3_debugmodel_lora


def _build_lora_model():
    """Build a debug Llama3 model with LoRA adapters on CPU."""
    config = llama3_debugmodel_lora()
    model_config = config.model_spec.model

    with torch.device("cpu"):
        model = model_config.build()

    with torch.no_grad():
        model.init_weights()

    # Freezing is automatic — LoRAConverter wraps non-target configs
    # with FrozenConfig, and LoRALinear.__init__ freezes its base weight.

    return model, model_config, config


def _get_trainable_sd(model):
    """Get state dict of only trainable (requires_grad) parameters."""
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _get_frozen_sd(model):
    """Get state dict of only frozen (not requires_grad) parameters."""
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not param.requires_grad
    }


def _get_full_sd(model):
    """Get state dict of all parameters."""
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }


class TestLoRAStateDictFiltering(unittest.TestCase):
    """Test trainable vs frozen parameter filtering with LoRA."""

    def setUp(self):
        self.model, self.model_config, self.config = _build_lora_model()

    def test_trainable_returns_only_lora_keys(self):
        sd = _get_trainable_sd(self.model)
        self.assertTrue(len(sd) > 0, "No trainable params found")
        for key in sd:
            self.assertIn(
                "lora_", key, f"Trainable sd should only have lora keys, got: {key}"
            )

    def test_frozen_returns_only_base_keys(self):
        sd = _get_frozen_sd(self.model)
        self.assertTrue(len(sd) > 0, "No frozen params found")
        for key in sd:
            self.assertNotIn(
                "lora_", key, f"Frozen sd should not have lora keys, got: {key}"
            )

    def test_trainable_plus_frozen_equals_full(self):
        trainable = _get_trainable_sd(self.model)
        frozen = _get_frozen_sd(self.model)
        full = _get_full_sd(self.model)
        self.assertEqual(
            set(trainable.keys()) | set(frozen.keys()),
            set(full.keys()),
        )
        self.assertEqual(
            set(trainable.keys()) & set(frozen.keys()),
            set(),
            "Trainable and frozen should be disjoint",
        )

    def test_lora_target_modules_applied_correctly(self):
        trainable = _get_trainable_sd(self.model)
        # Config targets wq, wkv, wo — verify those are present
        for key in trainable:
            parts = key.split(".")
            # Find the linear module name before lora_a/lora_b
            for i, part in enumerate(parts):
                if part in ("lora_a", "lora_b"):
                    parent = parts[i - 1]
                    self.assertIn(
                        parent,
                        {"wq", "wk", "wv", "wo"},
                        f"LoRA applied to unexpected module: {parent} in {key}",
                    )
                    break


class TestLoRAKeyRemapping(unittest.TestCase):
    """Test LoRAStateDictAdapter key name round-trip."""

    def setUp(self):
        from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

        _, model_config, config = _build_lora_model()
        sd_adapter = Llama3StateDictAdapter(model_config, config.hf_assets_path)
        self.from_hf = sd_adapter.from_hf_map

    def test_native_to_hf_key_format(self):
        native_sd = {
            "layers.0.attention.qkv_linear.wq.lora_a.weight": torch.randn(8, 256),
            "layers.0.attention.qkv_linear.wq.lora_b.weight": torch.randn(256, 8),
            "layers.0.attention.wo.lora_a.weight": torch.randn(8, 256),
            "layers.0.attention.wo.lora_b.weight": torch.randn(256, 8),
        }
        hf_sd = remap_lora_keys_to_hf(native_sd, self.from_hf)

        for key in hf_sd:
            self.assertTrue(
                key.startswith("base_model.model."),
                f"HF key should start with 'base_model.model.', got: {key}",
            )
            self.assertFalse(
                ".lora_a." in key or ".lora_b." in key,
                f"HF key should use lora_A/lora_B, got: {key}",
            )

    def test_round_trip_keys(self):
        native_sd = {
            "layers.0.attention.qkv_linear.wq.lora_a.weight": torch.randn(8, 256),
            "layers.0.attention.qkv_linear.wq.lora_b.weight": torch.randn(256, 8),
            "layers.1.attention.qkv_linear.wv.lora_a.weight": torch.randn(8, 256),
            "layers.1.attention.qkv_linear.wv.lora_b.weight": torch.randn(256, 8),
            "layers.0.attention.wo.lora_a.weight": torch.randn(8, 256),
            "layers.0.attention.wo.lora_b.weight": torch.randn(256, 8),
        }
        hf_sd = remap_lora_keys_to_hf(native_sd, self.from_hf)
        recovered = remap_lora_keys_from_hf(hf_sd, self.from_hf)

        self.assertEqual(
            set(native_sd.keys()),
            set(recovered.keys()),
            f"Round-trip key mismatch.\n"
            f"  Original: {sorted(native_sd.keys())}\n"
            f"  Recovered: {sorted(recovered.keys())}",
        )

    def test_round_trip_values(self):
        native_sd = {
            "layers.0.attention.qkv_linear.wq.lora_a.weight": torch.randn(8, 256),
            "layers.0.attention.qkv_linear.wq.lora_b.weight": torch.randn(256, 8),
        }
        hf_sd = remap_lora_keys_to_hf(native_sd, self.from_hf)
        recovered = remap_lora_keys_from_hf(hf_sd, self.from_hf)

        for key in native_sd:
            torch.testing.assert_close(
                native_sd[key],
                recovered[key],
                msg=f"Value mismatch for key: {key}",
            )

    def test_real_model_round_trip(self):
        """Round-trip using actual model state dict keys."""
        model, _, _ = _build_lora_model()
        trainable_sd = _get_trainable_sd(model)

        hf_sd = remap_lora_keys_to_hf(trainable_sd, self.from_hf)
        recovered = remap_lora_keys_from_hf(hf_sd, self.from_hf)

        self.assertEqual(
            set(trainable_sd.keys()),
            set(recovered.keys()),
            f"Real model round-trip key mismatch.\n"
            f"  Missing: {set(trainable_sd.keys()) - set(recovered.keys())}\n"
            f"  Extra: {set(recovered.keys()) - set(trainable_sd.keys())}",
        )
        for key in trainable_sd:
            torch.testing.assert_close(trainable_sd[key], recovered[key])


class TestLoRAPEFTSaveLoad(unittest.TestCase):
    """Test saving LoRA adapter in PEFT safetensors format and 2-phase reload."""

    def setUp(self):
        self.model, self.model_config, self.config = _build_lora_model()
        from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

        sd_adapter = Llama3StateDictAdapter(
            self.model_config, self.config.hf_assets_path
        )
        self.from_hf = sd_adapter.from_hf_map
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_load_peft_safetensors(self):
        """Save adapter as PEFT safetensors, reload, verify match."""
        try:
            from safetensors.torch import load_file, save_file
        except ImportError:
            self.skipTest("safetensors not installed")

        trainable_sd = _get_trainable_sd(self.model)
        self.assertTrue(len(trainable_sd) > 0)

        # Save in PEFT format
        peft_sd = remap_lora_keys_to_hf(trainable_sd, self.from_hf)
        peft_path = os.path.join(self.tmp_dir, "adapter_model.safetensors")
        save_file(peft_sd, peft_path)
        self.assertTrue(os.path.exists(peft_path))

        # Load and convert back
        loaded_peft = load_file(peft_path)
        recovered = remap_lora_keys_from_hf(loaded_peft, self.from_hf)

        self.assertEqual(set(trainable_sd.keys()), set(recovered.keys()))
        for key in trainable_sd:
            torch.testing.assert_close(trainable_sd[key], recovered[key])

    def test_two_phase_load(self):
        """Simulate 2-phase load: base weights + adapter from PEFT."""
        try:
            from safetensors.torch import load_file, save_file
        except ImportError:
            self.skipTest("safetensors not installed")

        # Set LoRA weights to a known value so we can verify they loaded
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.fill_(42.0)

        original_trainable = _get_trainable_sd(self.model)
        original_frozen = _get_frozen_sd(self.model)

        # Save adapter in PEFT format
        peft_sd = remap_lora_keys_to_hf(original_trainable, self.from_hf)
        peft_path = os.path.join(self.tmp_dir, "adapter_model.safetensors")
        save_file(peft_sd, peft_path)

        # --- Phase 1: Build fresh model, load base weights ---
        with torch.device("cpu"):
            fresh_model = self.model_config.build()
        with torch.no_grad():
            fresh_model.init_weights()

        # Load base weights (simulates HF checkpoint load)
        fresh_model.load_state_dict(original_frozen, strict=False)

        # Verify base matches
        fresh_frozen = _get_frozen_sd(fresh_model)
        for key in original_frozen:
            torch.testing.assert_close(
                original_frozen[key], fresh_frozen[key],
                msg=f"Base mismatch after phase 1: {key}",
            )

        # LoRA weights should NOT be 42.0 yet (fresh init)
        fresh_trainable_before = _get_trainable_sd(fresh_model)
        has_mismatch = any(
            not torch.equal(original_trainable[k], fresh_trainable_before[k])
            for k in original_trainable
        )
        self.assertTrue(
            has_mismatch,
            "Fresh model should have different LoRA weights before phase 2",
        )

        # --- Phase 2: Load adapter from PEFT ---
        loaded_peft = load_file(peft_path)
        native_adapter = remap_lora_keys_from_hf(loaded_peft, self.from_hf)
        fresh_model.load_state_dict(native_adapter, strict=False)

        # Verify adapter weights now match (should be 42.0)
        fresh_trainable = _get_trainable_sd(fresh_model)
        for key in original_trainable:
            torch.testing.assert_close(
                original_trainable[key], fresh_trainable[key],
                msg=f"Adapter mismatch after phase 2: {key}",
            )

        # Verify complete model matches
        original_full = _get_full_sd(self.model)
        fresh_full = _get_full_sd(fresh_model)
        for key in original_full:
            torch.testing.assert_close(
                original_full[key], fresh_full[key],
                msg=f"Full model mismatch: {key}",
            )


if __name__ == "__main__":
    unittest.main()
