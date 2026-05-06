# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch

from torchtitan.components.checkpoint import CheckpointManager, ModelWrapper
from torchtitan.components.lora import (
    LoRAConverter,
    remap_lora_keys_from_hf,
    remap_lora_keys_to_hf,
)
from torchtitan.models.llama3 import model_registry
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model import StateDictMode


class FakeOptimizersContainer:
    def __init__(self):
        self._fake_param = torch.tensor([1.0])

    def state_dict(self):
        return {"fake_param": self._fake_param}

    def load_state_dict(self, sd):
        if "fake_param" in sd:
            self._fake_param = sd["fake_param"]

    def init_cache_state_dict(self):
        pass


class FakeLRSchedulersContainer:
    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        pass


class FakeDataLoader:
    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        pass


def _build_lora_model():
    """Build a LoRA debug model and return (model, model_spec, model_config)."""
    model_spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=8, alpha=16.0, target_modules=["wq", "wkv", "wo"]
            ),
        ],
    )
    model_config = model_spec.model
    model = model_config.build()
    model.init_states()
    return model, model_spec, model_config


class TestLoRACheckpoint(unittest.TestCase):
    """Test LoRA key remapping, state dict modes, and checkpoint save/load."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model, self.model_spec, self.model_config = _build_lora_model()
        sd_adapter = Llama3StateDictAdapter(self.model_config, None)
        self.model.set_sd_adapter(sd_adapter, self.model_spec.converters)

        self.patcher_group = mock.patch(
            "torch.distributed.new_group", return_value="pg"
        )
        self.patcher_group.start()

    def tearDown(self):
        self.patcher_group.stop()
        shutil.rmtree(self.temp_dir)

    def test_peft_key_remapping_roundtrip(self):
        """Native LoRA keys -> PEFT HF keys -> native: keys and values preserved."""
        from_hf_map = self.model.sd_adapter.from_hf_map
        adapter_sd = {
            "layers.0.attention.qkv_linear.wq.lora_a.weight": torch.randn(8, 256),
            "layers.0.attention.qkv_linear.wq.lora_b.weight": torch.randn(256, 8),
            "layers.2.attention.wo.lora_a.weight": torch.randn(8, 256),
            "layers.2.attention.wo.lora_b.weight": torch.randn(256, 8),
        }
        hf_sd = remap_lora_keys_to_hf(adapter_sd, from_hf_map)
        for key in hf_sd:
            self.assertTrue(key.startswith("base_model.model."))
            self.assertTrue("lora_A" in key or "lora_B" in key)
        restored = remap_lora_keys_from_hf(hf_sd, from_hf_map)
        self.assertEqual(set(restored.keys()), set(adapter_sd.keys()))
        for k in adapter_sd:
            torch.testing.assert_close(restored[k], adapter_sd[k])

    def test_state_dict_modes(self):
        """FULL, TRAINABLE, BASE modes partition parameters correctly."""
        wrapper = ModelWrapper([self.model])

        wrapper.mode = StateDictMode.FULL
        full_keys = set(wrapper.state_dict().keys())

        wrapper.mode = StateDictMode.TRAINABLE
        trainable_keys = set(wrapper.state_dict().keys())

        wrapper.mode = StateDictMode.BASE
        base_keys = set(wrapper.state_dict().keys())

        # Trainable keys are only LoRA adapters
        for key in trainable_keys:
            self.assertTrue("lora_a" in key or "lora_b" in key)
        # Base keys have no LoRA adapters
        for key in base_keys:
            self.assertFalse("lora_a" in key or "lora_b" in key)
        # Disjoint and cover full
        self.assertEqual(base_keys & trainable_keys, set())
        self.assertEqual(base_keys | trainable_keys, full_keys)

    def _fake_save(self, state_dict, checkpoint_id, storage_writer=None):
        os.makedirs(checkpoint_id, exist_ok=True)
        sd_to_save = {}
        for key, val in state_dict.items():
            if hasattr(val, "state_dict"):
                sd_to_save[key] = val.state_dict()
            elif isinstance(val, torch.Tensor):
                sd_to_save[key] = val
        torch.save(sd_to_save, os.path.join(checkpoint_id, "state_dict.pt"))

    def _fake_load(self, states, checkpoint_id=None):
        path = os.path.join(checkpoint_id, "state_dict.pt")
        loaded = torch.load(path, weights_only=False)
        for key, val in loaded.items():
            if key in states and hasattr(states[key], "load_state_dict"):
                states[key].load_state_dict(val)
            elif key in states and isinstance(states[key], torch.Tensor):
                states[key].copy_(val)

    @mock.patch("torch.distributed.get_rank", return_value=0)
    @mock.patch("torchtitan.components.checkpoint.dcp.save")
    @mock.patch("torchtitan.components.checkpoint.dcp.load")
    def test_save_load_roundtrip(self, mock_load, mock_save, mock_rank):
        """Save full LoRA model, zero params, load back, verify match."""
        mock_save.side_effect = self._fake_save
        mock_load.side_effect = self._fake_load

        cfg = CheckpointManager.Config(
            enable=True,
            folder="",
            interval=1,
            keep_latest_k=0,
            last_save_model_only=False,
            export_dtype="float32",
        )
        manager = CheckpointManager(
            config=cfg,
            dataloader=FakeDataLoader(),
            model_parts=[self.model],
            optimizers=FakeOptimizersContainer(),
            lr_schedulers=FakeLRSchedulersContainer(),
            states={"train_state": torch.tensor([1.0])},
            base_folder=self.temp_dir,
        )

        orig_sd = {k: v.clone() for k, v in self.model.state_dict().items()}
        manager.save(curr_step=1)

        with torch.no_grad():
            for p in self.model.parameters():
                p.zero_()

        manager.load(step=1)

        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(v, orig_sd[k])
        manager.close()


if __name__ == "__main__":
    unittest.main()
