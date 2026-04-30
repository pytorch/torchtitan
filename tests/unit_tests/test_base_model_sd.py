# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from unittest import mock

import torch
import torch.nn as nn

from torchtitan.protocols.model import BaseModel, StateDictMode
from torchtitan.protocols.module import Module


class SimpleLinear(nn.Linear, Module):
    pass


class TinyModel(BaseModel):
    """Model simulating LoRA: base weights frozen, adapter weights trainable."""

    def __init__(self):
        super().__init__()
        self.base_layer = SimpleLinear(4, 4)
        self.base_layer.lora_a = SimpleLinear(4, 2, bias=False)
        self.base_layer.lora_b = SimpleLinear(2, 4, bias=False)
        # Freeze base, keep adapters trainable
        self.base_layer.weight.requires_grad_(False)
        self.base_layer.bias.requires_grad_(False)

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        def update_from_config(self, *, trainer_config, **kwargs):
            pass

        def get_nparams_and_flops(self, model, seq_len):
            return 0, 0


class PlainModel(BaseModel):
    """Model with all parameters trainable (no adapters)."""

    def __init__(self):
        super().__init__()
        self.linear = SimpleLinear(4, 4)

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        def update_from_config(self, *, trainer_config, **kwargs):
            pass

        def get_nparams_and_flops(self, model, seq_len):
            return 0, 0


def mock_get_sd(model, *, options=None):
    """Return plain state_dict (no DCP) for testing.

    Supports ``options.ignore_frozen_params`` to match DCP behavior.
    """
    sd = dict(model.state_dict())
    if options is not None and getattr(options, "ignore_frozen_params", False):
        frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
        sd = {k: v for k, v in sd.items() if k not in frozen}
    return sd


class TestGetSdFull(unittest.TestCase):
    @mock.patch(
        "torchtitan.protocols.model.get_model_state_dict", side_effect=mock_get_sd
    )
    def test_full_returns_all_keys(self, _):
        model = TinyModel()
        sd = model.get_sd(StateDictMode.FULL)
        self.assertIn("base_layer.weight", sd)
        self.assertIn("base_layer.bias", sd)
        self.assertIn("base_layer.lora_a.weight", sd)
        self.assertIn("base_layer.lora_b.weight", sd)


class TestGetSdTrainable(unittest.TestCase):
    @mock.patch(
        "torchtitan.protocols.model.get_model_state_dict", side_effect=mock_get_sd
    )
    def test_trainable_returns_only_grad_true(self, _):
        model = TinyModel()
        sd = model.get_sd(StateDictMode.TRAINABLE)
        self.assertNotIn("base_layer.weight", sd)
        self.assertNotIn("base_layer.bias", sd)
        self.assertIn("base_layer.lora_a.weight", sd)
        self.assertIn("base_layer.lora_b.weight", sd)

    @mock.patch(
        "torchtitan.protocols.model.get_model_state_dict", side_effect=mock_get_sd
    )
    def test_trainable_all_trainable_returns_all(self, _):
        """When all params are trainable (no adapters), TRAINABLE == FULL."""
        model = PlainModel()
        sd_full = model.get_sd(StateDictMode.FULL)
        sd_trainable = model.get_sd(StateDictMode.TRAINABLE)
        self.assertEqual(set(sd_full.keys()), set(sd_trainable.keys()))


class TestGetSdBase(unittest.TestCase):
    @mock.patch(
        "torchtitan.protocols.model.get_model_state_dict", side_effect=mock_get_sd
    )
    def test_base_returns_only_frozen(self, _):
        model = TinyModel()
        sd = model.get_sd(StateDictMode.BASE)
        self.assertIn("base_layer.weight", sd)
        self.assertIn("base_layer.bias", sd)
        self.assertNotIn("base_layer.lora_a.weight", sd)
        self.assertNotIn("base_layer.lora_b.weight", sd)

    @mock.patch(
        "torchtitan.protocols.model.get_model_state_dict", side_effect=mock_get_sd
    )
    def test_base_all_trainable_returns_all(self, _):
        """When all params are trainable, BASE returns all (nothing frozen)."""
        model = PlainModel()
        sd_full = model.get_sd(StateDictMode.FULL)
        sd_base = model.get_sd(StateDictMode.BASE)
        self.assertEqual(set(sd_full.keys()), set(sd_base.keys()))


class TestPPParts(unittest.TestCase):
    @mock.patch(
        "torchtitan.protocols.model.get_model_state_dict", side_effect=mock_get_sd
    )
    def test_full_merges_pp_parts(self, mock_fn):
        part0 = PlainModel()
        part1 = PlainModel()
        part1.layer2 = SimpleLinear(4, 4)
        del part1.linear
        part0._pp_parts = [part0, part1]

        sd = part0.get_sd(StateDictMode.FULL)
        self.assertIn("linear.weight", sd)
        self.assertIn("layer2.weight", sd)
        self.assertEqual(mock_fn.call_count, 2)

    @mock.patch("torchtitan.protocols.model.set_model_state_dict")
    def test_load_sd_loads_all_pp_parts(self, mock_set):
        part0 = PlainModel()
        part1 = PlainModel()
        part0._pp_parts = [part0, part1]

        part0.load_sd({"linear.weight": torch.zeros(4, 4)})
        self.assertEqual(mock_set.call_count, 2)

    @mock.patch(
        "torchtitan.protocols.model.get_model_state_dict", side_effect=mock_get_sd
    )
    def test_trainable_with_pp_and_adapters(self, _):
        """PP part0 has no adapters (all trainable), part1 has adapters."""
        part0 = PlainModel()
        part1 = TinyModel()
        part0._pp_parts = [part0, part1]

        sd = part0.get_sd(StateDictMode.TRAINABLE)
        # part0: all trainable → included
        self.assertIn("linear.weight", sd)
        # part1: only adapter trainable
        self.assertIn("base_layer.lora_a.weight", sd)
        self.assertIn("base_layer.lora_b.weight", sd)
        # part1 base: frozen → excluded
        self.assertNotIn("base_layer.weight", sd)


class TestNoPPDefault(unittest.TestCase):
    @mock.patch(
        "torchtitan.protocols.model.get_model_state_dict", side_effect=mock_get_sd
    )
    def test_no_pp_parts_uses_self(self, mock_fn):
        model = TinyModel()
        self.assertIsNone(model._pp_parts)
        model.get_sd(StateDictMode.FULL)
        self.assertEqual(mock_fn.call_count, 1)
        mock_fn.assert_called_with(model)


if __name__ == "__main__":
    unittest.main()
