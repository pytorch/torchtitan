# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v3 import MODEL_FLAVORS as DEEPSEEK_V3_MODEL_FLAVORS
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.models.kimi_k2_7 import MODEL_FLAVORS as KIMI_K2_7_MODEL_FLAVORS
from torchtitan.models.kimi_k2_7.state_dict_adapter import KimiK25StateDictAdapter
from torchtitan.models.kimi_k3 import MODEL_FLAVORS as KIMI_K3_MODEL_FLAVORS
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter


class MLAStateDictAdapterTest(unittest.TestCase):
    def test_hf_roundtrip_loads_nested_kv_projection(self) -> None:
        cases = (
            (DEEPSEEK_V3_MODEL_FLAVORS, DeepSeekV3StateDictAdapter, {}),
            (
                DEEPSEEK_V3_MODEL_FLAVORS,
                DeepSeekV3StateDictAdapter,
                {"num_mtp_layers": 1},
            ),
            (KIMI_K2_7_MODEL_FLAVORS, KimiK25StateDictAdapter, {}),
            (KIMI_K3_MODEL_FLAVORS, KimiK3StateDictAdapter, {}),
        )
        for flavors, adapter_cls, kwargs in cases:
            with self.subTest(adapter=adapter_cls.__name__, **kwargs):
                build_config, _ = flavors["debugmodel"]
                config = build_config(
                    attn_backend="flex",
                    seq_len=128,
                    parallelism=ParallelismConfig(),
                    **kwargs,
                )
                model = config.build()
                model.init_states()
                state_dict = model.state_dict()
                projection_keys = [
                    key
                    for key, _ in model.named_parameters()
                    if key.endswith(".inner_attention.wkv_b.weight")
                ]
                self.assertTrue(projection_keys)
                if kwargs.get("num_mtp_layers"):
                    self.assertIn(
                        "mtp_layers.0.attention.inner_attention.wkv_b.weight",
                        projection_keys,
                    )

                adapter = adapter_cls(config, hf_assets_path=None)
                hf_state_dict = adapter.to_hf(state_dict)
                for key in projection_keys:
                    self.assertIn(key, state_dict)
                    self.assertNotIn(key.replace(".inner_attention", ""), state_dict)
                    layer_group, layer_id, _ = key.split(".", 2)
                    layer_id = int(layer_id)
                    if layer_group == "mtp_layers":
                        layer_id += len(config.layers)
                    prefix = (
                        "language_model."
                        if adapter_cls
                        in (KimiK25StateDictAdapter, KimiK3StateDictAdapter)
                        else ""
                    )
                    hf_key = (
                        f"{prefix}model.layers.{layer_id}.self_attn.kv_b_proj.weight"
                    )
                    torch.testing.assert_close(
                        hf_state_dict[hf_key], state_dict[key], rtol=0, atol=0
                    )

                restored = adapter.from_hf(hf_state_dict)
                self.assertEqual(restored.keys(), state_dict.keys())
                for key, value in state_dict.items():
                    torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
                restored_model = config.build()
                restored_model.load_state_dict(restored, strict=True)
                restored_state = restored_model.state_dict()
                for key in projection_keys:
                    torch.testing.assert_close(
                        restored_state[key], state_dict[key], rtol=0, atol=0
                    )
