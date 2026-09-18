# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from torchtitan.components.checkpointer.base import ModelWrapper
from torchtitan.components.checkpointer.packed_hf_storage import (
    PackedPairHuggingFaceStorageReader,
)
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.models.deepseek_v4 import model_registry as deepseek_v4_model_registry
from torchtitan.models.deepseek_v4.model import DeepSeekV4Model
from torchtitan.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter
from torchtitan.models.gpt_oss import gptoss_configs
from torchtitan.models.gpt_oss.state_dict_adapter import GptOssStateDictAdapter
from torchtitan.models.kimi_k3 import model_registry as kimi_k3_model_registry
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter


class KimiK3StateDictAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        model_spec = kimi_k3_model_registry("debugmodel", seq_len=128)
        self.adapter = KimiK3StateDictAdapter(
            model_spec.model,
            hf_assets_path=None,
        )

    def test_quantized_load_uses_packed_pair_reader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps(
                    {
                        "text_config": {
                            "quantization_config": {
                                "format": "mxfp4-pack-quantized",
                                "quant_method": "compressed-tensors",
                                "config_groups": {
                                    "group_0": {
                                        "targets": ["Linear"],
                                        "weights": {"group_size": 32},
                                    }
                                },
                                "ignore": ["re:.*shared_experts.*"],
                            }
                        }
                    }
                )
            )
            reader = self.adapter.get_hf_storage_reader(
                directory,
                from_quantized=True,
            )

        self.assertIsInstance(reader, PackedPairHuggingFaceStorageReader)
        self.assertEqual(reader.spec.packed_suffix, ".weight_packed")
        self.assertEqual(reader.spec.scale_suffix, ".weight_scale")
        self.assertEqual(reader.spec.virtual_suffix, ".weight")
        self.assertEqual(reader.spec.block_size, 32)
        self.assertEqual(reader.spec.target_dtype, torch.bfloat16)
        self.assertTrue(
            reader.spec.is_target(
                "language_model.model.layers.1.block_sparse_moe.experts.3.w1.weight"
            )
        )
        self.assertFalse(
            reader.spec.is_target(
                "language_model.model.layers.1.block_sparse_moe.shared_experts.w1.weight"
            )
        )
        self.assertTrue(
            reader.spec.is_target(
                "language_model.model.layers.1.block_sparse_moe.experts.3.w4.weight"
            )
        )

    def test_quantized_load_rejects_missing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "missing quantization_config"):
                Path(directory, "config.json").write_text("{}")
                self.adapter.get_hf_storage_reader(directory, from_quantized=True)

    def test_unquantized_load_keeps_plain_reader(self) -> None:
        reader = self.adapter.get_hf_storage_reader(
            "/tmp/kimi-k3-checkpoint",
            from_quantized=False,
        )

        self.assertIs(type(reader), HuggingFaceStorageReader)


class DeepSeekV3StateDictAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls._owns_process_group = not dist.is_initialized()
        if cls._owns_process_group:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{cls._temporary_directory.name}/rendezvous",
                rank=0,
                world_size=1,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._owns_process_group:
            dist.destroy_process_group()
        cls._temporary_directory.cleanup()

    def test_to_hf_handles_replicated_grouped_experts(self) -> None:
        build_config, max_context_length = deepseekv3_configs["debugmodel"]
        config = build_config(
            attn_backend="flex", moe_comm_backend="standard", seq_len=max_context_length
        )
        adapter = DeepSeekV3StateDictAdapter(config, hf_assets_path=None)
        mesh = init_device_mesh(
            "cpu",
            (1, 1),
            mesh_dim_names=("replicate", "shard"),
        )
        local_weight = torch.arange(8 * 2 * 3, dtype=torch.float32).reshape(8, 2, 3)
        grouped_expert_weight = DTensor.from_local(
            local_weight,
            mesh,
            (Replicate(), Shard(0)),
            run_check=False,
        )

        hf_state_dict = adapter.to_hf(
            {"layers.1.moe.routed_experts.inner_experts.w1_EFD": grouped_expert_weight}
        )

        expected_keys = {
            f"model.layers.1.mlp.experts.{expert}.gate_proj.weight"
            for expert in range(8)
        }
        self.assertEqual(set(hf_state_dict), expected_keys)
        for expert in range(8):
            key = f"model.layers.1.mlp.experts.{expert}.gate_proj.weight"
            self.assertIsInstance(hf_state_dict[key], DTensor)
            torch.testing.assert_close(
                hf_state_dict[key].to_local(),
                local_weight[expert],
            )

    def test_roundtrip_preserves_mtp_expert_placements(self) -> None:
        build_config, _ = deepseekv3_configs["debugmodel"]
        config = build_config(
            attn_backend="flex",
            moe_comm_backend="standard",
            seq_len=128,
            num_mtp_layers=1,
        )
        adapter = DeepSeekV3StateDictAdapter(config, hf_assets_path=None)
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))
        local_weight = torch.arange(8 * 2 * 3, dtype=torch.float32).reshape(8, 2, 3)
        weight = DTensor.from_local(local_weight, mesh, (Shard(0),), run_check=False)
        key = "mtp_layers.0.moe.routed_experts.inner_experts.w1_EFD"
        restored = adapter.from_hf(adapter.to_hf({key: weight}))
        self.assertIsInstance(restored[key], DTensor)
        self.assertEqual(restored[key].placements, weight.placements)
        torch.testing.assert_close(restored[key], weight, rtol=0, atol=0)


class DeepSeekV4StateDictAdapterTest(unittest.TestCase):
    def test_full_model_roundtrip_with_optional_mtp(self) -> None:
        for num_mtp_layers in (0, 1, 2):
            with self.subTest(num_mtp_layers=num_mtp_layers):
                config = deepseek_v4_model_registry(
                    "debugmodel", seq_len=128, n_mtp_layers=num_mtp_layers
                ).model
                assert isinstance(config, DeepSeekV4Model.Config)
                model = config.build()
                model.init_states()
                state_dict = model.state_dict()
                for index, (key, value) in enumerate(state_dict.items()):
                    if key.endswith("attention.attn_sink.weight"):
                        value.copy_(
                            torch.arange(value.numel()).reshape(value.shape) + index
                        )
                    self.assertTrue(torch.isfinite(value).all(), key)

                adapter = DeepSeekV4StateDictAdapter(config, hf_assets_path=None)
                hf_state_dict = adapter.to_hf(state_dict)
                for depth in range(num_mtp_layers):
                    sink_key = f"mtp.{depth}.attn.attn_sink"
                    torch.testing.assert_close(
                        hf_state_dict[sink_key],
                        state_dict[
                            f"mtp_layers.{depth}.attention.attn_sink.weight"
                        ].squeeze(-1),
                        rtol=0,
                        atol=0,
                    )

                restored = adapter.from_hf(hf_state_dict)
                self.assertEqual(restored.keys(), state_dict.keys())
                for key, value in state_dict.items():
                    torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
                restored_model = config.build()
                restored_model.load_state_dict(restored, strict=True)

    def test_roundtrip_loads_sharded_mtp_experts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            owns_process_group = not dist.is_initialized()
            if owns_process_group:
                dist.init_process_group(
                    "gloo",
                    init_method=f"file://{directory}/rendezvous",
                    rank=0,
                    world_size=1,
                )
            try:
                config = deepseek_v4_model_registry(
                    "debugmodel", seq_len=128, n_mtp_layers=1
                ).model
                assert isinstance(config, DeepSeekV4Model.Config)
                model = config.build()
                model.init_states()
                mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))
                for key, value in model.state_dict().items():
                    if "moe.routed_experts.inner_experts" in key:
                        module_path, name = key.rsplit(".", 1)
                        weight = DTensor.from_local(
                            value.clone(), mesh, (Shard(0),), run_check=False
                        )
                        setattr(
                            model.get_submodule(module_path),
                            name,
                            torch.nn.Parameter(weight),
                        )
                adapter = DeepSeekV4StateDictAdapter(config, hf_assets_path=None)
                original = model.state_dict()
                restored = adapter.from_hf(adapter.to_hf(original))
                self.assertEqual(restored.keys(), original.keys())
                for key, value in original.items():
                    if isinstance(value, DTensor):
                        self.assertIsInstance(restored[key], DTensor)
                        self.assertEqual(restored[key].placements, value.placements)
                    torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
                ModelWrapper(model).load_state_dict(restored)
            finally:
                if owns_process_group:
                    dist.destroy_process_group()


class GptOssStateDictAdapterTest(unittest.TestCase):
    def test_full_model_roundtrip_preserves_all_expert_weights(self) -> None:
        build_config, _ = gptoss_configs["debugmodel"]
        config = build_config(
            moe_comm_backend="standard", attn_backend="flex", seq_len=128
        )
        # Keep the real four-layer model structure while making expert tensors
        # small enough for a CPU unit test.
        for layer_config in config.layers:
            assert layer_config.moe is not None
            layer_config.moe.routed_experts.inner_experts.hidden_dim = 16

        model = config.build()
        model.init_states()
        state_dict = model.state_dict()
        expert_bias_keys = {
            key for key in state_dict if key.endswith(".moe.expert_bias_E")
        }
        self.assertEqual(len(expert_bias_keys), len(config.layers))
        for key in expert_bias_keys:
            state_dict[key].fill_(1.0)

        adapter = GptOssStateDictAdapter(config, hf_assets_path=None)
        hf_state_dict = adapter.to_hf(state_dict)

        hf_expert_names = {
            "gate_up_proj_blocks",
            "gate_up_proj_bias",
            "down_proj_blocks",
            "down_proj_bias",
        }
        expected_hf_expert_keys = {
            f"model.layers.{layer_num}.mlp.experts.{name}"
            for layer_num in range(len(config.layers))
            for name in hf_expert_names
        }
        actual_hf_expert_keys = {key for key in hf_state_dict if ".mlp.experts." in key}
        self.assertEqual(actual_hf_expert_keys, expected_hf_expert_keys)
        self.assertFalse(any("expert_bias_E" in key for key in hf_state_dict))

        roundtrip_state_dict = adapter.from_hf(hf_state_dict)
        self.assertEqual(roundtrip_state_dict.keys(), state_dict.keys())
        for key, value in state_dict.items():
            if key in expert_bias_keys:
                torch.testing.assert_close(
                    roundtrip_state_dict[key], torch.zeros_like(value)
                )
            else:
                torch.testing.assert_close(
                    roundtrip_state_dict[key], value, rtol=0, atol=0
                )
