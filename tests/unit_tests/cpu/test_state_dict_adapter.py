# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torchtitan.components.checkpointer.base import ModelWrapper
from torchtitan.components.checkpointer.hf_storage import (
    HuggingFaceStorageReaderWithViews,
)
from torchtitan.models.deepseek_v3 import deepseekv3_configs
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.models.deepseek_v4 import model_registry as deepseek_v4_model_registry
from torchtitan.models.deepseek_v4.model import DeepSeekV4Model
from torchtitan.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter
from torchtitan.models.gpt_oss import gptoss_configs
from torchtitan.models.gpt_oss.state_dict_adapter import GptOssStateDictAdapter
from torchtitan.models.kimi_k3 import model_registry as kimi_k3_model_registry
from torchtitan.models.kimi_k3.quantization import MXFP4_QUANTIZATION_CONFIG
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.models.qwen3 import qwen3_configs
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from tests.unit_tests.cpu.mx_qat_test_utils import write_mixed_checkpoint_metadata


class NativeFusedLinearStateDictAdapterTest(unittest.TestCase):
    def test_stacked_helpers_support_nonleading_projection_dim(self) -> None:
        fused = torch.randn(3, 5, 2, 7)
        state_dict = {"experts.w13": fused}
        logical_keys = ("experts.w1", "experts.w3")

        StateDictAdapter._split_stacked_linear(
            state_dict,
            fused_key="experts.w13",
            logical_keys=logical_keys,
            dim=2,
        )

        self.assertNotIn("experts.w13", state_dict)
        torch.testing.assert_close(state_dict["experts.w1"], fused[:, :, 0, :])
        torch.testing.assert_close(state_dict["experts.w3"], fused[:, :, 1, :])

        StateDictAdapter._stack_logical_linears(
            state_dict,
            fused_key="experts.w13",
            logical_keys=logical_keys,
            dim=2,
        )

        self.assertEqual(set(state_dict), {"experts.w13"})
        torch.testing.assert_close(state_dict["experts.w13"], fused)


class Llama3FusedLinearStateDictAdapterTest(unittest.TestCase):
    def test_hf_roundtrip_converts_native_fused_feed_forward(self) -> None:
        build_config, max_context_length = llama3_configs["debugmodel"]
        config = build_config(attn_backend="flex", seq_len=max_context_length)
        model = Llama3Model(config)
        model.init_states()
        state_dict = model.state_dict()
        w13_key = "layers.0.feed_forward.w13.weight"

        self.assertIn(w13_key, state_dict)
        self.assertNotIn("layers.0.feed_forward.w1.weight", state_dict)
        self.assertNotIn("layers.0.feed_forward.w3.weight", state_dict)

        adapter = Llama3StateDictAdapter(config, hf_assets_path=None)
        hf_state_dict = adapter.to_hf(state_dict)
        torch.testing.assert_close(
            hf_state_dict["model.layers.0.mlp.gate_proj.weight"],
            state_dict[w13_key][0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            hf_state_dict["model.layers.0.mlp.up_proj.weight"],
            state_dict[w13_key][1],
            rtol=0,
            atol=0,
        )

        restored = adapter.from_hf(hf_state_dict)
        self.assertEqual(restored.keys(), state_dict.keys())
        torch.testing.assert_close(restored[w13_key], state_dict[w13_key])
        model.load_state_dict(restored, strict=True)


class Qwen3StateDictAdapterTest(unittest.TestCase):
    def test_hf_roundtrip_preserves_tied_embedding_shape(self) -> None:
        build_config, max_context_length = qwen3_configs["debugmodel"]
        config = build_config(attn_backend="flex", seq_len=max_context_length)
        model = Qwen3Model(config)
        model.init_states()
        state_dict = model.state_dict()

        self.assertEqual(state_dict["tok_embeddings.weight"].ndim, 2)
        self.assertEqual(state_dict["lm_head.weight"].ndim, 2)

        adapter = Qwen3StateDictAdapter(config, hf_assets_path=None)
        hf_state_dict = adapter.to_hf(state_dict)
        self.assertEqual(hf_state_dict["model.embed_tokens.weight"].ndim, 2)
        self.assertNotIn("lm_head.weight", hf_state_dict)

        restored = adapter.from_hf(hf_state_dict)
        self.assertEqual(restored["tok_embeddings.weight"].ndim, 2)
        self.assertEqual(restored["lm_head.weight"].ndim, 2)
        model.load_state_dict(restored, strict=True)


class KimiK3StateDictAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        model_config = kimi_k3_model_registry("debugmodel", seq_len=128)
        self.adapter = KimiK3StateDictAdapter(
            model_config,
            hf_assets_path=None,
        )

    def _write_checkpoint_metadata(self, path):
        return write_mixed_checkpoint_metadata(Path(path), self.adapter)

    def test_quantized_load_uses_packed_pair_reader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self._write_checkpoint_metadata(directory)
            reader = self.adapter.get_hf_storage_reader(
                directory,
                from_quantized=True,
            )

        self.assertIsInstance(reader, HuggingFaceStorageReaderWithViews)
        self.assertEqual(reader.spec.packed_suffix, ".weight_packed")
        self.assertEqual(reader.spec.scale_suffix, ".weight_scale")
        self.assertEqual(reader.spec.virtual_suffix, ".weight")
        self.assertEqual(reader.spec.block_size, 32)
        self.assertEqual(reader.spec.target_dtype, torch.bfloat16)
        self.assertTrue(
            "language_model.model.layers.1.block_sparse_moe.experts.3.w1.weight"
            in reader.spec.target_fqns
        )
        self.assertFalse(
            "language_model.model.layers.1.block_sparse_moe.shared_experts.w1.weight"
            in reader.spec.target_fqns
        )
        self.assertFalse(
            "language_model.model.layers.1.block_sparse_moe.experts.3.w4.weight"
            in reader.spec.target_fqns
        )
        self.assertFalse(
            "language_model.model.embed_tokens.weight" in reader.spec.target_fqns
        )
        self.assertFalse(
            "language_model.model.layers.1.block_sparse_moe.gate.weight"
            in reader.spec.target_fqns
        )
        self.assertNotIn(
            "language_model.model.layers.1.block_sparse_moe.routed_expert_up_proj.weight",
            reader.spec.target_fqns,
        )
        self.assertNotIn(
            "language_model.model.layers.1.mlp_res_proj.weight", reader.spec.target_fqns
        )

    def test_qat_selection_matches_import_including_dense_projections(self) -> None:
        from torchtitan.config.transform import MXQATTransform
        from torchtitan.quantization.mx_qat.checkpoint import MXFP4CheckpointPolicy

        mapping = self.adapter.hf_linear_weight_mapping()
        policy = MXFP4CheckpointPolicy.from_config(MXFP4_QUANTIZATION_CONFIG, mapping)
        model = self.adapter.kimi_config
        transform = MXQATTransform.from_weight_fqns(
            model,
            {mapping[key] for key in policy.weight_fqns if mapping[key] is not None},
        )
        model = transform.transform(model)
        self.adapter._validate_qat_policy(policy)
        self.assertTrue(type(model.layers[1].moe.routed_up)._owner._mx_qat)

    def test_qat_rejects_expert_only_selection_when_dense_weights_are_packed(
        self,
    ) -> None:
        from torchtitan.config.transform import MXQATTransform
        from torchtitan.quantization.mx_qat.checkpoint import MXFP4CheckpointPolicy

        policy = MXFP4CheckpointPolicy.from_config(
            MXFP4_QUANTIZATION_CONFIG, self.adapter.hf_linear_weight_mapping()
        )
        MXQATTransform().transform(self.adapter.kimi_config)
        with self.assertRaisesRegex(ValueError, "selection disagrees"):
            self.adapter._validate_qat_policy(policy)

    def test_vision_policy_uses_runtime_layer_names(self) -> None:
        from torchtitan.quantization.mx_qat.checkpoint import MXFP4CheckpointPolicy

        mapping = self.adapter.hf_linear_weight_mapping()
        name = "vision_tower.encoder.blocks.0.mlp.fc0.weight"
        self.assertEqual(mapping[name], "vision_encoder.layers.0.mlp.linear_fc1.weight")
        policy = MXFP4CheckpointPolicy.from_config(MXFP4_QUANTIZATION_CONFIG, mapping)
        self.assertNotIn(name, policy.weight_fqns)
        quantization = deepcopy(MXFP4_QUANTIZATION_CONFIG)
        quantization["ignore"] = []
        self.assertIn(
            name, MXFP4CheckpointPolicy.from_config(quantization, mapping).weight_fqns
        )

    def test_qat_recipe_is_valid_with_and_without_initial_checkpoint(self) -> None:
        from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel_mx_qat

        recipe = kimi_k3_debugmodel_mx_qat(seq_len=16)
        self.assertFalse(recipe.checkpointer.initial_load_in_hf_quantized)
        with tempfile.TemporaryDirectory() as directory:
            self._write_checkpoint_metadata(directory)
            recipe = kimi_k3_debugmodel_mx_qat(seq_len=16, checkpoint_path=directory)
            self.assertTrue(recipe.checkpointer.initial_load_in_hf_quantized)
            self.assertEqual(recipe.checkpointer.initial_load_path, directory)
            self.assertFalse(
                getattr(
                    type(recipe.model.layers[1].moe.routed_up)._owner, "_mx_qat", False
                )
            )

    def test_manifest_qat_rejects_mixed_experts_in_one_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_checkpoint_metadata(directory)
            prefix = "language_model.model.layers.1.block_sparse_moe.experts.0.w1"
            del manifest[prefix + ".weight_packed"]
            del manifest[prefix + ".weight_scale"]
            manifest[prefix + ".weight"] = "dense.safetensors"
            Path(directory, "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": manifest})
            )
            policy = self.adapter.mxfp4_policy(directory)
            with self.assertRaisesRegex(
                ValueError, "cannot mix packed and BF16 experts"
            ):
                self.adapter.qat_weight_fqns(policy)

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
                )
                assert isinstance(config, DeepSeekV4Model.Config)
                model = config.build()
                model.init_states()
                state_dict = model.state_dict()
                for index, (key, value) in enumerate(state_dict.items()):
                    if key.endswith("attention.attn_sink.weight"):
                        self.assertEqual(value.ndim, 2)
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
                )
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


class Llama3DTensorStateDictAdapterTest(unittest.TestCase):
    """Regression tests for the q/k permute's handling of DTensor inputs.

    ``_permute``/``_reverse_permute`` do a head-splitting ``view()`` on the
    q/k projection weights. In the live save/load path those weights are
    DTensors that FSDP shards along dim 0, the exact dim the view unflattens,
    and an FSDP degree that does not evenly divide the head count makes the
    unflatten invalid on the (still-sharded) DTensor directly.
    """

    def test_permute_does_not_raise_when_fsdp_degree_exceeds_head_count(self) -> None:
        # llama3 debugmodel has 16 attention heads and 16 KV heads. Sharding its
        # packed QKV weight over a 32-rank mesh reproduces the production
        # failure: the QKV group reshape cannot be applied to a dim-0 shard
        # when the FSDP degree exceeds the KV-head count. A "fake" process
        # group is enough here; no real collectives run, but DTensor shape
        # propagation still executes.
        world_size = 32
        dist.init_process_group(
            "fake", store=FakeStore(), rank=0, world_size=world_size
        )
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))

        build_config, max_context_length = llama3_configs["debugmodel"]
        config = build_config(attn_backend="flex", seq_len=max_context_length)
        adapter = Llama3StateDictAdapter(config, hf_assets_path=None)

        qkv_config = config.layers[0].attention.qkv_linear
        qkv_out_features = qkv_config.wqkv.out_features
        full = torch.arange(qkv_out_features * config.dim, dtype=torch.float32).reshape(
            qkv_out_features, config.dim
        )
        local_shard = full[: qkv_out_features // world_size].clone()
        sharded_weight = DTensor.from_local(
            local_shard, mesh, (Shard(0),), run_check=False
        )

        hf_state_dict = adapter.to_hf(
            {"layers.0.attention.qkv_linear.wqkv.weight": sharded_weight}
        )
        out = hf_state_dict["model.layers.0.self_attn.q_proj.weight"]
        self.assertIsInstance(out, DTensor)
        self.assertEqual(out.shape, torch.Size((256, 256)))

        restored = adapter.from_hf(hf_state_dict)
        native = restored["layers.0.attention.qkv_linear.wqkv.weight"]
        self.assertIsInstance(native, DTensor)
        self.assertEqual(native.placements, sharded_weight.placements)
        self.assertEqual(native.to_local().shape, local_shard.shape)

    def test_permute_roundtrip_preserves_native_dtensor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group(
                "gloo",
                init_method=f"file://{directory}/rendezvous",
                rank=0,
                world_size=1,
            )
            try:
                mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
                build_config, max_context_length = llama3_configs["debugmodel"]
                config = build_config(attn_backend="flex", seq_len=max_context_length)
                adapter = Llama3StateDictAdapter(config, hf_assets_path=None)

                qkv_config = config.layers[0].attention.qkv_linear
                qkv_out_features = qkv_config.wqkv.out_features
                full = torch.arange(
                    qkv_out_features * config.dim, dtype=torch.float32
                ).reshape(qkv_out_features, config.dim)
                weight = DTensor.from_local(full, mesh, (Shard(0),), run_check=False)
                key = "layers.0.attention.qkv_linear.wqkv.weight"

                restored = adapter.from_hf(adapter.to_hf({key: weight}))
                out = restored[key]
                self.assertIsInstance(out, DTensor)
                self.assertEqual(out.placements, weight.placements)
                torch.testing.assert_close(out.to_local(), full, rtol=0, atol=0)
            finally:
                dist.destroy_process_group()
