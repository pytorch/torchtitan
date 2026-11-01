# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import (
    HuggingFaceStorageReader,
    HuggingFaceStorageWriter,
)
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from torchtitan.components.checkpointer.base import ModelWrapper

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.base.state_dict_adapter import StateDictAdapter
from torchtitan.models.deepseek_v3 import MODEL_FLAVORS as DEEPSEEK_V3_MODEL_FLAVORS
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.models.deepseek_v4 import (
    build_model_config as build_deepseek_v4_model_config,
)
from torchtitan.models.deepseek_v4.model import DeepSeekV4Model
from torchtitan.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter
from torchtitan.models.flux import MODEL_FLAVORS as FLUX_MODEL_FLAVORS
from torchtitan.models.flux.state_dict_adapter import FluxStateDictAdapter
from torchtitan.models.gpt_oss import MODEL_FLAVORS as GPT_OSS_MODEL_FLAVORS
from torchtitan.models.gpt_oss.state_dict_adapter import GptOssStateDictAdapter
from torchtitan.models.kimi_k3.flavors import _kimi_k3_config, _vision_encoder_config
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from torchtitan.models.llama3 import MODEL_FLAVORS as LLAMA3_MODEL_FLAVORS
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.models.muse_glimmer import (
    build_model_config as build_muse_glimmer_model_config,
)
from torchtitan.models.muse_glimmer.state_dict_adapter import (
    MuseGlimmerStateDictAdapter,
)
from torchtitan.models.qwen3 import MODEL_FLAVORS as QWEN3_MODEL_FLAVORS
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter


class MuseGlimmerVisionStateDictAdapterTest(unittest.TestCase):
    def test_hf_native_roundtrip_and_strict_load(self) -> None:
        config = build_muse_glimmer_model_config(
            "debugmodel_mm", seq_len=8, parallelism=ParallelismConfig()
        )
        model = config.build()
        model.init_states()
        state = model.state_dict()
        for key, tensor in state.items():
            if key.startswith("vision_encoder.layers.") and key.endswith(
                ("wq.bias", "wk.bias")
            ):
                tensor.copy_(torch.arange(tensor.numel(), dtype=tensor.dtype))
        adapter = MuseGlimmerStateDictAdapter(config, hf_assets_path=None)
        hf = adapter.to_hf(state)
        assert config.vision_encoder is not None
        heads = config.vision_encoder.num_heads
        for projection, native in (("q", "wq"), ("k", "wk")):
            for parameter in ("weight", "bias"):
                value = state[f"vision_encoder.layers.0.attn.{native}.{parameter}"]
                expected = (
                    value.reshape(heads, -1, 2, *value.shape[1:])
                    .transpose(1, 2)
                    .reshape_as(value)
                )
                torch.testing.assert_close(
                    hf[
                        f"model.vision_tower.layers.0.attn.{projection}_proj.{parameter}"
                    ],
                    expected,
                    rtol=0,
                    atol=0,
                )
        restored = adapter.from_hf(hf)
        self.assertEqual(restored.keys(), state.keys())
        for key, tensor in state.items():
            torch.testing.assert_close(restored[key], tensor, rtol=0, atol=0, msg=key)
        config.build().load_state_dict(restored, strict=True)


class HuggingFaceStorageReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        llama_build_config, _ = LLAMA3_MODEL_FLAVORS["debugmodel"]
        deepseek_build_config, _ = DEEPSEEK_V3_MODEL_FLAVORS["debugmodel"]
        gpt_oss_build_config, _ = GPT_OSS_MODEL_FLAVORS["debugmodel"]
        self.adapters = (
            Llama3StateDictAdapter(
                llama_build_config(
                    attn_backend="flex", seq_len=128, parallelism=ParallelismConfig()
                ),
                hf_assets_path=None,
            ),
            DeepSeekV3StateDictAdapter(
                deepseek_build_config(
                    attn_backend="flex", seq_len=128, parallelism=ParallelismConfig()
                ),
                hf_assets_path=None,
            ),
            GptOssStateDictAdapter(
                gpt_oss_build_config(
                    attn_backend="flex", seq_len=128, parallelism=ParallelismConfig()
                ),
                hf_assets_path=None,
            ),
        )

    def test_unquantized_readers_use_two_threads(self) -> None:
        for adapter in self.adapters:
            with self.subTest(adapter=type(adapter).__name__):
                reader = adapter.get_hf_storage_reader("checkpoint")
                self.assertEqual(reader.thread_count, 2)

    def test_thread_count_can_be_overridden(self) -> None:
        for adapter in self.adapters:
            with self.subTest(adapter=type(adapter).__name__):
                reader = adapter.get_hf_storage_reader(
                    "checkpoint",
                    thread_count=3,
                )
                self.assertEqual(reader.thread_count, 3)

    def test_quantized_reader_thread_count_can_be_overridden(self) -> None:
        for adapter in self.adapters[1:]:
            with self.subTest(adapter=type(adapter).__name__):
                default_reader = adapter.get_hf_storage_reader(
                    "checkpoint",
                    from_quantized=True,
                )
                self.assertEqual(default_reader.thread_count, 4)
                reader = adapter.get_hf_storage_reader(
                    "checkpoint",
                    from_quantized=True,
                    thread_count=3,
                )
                self.assertEqual(reader.thread_count, 3)


def _small_kimi_k3_config() -> KimiK3Model.Config:
    dim = 64
    return _kimi_k3_config(
        max_context_length=128,
        parallelism=ParallelismConfig(),
        dim=dim,
        vocab_size=32,
        num_layers=2,
        full_attention_layers={1},
        attn_res_block_size=1,
        num_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        kda_head_dim=128,
        conv_kernel_size=3,
        dense_hidden_dim=128,
        latent_dim=32,
        expert_hidden_dim=32,
        num_experts=2,
        top_k=1,
        num_shared_experts=1,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=48,
            qkv_dim=48,
            hidden_dim=96,
            num_layers=1,
            num_heads=3,
            patch_size=2,
            merge_kernel_size=(2, 2),
            init_pos_emb_height=2,
            init_pos_emb_width=2,
            max_num_frames=1,
        ),
        attn_backend="flex",
    )


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
        build_config, max_context_length = LLAMA3_MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
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
        build_config, max_context_length = QWEN3_MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
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


class UnsupportedQuantizedKeysTest(unittest.TestCase):
    def test_adapters_without_quantized_loads_reject_quantized_layout(self) -> None:
        adapters = [
            FluxStateDictAdapter(
                FLUX_MODEL_FLAVORS["flux-debug"](), hf_assets_path=None
            )
        ]
        for flavors, adapter_cls in (
            (LLAMA3_MODEL_FLAVORS, Llama3StateDictAdapter),
            (QWEN3_MODEL_FLAVORS, Qwen3StateDictAdapter),
        ):
            build_config, max_context_length = flavors["debugmodel"]
            config = build_config(
                attn_backend="flex",
                seq_len=max_context_length,
                parallelism=ParallelismConfig(),
            )
            adapters.append(adapter_cls(config, hf_assets_path=None))

        for adapter in adapters:
            with self.subTest(adapter=type(adapter).__name__):
                with self.assertRaisesRegex(ValueError, "does not support loading"):
                    adapter.to_hf({}, to_quantized_layout=True)
                with self.assertRaisesRegex(ValueError, "does not support loading"):
                    adapter.from_hf({}, is_quantized_layout=True)


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

    def test_released_index_drops_unsaved_entries(self) -> None:
        build_config, max_context_length = DEEPSEEK_V3_MODEL_FLAVORS["debugmodel"]
        for num_mtp_layers in (0, 1):
            config = build_config(
                attn_backend="flex",
                seq_len=max_context_length,
                num_mtp_layers=num_mtp_layers,
                parallelism=ParallelismConfig(),
            )
            mtp_layer = f"model.layers.{len(config.layers)}"
            weight_map = {
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight_scale_inv": (
                    "model-00001-of-00002.safetensors"
                ),
                f"{mtp_layer}.eh_proj.weight": "model-00002-of-00002.safetensors",
                "model.norm.weight": "model-00002-of-00002.safetensors",
            }
            with tempfile.TemporaryDirectory() as hf_assets_path:
                with open(f"{hf_assets_path}/model.safetensors.index.json", "w") as f:
                    json.dump({"weight_map": weight_map}, f)
                adapter = DeepSeekV3StateDictAdapter(
                    config, hf_assets_path=hf_assets_path
                )
            with torch.device("meta"):
                hf_state_dict = adapter.to_hf(config.build().state_dict())

            expected = {
                "model.layers.0.self_attn.q_proj.weight": 1,
                "model.norm.weight": 2,
            }
            if num_mtp_layers:
                expected[f"{mtp_layer}.eh_proj.weight"] = 2
            with self.subTest(num_mtp_layers=num_mtp_layers):
                self.assertEqual(
                    adapter.fqn_to_index_mapping_for_save(hf_state_dict), expected
                )

    def test_mtp_layers_save_independent_embedding_and_head_copies(self) -> None:
        build_config, max_context_length = DEEPSEEK_V3_MODEL_FLAVORS["debugmodel"]
        for num_mtp_layers in (1, 2):
            with self.subTest(num_mtp_layers=num_mtp_layers):
                config = build_config(
                    attn_backend="flex",
                    seq_len=max_context_length,
                    num_mtp_layers=num_mtp_layers,
                    parallelism=ParallelismConfig(),
                )
                model = config.build()
                model.init_states()
                state_dict = model.state_dict()
                adapter = DeepSeekV3StateDictAdapter(config, hf_assets_path=None)

                hf_state_dict = adapter.to_hf(state_dict)
                copy_sources = {
                    f"model.layers.{len(config.layers) + mtp_layer}.{suffix}": tt_key
                    for mtp_layer in range(num_mtp_layers)
                    for suffix, tt_key in (
                        ("embed_tokens.weight", "tok_embeddings.weight"),
                        ("shared_head.head.weight", "lm_head.weight"),
                    )
                }
                # Exercise serialization of the shared tensors independently of
                # unrelated expert-weight layout conversions.
                exported = {
                    key: hf_state_dict[key]
                    for key in (
                        "model.embed_tokens.weight",
                        "lm_head.weight",
                        *copy_sources,
                    )
                }
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    dcp.save(
                        exported,
                        storage_writer=HuggingFaceStorageWriter(
                            path=checkpoint_dir,
                            save_distributed=True,
                            enable_consolidation=True,
                        ),
                    )
                    restored = {
                        key: torch.empty_like(value) for key, value in exported.items()
                    }
                    dcp.load(
                        restored,
                        storage_reader=HuggingFaceStorageReader(checkpoint_dir),
                    )
                for key, value in exported.items():
                    torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
                for hf_key, tt_key in copy_sources.items():
                    self.assertNotEqual(
                        hf_state_dict[hf_key].untyped_storage().data_ptr(),
                        state_dict[tt_key].untyped_storage().data_ptr(),
                    )
                    torch.testing.assert_close(
                        hf_state_dict[hf_key], state_dict[tt_key], rtol=0, atol=0
                    )
                self.assertEqual(
                    adapter.from_hf(hf_state_dict).keys(), state_dict.keys()
                )

    def test_to_hf_handles_replicated_grouped_experts(self) -> None:
        build_config, max_context_length = DEEPSEEK_V3_MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        adapter = DeepSeekV3StateDictAdapter(config, hf_assets_path=None)
        mesh = init_device_mesh(
            "cpu",
            (1, 1),
            mesh_dim_names=("replicate", "shard"),
        )
        local_weight = torch.arange(8 * 2 * 2 * 3, dtype=torch.float32).reshape(
            8, 2, 2, 3
        )
        grouped_expert_weight = DTensor.from_local(
            local_weight,
            mesh,
            (Replicate(), Shard(0)),
            run_check=False,
        )

        hf_state_dict = adapter.to_hf(
            {"layers.1.feed_forward.routed_experts.w13.weight": grouped_expert_weight}
        )

        expected_keys = {
            f"model.layers.1.mlp.experts.{expert}.{projection}_proj.weight"
            for expert in range(8)
            for projection in ("gate", "up")
        }
        self.assertEqual(set(hf_state_dict), expected_keys)
        for expert in range(8):
            for index, projection in enumerate(("gate", "up")):
                key = f"model.layers.1.mlp.experts.{expert}.{projection}_proj.weight"
                self.assertIsInstance(hf_state_dict[key], DTensor)
                torch.testing.assert_close(
                    hf_state_dict[key].to_local(),
                    local_weight[expert, index],
                )

    def test_roundtrip_preserves_mtp_expert_placements(self) -> None:
        build_config, _ = DEEPSEEK_V3_MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=128,
            num_mtp_layers=1,
            parallelism=ParallelismConfig(),
        )
        adapter = DeepSeekV3StateDictAdapter(config, hf_assets_path=None)
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))
        local_weight = torch.arange(8 * 2 * 2 * 3, dtype=torch.float32).reshape(
            8, 2, 2, 3
        )
        key = "mtp_layers.0.feed_forward.routed_experts.w13.weight"
        for placement in (Shard(0), Shard(2)):
            with self.subTest(placement=placement):
                weight = DTensor.from_local(
                    local_weight, mesh, (placement,), run_check=False
                )
                restored = adapter.from_hf(adapter.to_hf({key: weight}))
                self.assertIsInstance(restored[key], DTensor)
                self.assertEqual(restored[key].placements, weight.placements)
                torch.testing.assert_close(restored[key], weight, rtol=0, atol=0)


class KimiK3StateDictAdapterTest(unittest.TestCase):
    def test_a_log_padding_is_only_for_quantized_layout(self) -> None:
        a_log = torch.tensor([1.0, 2.0])
        state_dict = {"layers.0.attention.A_log": a_log}
        for quantized_layout in (False, True):
            with self.subTest(quantized_layout=quantized_layout):
                adapter = KimiK3StateDictAdapter(
                    _small_kimi_k3_config(), hf_assets_path=None
                )
                hf_state_dict = adapter.to_hf(
                    state_dict, to_quantized_layout=quantized_layout
                )
                expected = (
                    torch.cat((a_log, torch.zeros(126))) if quantized_layout else a_log
                )
                torch.testing.assert_close(
                    hf_state_dict["language_model.model.layers.0.self_attn.A_log"],
                    expected,
                    rtol=0,
                    atol=0,
                )
                restored = adapter.from_hf(
                    hf_state_dict, is_quantized_layout=quantized_layout
                )
                torch.testing.assert_close(
                    restored["layers.0.attention.A_log"], a_log, rtol=0, atol=0
                )

    def test_first_pipeline_stage_synthesizes_unused_residual_weights(self) -> None:
        adapter = KimiK3StateDictAdapter(_small_kimi_k3_config(), hf_assets_path=None)
        state_dict = {
            "layers.0.ffn_res_norm.weight": torch.arange(64, dtype=torch.bfloat16),
            "layers.0.ffn_res_proj.weight": torch.arange(
                64, dtype=torch.bfloat16
            ).reshape(1, 64),
        }

        hf_state_dict = adapter.to_hf(state_dict)

        prefix = "language_model.model.layers.0"
        torch.testing.assert_close(
            hf_state_dict[f"{prefix}.self_attention_res_norm.weight"],
            torch.ones(64, dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            hf_state_dict[f"{prefix}.self_attention_res_proj.weight"],
            torch.zeros(1, 64, dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )
        restored_state_dict = adapter.from_hf(hf_state_dict)
        self.assertEqual(restored_state_dict.keys(), state_dict.keys())
        for key, value in state_dict.items():
            torch.testing.assert_close(restored_state_dict[key], value, rtol=0, atol=0)

    def test_later_pipeline_stage_omits_layer_zero_residual_weights(self) -> None:
        adapter = KimiK3StateDictAdapter(_small_kimi_k3_config(), hf_assets_path=None)
        state_dict = {
            "layers.1.attention_res_norm.weight": torch.arange(
                64, dtype=torch.bfloat16
            ),
            "layers.1.attention_res_proj.weight": torch.arange(
                64, dtype=torch.bfloat16
            ).reshape(1, 64),
        }

        hf_state_dict = adapter.to_hf(state_dict)

        prefix = "language_model.model.layers.1"
        self.assertEqual(
            set(hf_state_dict),
            {
                f"{prefix}.self_attention_res_norm.weight",
                f"{prefix}.self_attention_res_proj.weight",
            },
        )
        restored_state_dict = adapter.from_hf(hf_state_dict)
        self.assertEqual(restored_state_dict.keys(), state_dict.keys())
        for key, value in state_dict.items():
            torch.testing.assert_close(restored_state_dict[key], value, rtol=0, atol=0)

    def test_quantized_names_round_trip(self) -> None:
        config = _small_kimi_k3_config()
        with torch.device("meta"):
            state_dict = config.build().state_dict()
        adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)

        hf_state_dict = adapter.to_hf(state_dict, to_quantized_layout=True)

        prefix = "language_model.model.layers.1.block_sparse_moe"
        for name in ("w1", "w2", "w3"):
            self.assertIn(f"{prefix}.experts.0.{name}.weight_packed", hf_state_dict)
            self.assertNotIn(f"{prefix}.experts.0.{name}.weight", hf_state_dict)
        self.assertIn(f"{prefix}.shared_experts.gate_proj.weight", hf_state_dict)
        self.assertEqual(
            adapter.to_hf(state_dict).keys(),
            {key.removesuffix("_packed") for key in hf_state_dict},
        )
        self.assertEqual(
            adapter.from_hf(hf_state_dict, is_quantized_layout=True).keys(),
            state_dict.keys(),
        )
        with self.assertRaisesRegex(ValueError, "weight_packed"):
            adapter.from_hf(hf_state_dict)
        with self.assertRaisesRegex(ValueError, r"experts\.0\.w1\.weight'"):
            adapter.from_hf(adapter.to_hf(state_dict), is_quantized_layout=True)

    def test_released_index_maps_to_unquantized_names(self) -> None:
        experts = "language_model.model.layers.1.block_sparse_moe.experts.0"
        weight_map = {
            f"{experts}.w1.weight_packed": "model-00001-of-00002.safetensors",
            f"{experts}.w1.weight_scale": "model-00001-of-00002.safetensors",
            "language_model.model.norm.weight": "model-00002-of-00002.safetensors",
            "vision_tower.patch_embed.proj.weight": "model-00001-of-00002.safetensors",
            "mm_projector.proj.0.weight": "model-00002-of-00002.safetensors",
        }
        config = _small_kimi_k3_config()
        with tempfile.TemporaryDirectory() as hf_assets_path:
            with open(f"{hf_assets_path}/model.safetensors.index.json", "w") as f:
                json.dump({"weight_map": weight_map}, f)
            adapter = KimiK3StateDictAdapter(config, hf_assets_path=hf_assets_path)
        with torch.device("meta"):
            hf_state_dict = adapter.to_hf(config.build().state_dict())

        self.assertEqual(
            adapter.fqn_to_index_mapping_for_save(hf_state_dict),
            {
                f"{experts}.w1.weight": 1,
                "language_model.model.norm.weight": 2,
                "vision_tower.patch_embed.proj.weight": 1,
                "mm_projector.proj.0.weight": 2,
            },
        )

    def test_text_only_export_consolidates_without_vision_entries(self) -> None:
        config = _small_kimi_k3_config()
        config.vision_encoder = None
        hf_key = "language_model.model.norm.weight"
        weight_map = {
            hf_key: "model-00001-of-00002.safetensors",
            "vision_tower.patch_embed.proj.weight": "model-00002-of-00002.safetensors",
            "mm_projector.proj.0.weight": "model-00002-of-00002.safetensors",
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            with open(f"{checkpoint_dir}/model.safetensors.index.json", "w") as f:
                json.dump({"weight_map": weight_map}, f)
            adapter = KimiK3StateDictAdapter(config, hf_assets_path=checkpoint_dir)
            exported = adapter.to_hf(
                {"norm.weight": torch.arange(64, dtype=torch.float32)}
            )
            sharded_dir = f"{checkpoint_dir}/sharded"
            dcp.save(
                exported,
                storage_writer=HuggingFaceStorageWriter(
                    path=sharded_dir,
                    save_distributed=True,
                    fqn_to_index_mapping=adapter.fqn_to_index_mapping,
                    enable_consolidation=False,
                ),
            )
            fqn_to_index_mapping = adapter.fqn_to_index_mapping_for_save(exported)
            consolidate_safetensors_files_on_every_rank(
                input_dir=sharded_dir,
                output_dir=checkpoint_dir,
                fqn_to_index_mapping=fqn_to_index_mapping,
            )
            restored = {hf_key: torch.empty_like(exported[hf_key])}
            dcp.load(restored, storage_reader=HuggingFaceStorageReader(checkpoint_dir))
        self.assertEqual(fqn_to_index_mapping, {hf_key: 1})
        torch.testing.assert_close(restored[hf_key], exported[hf_key], rtol=0, atol=0)


class KimiK3StateDictAdapterUnevenShardingTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 3

    @with_comms
    def test_kda_parameters_round_trip(self) -> None:
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        full_a_log = torch.arange(2, dtype=torch.float32, device=self.device_type)
        sharded_a_log = distribute_tensor(full_a_log, mesh, (Shard(0),))
        full_bias = torch.arange(
            2 * 128,
            dtype=torch.float32,
            device=self.device_type,
        ).reshape(2, 128)
        sharded_bias = distribute_tensor(full_bias, mesh, (Shard(0),))

        for quantized_layout in (False, True):
            with self.subTest(quantized_layout=quantized_layout):
                adapter = KimiK3StateDictAdapter(
                    _small_kimi_k3_config(), hf_assets_path=None
                )
                hf_state_dict = adapter.to_hf(
                    {
                        "layers.0.attention.A_log": sharded_a_log,
                        "layers.0.attention.dt_bias": sharded_bias,
                        "layers.1.attention_res_norm.weight": torch.ones(
                            1, device=self.device_type
                        ),
                        "layers.1.attention_res_proj.weight": torch.ones(
                            1, device=self.device_type
                        ),
                    },
                    to_quantized_layout=quantized_layout,
                )
                hf_a_log = hf_state_dict[
                    "language_model.model.layers.0.self_attn.A_log"
                ]
                self.assertIsInstance(hf_a_log, DTensor)
                self.assertEqual(hf_a_log.placements, (Replicate(),))
                expected_a_log = (
                    torch.cat((full_a_log, torch.zeros(126, device=self.device_type)))
                    if quantized_layout
                    else full_a_log
                )
                torch.testing.assert_close(
                    hf_a_log.full_tensor(), expected_a_log, rtol=0, atol=0
                )
                hf_bias = hf_state_dict[
                    "language_model.model.layers.0.self_attn.dt_bias"
                ]
                self.assertIsInstance(hf_bias, DTensor)
                self.assertEqual(hf_bias.placements, (Replicate(),))
                torch.testing.assert_close(
                    hf_bias.full_tensor(), full_bias.reshape(-1), rtol=0, atol=0
                )

                restored_state_dict = adapter.from_hf(
                    hf_state_dict, is_quantized_layout=quantized_layout
                )
                restored_a_log = restored_state_dict["layers.0.attention.A_log"]
                self.assertIsInstance(restored_a_log, DTensor)
                self.assertEqual(restored_a_log.placements, (Shard(0),))
                torch.testing.assert_close(
                    restored_a_log.full_tensor(), full_a_log, rtol=0, atol=0
                )

                restored_bias = restored_state_dict["layers.0.attention.dt_bias"]
                self.assertIsInstance(restored_bias, DTensor)
                self.assertEqual(restored_bias.placements, (Shard(0),))
                torch.testing.assert_close(
                    restored_bias.full_tensor(), full_bias, rtol=0, atol=0
                )


class DeepSeekV4StateDictAdapterTest(unittest.TestCase):
    def test_full_model_roundtrip_with_optional_mtp(self) -> None:
        for num_mtp_layers in (0, 1, 2):
            with self.subTest(num_mtp_layers=num_mtp_layers):
                config = build_deepseek_v4_model_config(
                    "debugmodel",
                    seq_len=128,
                    n_mtp_layers=num_mtp_layers,
                    parallelism=ParallelismConfig(),
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
                self.assertFalse(
                    any("routed_experts.w13" in key for key in hf_state_dict)
                )
                self.assertTrue(
                    any(
                        ".ffn.experts." in key and key.endswith(".w1.weight")
                        for key in hf_state_dict
                    )
                )
                self.assertTrue(
                    any(
                        ".ffn.experts." in key and key.endswith(".w3.weight")
                        for key in hf_state_dict
                    )
                )
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
                config = build_deepseek_v4_model_config(
                    "debugmodel",
                    seq_len=128,
                    n_mtp_layers=1,
                    parallelism=ParallelismConfig(),
                )
                assert isinstance(config, DeepSeekV4Model.Config)
                model = config.build()
                model.init_states()
                mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))
                for key, value in model.state_dict().items():
                    if ".feed_forward.routed_experts.w13." in key or key.endswith(
                        ".feed_forward.routed_experts.w2.weight"
                    ):
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
        build_config, _ = GPT_OSS_MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex", seq_len=128, parallelism=ParallelismConfig()
        )
        # Keep the real four-layer model structure while making expert tensors
        # small enough for a CPU unit test.
        for layer_config in config.layers:
            assert layer_config.feed_forward is not None
            layer_config.feed_forward.routed_experts.w13.out_features = 16
            layer_config.feed_forward.routed_experts.w2.in_features = 16
            # The debug flavor uses the batch-wise aux loss without expert
            # bias; enable the bias to cover its HF round trip.
            layer_config.feed_forward.load_balance_coeff = 1e-3

        model = config.build()
        model.init_states()
        state_dict = model.state_dict()
        expert_bias_keys = {
            key for key in state_dict if key.endswith(".feed_forward.expert_bias_E")
        }
        self.assertEqual(len(expert_bias_keys), len(config.layers))
        for key in expert_bias_keys:
            state_dict[key].fill_(1.0)

        adapter = GptOssStateDictAdapter(config, hf_assets_path=None)
        hf_state_dicts = {}
        for quantized_layout, weight_suffix in ((False, ""), (True, "_blocks")):
            with self.subTest(quantized_layout=quantized_layout):
                hf_state_dict = adapter.to_hf(
                    state_dict, to_quantized_layout=quantized_layout
                )
                hf_state_dicts[quantized_layout] = hf_state_dict

                hf_expert_names = {
                    f"gate_up_proj{weight_suffix}",
                    "gate_up_proj_bias",
                    f"down_proj{weight_suffix}",
                    "down_proj_bias",
                }
                expected_hf_expert_keys = {
                    f"model.layers.{layer_num}.mlp.experts.{name}"
                    for layer_num in range(len(config.layers))
                    for name in hf_expert_names
                }
                actual_hf_expert_keys = {
                    key for key in hf_state_dict if ".mlp.experts." in key
                }
                self.assertEqual(actual_hf_expert_keys, expected_hf_expert_keys)
                self.assertFalse(any("expert_bias_E" in key for key in hf_state_dict))

                roundtrip_state_dict = adapter.from_hf(
                    hf_state_dict, is_quantized_layout=quantized_layout
                )
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

        # Unquantized HF experts are [in, out]; MXFP4 blocks dequantize to [out, in].
        dim = state_dict["tok_embeddings.weight"].shape[1]
        experts = "model.layers.0.mlp.experts"
        for name in ("gate_up_proj", "down_proj"):
            unquantized = hf_state_dicts[False][f"{experts}.{name}"]
            torch.testing.assert_close(
                hf_state_dicts[True][f"{experts}.{name}_blocks"],
                unquantized.transpose(1, 2),
                rtol=0,
                atol=0,
            )
        self.assertEqual(hf_state_dicts[False][f"{experts}.gate_up_proj"].shape[1], dim)
        self.assertEqual(hf_state_dicts[False][f"{experts}.down_proj"].shape[2], dim)

        # from_hf reads only the names for its quantized flag, so a mismatched
        # flag leaves the expert weights out instead of loading the wrong layout.
        for quantized_layout in (False, True):
            mismatched = adapter.from_hf(
                hf_state_dicts[not quantized_layout],
                is_quantized_layout=quantized_layout,
            )
            self.assertNotIn(
                "layers.0.feed_forward.routed_experts.w13.weight", mismatched
            )
            self.assertNotIn(
                "layers.0.feed_forward.routed_experts.w2.weight", mismatched
            )

    def test_released_index_maps_to_unquantized_names(self) -> None:
        build_config, _ = GPT_OSS_MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex", seq_len=128, parallelism=ParallelismConfig()
        )
        experts = "model.layers.0.mlp.experts"
        weight_map = {
            f"{experts}.gate_up_proj_blocks": "model-00001-of-00002.safetensors",
            f"{experts}.gate_up_proj_scales": "model-00001-of-00002.safetensors",
            f"{experts}.gate_up_proj_bias": "model-00001-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
        }
        with tempfile.TemporaryDirectory() as hf_assets_path:
            with open(f"{hf_assets_path}/model.safetensors.index.json", "w") as f:
                json.dump({"weight_map": weight_map}, f)
            adapter = GptOssStateDictAdapter(config, hf_assets_path=hf_assets_path)
        with torch.device("meta"):
            hf_state_dict = adapter.to_hf(config.build().state_dict())

        self.assertEqual(
            adapter.fqn_to_index_mapping_for_save(hf_state_dict),
            {
                f"{experts}.gate_up_proj": 1,
                f"{experts}.gate_up_proj_bias": 1,
                "model.norm.weight": 2,
            },
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

        build_config, max_context_length = LLAMA3_MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
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
                build_config, max_context_length = LLAMA3_MODEL_FLAVORS["debugmodel"]
                config = build_config(
                    attn_backend="flex",
                    seq_len=max_context_length,
                    parallelism=ParallelismConfig(),
                )
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
