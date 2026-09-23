# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from safetensors.torch import save_file
from scripts.checkpoint_conversion.validate_kimi_k3_mxfp4_checkpoint import (
    validate_checkpoint,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torchtitan.components.checkpointer.hf_storage import (
    HuggingFaceStorageReaderWithViews,
    LogicalPrefixSpec,
)
from torchtitan.models.kimi_k3 import model_registry as kimi_k3_model_registry
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter


def _run_kimi_uneven_dt_bias_roundtrip(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        config = kimi_k3_model_registry("debugmodel", seq_len=128)
        delta_config = config.layers[1].delta_attention
        assert delta_config is not None
        delta_config.num_heads = 1
        adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)

        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
        full_dt_bias = torch.arange(128, dtype=torch.bfloat16).reshape(1, 128)
        local_dt_bias = full_dt_bias if rank == 0 else full_dt_bias[:0]
        sharded_dt_bias = DTensor.from_local(
            local_dt_bias,
            mesh,
            (Shard(0),),
            run_check=False,
            shape=torch.Size((1, 128)),
            stride=(128, 1),
        )

        hf_state_dict = adapter.to_hf(
            {
                "layers.1.delta_attention.dt_bias": sharded_dt_bias,
                "layers.1.attention_res_norm.weight": torch.ones(1),
                "layers.1.attention_res_proj.weight": torch.ones(1),
            }
        )
        hf_dt_bias = hf_state_dict["language_model.model.layers.1.self_attn.dt_bias"]
        assert isinstance(hf_dt_bias, DTensor)
        assert hf_dt_bias.shape == torch.Size((128,))
        assert hf_dt_bias.placements == (Shard(0),)

        # Load a new bias and a padded A_log through the real two-rank planner.
        checkpoint = Path(rendezvous).parent
        loaded_bias = full_dt_bias + 100
        if rank == 0:
            save_file(
                {
                    "language_model.model.layers.1.self_attn.dt_bias": loaded_bias.flatten(),
                    "A_log": torch.cat((torch.arange(96).float(), torch.zeros(32))),
                },
                checkpoint / "model.safetensors",
            )
        dist.barrier()
        a_log = DTensor.from_local(
            torch.empty(48),
            mesh,
            (Shard(0),),
            run_check=False,
            shape=torch.Size((96,)),
            stride=(1,),
        )
        dcp.load(
            {
                "language_model.model.layers.1.self_attn.dt_bias": hf_dt_bias,
                "A_log": a_log,
            },
            storage_reader=HuggingFaceStorageReaderWithViews(
                str(checkpoint), logical_prefixes={"A_log": LogicalPrefixSpec(96, 128)}
            ),
        )
        torch.testing.assert_close(
            a_log.to_local(),
            torch.arange(rank * 48, (rank + 1) * 48).float(),
            rtol=0,
            atol=0,
        )

        restored = adapter.from_hf(hf_state_dict)["layers.1.delta_attention.dt_bias"]
        assert isinstance(restored, DTensor)
        assert restored.shape == torch.Size((1, 128))
        assert restored.placements == (Shard(0),)
        assert restored.device_mesh == mesh
        assert restored.to_local().shape == local_dt_bias.shape
        torch.testing.assert_close(
            restored.to_local(),
            loaded_bias if rank == 0 else loaded_bias[:0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(restored.full_tensor(), loaded_bias, rtol=0, atol=0)
        # Other placement layouts are rejected before any collective reshape.
        invalid = DTensor.from_local(
            full_dt_bias[:, :64],
            mesh,
            (Shard(1),),
            run_check=False,
            shape=torch.Size((1, 128)),
            stride=(128, 1),
        )
        try:
            adapter._reshape_dt_bias(invalid, (-1,))
        except ValueError as error:
            assert "supports only" in str(error)
        else:
            raise AssertionError("Expected unsupported placement rejection")
    finally:
        dist.destroy_process_group()


class KimiK3CheckpointTest(unittest.TestCase):
    def test_preflight_checks_nonpacked_shapes_against_model(self):
        key = "language_model.model.layers.1.self_attn.dt_bias"
        with tempfile.TemporaryDirectory() as directory:
            save_file({key: torch.zeros(1)}, Path(directory) / "model.safetensors")
            with self.assertRaisesRegex(
                ValueError, "mismatches=.*dt_bias.*checkpoint.*model"
            ):
                validate_checkpoint(
                    Path(directory), model_flavor="debugmodel", from_quantized=False
                )

    @unittest.skipUnless(dist.is_gloo_available(), "Requires Gloo")
    def test_dt_bias_with_fewer_heads_than_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _run_kimi_uneven_dt_bias_roundtrip,
                args=(f"{directory}/rendezvous",),
                nprocs=2,
                join=True,
            )

    def test_unquantized_reader_normalizes_release_padding(self):
        config = kimi_k3_model_registry("debugmodel", seq_len=128)
        config.layers[1].delta_attention.num_heads = 96
        adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)
        key = "language_model.model.layers.1.self_attn.A_log"
        prefix = torch.arange(96, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {key: torch.cat((prefix, torch.zeros(32)))},
                Path(directory) / "model.safetensors",
            )
            reader = adapter.get_hf_storage_reader(directory, from_quantized=False)
            self.assertIsInstance(reader, HuggingFaceStorageReaderWithViews)
            self.assertIsNone(reader.spec)
            destination = {key: torch.empty_like(prefix)}
            dcp.load(destination, storage_reader=reader)
        restored = adapter.from_hf(destination)
        torch.testing.assert_close(
            restored["layers.1.delta_attention.A_log"], prefix, rtol=0, atol=0
        )
        # Export the model's canonical vector, never the release's padded shape.
        restored["layers.1.attention_res_norm.weight"] = torch.ones(1)
        restored["layers.1.attention_res_proj.weight"] = torch.ones(1)
        self.assertEqual(adapter.to_hf(restored)[key].shape, (96,))


if __name__ == "__main__":
    unittest.main()
