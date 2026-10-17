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
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torchtitan.models.kimi_k3 import model_registry as kimi_k3_model_registry
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter


def _run_kimi_vision_qkv_roundtrip(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        config = kimi_k3_model_registry("debugmodel", enable_sp=False, seq_len=128)
        adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))
        full = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(24, 8)
        keys = tuple(f"vision_encoder.layers.0.attn.w{p}.weight" for p in "qkv")
        state = {
            key: DTensor.from_local(
                part.chunk(2, dim=0)[rank].clone(), mesh, (Shard(0),)
            )
            for key, part in zip(keys, full.chunk(3), strict=True)
        }
        # The adapter uses layer 1 to synthesize unused layer-0 placeholders.
        state["layers.1.attention_res_norm.weight"] = torch.ones(8)
        state["layers.1.attention_res_proj.weight"] = torch.zeros(1, 8)
        hf_key = "vision_tower.encoder.blocks.0.wqkv.weight"
        hf_state = {hf_key: adapter.to_hf(state)[hf_key]}
        # Do not let cat select column shards, requiring a row/column all-to-all
        # when loaded projections are copied back to FSDP's native row shards.
        assert hf_state[hf_key].placements == (Replicate(),)
        torch.testing.assert_close(hf_state[hf_key].to_local(), full, rtol=0, atol=0)

        loaded = full + 1000
        checkpoint = Path(rendezvous).parent
        if rank == 0:
            save_file({hf_key: loaded}, checkpoint / "model.safetensors")
        dist.barrier()
        dcp.load(hf_state, storage_reader=HuggingFaceStorageReader(str(checkpoint)))
        # Also cover an independently supplied column-sharded HF checkpoint.
        column_sharded = DTensor.from_local(
            loaded.chunk(2, dim=1)[rank].contiguous(), mesh, (Shard(1),)
        )
        for hf_value in (hf_state[hf_key], column_sharded):
            restored = adapter.from_hf({hf_key: hf_value})
            for key, expected in zip(keys, loaded.chunk(3), strict=True):
                assert restored[key].placements == (Replicate(),)
                torch.testing.assert_close(
                    restored[key].to_local(), expected, rtol=0, atol=0
                )
                state[key].copy_(restored[key])
                torch.testing.assert_close(
                    state[key].to_local(),
                    expected.chunk(2, dim=0)[rank],
                    rtol=0,
                    atol=0,
                )
    finally:
        dist.destroy_process_group()


class KimiK3VisionCheckpointTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_gloo_available(), "Requires Gloo")
    def test_vision_qkv_checkpoint_avoids_row_column_redistribution(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _run_kimi_vision_qkv_roundtrip,
                args=(f"{directory}/rendezvous",),
                nprocs=2,
                join=True,
            )
