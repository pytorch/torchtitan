# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import load
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, ReadItem

from torchtitan.models.deepseek_v4_1.checkpoint_storage import (
    BlockScaledHuggingFaceStorageReader,
    BlockScaledLayout,
)


def test_quantized_metadata_and_unaligned_slices(tmp_path):
    fp8 = (
        torch.arange(40 * 64).remainder(50).float().view(40, 64).to(torch.float8_e4m3fn)
    )
    packed = torch.arange(40 * 32).remainder(256).byte().view(40, 32)
    fp8_scales = torch.tensor([[0.5, 2.0], [4.0, 1.0]])
    fp4_scales = torch.arange(80).view(40, 2).float().add(1) / 8
    save_file(
        {
            "linear.weight": fp8,
            "expert.weight": packed,
            "norm.weight": torch.arange(40).float(),
        },
        tmp_path / "weights.safetensors",
    )
    # Scales can reside in a different shard from their weights.
    save_file(
        {"linear.scale": fp8_scales, "expert.scale": fp4_scales},
        tmp_path / "scales.safetensors",
    )
    weight_map = {
        "linear.weight": "weights.safetensors",
        "expert.weight": "weights.safetensors",
        "linear.scale": "scales.safetensors",
        "expert.scale": "scales.safetensors",
        "norm.weight": "weights.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    reader = BlockScaledHuggingFaceStorageReader(
        str(tmp_path),
        layouts={
            "linear.weight": BlockScaledLayout("linear.scale", 32, 32),
            "expert.weight": BlockScaledLayout("expert.scale", 1, 32, True),
        },
    )
    metadata = reader.read_metadata()
    assert metadata.state_dict_metadata["expert.weight"].size == (40, 64)
    assert metadata.state_dict_metadata["expert.weight"].chunks[0].sizes == (40, 64)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    expert = torch.empty(40, 64)
    expert[:, ::2], expert[:, 1::2] = (
        lut[(packed & 15).long()],
        lut[(packed >> 4).long()],
    )
    expert *= fp4_scales.repeat_interleave(32, -1)
    linear = fp8.float() * fp8_scales.repeat_interleave(32, 0)[:40].repeat_interleave(
        32, 1
    )
    state = {
        "linear.weight": torch.empty(40, 64),
        "expert.weight": torch.empty(40, 64),
        "norm.weight": torch.empty(40),
    }
    load(state, storage_reader=reader, no_dist=True)
    torch.testing.assert_close(state["linear.weight"], linear, rtol=0, atol=0)
    torch.testing.assert_close(state["expert.weight"], expert, rtol=0, atol=0)
    torch.testing.assert_close(
        state["norm.weight"], torch.arange(40).float(), rtol=0, atol=0
    )

    class Planner:
        def __init__(self):
            self.tensor = torch.empty(7, 41)

        def resolve_tensor(self, req):
            return self.tensor

        def commit_tensor(self, req, tensor):
            pass

    for key, expected in (("linear.weight", linear), ("expert.weight", expert)):
        planner = Planner()
        request = ReadItem(
            type=LoadItemType.TENSOR,
            dest_index=MetadataIndex(key),
            dest_offsets=torch.Size((0, 0)),
            storage_index=MetadataIndex(key),
            storage_offsets=torch.Size((30, 7)),
            lengths=torch.Size((7, 41)),
        )
        reader.read_data(LoadPlan([request]), planner).wait()
        torch.testing.assert_close(
            planner.tensor, expected[30:37, 7:48], rtol=0, atol=0
        )
