# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from torchtitan.rl import model_state_dict_delta
from torchtitan.rl.model_state_dict_delta import (
    _shard_to_destination,
    ModelStateDictDeltaPublisher,
    ModelStateDictDeltaReader,
)


class _FakeTorchStore:
    """TorchStore state dict API for plain tensors on one publisher."""

    def __init__(self) -> None:
        self.stored: dict[str, dict[str, torch.Tensor]] = {}
        self.full_reads: list[set[str]] = []
        self.fail_full_reads = False

    async def put_state_dict(
        self, state_dict, key, *, transfer_dtype=None, preserve_dtype_keys=frozenset()
    ):
        self.stored[key] = {
            name: (
                value.to(transfer_dtype)
                if transfer_dtype is not None
                and name not in preserve_dtype_keys
                and value.is_floating_point()
                else value.clone()
            )
            for name, value in state_dict.items()
        }

    async def get_state_dict(self, key, user_state_dict, strict=True):
        if key == "model_state_dict":
            if self.fail_full_reads:
                raise RuntimeError("full read failed")
            self.full_reads.append(set(user_state_dict))
        for name, value in user_state_dict.items():
            value.copy_(self.stored[key][name])
        return user_state_dict

    async def get_state_dict_layout(self, key):
        return {
            name: [SimpleNamespace(offsets=[0] * value.ndim, local_shape=value.shape)]
            for name, value in self.stored[key].items()
        }


def _pull(store, publisher, reader, trainer_state_dict, generator_state_dict):
    async def run():
        with patch.multiple(
            model_state_dict_delta.ts,
            put_state_dict=store.put_state_dict,
            get_state_dict=store.get_state_dict,
            get_state_dict_layout=store.get_state_dict_layout,
        ):
            await publisher.put(trainer_state_dict)
            await reader.pull(generator_state_dict)

    asyncio.run(run())


# Several chunks per tensor, so changes and overflow carry across chunks.
@patch.object(model_state_dict_delta, "_CHUNK_NUMEL", 16)
def test_generator_tracks_trainer_bitwise_and_falls_back_to_full_reads():
    torch.manual_seed(0)
    trainer_state_dict = {
        "weight": torch.randn(16, 8),
        "norm.scale": torch.randn(8),
    }
    generator_state_dict = {
        "weight": torch.zeros(16, 8, dtype=torch.bfloat16),
        "norm.scale": torch.zeros(8),
    }
    store = _FakeTorchStore()
    publisher = ModelStateDictDeltaPublisher(
        trainer_state_dict,
        key="model_state_dict",
        max_fraction=0.25,
        transfer_dtype=torch.bfloat16,
        preserve_dtype_keys=frozenset({"norm.scale"}),
    )
    reader = ModelStateDictDeltaReader("model_state_dict")

    def check(full_reads):
        _pull(store, publisher, reader, trainer_state_dict, generator_state_dict)
        assert (store.full_reads.pop() if store.full_reads else set()) == full_reads
        assert torch.equal(
            generator_state_dict["weight"], trainer_state_dict["weight"].bfloat16()
        )
        assert torch.equal(
            generator_state_dict["norm.scale"], trainer_state_dict["norm.scale"]
        )

    # The first version is read in full.
    check({"weight", "norm.scale"})
    # A few changed elements are applied from the rows.
    trainer_state_dict["weight"][3, 2] += 1
    trainer_state_dict["weight"][15, 7] -= 1
    check(set())
    # A change under the wire precision is not sent.
    trainer_state_dict["weight"][0, 0] *= 1 + 1e-6
    check(set())
    # More changed elements than fit in a row read the tensor in full.
    trainer_state_dict["weight"] += 1
    trainer_state_dict["norm.scale"][0] += 1
    check({"weight"})
    # A restarted trainer draws a new epoch, so its versions are not trusted.
    publisher = ModelStateDictDeltaPublisher(
        trainer_state_dict,
        key="model_state_dict",
        max_fraction=0.25,
        transfer_dtype=torch.bfloat16,
        preserve_dtype_keys=frozenset({"norm.scale"}),
    )
    check({"weight", "norm.scale"})
    trainer_state_dict["weight"][2, 2] += 1
    check(set())
    # A failed full read is not trusted as the base of the next delta.
    trainer_state_dict["weight"] += 1
    store.fail_full_reads = True
    with pytest.raises(RuntimeError, match="full read failed"):
        _pull(store, publisher, reader, trainer_state_dict, generator_state_dict)
    store.fail_full_reads = False
    trainer_state_dict["weight"][1, 1] += 1
    check({"weight"})


def test_shard_to_destination_maps_other_shardings():
    # A [4, 6] tensor published as two row shards, read by a column shard.
    global_tensor = torch.arange(24).reshape(4, 6)
    destination = torch.zeros(4, 3, dtype=torch.long)
    for shard_offsets in [(0, 0), (2, 0)]:
        shard = global_tensor[shard_offsets[0] : shard_offsets[0] + 2]
        indices = torch.arange(shard.numel())
        indices, values = _shard_to_destination(
            indices, shard.reshape(-1), shard_offsets, (2, 6), (0, 3), (4, 3)
        )
        destination.view(-1)[indices] = values
    assert torch.equal(destination, global_tensor[:, 3:])
