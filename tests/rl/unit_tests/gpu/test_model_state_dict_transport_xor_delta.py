# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that XOR delta pushes leave the generators' prefetch buffers bitwise equal
to whole pushes, over many pushes, with deltas across window boundaries, packed
keys, and the fallbacks to whole reads, over an in-process fake transport."""

import asyncio
import itertools

import pytest
import torch

pytest.importorskip("nvidia.nvcomp")

from torchtitan.rl import model_state_dict_transport as m
from torchtitan.rl.model_state_dict_transport import ModelStateDictTransport


class _FakeRemoteBuffer:
    def __init__(self, data, url):
        self.data = data
        self.url = url

    def slice(self, offset, nbytes):
        assert offset + nbytes <= self.data.numel()
        return _FakeRemoteBuffer(self.data[offset : offset + nbytes], self.url)


class _FakeMemory:
    def __init__(self, tensor, url):
        self.data = tensor.view(-1).view(torch.uint8)
        self.url = url

    def to_remote_buffer(self):
        return _FakeRemoteBuffer(self.data, self.url)

    def to_mutable_view(self, offset, nbytes):
        assert offset + nbytes <= self.data.numel()
        return self.data[offset : offset + nbytes]


class _FakeTransport:
    _urls = itertools.count()

    def __init__(self):
        self.url = str(next(self._urls)).encode()
        self.connected = set()
        self.bytes_read = 0

    def register_memory(self, tensor):
        assert tensor.numel() > 0 and tensor.is_contiguous()
        return _FakeMemory(tensor, self.url)

    def bind(self):
        return self.url

    def connect(self, url):
        self.connected.add(url)

    def read(self, local, remote, *, async_op):
        assert remote.url in self.connected
        assert local.shape == remote.data.shape
        local.copy_(remote.data)
        self.bytes_read += local.numel()


async def _wait_all(works, *, timeout):
    pass


def _index(offset, shape):
    return tuple(slice(o, o + n) for o, n in zip(offset, shape))


@pytest.fixture
def small_windows(monkeypatch):
    """Windows and chunks small enough that every tensor spans several."""
    monkeypatch.setattr(m, "_DELTA_WINDOW_NBYTES", 1 << 16)
    monkeypatch.setattr(m, "_DELTA_CHUNK_NBYTES", 1 << 12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_replicas,resident", [(1, False), (2, True)])
def test_xor_delta_pushes(monkeypatch, small_windows, num_replicas, resident):
    """Trainer FSDP 2 on dim 0 (HSDP with two replicas: each piece staged twice);
    generator TP 2 on dim 1 (packed pieces) for BF16 ``w`` and FP32 ``bias``, and
    split on dim 0 like experts for FP32 ``e``, which loads XOR in place on the
    GPU when resident. Each push changes a few elements of the trainer's FP32
    weights."""
    boxes = {}
    monkeypatch.setattr(
        m, "_global_offset_and_local_shape", lambda tensor: boxes[id(tensor)]
    )
    transports = []

    def new_transport(backend):
        transports.append(_FakeTransport())
        return transports[-1]

    monkeypatch.setattr(m, "new_transport", new_transport)
    monkeypatch.setattr(m, "wait_all", _wait_all)
    config = ModelStateDictTransport.Config(xor_delta_codec="zstd")
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(0)
    shapes = {"w": (512, 256), "bias": (512, 4), "e": (512, 64)}
    # Generator split dim per key.
    split = {"w": 1, "bias": 1, "e": 0}
    full = {
        key: torch.randn(shape, device=device, generator=generator)
        for key, shape in shapes.items()
    }
    dtypes = {"w": torch.bfloat16, "bias": torch.float32, "e": torch.float32}

    trainer_sds = []
    for rank in range(2 * num_replicas):
        sd = {}
        for key, shape in shapes.items():
            offset = (rank % 2 * shape[0] // 2, 0)
            box_shape = (shape[0] // 2, shape[1])
            sd[key] = full[key][_index(offset, box_shape)]
            boxes[id(sd[key])] = (offset, box_shape)
        trainer_sds.append(sd)
    generator_sds = []
    for rank in range(2):
        sd = {}
        for key, shape in shapes.items():
            d = split[key]
            offset = tuple(rank * n // 2 if i == d else 0 for i, n in enumerate(shape))
            box_shape = tuple(n // 2 if i == d else n for i, n in enumerate(shape))
            sd[key] = torch.zeros(box_shape, dtype=dtypes[key], pin_memory=True)
            boxes[id(sd[key])] = (offset, box_shape)
        generator_sds.append(sd)
    rank_layouts = [m.shard_layout(sd) for sd in generator_sds]

    trainers, sources = [], []
    for sd in trainer_sds:
        trainer = ModelStateDictTransport(config)
        sources.append(
            trainer.register_trainer_source(
                sd,
                generator_rank_layouts=rank_layouts,
                transfer_dtype=torch.bfloat16,
                preserve_dtype_keys=frozenset({"bias", "e"}),
            )
        )
        trainers.append(trainer)
    generators = []
    for rank, sd in enumerate(generator_sds):
        transport = ModelStateDictTransport(config)
        transport.connect_generator(
            sd,
            sources=sources,
            rank=rank,
            generator_index=0,
            resident_keys=frozenset({"e"}) if resident else frozenset(),
        )
        generators.append(transport)
    # Load destinations, kept across loads as a resident key's must be.
    loaded_sds = [
        {key: torch.empty_like(t, device=device) for key, t in sd.items()}
        for sd in generator_sds
    ]

    def push_and_pull(trainer_sds):
        for trainer, sd in zip(trainers, trainer_sds):
            trainer.stage(sd)
            torch.cuda.synchronize()
            asyncio.run(trainer.finish_stage())
        read = []
        for transport, generator_transport in zip(
            generators, transports[len(trainers) :]
        ):
            before = generator_transport.bytes_read
            asyncio.run(transport.read())
            read.append(generator_transport.bytes_read - before)
        for rank, (transport, loaded) in enumerate(zip(generators, loaded_sds)):
            transport.copy_prefetched_(loaded)
            for key, tensor in loaded.items():
                d = split[key]
                index = tuple(
                    slice(rank * n, (rank + 1) * n) if i == d else slice(None)
                    for i, n in enumerate(tensor.shape)
                )
                expected = full[key][index]
                torch.testing.assert_close(
                    tensor, expected.to(dtypes[key]), rtol=0, atol=0
                )
        return read

    whole = push_and_pull(trainer_sds)
    for step in range(6):
        # Few changed elements: after the first push, reads are deltas, far
        # smaller than whole reads.
        for key in shapes:
            changed = (
                torch.rand(full[key].shape, device=device, generator=generator) < 0.01
            )
            full[key] += changed * torch.randn(
                full[key].shape, device=device, generator=generator
            )
        read = push_and_pull(trainer_sds)
        assert all(r < w / 4 for r, w in zip(read, whole)), (step, read, whole)

    # Every element changes: the delta outgrows the delta buffer, and generators
    # read the push whole.
    for key in shapes:
        full[key].copy_(
            torch.randn(full[key].shape, device=device, generator=generator)
        )
    read = push_and_pull(trainer_sds)
    assert all(r >= w for r, w in zip(read, whole)), (read, whole)
    # Back to deltas.
    full["w"][0, 0] += 1
    read = push_and_pull(trainer_sds)
    assert all(r < w / 4 for r, w in zip(read, whole)), (read, whole)

    # A generator that missed a push reads the next one whole.
    for trainer, sd in zip(trainers, trainer_sds):
        trainer.stage(sd)
        torch.cuda.synchronize()
        asyncio.run(trainer.finish_stage())
    full["w"][1, 1] += 1
    read = push_and_pull(trainer_sds)
    assert all(r >= w for r, w in zip(read, whole)), (read, whole)
