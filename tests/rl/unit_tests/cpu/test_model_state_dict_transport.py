# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks that staging, the read plan, the reads and the load move every
element of a generator shard from the right trainer rank to the right place,
across trainer FSDP/HSDP and generator TP shardings, over an in-process fake
transport."""

import asyncio
import itertools
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Partial, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.fake_pg import FakeStore

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
        self.registered = []
        self.reads = []

    def register_memory(self, tensor):
        assert tensor.numel() > 0 and tensor.is_contiguous()
        self.registered.append(tensor)
        return _FakeMemory(tensor, self.url)

    def bind(self):
        return self.url

    def connect(self, url):
        self.connected.add(url)

    def read(self, local, remote, *, async_op):
        assert remote.url in self.connected
        assert local.shape == remote.data.shape
        self.reads.append((local, remote))
        local.copy_(remote.data)
        return _FakeWork() if async_op else 0


class _FakeWork:
    def wait(self, timeout=None):
        return True


async def _wait_all(works, *, timeout):
    pass


def _boxes(shape, num_shards, dim):
    """Offset and shape of each even shard of ``shape`` along ``dim``."""
    size = shape[dim] // num_shards
    return [
        (
            tuple(i * size if d == dim else 0 for d in range(len(shape))),
            tuple(size if d == dim else n for d, n in enumerate(shape)),
        )
        for i in range(num_shards)
    ]


def _index(offset, shape):
    return tuple(slice(o, o + n) for o, n in zip(offset, shape))


def _transfer(
    monkeypatch,
    trainer_shards,
    generator_shards,
    *,
    generator_index=0,
    transfer_dtype=None,
    preserve_dtype_keys=frozenset(),
    update=None,
):
    """Register and stage each trainer rank's ``{key: (local tensor, box)}``, then
    connect, read and load each generator rank's ``{key: (box, dtype)}``; returns
    the generator ranks' loaded state dicts. With ``update``, which changes the
    trainer's local tensors in place, push and load a second time with delta
    sync."""
    boxes = {}
    monkeypatch.setattr(
        m,
        "_global_offset_and_local_shape",
        lambda tensor: boxes[id(tensor)],
    )
    config = ModelStateDictTransport.Config(
        backend="nixl",
        sync_mode="full" if update is None else "delta",
        read_timeout_s=5.0,
    )

    def new_transport(backend):
        assert backend == config.backend
        return _FakeTransport()

    async def wait_all(works, *, timeout):
        assert timeout == config.read_timeout_s

    monkeypatch.setattr(m, "new_transport", new_transport)
    monkeypatch.setattr(m, "wait_all", wait_all)

    generator_sds = []
    for shards in generator_shards:
        sd = {}
        for key, (box, dtype) in shards.items():
            sd[key] = torch.zeros(box[1], dtype=dtype)
            boxes[id(sd[key])] = box
        generator_sds.append(sd)
    rank_layouts = [m.shard_layout(sd) for sd in generator_sds]

    sources, trainer_metadata, trainers = [], [], []
    for shards in trainer_shards:
        sd = {key: local for key, (local, _) in shards.items()}
        for key, (local, box) in shards.items():
            boxes[id(local)] = box
        trainer = config.build()
        sources.append(
            trainer.register_trainer_source(
                sd,
                generator_rank_layouts=rank_layouts,
                transfer_dtype=transfer_dtype,
                preserve_dtype_keys=preserve_dtype_keys,
            )
        )
        trainer.stage(sd)
        trainer_metadata.append(trainer.finish_stage())
        trainers.append(trainer)

    generators = []
    for rank, sd in enumerate(generator_sds):
        generator = config.build()
        generator.connect_generator(
            sd,
            load_groups=[list(sd)],
            sources=sources,
            rank=rank,
            generator_index=generator_index,
        )
        generators.append(generator)

    def pull() -> list[dict[str, torch.Tensor]]:
        loaded_sds = []
        for generator, sd in zip(generators, generator_sds):
            asyncio.run(generator.read(tuple(trainer_metadata)))
            loaded = {key: torch.full_like(tensor, -1) for key, tensor in sd.items()}
            generator.copy_prefetched_(loaded)
            loaded_sds.append(loaded)
        return loaded_sds

    loaded_sds = pull()
    if update is None:
        return loaded_sds
    update()
    trainer_metadata = []
    for trainer, shards in zip(trainers, trainer_shards):
        trainer.stage({key: local for key, (local, _) in shards.items()})
        trainer_metadata.append(trainer.finish_stage())
    assert all(metadata.delta_records for metadata in trainer_metadata)
    return pull()


def _transfer_w(monkeypatch, trainer_locals, generator_boxes, *, dtype, **kwargs):
    """``_transfer`` of one key ``w`` from (local tensor, box) pairs to boxes."""
    sds = _transfer(
        monkeypatch,
        [{"w": local_box} for local_box in trainer_locals],
        [{"w": (box, dtype)} for box in generator_boxes],
        transfer_dtype=dtype,
        **kwargs,
    )
    return [sd["w"] for sd in sds]


@pytest.mark.parametrize("generator_tp,generator_dim", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("generator_index", [0, 1])
def test_hsdp_to_tp(monkeypatch, generator_tp, generator_dim, generator_index):
    full = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6)
    # HSDP 3x4: three replicas of an FSDP-4 row sharding. The replicas hold
    # different values only to show which one each generator rank reads.
    trainer_locals = [
        (full[_index(*box)] + 100 * replica, box)
        for replica in range(3)
        for box in _boxes(full.shape, 4, 0)
    ]
    generator_boxes = _boxes(full.shape, generator_tp, generator_dim)
    outputs = _transfer_w(
        monkeypatch,
        trainer_locals,
        generator_boxes,
        dtype=torch.bfloat16,
        generator_index=generator_index,
    )
    for rank, (out, box) in enumerate(zip(outputs, generator_boxes)):
        # Generator ranks take the replicas round-robin, across generators too.
        replica = (generator_index * generator_tp + rank) % 3
        expected = (full[_index(*box)] + 100 * replica).to(torch.bfloat16)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("generator_tp,generator_dim", [(1, 1), (2, 1), (2, 2)])
def test_stacked_pieces(monkeypatch, generator_tp, generator_dim):
    # A stacked [2, F, D] weight sharded on F: each [2, F/4, ...] piece spans
    # both stacked linears, so it is read packed and unpacked on load.
    full = torch.arange(2 * 8 * 6, dtype=torch.float32).reshape(2, 8, 6)
    generator_boxes = _boxes(full.shape, generator_tp, generator_dim)
    outputs = _transfer_w(
        monkeypatch,
        [(full[_index(*box)], box) for box in _boxes(full.shape, 4, 1)],
        generator_boxes,
        dtype=torch.float32,
    )
    for out, box in zip(outputs, generator_boxes):
        torch.testing.assert_close(out, full[_index(*box)], rtol=0, atol=0)


def test_replicated_pieces(monkeypatch):
    # Generator ranks that receive the same piece share one staged copy.
    full = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6)
    trainer_box = _boxes(full.shape, 4, 0)[1]
    shard = m.TensorShard((0, 0), (8, 6), torch.float32)
    pieces, nbytes = m._staged_pieces({"w": trainer_box}, [{"w": shard}] * 2)
    assert pieces[0] == pieces[1] and nbytes == 2 * 6 * 4
    outputs = _transfer_w(
        monkeypatch,
        [(full[_index(*box)], box) for box in _boxes(full.shape, 4, 0)],
        [((0, 0), (8, 6))] * 2,
        dtype=torch.float32,
    )
    for out in outputs:
        torch.testing.assert_close(out, full, rtol=0, atol=0)


def test_missing_coverage(monkeypatch):
    full = torch.zeros(8, 6)
    with pytest.raises(ValueError, match="covers"):
        _transfer_w(
            monkeypatch,
            [(full[_index(*box)], box) for box in _boxes(full.shape, 4, 0)[:3]],
            [((0, 0), (8, 6))],
            dtype=torch.float32,
        )


def test_overlapping_pieces(monkeypatch):
    # Rows 0-1, 1-2 and 4-7: 8 rows in total, but row 3 is missing.
    full = torch.zeros(8, 6)
    boxes = [((0, 0), (2, 6)), ((1, 0), (2, 6)), ((4, 0), (4, 6))]
    with pytest.raises(ValueError, match="overlapping"):
        _transfer_w(
            monkeypatch,
            [(full[_index(*box)], box) for box in boxes],
            [((0, 0), (8, 6))],
            dtype=torch.float32,
        )


def test_mixed_dtypes(monkeypatch):
    # An odd-sized bf16 key, then a key kept in fp32: the second piece starts at
    # an aligned offset and keeps its dtype.
    a = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4) / 3
    (out,) = _transfer(
        monkeypatch,
        [
            {
                "a": (a[i : i + 1], ((i, 0), (1, 3))),
                "b": (b[i : i + 1], ((i, 0), (1, 4))),
            }
            for i in range(2)
        ],
        [
            {
                "a": (((0, 0), (2, 3)), torch.bfloat16),
                "b": (((0, 0), (2, 4)), torch.float32),
            }
        ],
        transfer_dtype=torch.bfloat16,
        preserve_dtype_keys=frozenset({"b"}),
    )
    torch.testing.assert_close(out["a"], a.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(out["b"], b, rtol=0, atol=0)


def test_dtype_mismatch(monkeypatch):
    w = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="receives"):
        _transfer(
            monkeypatch,
            [{"w": (w, ((0, 0), (2, 3)))}],
            [{"w": (((0, 0), (2, 3)), torch.float32)}],
            transfer_dtype=torch.bfloat16,
        )


def test_no_reads(monkeypatch):
    # A generator rank whose shards are all empty has nothing to read.
    (out,) = _transfer(
        monkeypatch,
        [{"w": (torch.zeros(2, 3), ((0, 0), (2, 3)))}],
        [{"w": (((0, 0), (0, 3)), torch.float32)}],
    )
    assert out["w"].shape == (0, 3)


def test_full_sync_is_default(monkeypatch):
    config = ModelStateDictTransport.Config(backend="nixl", read_timeout_s=5.0)
    monkeypatch.setattr(m, "new_transport", lambda backend: _FakeTransport())
    monkeypatch.setattr(m, "wait_all", _wait_all)

    initial = {"w": torch.arange(8, dtype=torch.float32)}
    prefetched = {"w": torch.zeros_like(initial["w"])}
    layouts = [m.shard_layout(prefetched)]
    trainer = config.build()
    assert type(trainer) is ModelStateDictTransport
    delta_config = ModelStateDictTransport.Config(backend="nixl", sync_mode="delta")
    assert type(delta_config.build()) is m.ModelStateDictDeltaTransport
    # Constructing the full-sync class for delta sync would silently read whole.
    with pytest.raises(ValueError, match="Config.build"):
        ModelStateDictTransport(delta_config)
    source = trainer.register_trainer_source(
        initial,
        generator_rank_layouts=layouts,
        transfer_dtype=None,
        preserve_dtype_keys=frozenset(),
    )
    assert len(trainer._transport.registered) == 1

    trainer.stage(initial)
    metadata = trainer.finish_stage()
    generator = config.build()
    generator.connect_generator(
        prefetched,
        load_groups=[list(prefetched)],
        sources=[source],
        rank=0,
        generator_index=0,
    )
    asyncio.run(generator.read((metadata,)))

    target = {"w": initial["w"] + 10}
    trainer.stage(target)
    metadata = trainer.finish_stage()
    asyncio.run(generator.read((metadata,)))
    assert len(generator._transport.reads) == 2
    torch.testing.assert_close(prefetched["w"], target["w"], rtol=0, atol=0)


@pytest.mark.parametrize("dim", [0, 1])
def test_delta_loads_changes(monkeypatch, dim):
    """A delta load XORs the changes into a key read in place (row pieces) and
    into a packed key (column pieces), with pieces spanning windows."""
    monkeypatch.setattr(m, "_DELTA_WINDOW_NBYTES", 32)
    monkeypatch.setattr(m, "_DELTA_CAPACITY_DIVISOR", 1)
    full = torch.arange(64, dtype=torch.bfloat16).view(8, 8)
    boxes = _boxes(full.shape, 2, dim)
    trainer_locals = [full[_index(*box)].clone() for box in boxes]

    def update():
        trainer_locals[0].view(-1)[[0, 5, 20]] += 1
        trainer_locals[1].view(-1)[-1] -= 1

    (loaded,) = _transfer(
        monkeypatch,
        [{"w": (local, box)} for local, box in zip(trainer_locals, boxes)],
        [{"w": (((0, 0), (8, 8)), torch.bfloat16)}],
        update=update,
    )
    expected = torch.empty_like(full)
    for local, box in zip(trainer_locals, boxes):
        expected[_index(*box)] = local
    torch.testing.assert_close(loaded["w"], expected, rtol=0, atol=0)


def _load(generator, prefetched):
    """Load the prefetched model state dict into a new state dict."""
    loaded = {key: torch.full_like(value, -1) for key, value in prefetched.items()}
    generator.copy_prefetched_(loaded)
    return loaded


def test_delta_reads_only_changed_pieces(monkeypatch):
    config = ModelStateDictTransport.Config(
        backend="nixl", sync_mode="delta", read_timeout_s=5.0
    )
    monkeypatch.setattr(m, "new_transport", lambda backend: _FakeTransport())
    monkeypatch.setattr(m, "wait_all", _wait_all)
    # Each piece below can send every word: the records of pieces this small are
    # mostly zstd frame overhead.
    monkeypatch.setattr(m, "_DELTA_CAPACITY_DIVISOR", 1)
    initial = {
        "unchanged": torch.arange(128, dtype=torch.bfloat16),
        "changed": torch.arange(128, dtype=torch.bfloat16) + 200,
        "many_changes": torch.arange(128, dtype=torch.bfloat16) + 400,
    }
    prefetched = {key: torch.zeros_like(value) for key, value in initial.items()}
    layouts = [m.shard_layout(prefetched)]

    trainer = config.build()
    source = trainer.register_trainer_source(
        initial,
        generator_rank_layouts=layouts,
        transfer_dtype=None,
        preserve_dtype_keys=frozenset(),
    )
    # One dense staging buffer and one delta buffer.
    assert len(trainer._transport.registered) == 2
    trainer.stage(initial)
    metadata = trainer.finish_stage()
    assert metadata.delta_records is None

    generator = config.build()
    generator.connect_generator(
        prefetched,
        load_groups=[list(prefetched)],
        sources=[source],
        rank=0,
        generator_index=0,
    )
    # The prefetch buffers and the buffer records are received in.
    assert len(generator._transport.registered) == len(prefetched) + 1
    reads = generator._transport.reads
    asyncio.run(generator.read((metadata,)))
    assert len(reads) == 3
    for key in initial:
        torch.testing.assert_close(prefetched[key], initial[key], rtol=0, atol=0)

    target = {key: value.clone() for key, value in initial.items()}
    target["changed"][3] += 1
    target["many_changes"][:125].neg_()
    trainer.stage(target)
    metadata = trainer.finish_stage()
    assert metadata.push == 1
    # The unchanged piece is not sent.
    assert len(metadata.delta_records) == 2

    asyncio.run(generator.read((metadata,)))
    # The two changed pieces' records are adjacent in the delta buffer: one read.
    assert len(reads) == 3 + 1
    loaded = _load(generator, prefetched)
    for key in target:
        torch.testing.assert_close(loaded[key], target[key], rtol=0, atol=0)

    # The trainer's staging buffer and the generator's prefetch buffers both
    # advance, so the following push is relative to `target`.
    next_target = {key: value.clone() for key, value in target.items()}
    next_target["changed"][17] += 2
    next_target["changed"][100] -= 3
    trainer.stage(next_target)
    metadata = trainer.finish_stage()
    asyncio.run(generator.read((metadata,)))
    assert len(reads) == 3 + 1 + 1
    # The read waited for the loaded changes to reach the prefetch buffers.
    for key in target:
        torch.testing.assert_close(prefetched[key], target[key], rtol=0, atol=0)

    # The next read first applies the changes of a read that was not loaded.
    last_target = {key: value.clone() for key, value in next_target.items()}
    last_target["many_changes"][3] += 64
    trainer.stage(last_target)
    metadata = trainer.finish_stage()
    asyncio.run(generator.read((metadata,)))
    assert len(reads) == 3 + 1 + 1 + 1
    for key in next_target:
        torch.testing.assert_close(prefetched[key], next_target[key], rtol=0, atol=0)
    loaded = _load(generator, prefetched)
    for key in last_target:
        torch.testing.assert_close(loaded[key], last_target[key], rtol=0, atol=0)

    # A generator that missed a push reads the next one whole.
    trainer.stage(target)
    trainer.finish_stage()
    trainer.stage(initial)
    metadata = trainer.finish_stage()
    asyncio.run(generator.read((metadata,)))
    assert len(reads) == 3 + 1 + 1 + 1 + 3
    for key in initial:
        torch.testing.assert_close(prefetched[key], initial[key], rtol=0, atol=0)


def test_delta_reads_whole_past_capacity(monkeypatch):
    """A window with more changed words than it sends makes the push read whole."""
    config = ModelStateDictTransport.Config(
        backend="nixl", sync_mode="delta", read_timeout_s=5.0
    )
    monkeypatch.setattr(m, "new_transport", lambda backend: _FakeTransport())
    monkeypatch.setattr(m, "wait_all", _wait_all)
    # One window of 128 words, which sends at most 8.
    initial = {"w": torch.arange(128, dtype=torch.bfloat16)}
    prefetched = {"w": torch.zeros_like(initial["w"])}
    trainer = config.build()
    source = trainer.register_trainer_source(
        initial,
        generator_rank_layouts=[m.shard_layout(prefetched)],
        transfer_dtype=None,
        preserve_dtype_keys=frozenset(),
    )
    trainer.stage(initial)
    metadata = trainer.finish_stage()
    generator = config.build()
    generator.connect_generator(
        prefetched,
        load_groups=[list(prefetched)],
        sources=[source],
        rank=0,
        generator_index=0,
    )
    asyncio.run(generator.read((metadata,)))
    target = {"w": initial["w"].clone()}
    target["w"][:9] += 1
    trainer.stage(target)
    metadata = trainer.finish_stage()
    assert metadata.delta_records is None
    asyncio.run(generator.read((metadata,)))
    torch.testing.assert_close(prefetched["w"], target["w"], rtol=0, atol=0)
    torch.testing.assert_close(_load(generator, prefetched)["w"], target["w"])


def test_delta_pieces_span_windows(monkeypatch):
    """A window holding several pieces, and a piece spanning windows, of mixed
    dtypes, each sent as one record."""
    config = ModelStateDictTransport.Config(
        backend="nixl", sync_mode="delta", read_timeout_s=5.0
    )
    monkeypatch.setattr(m, "new_transport", lambda backend: _FakeTransport())
    monkeypatch.setattr(m, "wait_all", _wait_all)
    monkeypatch.setattr(m, "_DELTA_WINDOW_NBYTES", 160)
    monkeypatch.setattr(m, "_DELTA_CAPACITY_DIVISOR", 1)
    initial = {
        "a": torch.arange(40, dtype=torch.bfloat16),
        "b": torch.arange(100, dtype=torch.float32),
        "c": torch.arange(24, dtype=torch.bfloat16),
    }
    prefetched = {key: torch.zeros_like(value) for key, value in initial.items()}
    trainer = config.build()
    source = trainer.register_trainer_source(
        initial,
        generator_rank_layouts=[m.shard_layout(prefetched)],
        transfer_dtype=None,
        preserve_dtype_keys=frozenset(),
    )
    trainer.stage(initial)
    metadata = trainer.finish_stage()
    generator = config.build()
    generator.connect_generator(
        prefetched,
        load_groups=[list(prefetched)],
        sources=[source],
        rank=0,
        generator_index=0,
    )
    asyncio.run(generator.read((metadata,)))
    target = {key: value.clone() for key, value in initial.items()}
    target["a"][39] += 1
    target["b"][[0, 50, 99]] += 0.5
    target["c"][0] -= 1
    trainer.stage(target)
    metadata = trainer.finish_stage()
    assert metadata.delta_records
    asyncio.run(generator.read((metadata,)))
    loaded = _load(generator, prefetched)
    for key in target:
        torch.testing.assert_close(loaded[key], target[key], rtol=0, atol=0)


def test_shared_rank_layouts():
    a = m.TensorShard((0,), (2,), torch.float32)
    b = m.TensorShard((2,), (2,), torch.float32)
    assert m.shared_rank_layouts([[(1, {"w": b}), (0, {"w": a})]] * 2) == [
        {"w": a},
        {"w": b},
    ]
    with pytest.raises(ValueError, match="differ"):
        m.shared_rank_layouts([[(0, {"w": a})], [(0, {"w": b})]])


@pytest.mark.parametrize(
    "placements,shape,match",
    [
        ([Partial(), Shard(0)], (8, 6), "Partial"),
        ([_StridedShard(0, split_factor=3), Shard(0)], (8, 6), "Uneven"),
    ],
)
def test_box_rejects(placements, shape, match):
    dist.init_process_group("fake", rank=0, world_size=4, store=FakeStore())
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        tensor = DTensor.from_local(
            torch.empty(2, 6), mesh, placements, shape=torch.Size(shape), stride=(6, 1)
        )
        with pytest.raises(ValueError, match=match):
            m._global_offset_and_local_shape(tensor)
    finally:
        dist.destroy_process_group()


def _fsdp_over_tp(full, mesh, dim):
    """FSDP2 over TP on the same dim: each rank holds its FSDP chunk of its TP
    chunk."""
    dp, tp = mesh.get_coordinate()
    local = full.chunk(2, dim)[tp].chunk(2, dim)[dp].contiguous()
    return DTensor.from_local(
        local,
        mesh,
        [_StridedShard(dim, split_factor=2), Shard(dim)],
        shape=full.shape,
        stride=full.stride(),
    )


class TestStageDTensors(DTensorTestBase):
    """Stages real FSDP 2 x TP 2 DTensors on 4 gloo ranks and reads them into
    generator TP 2 shards, checking packing and the boxes against the shards
    DTensor and FSDP2 actually hold."""

    @property
    def world_size(self) -> int:
        return 4

    @property
    def backend(self) -> str:
        return "gloo"

    @property
    def device_type(self) -> str:
        return "cpu"

    @with_comms
    def test_fsdp_tp_to_tp(self):
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        full = {
            # Row-parallel: FSDP Shard(0), TP Shard(1).
            "row": torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6),
            # Colwise: FSDP over TP on dim 0.
            "col": torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6) + 1000,
            # Stacked [2, F, D] w13: FSDP over TP on dim 1.
            "w13": torch.arange(2 * 8 * 6, dtype=torch.float32).reshape(2, 8, 6) + 2000,
            # Row-parallel with 2 rows: split on dim 0, the generator holds one
            # row, so column pieces are still read in place.
            "small": torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6) + 3000,
        }
        sd = {
            "row": distribute_tensor(full["row"], mesh, [Shard(0), Shard(1)]),
            "col": _fsdp_over_tp(full["col"], mesh, 0),
            "w13": _fsdp_over_tp(full["w13"], mesh, 1),
            "small": distribute_tensor(full["small"], mesh, [Shard(0), Shard(1)]),
        }
        boxes = {}
        global_offset_and_local_shape = m._global_offset_and_local_shape
        with (
            patch.object(m, "new_transport", lambda backend: _FakeTransport()),
            patch.object(m, "wait_all", _wait_all),
            patch.object(
                m,
                "_global_offset_and_local_shape",
                lambda tensor: boxes.get(id(tensor))
                or global_offset_and_local_shape(tensor),
            ),
        ):
            # Generator TP dims per key, and the keys read packed: as its
            # trainer TP, and all on dim 0.
            for dims, packed in (
                ({"row": 1, "col": 0, "w13": 1, "small": 1}, {"w13"}),
                (dict.fromkeys(full, 0), {"row"}),
            ):
                rank_layouts = [
                    {
                        key: m.TensorShard(
                            *_boxes(tuple(full[key].shape), 2, dim)[rank],
                            torch.float32,
                        )
                        for key, dim in dims.items()
                    }
                    for rank in range(2)
                ]
                config = ModelStateDictTransport.Config()
                trainer = ModelStateDictTransport(config)
                source = trainer.register_trainer_source(
                    sd,
                    generator_rank_layouts=rank_layouts,
                    transfer_dtype=None,
                    preserve_dtype_keys=frozenset(),
                )
                trainer.stage(sd)
                metadata = trainer.finish_stage()
                sources = [None] * self.world_size
                dist.all_gather_object(sources, source)
                metadata_by_source = [None] * self.world_size
                dist.all_gather_object(metadata_by_source, metadata)

                for rank, layout in enumerate(rank_layouts):
                    generator_sd = {}
                    for key, shard in layout.items():
                        generator_sd[key] = torch.zeros(shard.shape)
                        boxes[id(generator_sd[key])] = (shard.offset, shard.shape)
                    generator = ModelStateDictTransport(config)
                    generator.connect_generator(
                        generator_sd,
                        load_groups=[list(generator_sd)],
                        sources=sources,
                        rank=rank,
                        generator_index=0,
                    )
                    asyncio.run(generator.read(tuple(metadata_by_source)))
                    self.assertEqual(set(generator._packed), packed)
                    loaded = {key: torch.zeros(s.shape) for key, s in layout.items()}
                    generator.copy_prefetched_(loaded)
                    for key, shard in layout.items():
                        torch.testing.assert_close(
                            loaded[key],
                            full[key][_index(shard.offset, shard.shape)],
                            rtol=0,
                            atol=0,
                        )
