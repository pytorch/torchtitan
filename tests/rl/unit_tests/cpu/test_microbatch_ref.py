# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `MicrobatchRef` and `MicrobatchStream`, with Monarch's RDMA classes faked."""

import asyncio
import contextvars
import dataclasses
import time
from types import SimpleNamespace

import pytest
import torch

from torchtitan.rl.distributed import microbatch_ref
from torchtitan.rl.distributed.microbatch_ref import (
    _NUM_MICROBATCHES_READ_AHEAD,
    _TensorRef,
    MicrobatchRef,
    MicrobatchStream,
)
from torchtitan.rl.types import TrainingMicrobatch


class _FakeRDMABuffer:
    """Stands in for `monarch.rdma.RDMABuffer`: keeps the controller's byte view."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data
        self.dropped = False

    async def drop(self) -> None:
        self.dropped = True


class _FakeRDMAAction:
    """Stands in for `monarch.rdma.RDMAAction`: copies every queued read on `submit()`."""

    def __init__(self) -> None:
        self._reads: list[tuple[torch.Tensor, _FakeRDMABuffer]] = []

    def read_remote(self, dst: torch.Tensor, src: _FakeRDMABuffer) -> "_FakeRDMAAction":
        self._reads.append((dst, src))
        return self

    def submit(self) -> SimpleNamespace:
        for dst, src in self._reads:
            dst.copy_(src.data)
        return SimpleNamespace(get=lambda: None)


def _microbatch() -> TrainingMicrobatch:
    return TrainingMicrobatch(
        input=torch.tensor([3, 5, 7]),
        labels=torch.tensor([5, 7, 9]),
        positions=torch.tensor([0, 1, 2]),
        padding_mask=torch.tensor([False, False, True]),
        loss_token_counts=torch.tensor(2),
        routing_token_counts=torch.tensor([2, 2]),
        generator_logprobs=torch.tensor([-0.5, -1.5, 0.0]),
        temperature=torch.tensor([1.0, 1.0, 1.0]),
        loss_mask=torch.tensor([True, True, False]),
        advantages=torch.tensor([0.25, 0.25, 0.0]),
        model_kwargs={
            "routed_expert_ids": torch.arange(12, dtype=torch.uint8).view(3, 2, 2),
            "topk_logprobs": torch.zeros(0, 4),
        },
    )


def test_microbatch_ref_round_trips_every_field_and_drops_its_buffers(
    monkeypatch,
) -> None:
    monkeypatch.setattr(microbatch_ref, "RDMABuffer", _FakeRDMABuffer)
    monkeypatch.setattr(microbatch_ref, "RDMAAction", _FakeRDMAAction)
    microbatch = _microbatch()

    ref = MicrobatchRef.register(microbatch)
    read_microbatch = ref.read()

    # Tensors nested in model_kwargs become refs too; an empty tensor goes by value.
    assert isinstance(ref.fields["input"], _TensorRef)
    assert isinstance(ref.fields["model_kwargs"]["routed_expert_ids"], _TensorRef)
    assert isinstance(ref.fields["model_kwargs"]["topk_logprobs"], torch.Tensor)
    for field in dataclasses.fields(microbatch):
        expected = getattr(microbatch, field.name)
        actual = getattr(read_microbatch, field.name)
        if field.name == "model_kwargs":
            assert expected.keys() == actual.keys()
            for key in expected:
                assert torch.equal(expected[key], actual[key])
        else:
            assert torch.equal(expected, actual)
            assert expected.dtype == actual.dtype
            assert expected.data_ptr() != actual.data_ptr()

    asyncio.run(ref.drop())
    tensor_refs = [
        value for value in ref.fields.values() if isinstance(value, _TensorRef)
    ]
    tensor_refs.append(ref.fields["model_kwargs"]["routed_expert_ids"])
    assert len(tensor_refs) == 11
    assert all(tensor_ref.buffer.dropped for tensor_ref in tensor_refs)


def _ref(index: int, started: list[int], read_s: float = 0.0) -> SimpleNamespace:
    """A ref whose `read()` records its index, then takes `read_s` seconds."""

    def read() -> int:
        started.append(index)
        time.sleep(read_s)
        return index

    return SimpleNamespace(read=read)


def test_microbatch_stream_yields_in_order_and_reads_two_ahead() -> None:
    started: list[int] = []
    stream = MicrobatchStream([_ref(index, started) for index in range(6)])

    assert len(stream) == 6
    for [index] in stream:
        # Wait for the expected read, then 20 ms more, so a read too far ahead shows.
        expected = min(index + _NUM_MICROBATCHES_READ_AHEAD, 5)
        deadline = time.monotonic() + 5
        while max(started) < expected and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.02)
        assert max(started) == expected
    assert started == list(range(6))


def test_microbatch_stream_read_thread_sees_caller_context() -> None:
    step = contextvars.ContextVar("step")
    step.set(7)
    stream = MicrobatchStream([SimpleNamespace(read=step.get)])

    assert list(stream) == [[7]]


def test_microbatch_stream_reraises_read_error_at_its_position() -> None:
    def fail() -> int:
        raise ValueError("read failed")

    stream = MicrobatchStream(
        [SimpleNamespace(read=lambda: 0), SimpleNamespace(read=fail)]
    )
    iterator = iter(stream)

    assert next(iterator) == [0]
    with pytest.raises(ValueError, match="read failed"):
        next(iterator)


def test_microbatch_stream_skips_queued_reads_when_training_stops() -> None:
    started: list[int] = []
    stream = MicrobatchStream([_ref(index, started, read_s=0.5) for index in range(3)])
    iterator = iter(stream)

    assert next(iterator) == [0]
    # Read 1 is running and read 2 is queued: closing waits for read 1 and cancels read 2.
    iterator.close()

    assert 2 not in started
