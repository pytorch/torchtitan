# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `MonarchRDMAObject` and for prompts sent as RDMA objects from a rollout worker
through the router to generator rank 0 (`RolloutWorker.Config.rdma_min_prompt_bytes`), with
Monarch's RDMA classes faked except in the tests over real Monarch (TCP)."""

import asyncio
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from monarch.actor import Actor, endpoint, this_host

from torchtitan.rl.actors.monarch import rdma
from torchtitan.rl.actors.monarch.rdma import (
    _ArrayRef,
    _BytesRef,
    _TensorRef,
    MonarchRDMAObject,
)
from torchtitan.rl.actors.monarch.router import InterGeneratorRouterActor
from torchtitan.rl.actors.rdma import RDMAObject
from torchtitan.rl.actors.system import ServiceHandle
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.rollout.rollouter import RolloutWorker
from torchtitan.rl.rollout.types import Completion, Prompt
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter


class _FakeRDMABuffer:
    """Stands in for `monarch.rdma.RDMABuffer`: keeps the owner's byte view."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data
        self.dropped = False

    def size(self) -> int:
        return self.data.numel()

    async def drop(self) -> None:
        self.dropped = True


class _FakeRDMAAction:
    """Stands in for `monarch.rdma.RDMAAction`: copies every queued read on `submit()`."""

    def __init__(self) -> None:
        self._reads: list[tuple[torch.Tensor, _FakeRDMABuffer]] = []

    def read_remote(self, dst: torch.Tensor, src: _FakeRDMABuffer) -> "_FakeRDMAAction":
        self._reads.append((dst, src))
        return self

    async def submit(self) -> None:
        for dst, src in self._reads:
            dst.copy_(src.data)


@dataclasses.dataclass(frozen=True, slots=True)
class _File:
    name: str
    data: bytes


@dataclasses.dataclass(frozen=True, slots=True)
class _Payload:
    """A value with every kind of leaf, large and small, inside dataclasses and containers."""

    tensor: torch.Tensor
    array: np.ndarray
    files: tuple[_File, ...]
    extras: dict
    note: str | None = None


def _payload() -> _Payload:
    return _Payload(
        tensor=torch.arange(64, dtype=torch.float32).view(8, 8),  # 256 B
        array=np.arange(32, dtype=np.int32),  # 128 B
        files=(
            _File(name="large", data=bytes(range(200))),
            _File(name="small", data=b"x"),
        ),
        extras={"small_tensor": torch.tensor([1, 2]), "count": 3},
    )


def test_rdma_object_round_trips_tensor_array_and_bytes_leaves_and_drops_their_buffers(
    monkeypatch,
) -> None:
    monkeypatch.setattr(rdma, "RDMABuffer", _FakeRDMABuffer)
    monkeypatch.setattr(rdma, "RDMAAction", _FakeRDMAAction)
    payload = _payload()

    registered = MonarchRDMAObject.register(payload, min_bytes=100)

    # Leaves of at least 100 bytes stay in the owner; the rest travel with the handle.
    registered_value = registered.value
    assert isinstance(registered_value.tensor, _TensorRef)
    assert isinstance(registered_value.array, _ArrayRef)
    assert isinstance(registered_value.files[0].data, _BytesRef)
    assert registered_value.files[1].data == b"x"
    assert isinstance(registered_value.extras["small_tensor"], torch.Tensor)
    assert registered_value.extras["count"] == 3
    assert isinstance(registered, RDMAObject) and not isinstance(payload, RDMAObject)

    read = asyncio.run(registered.read())
    assert torch.equal(read.tensor, payload.tensor)
    assert read.tensor.dtype == payload.tensor.dtype
    np.testing.assert_array_equal(read.array, payload.array)
    assert read.array.dtype == payload.array.dtype
    assert read.files == payload.files
    assert torch.equal(read.extras["small_tensor"], payload.extras["small_tensor"])
    assert read.extras["count"] == 3 and read.note is None

    asyncio.run(registered.drop())
    buffers = [
        registered_value.tensor.buffer,
        registered_value.array.buffer,
        registered_value.files[0].data.buffer,
    ]
    assert all(buffer.dropped for buffer in buffers)


class _Reader(Actor):
    """Reads a registered value in another process."""

    @endpoint
    async def read(self, registered: MonarchRDMAObject) -> object:
        return await registered.read()


def test_monarch_rdma_object_round_trips_every_leaf_kind(monkeypatch) -> None:
    # Real Monarch RDMA over TCP (works on any CI host; the reader process inherits the env var).
    monkeypatch.setenv("MONARCH_RDMA_DISABLE_IBVERBS", "1")
    payload = _payload()

    async def run() -> _Payload:
        mesh = this_host().spawn_procs(per_host={"procs": 1})
        registered = MonarchRDMAObject.register(payload, min_bytes=100)
        try:
            reader = mesh.spawn("reader", _Reader)
            return await reader.read.call_one(registered)
        finally:
            await registered.drop()
            await mesh.stop()

    read = asyncio.run(run())
    assert torch.equal(read.tensor, payload.tensor)
    np.testing.assert_array_equal(read.array, payload.array)
    assert read.files == payload.files


@dataclasses.dataclass
class _FakeRDMAPrompt:
    """Stands in for a registered prompt: keeps the prompt and records the drop."""

    prompt: Prompt
    min_bytes: int
    dropped: bool = False

    async def read(self) -> Prompt:
        return self.prompt

    async def drop(self) -> None:
        self.dropped = True


def _prompt(num_tokens: int) -> Prompt:
    return Prompt(token_ids=np.arange(num_tokens, dtype=np.int32))


def _worker(rdma_min_prompt_bytes: int | None) -> RolloutWorker:
    return RolloutWorker(
        SimpleNamespace(
            rubric=SimpleNamespace(build=lambda: None),
            message_env=None,
            token_env=None,
            advantage=SimpleNamespace(build=lambda: None),
            rdma_min_prompt_bytes=rdma_min_prompt_bytes,
        ),
        register_rdma_object=_FakeRDMAPrompt,
    )


class _RecordingGenerateFn:
    def __init__(self, error: Exception | None = None, block: bool = False) -> None:
        self.prompt = None
        self.error = error
        self.block = block
        self.started = asyncio.Event()

    async def __call__(self, prompt, **kwargs) -> str:
        self.prompt = prompt
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        return "completion"


def _generate(worker: RolloutWorker, generate_fn, prompt: Prompt):
    return worker._generate(
        generate_fn,
        prompt,
        request_id="group=0/rollout=0/turn=0",
        group_id=0,
        routing_session_id="group=0/rollout=0",
        sampling_config=SamplingConfig(),
    )


def test_worker_sends_a_long_prompt_as_an_rdma_object_and_drops_it() -> None:
    generate_fn = _RecordingGenerateFn()
    prompt = _prompt(num_tokens=4)  # 16 bytes

    result = asyncio.run(_generate(_worker(16), generate_fn, prompt))

    assert result == "completion"
    sent = generate_fn.prompt
    assert isinstance(sent, _FakeRDMAPrompt)
    assert sent.prompt is prompt and sent.min_bytes == 16
    # Generator rank 0 has read the prompt once `generate` returns.
    assert sent.dropped


@pytest.mark.parametrize("rdma_min_prompt_bytes", [None, 17])
def test_worker_sends_a_short_prompt_in_the_call(rdma_min_prompt_bytes) -> None:
    generate_fn = _RecordingGenerateFn()
    prompt = _prompt(num_tokens=4)

    asyncio.run(_generate(_worker(rdma_min_prompt_bytes), generate_fn, prompt))

    assert generate_fn.prompt is prompt


def test_worker_drops_the_prompt_when_generate_fails() -> None:
    generate_fn = _RecordingGenerateFn(error=RuntimeError("generator died"))

    with pytest.raises(RuntimeError, match="generator died"):
        asyncio.run(_generate(_worker(1), generate_fn, _prompt(num_tokens=1)))

    assert generate_fn.prompt.dropped


def test_worker_drops_the_prompt_when_cancelled() -> None:
    generate_fn = _RecordingGenerateFn(block=True)

    async def run() -> None:
        call = asyncio.create_task(
            _generate(_worker(1), generate_fn, _prompt(num_tokens=1))
        )
        await generate_fn.started.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    asyncio.run(run())

    assert generate_fn.prompt.dropped


def test_worker_rejects_rdma_without_a_registration() -> None:
    with pytest.raises(ValueError, match="must be None or >= 1"):
        RolloutWorker.Config(rubric=None, message_env=None, rdma_min_prompt_bytes=0)
    with pytest.raises(ValueError, match="no register_rdma_object"):
        RolloutWorker(
            SimpleNamespace(rdma_min_prompt_bytes=1024),
        )


def test_router_forwards_an_rdma_prompt_without_reading_it() -> None:
    received = []

    async def generate(prompt, **kwargs):
        received.append(prompt)
        return "completion"

    rank0 = SimpleNamespace(generate=SimpleNamespace(call_one=generate))
    router = InterGeneratorRouter(
        InterGeneratorRouter.Config(),
        generators=[ServiceHandle(all_ranks=rank0, rank0=rank0)],
    )
    sent = _FakeRDMAPrompt(_prompt(num_tokens=4), min_bytes=1)

    result = asyncio.run(
        router.generate(
            sent,
            request_id="r",
            group_id=0,
            routing_session_id="s",
            sampling_config=None,
            metrics_prefix="generator",
        )
    )

    assert result == "completion" and received == [sent]


def test_generator_reads_an_rdma_prompt() -> None:
    # `Generator.generate` without building vLLM: the engine loop records the request.
    generator = object.__new__(Generator)
    generator.config = SimpleNamespace(sampling=SamplingConfig(stop_token_ids=[9]))
    requests = []

    async def generate(request, *, metrics_prefix):
        requests.append(request)
        return "completion"

    generator._engine_loop = SimpleNamespace(generate=generate)

    result = asyncio.run(
        generator.generate(
            _FakeRDMAPrompt(_prompt(num_tokens=3), min_bytes=1),
            request_id="r",
            group_id=0,
            routing_session_id="s",
        )
    )

    assert result == "completion"
    [request] = requests
    assert request.prompt_token_ids == [0, 1, 2]


class _GeneratorRank0(Actor):
    """Generator rank 0's `generate` endpoint: reads the prompt and echoes its token ids."""

    @endpoint
    async def generate(self, prompt, *, request_id: str, **kwargs) -> Completion:
        # Only the handle crossed the worker -> router -> generator hops.
        assert isinstance(prompt.value.token_ids, _ArrayRef)
        prompt = await prompt.read()
        return Completion(
            min_policy_version=0,
            max_policy_version=0,
            request_id=request_id,
            token_ids=prompt.token_ids.tolist(),
            token_logprobs=[],
        )


def test_worker_prompt_reaches_generator_through_the_router_as_a_handle(
    monkeypatch,
) -> None:
    # Real Monarch RDMA over TCP: this process is the rollout worker that owns the prompt's buffer.
    monkeypatch.setenv("MONARCH_RDMA_DISABLE_IBVERBS", "1")
    worker = RolloutWorker(
        SimpleNamespace(
            rubric=SimpleNamespace(build=lambda: None),
            message_env=None,
            token_env=None,
            advantage=SimpleNamespace(build=lambda: None),
            rdma_min_prompt_bytes=1024,
        ),
        register_rdma_object=MonarchRDMAObject.register,
    )
    prompt = _prompt(num_tokens=512)  # 2 KiB

    async def run() -> Completion:
        generator_mesh = this_host().spawn_procs(per_host={"procs": 1})
        router_mesh = this_host().spawn_procs(per_host={"procs": 1})
        try:
            generator = generator_mesh.spawn("generator", _GeneratorRank0)
            router = router_mesh.spawn(
                "router",
                InterGeneratorRouterActor,
                InterGeneratorRouter.Config(),
                generators=[ServiceHandle(all_ranks=generator, rank0=generator)],
            )

            async def generate_fn(prompt, **kwargs):
                return await router.generate.call_one(
                    prompt, metrics_prefix="generator", **kwargs
                )

            return await _generate(worker, generate_fn, prompt)
        finally:
            await router_mesh.stop()
            await generator_mesh.stop()

    completion = asyncio.run(run())
    assert completion.token_ids == prompt.token_ids.tolist()
