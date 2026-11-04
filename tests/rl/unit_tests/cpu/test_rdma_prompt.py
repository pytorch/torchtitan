# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for sending prompt image files to generator rank 0 as RDMA buffers
(`Controller.Config.rdma_min_image_bytes`), with fakes and once over real Monarch (TCP)."""

import asyncio
import dataclasses
from collections import OrderedDict
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from monarch.actor import Actor, endpoint, this_host

from torchtitan.components.data.multimodal.utils.image import ImageSpec
from torchtitan.rl.actors.monarch import rdma_prompt
from torchtitan.rl.actors.monarch.rdma_prompt import MonarchRDMAPrompt
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.rdma_prompt import RDMAPrompt
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.rollout.types import Prompt, PromptImage


def _image(key: str, data: bytes) -> PromptImage:
    return PromptImage(
        key=key, path=f"{key}.png", grid_thw=(1, 4, 4), offset=1, length=4, data=data
    )


@dataclasses.dataclass
class _FakeRDMAPrompt:
    """Stands in for `MonarchRDMAPrompt`: keeps the prompt and records the drop."""

    prompt: Prompt
    min_bytes: int
    dropped: bool = False

    async def read(self) -> Prompt:
        return self.prompt

    async def drop(self) -> None:
        self.dropped = True


def _register(prompt: Prompt, *, min_bytes: int) -> _FakeRDMAPrompt:
    return _FakeRDMAPrompt(prompt, min_bytes)


class _FakeRouter:
    def __init__(self, error: Exception | None = None, block: bool = False) -> None:
        self.prompt = None
        self.error = error
        self.block = block
        self.started = asyncio.Event()
        self.generate = SimpleNamespace(call_one=self._generate)

    async def _generate(self, prompt, **kwargs):
        self.prompt = prompt
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        return "completion"


def _generate_fn(router, rdma_min_image_bytes: int | None, register=_register):
    controller = SimpleNamespace(
        generator_router=router,
        config=SimpleNamespace(rdma_min_image_bytes=rdma_min_image_bytes),
        actor_system=SimpleNamespace(register_rdma_prompt=register),
    )
    return Controller._make_generate_fn(controller, metrics_prefix="generator")


def test_generate_fn_sends_a_prompt_with_large_files_as_rdma_and_drops_it() -> None:
    prompt = Prompt(
        token_ids=[1], images=(_image("small", b"123"), _image("large", b"12345678"))
    )
    router = _FakeRouter()

    result = asyncio.run(
        _generate_fn(router, rdma_min_image_bytes=4)(prompt, request_id="r", group_id=0)
    )

    assert result == "completion"
    sent = router.prompt
    assert isinstance(sent, RDMAPrompt)
    assert sent.prompt is prompt and sent.min_bytes == 4
    # Rank 0 has read the files once `generate` returns.
    assert sent.dropped


@pytest.mark.parametrize("rdma_min_image_bytes", [None, 4])
def test_generate_fn_sends_a_prompt_without_large_files_in_the_call(
    rdma_min_image_bytes,
) -> None:
    prompt = Prompt(token_ids=[1], images=(_image("small", b"123"),))
    router = _FakeRouter()

    asyncio.run(
        _generate_fn(router, rdma_min_image_bytes)(prompt, request_id="r", group_id=0)
    )

    assert router.prompt is prompt


def test_generate_fn_drops_the_files_when_generate_fails() -> None:
    router = _FakeRouter(error=RuntimeError("generator died"))

    with pytest.raises(RuntimeError, match="generator died"):
        asyncio.run(
            _generate_fn(router, rdma_min_image_bytes=1)(
                Prompt(token_ids=[1], images=(_image("a", b"12"),)),
                request_id="r",
                group_id=0,
            )
        )

    assert router.prompt.dropped


def test_generate_fn_drops_the_files_when_cancelled() -> None:
    router = _FakeRouter(block=True)

    async def run() -> None:
        call = asyncio.create_task(
            _generate_fn(router, rdma_min_image_bytes=1)(
                Prompt(token_ids=[1], images=(_image("a", b"12"),)),
                request_id="r",
                group_id=0,
            )
        )
        await router.started.wait()
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

    asyncio.run(run())

    assert router.prompt.dropped


def test_generate_fn_with_the_monarch_registration_pickles() -> None:
    # Monarch pickles the GenerateFn into every rollout worker call.
    generate_fn = _generate_fn(
        None, rdma_min_image_bytes=1024, register=MonarchRDMAPrompt.register
    )

    assert callable(cloudpickle.loads(cloudpickle.dumps(generate_fn)))


class _FakeRDMABuffer:
    """Stands in for `monarch.rdma.RDMABuffer`: keeps the owner's bytes."""

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


def test_rdma_prompt_round_trips_the_large_files_and_drops_their_buffers(
    monkeypatch,
) -> None:
    monkeypatch.setattr(rdma_prompt, "RDMABuffer", _FakeRDMABuffer)
    monkeypatch.setattr(rdma_prompt, "RDMAAction", _FakeRDMAAction)
    prompt = Prompt(
        token_ids=[1, 2],
        images=(
            _image("a", b"\x89PNG large"),
            _image("b", b"sm"),
            _image("c", b"big!"),
        ),
    )

    registered = MonarchRDMAPrompt.register(prompt, min_bytes=4)

    assert isinstance(registered, RDMAPrompt)
    assert not isinstance(prompt, RDMAPrompt)
    # Only the small file travels with the handle.
    assert [image.data for image in registered.prompt.images] == [b"", b"sm", b""]
    assert asyncio.run(registered.read()) == prompt
    asyncio.run(registered.drop())
    assert sorted(registered.buffers) == [0, 2]
    assert all(buffer.dropped for buffer in registered.buffers.values())


def test_generator_reads_an_rdma_prompt() -> None:
    # `Generator.generate` without building vLLM: the engine loop records the request.
    generator = object.__new__(Generator)
    generator.config = SimpleNamespace(sampling=SamplingConfig(stop_token_ids=[9]))
    generator._image_files = OrderedDict()
    requests = []

    async def generate(request, *, metrics_prefix):
        requests.append(request)
        return "completion"

    generator._engine_loop = SimpleNamespace(generate=generate)
    sent = _FakeRDMAPrompt(
        Prompt(token_ids=[1, 9, 9, 9, 9], images=(_image("a", b"12"),)), min_bytes=1
    )

    result = asyncio.run(
        generator.generate(sent, request_id="r", group_id=0, routing_session_id="s")
    )

    assert result == "completion"
    [request] = requests
    assert request.prompt == sent.prompt


class _GeneratorRank0(Actor):
    """Reads a prompt as generator rank 0 does."""

    @endpoint
    async def read(self, rdma_prompt: MonarchRDMAPrompt) -> Prompt:
        return await rdma_prompt.read()


def test_monarch_rdma_prompt_round_trips_the_large_files(monkeypatch) -> None:
    # Real Monarch RDMA over TCP (works on any CI host; the generator process inherits the env var).
    monkeypatch.setenv("MONARCH_RDMA_DISABLE_IBVERBS", "1")
    prompt = Prompt(
        token_ids=[1, 2],
        images=(
            _image("a", bytes(range(256)) * 4),
            _image("b", b"\xff\xd8\xff"),
            _image("c", bytes(range(255, -1, -1)) * 2),
        ),
    )

    async def run() -> Prompt:
        mesh = this_host().spawn_procs(per_host={"procs": 1})
        registered = MonarchRDMAPrompt.register(prompt, min_bytes=4)
        try:
            rank0 = mesh.spawn("rank0", _GeneratorRank0)
            return await rank0.read.call_one(registered)
        finally:
            await registered.drop()
            await mesh.stop()

    assert asyncio.run(run()) == prompt


def test_controller_rejects_rdma_min_image_bytes_below_1_or_without_image_files() -> (
    None
):
    from torchtitan_recipes.tests.rl.alphabet_sort import rl_grpo_qwen3_5_debug_varlen

    config = rl_grpo_qwen3_5_debug_varlen(seq_len=128)
    image_engine = dataclasses.replace(config.generator.engine, image_spec=ImageSpec())
    image_generator = dataclasses.replace(config.generator, engine=image_engine)

    with pytest.raises(ValueError, match="must be None or >= 1"):
        dataclasses.replace(config, generator=image_generator, rdma_min_image_bytes=0)
    with pytest.raises(ValueError, match="prompts carry no image files"):
        dataclasses.replace(config, rdma_min_image_bytes=1024)
