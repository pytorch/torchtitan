# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client side of the generators' HTTP endpoint: every generate call goes caller -> generator."""

from __future__ import annotations

import asyncio
import dataclasses
import io
import weakref
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.generator.generation_server import (
    GROUP_ID_SAMPLING_PARAM,
    METRICS_PREFIX_SAMPLING_PARAM,
    response_to_completion,
    SESSION_ID_HEADER,
    TORCHTITAN_RESPONSE_FORMAT,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.rollout.types import Completion
from torchtitan.rl.routing.inter_generator import GeneratorEndpoint


@dataclass(frozen=True)
class GeneratorClient:
    """Sends generate calls to the generators over HTTP, taking a lease from the router for each.

    Picklable: it holds only the router handle, so the controller ships it to rollout workers.

        native rollout, per turn:  router.acquire(session) -> POST generator -> router.release
        native rollout, at end:    release_session
        Verifiers rollout, once:   acquire_rollout -> agent posts every turn -> release_rollout
        Verifiers group, at end:   fetch_routed_expert_ids, once per trained rollout

    Example:

        client = GeneratorClient(router=generator_router, metrics_prefix="generator")
        completion = await client(
            [1, 2, 3],
            request_id="group=7/rollout=3/turn=0",
            group_id=7,
            routing_session_id="group=7/rollout=3",
            sampling_config=sampling,
        )
    """

    router: Any
    """The `InterGeneratorRouter` actor handle."""

    metrics_prefix: str = "generator"
    """Prefix of the generator metrics on every `Completion`, e.g. `"validation_generator"`."""

    @sl.log_trace_span("generate")
    async def __call__(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str,
        group_id: int,
        routing_session_id: str | None = None,
        sampling_config: SamplingConfig | None = None,
    ) -> Completion:
        """Generate one completion; the `GenerateFn` protocol. `sampling_config` is required: the
        server reads the stop token ids from it."""
        if sampling_config is None:
            raise ValueError(f"{request_id}: sampling_config is required over HTTP")
        body = {
            "request_id": request_id,
            "token_ids": list(prompt_token_ids),
            "sampling_params": {
                **dataclasses.asdict(sampling_config),
                GROUP_ID_SAMPLING_PARAM: group_id,
                METRICS_PREFIX_SAMPLING_PARAM: self.metrics_prefix,
            },
            "response_format": TORCHTITAN_RESPONSE_FORMAT,
        }
        endpoint = await self.router.acquire.call_one(
            session_id=routing_session_id, group_id=group_id
        )
        try:
            async with _http_session().post(
                f"{endpoint.url}/inference/v1/generate",
                json=body,
                headers={SESSION_ID_HEADER: routing_session_id or request_id},
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"{endpoint.url} answered {request_id} with {response.status}: "
                        f"{await response.text()}"
                    )
                return response_to_completion(await response.json())
        finally:
            await self.router.release.call_one(index=endpoint.index, group_id=group_id)

    async def release_session(self, *, routing_session_id: str) -> None:
        """Drop a finished rollout's sticky routing and its held KV; the `ReleaseSessionFn` protocol."""
        await self.router.release_session.call_one(
            routing_session_id=routing_session_id
        )

    async def acquire_rollout(self, *, group_id: int) -> GeneratorEndpoint:
        """Take a lease on a generator for a whole rollout whose agent calls it directly."""
        return await self.router.acquire.call_one(session_id=None, group_id=group_id)

    async def release_rollout(
        self,
        endpoint: GeneratorEndpoint,
        *,
        group_id: int,
        session_ids: list[str],
    ) -> dict[str, VerifiersGenerationMetadata]:
        """Release a rollout's sessions on its generator, returning what they folded, then the lease."""
        return await self.router.release_rollout.call_one(
            index=endpoint.index, group_id=group_id, session_ids=session_ids
        )

    async def fetch_routed_expert_ids(
        self, url: str, *, session_id: str, handles: np.ndarray
    ) -> np.ndarray:
        """Fetch the router-replay rows `[len(handles), num_layers, top_k]` the generator at `url`
        holds for a session's handles, uint8 `[len(handles), 4]` (see `GenerationServer`)."""
        async with _http_session().post(
            f"{url}/inference/v1/routed_experts",
            data=handles.tobytes(),
            headers={SESSION_ID_HEADER: session_id},
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"{url} answered the routed-experts fetch of session {session_id} with "
                    f"{response.status}: {await response.text()}"
                )
            return np.load(io.BytesIO(await response.read()))


# One HTTP session per event loop, shared by every client object in it: a session per call
# would open a TCP connection per call.
# TODO: close these sessions when the process stops; today each one logs "Unclosed client session".
_http_sessions: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, aiohttp.ClientSession
] = weakref.WeakKeyDictionary()


def _http_session() -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    session = _http_sessions.get(loop)
    if session is None or session.closed:
        session = aiohttp.ClientSession(
            # The default pool caps at 100 connections; a worker can have more calls in flight.
            connector=aiohttp.TCPConnector(limit=0),
            # The default 300 s total would fail the longest generations; measured turns reach 430 s.
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
        )
        _http_sessions[loop] = session
    return session
