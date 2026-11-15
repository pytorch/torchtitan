# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The vLLM endpoint that Verifiers calls, served by TitanRL's `generate_fn`."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from aiohttp import web

from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import Completion, GenerateFn, Prompt

if TYPE_CHECKING:
    from torchtitan.rl.generator.types import SamplingConfig

# Stands in for the logprobs content list; `_generate` splices the real list in.
_LOGPROBS_CONTENT_PLACEHOLDER = "<logprobs content>"


@dataclass(slots=True)
class GenerationSession:
    """One Verifiers trace's model calls, as this server saw them."""

    group_id: int
    """The rollout group of the trace; its calls share the group's prefix cache."""

    num_calls: int = 0
    """Calls started so far. The next call's request id is `<session id>/call=<num_calls>`."""

    # TODO: per-token policy versions (Verifiers' `weight_versions`) need the generator to report
    #   where in a completion the weights changed.
    min_policy_version: int | None = None
    """Oldest policy version among the session's completions; None before the first."""

    max_policy_version: int | None = None
    """Newest policy version among the session's completions; None before the first."""

    metrics: list[m.Metric] = field(default_factory=list)
    """Generator metrics of every completion, merged with `m.fold`."""

    calls_in_flight: set[asyncio.Future[Completion | None]] = field(default_factory=set)
    """The session's `generate_fn` calls that haven't returned yet."""


class GenerationServer:
    """Serves TitanRL's `generate_fn` as the vLLM endpoint that Verifiers' train client calls.

    Each rollout worker runs one, on loopback:

        Verifiers' train client --POST /inference/v1/generate--> GenerationServer
                                  (X-Session-ID: the trace id)        |
                                                                      generate_fn --> generator
                                <---- token ids + logprobs ----------  <-- Completion --

    Verifiers keeps each completion's tokens and logprobs in its trace. This server keeps what a
    trace has no field for: each session's group, policy versions and generator metrics.
    """

    # TODO: drop this server once each generator serves this endpoint; what it keeps per session
    # then needs another home.

    MODEL_NAME = "torchtitan"

    def __init__(self, *, max_rollout_tokens: int) -> None:
        self._max_rollout_tokens = max_rollout_tokens
        self._generate_fn: GenerateFn
        self._runner: web.AppRunner | None = None
        self._sessions: dict[str, GenerationSession] = {}

    async def start(self, generate_fn: GenerateFn) -> str:
        """Listen on a free loopback port; return the base URL for Verifiers' client."""
        self._generate_fn = generate_fn
        # aiohttp's 1 MiB default refuses a prompt past about 150K tokens (7 bytes of JSON per
        # 6-digit token id). 1 GiB is Verifiers' own limit for its interception server.
        app = web.Application(client_max_size=1024**3)
        app.router.add_get("/v1/models", self._list_models)
        app.router.add_post("/inference/v1/generate", self._generate)
        # access_log=None: aiohttp would log one line per model call.
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        # Port 0: the OS picks a free one, so the workers on one host don't collide.
        await web.TCPSite(self._runner, host="127.0.0.1", port=0).start()
        host, port = self._runner.addresses[0][:2]
        return f"http://{host}:{port}/v1"

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def open_session(self, session_id: str, *, group_id: int) -> None:
        """Accept model calls for `session_id`, a Verifiers trace id."""
        self._sessions[session_id] = GenerationSession(group_id=group_id)

    async def close_session(self, session_id: str) -> GenerationSession:
        """Refuse further calls for `session_id`, wait for its calls in flight, and return what
        its calls left here.

        Why wait: aiohttp doesn't cancel a handler when its client hangs up, e.g. an agent past
        its deadline. The handler's `generate_fn` call still runs, so the routing session it uses
        must outlive it.
        """
        session = self._sessions.pop(session_id)
        if session.calls_in_flight:
            await asyncio.wait(session.calls_in_flight)
        return session

    async def _list_models(self, request: web.Request) -> web.Response:
        # Verifiers' renderer refuses a prompt longer than `max_model_len` before sending it.
        # One less than the context length leaves room for one output token, as vLLM needs.
        model_card = {
            "id": self.MODEL_NAME,
            "object": "model",
            "max_model_len": self._max_rollout_tokens - 1,
        }
        return web.json_response({"object": "list", "data": [model_card]})

    async def _generate(self, request: web.Request) -> web.Response:
        """Run one model call through `generate_fn`. Any failure answers 500.

        Example request and reply (one completion of 2 tokens):

            headers  {"X-Session-ID": "9f3c..."}
            body     {"model": "torchtitan", "token_ids": [151644, 872, ...],
                      "sampling_params": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512,
                                          "stop_token_ids": [151645], "logprobs": 1, ...}}
            reply    {"request_id": "9f3c.../call=0",
                      "choices": [{"token_ids": [9707, 151645], "finish_reason": "stop",
                                   "logprobs": {"content": [{"token": "token_id:9707",
                                                             "logprob": -0.01}, ...]}}]}
        """
        session_id = request.headers["X-Session-ID"]
        # TODO: parse with a faster JSON library (e.g. orjson): this takes about 8 ms at 128K tokens.
        body = await request.json()
        # Looked up after the body is read, with no await before the call joins
        # `calls_in_flight`: a session closed meanwhile raises KeyError (a 500) and calls nothing.
        session = self._sessions[session_id]
        request_id = f"{session_id}/call={session.num_calls}"
        session.num_calls += 1
        call = asyncio.ensure_future(
            self._generate_fn(
                # TODO: send the generator only this call's new tokens, keyed by the routing
                #   session: each call re-sends the whole prompt. A core change to the generator.
                Prompt(token_ids=body["token_ids"]),
                request_id=request_id,
                group_id=session.group_id,
                routing_session_id=session_id,
                sampling_config=_sampling_config(body["sampling_params"]),
            )
        )
        session.calls_in_flight.add(call)
        call.add_done_callback(session.calls_in_flight.discard)
        completion = await call
        if completion is None or completion.finish_reason not in ("stop", "length"):
            raise RuntimeError(
                f"{request_id}: the generator returned no usable completion"
            )

        if session.min_policy_version is None or session.max_policy_version is None:
            session.min_policy_version = completion.min_policy_version
            session.max_policy_version = completion.max_policy_version
        else:
            session.min_policy_version = min(
                session.min_policy_version, completion.min_policy_version
            )
            session.max_policy_version = max(
                session.max_policy_version, completion.max_policy_version
            )
        # Fold, so a long rollout holds one record per metric however many calls it makes.
        session.metrics = m.fold([*session.metrics, *completion.metrics])

        # TODO: when score centering's top-k logprobs come, add them here for the completion only:
        #   a per-token array must not ride the prompt, which every call re-sends.
        # Text, not one dict per token: 1.4x faster, and no GC pass every ~700 tokens.
        # A finite Python float's repr is valid JSON; vLLM's logprobs are Python floats.
        logprobs_content = ",".join(
            f'{{"token":"token_id:{token_id}","logprob":{logprob!r}}}'
            for token_id, logprob in zip(
                completion.token_ids, completion.token_logprobs, strict=True
            )
        )
        choice = {
            "index": 0,
            "token_ids": completion.token_ids,
            "finish_reason": completion.finish_reason,
            "logprobs": {"content": _LOGPROBS_CONTENT_PLACEHOLDER},
        }
        # TODO: add `usage` (token counts) before the Verifiers pin moves past 0.4.1. Verifiers
        #   main requires it (PrimeIntellect-ai/verifiers#2542). 0.4.0 and 0.4.1 count the
        #   tokens themselves and ignore it, so no test can check it yet.
        response_text = json.dumps({"request_id": request_id, "choices": [choice]})
        head, _, tail = response_text.rpartition(
            json.dumps(_LOGPROBS_CONTENT_PLACEHOLDER)
        )
        return web.Response(
            text=f"{head}[{logprobs_content}]{tail}", content_type="application/json"
        )


def _sampling_config(sampling_params: dict[str, Any]) -> SamplingConfig:
    """TitanRL's sampling config for the vLLM sampling params of one request. Raises on a param
    the generator can't apply, e.g. an env's `repetition_penalty`.

    Example:
        {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512, "seed": 7, "stop_token_ids": [2],
         "top_k": -1, "min_p": 0.0, "logprobs": 1, "skip_special_tokens": False}
        -> SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=512, seed=7, stop_token_ids=[2])
    """
    # Imported here: the generator module imports vLLM.
    from torchtitan.rl.generator.types import SamplingConfig

    # TODO: TitanRL's generator doesn't support other top_k or min_p, or top-k logprobs, yet.
    #   Raise early.
    for name, no_op_value in (("top_k", -1), ("min_p", 0.0), ("logprobs", 1)):
        if sampling_params.get(name, no_op_value) != no_op_value:
            raise ValueError(
                f"sampling_params.{name}={sampling_params[name]!r} is unsupported"
            )
    # The params the renderer client sends: applied below, checked above, or not needed here
    # (`routed_experts_prompt_start`: the TODO below). The generator can't apply any other, e.g.
    # an env's `repetition_penalty`, so raise.
    # TODO: check an env's agent sampling keys when the env loads, so a bad key fails its first
    #   group with this message, not every model call with an HTTP 500.
    known_params = {
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "stop_token_ids",
        "top_k",
        "min_p",
        "logprobs",
        "skip_special_tokens",
        "routed_experts_prompt_start",
    }
    if unsupported := sorted(set(sampling_params) - known_params):
        raise ValueError(f"unsupported sampling parameters: {unsupported}")
    # TODO: pass `routed_experts_prompt_start` to the generator once it replays expert routing:
    #   it marks where this call's new tokens start, so the reply needs routing rows only for
    #   those tokens. Recipes with replay also set
    #   `async_loop.batcher.rdma_min_tensor_bytes`, so a step's rows don't ride one Monarch message.
    return SamplingConfig(
        temperature=sampling_params["temperature"],
        top_p=sampling_params["top_p"],
        max_tokens=sampling_params["max_tokens"],
        seed=sampling_params.get("seed"),
        stop_token_ids=sampling_params["stop_token_ids"],
    )
