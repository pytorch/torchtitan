# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A generator's HTTP endpoint: token ids in, token ids out, in vLLM's token-in format."""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import logging
import resource
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np
from aiohttp import web

from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import Completion

logger = logging.getLogger(__name__)

SESSION_ID_HEADER = "X-Session-ID"

# Stands in for the logprobs content list; the handler splices the real list in.
_LOGPROBS_CONTENT_PLACEHOLDER = "<logprobs content>"
# Stands in for the routed-experts base64; the handler splices the bytes in.
_ROUTED_EXPERTS_PLACEHOLDER = "<routed experts data>"

GROUP_ID_SAMPLING_PARAM = "torchtitan_group_id"
"""Sampling parameter carrying a request's rollout group id.

Verifiers forwards extra sampling fields verbatim in `sampling_params`, so the
rollouter tags every request of a group with its id.
"""

METRICS_PREFIX_SAMPLING_PARAM = "torchtitan_metrics_prefix"
"""Optional sampling parameter naming the prefix of the generator metrics, e.g.
`"validation_generator"`; `"generator"` when absent."""

TORCHTITAN_RESPONSE_FORMAT = "torchtitan"
"""Value of a request's top-level `response_format` that asks for the compact response."""

_METRIC_TYPES = {
    value_type.__name__: value_type for value_type in (m.Mean, m.Max, m.Min, m.Sum)
}


@dataclass(frozen=True, slots=True)
class VerifiersGenerationMetadata:
    """TorchTitan generation data aggregated over one Verifiers rollout.

    Verifiers sends the trace ID as the generation session ID, but does not
    retain TorchTitan policy versions, generator metrics or top-k logprobs in the
    returned trace. The generation server therefore records them per rollout.
    """

    min_policy_version: int
    """Oldest policy version used to generate the completion."""

    max_policy_version: int
    """Newest policy version used to generate the completion."""

    metrics: list[m.Metric]
    """Metrics from every generation in the rollout, merged with `m.fold`."""

    topk_by_generation: dict[
        tuple[int, tuple[int, ...]], tuple[np.ndarray, np.ndarray]
    ] = field(default_factory=dict)
    """The generator's top-k ids and logprobs `[num_completion_tokens, k]` per generation,
    keyed by `(len(prompt_token_ids), completion_token_ids)`; empty unless
    `SamplingConfig.num_topk_logprobs` > 0.

    Example:

        {(2, (12, 13)): (array([[12, 5], [13, 6]]), array([[-0.1, -2.4], [-0.2, -1.9]]))}
    """


class GenerationServer:
    """Serve a generator's `generate` over HTTP, so every caller reaches the generator directly.

    One request shape (vLLM's token-in `POST /inference/v1/generate`), two response formats:

        default                    vLLM's token-in response, which Verifiers' client reads. Verifiers
                                   drops other fields, so the server folds each session's policy
                                   versions, metrics and top-k for `pop_generation_metadata`.
        "response_format":         token ids, logprobs, top-k as raw arrays; versions, metrics in one
          "torchtitan"             `torchtitan` block; 200 whenever the engine produced a completion.

    The server mints every request id, so no client can send an id the engine already runs.
    It listens on every interface and is unauthenticated.

    Args:
        generate: The generator's `generate`, called once per request on this event loop.
        max_context_length: The engine's context length; prompts of this many tokens or more get a 400.
    """

    model_id = "torchtitan"

    def __init__(
        self,
        *,
        generate: Callable[..., Awaitable[Completion]],
        max_context_length: int,
    ) -> None:
        self.generate = generate
        self.max_context_length = max_context_length
        self.runner: web.AppRunner | None = None
        self.bound_port: int | None = None
        self._request_counter = itertools.count()
        self.generation_metadata: dict[str, VerifiersGenerationMetadata] = {}
        # Group id -> the sessions with folded metadata, so `drop_generation_metadata` can free them.
        self._group_sessions: dict[int, set[str]] = {}
        # This process (the engine's) GC time and event-loop lag, attached to each completion's
        # metrics. The lag task needs the running loop, so `start` creates it.
        self._gc_timer = m.GCTimer("generator")
        self._loop_lag: m.EventLoopLag | None = None

    @property
    def port(self) -> int:
        if self.bound_port is None:
            raise RuntimeError("GenerationServer has not started")
        return self.bound_port

    @property
    def url(self) -> str:
        """Base URL other hosts can reach, e.g. `"http://host-3.example.com:41233"`."""
        return f"http://{socket.getfqdn()}:{self.port}"

    async def start(self) -> None:
        if self.runner is not None:
            return
        # The default 1 MiB cap is ~130k prompt tokens as JSON; long-context recipes go past it.
        app = web.Application(client_max_size=64 * 1024**2)
        app.router.add_get("/healthz", self._handle_health_request)
        app.router.add_get("/v1/models", self._handle_models_request)
        app.router.add_post("/inference/v1/generate", self._handle_generate_request)
        # A handler whose client hung up is cancelled, so a request still on the engine loop's queue
        # never runs. No access log: one line per request would run in the engine's process.
        runner = web.AppRunner(app, handler_cancellation=True, access_log=None)
        await runner.setup()
        sock = _listening_socket()
        # A burst of new calls (one connection each) can pass aiohttp's default of 128 pending
        # connects. The site calls listen() on `sock` again, so the backlog is set here.
        await web.SockSite(runner, sock, backlog=4096).start()
        self.runner = runner
        self.bound_port = int(sock.getsockname()[1])
        self._loop_lag = m.EventLoopLag("generator")
        # Every in-flight call holds a connection, so a low open-file limit caps concurrency.
        logger.info(
            "Generation server listening on %s (open-file limit %d)",
            self.url,
            resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        )

    async def close(self) -> None:
        runner = self.runner
        if runner is not None:
            await runner.cleanup()
        self._gc_timer.close()
        if self._loop_lag is not None:
            self._loop_lag.close()
            self._loop_lag = None
        self.runner = None
        self.bound_port = None
        self.generation_metadata.clear()
        self._group_sessions.clear()

    def pop_generation_metadata(
        self, session_ids: list[str]
    ) -> dict[str, VerifiersGenerationMetadata]:
        """Detach the generation metadata folded for these sessions; a session without any is left out."""
        return {
            session_id: self.generation_metadata.pop(session_id)
            for session_id in session_ids
            if session_id in self.generation_metadata
        }

    def drop_generation_metadata(self, group_ids: list[int]) -> None:
        """Drop the folded metadata of every session of these groups, including sessions nobody
        popped (e.g. a retried Verifiers trace)."""
        for group_id in group_ids:
            for session_id in self._group_sessions.pop(group_id, ()):
                self.generation_metadata.pop(session_id, None)

    async def _handle_health_request(self, request: web.Request) -> web.Response:
        del request
        return web.json_response({"status": "ok"})

    async def _handle_models_request(self, request: web.Request) -> web.Response:
        """Return OpenAI-compatible model metadata for Verifiers clients."""
        del request
        model: dict[str, object] = {
            "id": self.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "torchtitan",
            "max_model_len": self.max_context_length,
        }
        return web.json_response({"object": "list", "data": [model]})

    async def _handle_generate_request(self, request: web.Request) -> web.Response:
        session_id = request.headers.get(SESSION_ID_HEADER)
        if not session_id:
            return web.json_response(
                {"error": f"missing {SESSION_ID_HEADER} header"}, status=400
            )

        try:
            body = await request.json()
            prompt_token_ids = _validate_token_ids(
                body.get("token_ids"), field_name="token_ids"
            )
            # vLLM needs room for one output token. Keep vLLM's wording: Verifiers'
            # harnesses match it as a context overflow.
            if len(prompt_token_ids) >= self.max_context_length:
                raise ValueError(
                    f"This model's maximum context length is {self.max_context_length} "
                    f"tokens, and the prompt has {len(prompt_token_ids)} tokens, "
                    "leaving no room for an output token"
                )
            sampling_params = body.get("sampling_params")
            if not isinstance(sampling_params, dict):
                raise ValueError("sampling_params must be an object")
            sampling_params = dict(sampling_params)
            group_id = sampling_params.pop(GROUP_ID_SAMPLING_PARAM, None)
            if isinstance(group_id, bool) or not isinstance(group_id, int):
                raise ValueError(
                    f"sampling_params.{GROUP_ID_SAMPLING_PARAM} must be an integer"
                )
            metrics_prefix = sampling_params.pop(
                METRICS_PREFIX_SAMPLING_PARAM, "generator"
            )
            if not isinstance(metrics_prefix, str):
                raise ValueError(
                    f"sampling_params.{METRICS_PREFIX_SAMPLING_PARAM} must be a string"
                )
            # Router replay: return expert rows from this prompt position on; earlier rows came
            # with the previous turn. vLLM asserts the range inside engine.step, where a failure
            # stops the engine for every request.
            routed_experts_prompt_start = sampling_params.pop(
                "routed_experts_prompt_start", 0
            )
            if isinstance(routed_experts_prompt_start, bool) or not (
                isinstance(routed_experts_prompt_start, int)
                and 0 <= routed_experts_prompt_start < len(prompt_token_ids)
            ):
                raise ValueError(
                    "sampling_params.routed_experts_prompt_start must be an integer in "
                    f"[0, {len(prompt_token_ids)})"
                )
            sampling = _parse_sampling_config(sampling_params)
            sampling.routed_experts_prompt_start = routed_experts_prompt_start
            if body.get("features") is not None:
                raise ValueError("multimodal features are not supported")
            client_request_id = body.get("request_id")
            response_format = body.get("response_format")
        except (TypeError, ValueError) as error:
            return web.json_response({"error": str(error)}, status=400)
        # Keep only the prompt alive while the engine generates, not the whole parsed body.
        del body

        # The counter keeps every id unique, even when a client sends the same id twice.
        request_id = (
            f"{client_request_id or session_id}/http={next(self._request_counter)}"
        )
        try:
            completion = await self.generate(
                prompt_token_ids,
                request_id=request_id,
                group_id=group_id,
                routing_session_id=session_id,
                sampling_config=sampling,
                metrics_prefix=metrics_prefix,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("TorchTitan generation failed for %s", request_id)
            return web.json_response({"error": str(error)}, status=500)

        if completion is None:
            return web.json_response(
                {"error": f"generation returned no completion for {request_id}"},
                status=502,
            )
        if len(completion.token_ids) != len(completion.token_logprobs):
            return web.json_response(
                {"error": "completion token IDs and logprobs have different lengths"},
                status=500,
            )
        # Validation (negative group ids) logs at the same W&B step under these keys: leave this
        # process's GC time and loop lag for the next training completion.
        if group_id >= 0:
            completion.metrics = [
                *completion.metrics,
                *self._gc_timer.flush(),
                *self._loop_lag.flush(),  # set by `start`, before any request
            ]
        else:
            # Validation never trains, so its router-replay rows (one per token) would only fill
            # the callers' memory until the pass ends.
            completion.routed_expert_ids = None
        if response_format == TORCHTITAN_RESPONSE_FORMAT:
            return web.json_response(completion_to_response(completion))
        if completion.finish_reason not in ("stop", "length"):
            return web.json_response(
                {
                    "error": "generation finished without a usable completion: "
                    f"{completion.finish_reason}"
                },
                status=502,
            )

        previous = self.generation_metadata.get(session_id)
        topk_by_generation = {} if previous is None else previous.topk_by_generation
        if completion.topk_token_ids is not None:
            key = (len(prompt_token_ids), tuple(completion.token_ids))
            topk_by_generation = {
                **topk_by_generation,
                key: (completion.topk_token_ids, completion.topk_logprobs),
            }
        self.generation_metadata[session_id] = VerifiersGenerationMetadata(
            min_policy_version=(
                completion.min_policy_version
                if previous is None
                else min(previous.min_policy_version, completion.min_policy_version)
            ),
            max_policy_version=(
                completion.max_policy_version
                if previous is None
                else max(previous.max_policy_version, completion.max_policy_version)
            ),
            # Fold so a long rollout holds one set of generator records, not one per
            # generation: every full GC walks them.
            metrics=m.fold(
                completion.metrics
                if previous is None
                else [*previous.metrics, *completion.metrics]
            ),
            topk_by_generation=topk_by_generation,
        )
        self._group_sessions.setdefault(group_id, set()).add(session_id)
        # Text, not one dict per token: 1.4x faster, and no GC pass every ~700 tokens.
        # A finite Python float's repr is valid JSON; vLLM's logprobs are Python floats.
        logprobs_content = ",".join(
            f'{{"token":"token_id:{token_id}","logprob":{logprob!r}}}'
            for token_id, logprob in zip(
                completion.token_ids, completion.token_logprobs
            )
        )
        response_text = json.dumps(
            {
                "request_id": completion.request_id,
                "choices": [
                    {
                        "index": 0,
                        "token_ids": completion.token_ids,
                        "logprobs": {"content": _LOGPROBS_CONTENT_PLACEHOLDER},
                        "finish_reason": completion.finish_reason,
                        **(
                            {}
                            if completion.routed_expert_ids is None
                            else {
                                "routed_experts": {
                                    "data": _ROUTED_EXPERTS_PLACEHOLDER,
                                    "shape": list(completion.routed_expert_ids.shape),
                                    "start": routed_experts_prompt_start,
                                    "dtype": str(completion.routed_expert_ids.dtype),
                                }
                            }
                        ),
                    }
                ],
                "prompt_logprobs": None,
                "kv_transfer_params": None,
            },
            # Compact, so the client's fast path finds `"routed_experts":{"data":"`.
            separators=(",", ":"),
        )
        head, _, tail = response_text.rpartition(
            json.dumps(_LOGPROBS_CONTENT_PLACEHOLDER)
        )
        response_text = f"{head}[{logprobs_content}]{tail}"
        if completion.routed_expert_ids is None:
            return web.Response(text=response_text, content_type="application/json")
        # Base64 needs no JSON escaping: splice it in as bytes instead of json.dumps scanning
        # and copying it (~33 MB at a 64k-token context; 112 -> 27 ms per response).
        head, _, tail = response_text.rpartition(
            json.dumps(_ROUTED_EXPERTS_PLACEHOLDER)
        )
        body = b"".join(
            (
                head.encode(),
                b'"',
                base64.b64encode(np.ascontiguousarray(completion.routed_expert_ids)),
                b'"',
                tail.encode(),
            )
        )
        return web.Response(body=body, content_type="application/json", charset="utf-8")


def _encode_metrics(metrics: list[m.Metric]) -> list[list]:
    """Write metrics as JSON rows, e.g. `Metric("generator/queue_time_ms", Mean(3.0, 2.0))` ->
    `["generator/queue_time_ms", "Mean", 3.0, 2.0]`. Raises for types other than Mean/Max/Min/Sum."""
    rows = []
    for metric in metrics:
        kind = type(metric.value).__name__
        if kind not in _METRIC_TYPES:
            raise TypeError(f"cannot send the {kind} metric {metric.key!r} over HTTP")
        rows.append([metric.key, kind, *vars(metric.value).values()])
    return rows


def _decode_metrics(rows: list[list]) -> list[m.Metric]:
    """Inverse of `_encode_metrics`."""
    return [m.Metric(key, _METRIC_TYPES[kind](*fields)) for key, kind, *fields in rows]


def completion_to_response(completion: Completion) -> dict:
    """The `"response_format": "torchtitan"` response. Raw float64 logprobs round-trip every
    value exactly, -inf and NaN included."""
    return {
        "request_id": completion.request_id,
        "torchtitan": {
            "token_ids": _base64(np.asarray(completion.token_ids, dtype=np.int32)),
            "logprobs": _base64(
                np.asarray(completion.token_logprobs, dtype=np.float64)
            ),
            "finish_reason": completion.finish_reason,
            "min_policy_version": completion.min_policy_version,
            "max_policy_version": completion.max_policy_version,
            "metrics": _encode_metrics(completion.metrics),
            "routed_expert_ids": (
                None
                if completion.routed_expert_ids is None
                else {
                    "data": _base64(completion.routed_expert_ids),
                    "shape": list(completion.routed_expert_ids.shape),
                    "dtype": str(completion.routed_expert_ids.dtype),
                }
            ),
            **(
                {}
                if completion.topk_token_ids is None
                else {
                    "topk_token_ids": _base64(completion.topk_token_ids),
                    "topk_logprobs": _base64(completion.topk_logprobs),
                }
            ),
        },
    }


def response_to_completion(payload: dict) -> Completion:
    """Inverse of `completion_to_response`."""
    block = payload["torchtitan"]
    token_ids = np.frombuffer(base64.b64decode(block["token_ids"]), dtype=np.int32)
    topk_token_ids = topk_logprobs = None
    if "topk_token_ids" in block:
        # [num_tokens, k]: k from the flat length; .copy() makes the read-only buffer writable.
        topk_token_ids = (
            np.frombuffer(base64.b64decode(block["topk_token_ids"]), dtype=np.int32)
            .reshape(len(token_ids), -1)
            .copy()
        )
        topk_logprobs = (
            np.frombuffer(base64.b64decode(block["topk_logprobs"]), dtype=np.float32)
            .reshape(len(token_ids), -1)
            .copy()
        )
    return Completion(
        min_policy_version=block["min_policy_version"],
        max_policy_version=block["max_policy_version"],
        request_id=payload["request_id"],
        token_ids=token_ids.tolist(),
        token_logprobs=np.frombuffer(
            base64.b64decode(block["logprobs"]), dtype=np.float64
        ).tolist(),
        topk_token_ids=topk_token_ids,
        topk_logprobs=topk_logprobs,
        finish_reason=block["finish_reason"],
        routed_expert_ids=(
            None
            if (routed := block["routed_expert_ids"]) is None
            else np.frombuffer(
                base64.b64decode(routed["data"]), dtype=routed["dtype"]
            ).reshape(routed["shape"])
        ),
        metrics=_decode_metrics(block["metrics"]),
    )


def _base64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array)).decode("ascii")


def _listening_socket() -> socket.socket:
    """One socket on a free port for IPv4 and IPv6, so the one advertised port works for either
    family."""
    if socket.has_dualstack_ipv6():
        return socket.create_server(
            ("::", 0), family=socket.AF_INET6, dualstack_ipv6=True
        )
    return socket.create_server(("0.0.0.0", 0))  # IPv6 is disabled on this host


def _validate_token_ids(value: object, *, field_name: str) -> list[int]:
    """Validate an untyped JSON value as integer token IDs and return it."""
    # type(), not isinstance(): bool is an int subclass. map and set run in C.
    if not isinstance(value, list) or not set(map(type, value)) <= {int}:
        raise ValueError(f"{field_name} must be a list of integer token IDs")
    return value


def _parse_sampling_config(value: object):
    """Convert Verifiers' vLLM sampling payload to TorchTitan config."""
    from torchtitan.rl.generator.types import SamplingConfig

    if not isinstance(value, dict):
        raise ValueError("sampling_params must be an object")
    supported = {
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "stop_token_ids",
        "num_topk_logprobs",
    }
    protocol_fields = {
        "logprobs",
        "skip_special_tokens",
    }
    unsupported = set(value) - supported - protocol_fields
    if unsupported:
        raise ValueError(f"unsupported sampling parameters: {sorted(unsupported)}")

    # The generator stops only on these ids; the renderer client always sends them.
    stop_token_ids = _validate_token_ids(
        value.get("stop_token_ids"),
        field_name="stop_token_ids",
    )
    # The rollouter always sends these. Without them the generator would silently use
    # SamplingConfig's defaults (temperature 0.8, max_tokens 100).
    missing = {"temperature", "top_p", "max_tokens"} - set(value)
    if missing:
        raise ValueError(f"sampling_params must include {sorted(missing)}")
    return SamplingConfig(
        temperature=float(value["temperature"]),
        top_p=float(value["top_p"]),
        max_tokens=int(value["max_tokens"]),
        seed=value.get("seed"),
        stop_token_ids=stop_token_ids,
        num_topk_logprobs=int(value.get("num_topk_logprobs", 0)),
    )
