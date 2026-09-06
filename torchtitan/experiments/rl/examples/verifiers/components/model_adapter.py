# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from aiohttp import web

from torchtitan.experiments.rl.rollout.types import GenerateFn

logger = logging.getLogger(__name__)

_SESSION_ID_HEADER = "X-Session-ID"


@dataclass(frozen=True, slots=True)
class VerifiersGenerationMetadata:
    """TorchTitan completion data that Verifiers does not retain in its trace.

    ``GenerationServer`` stores these entries by request ID under the Verifiers
    session ID. Trace conversion uses the request ID recorded for each Verifiers
    node to attach this metadata to the corresponding rollout turn.
    """

    min_policy_version: int
    """Oldest policy version used to generate the completion."""

    max_policy_version: int
    """Newest policy version used to generate the completion."""

    metrics: list
    """Generator metrics attached to the corresponding rollout turn."""


class GenerationServer:
    """Expose a TorchTitan ``GenerateFn`` through Verifiers' model API.

    A Verifiers environment calls a named HTTP model endpoint, while TitanRL
    supplies an in-process ``GenerateFn`` backed by its generator router. This
    server exposes Verifiers' token-generation endpoint, forwards each request
    to that function, and retains TorchTitan policy-version and metric metadata
    that the resulting Verifiers trace does not carry.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        model_id: str,
        max_model_len: int,
    ) -> None:
        self.host = host
        self.requested_port = port
        self.model_id = model_id
        self.max_model_len = max_model_len
        self.generate_fn: GenerateFn | None = None
        self.runner: web.AppRunner | None = None
        self.bound_port: int | None = None
        self.request_counts: dict[str, int] = {}
        self.generation_metadata: dict[str, dict[str, VerifiersGenerationMetadata]] = {}

    @property
    def port(self) -> int:
        if self.bound_port is None:
            raise RuntimeError("GenerationServer has not started")
        return self.bound_port

    def set_generate_fn(self, generate_fn: GenerateFn) -> None:
        self.generate_fn = generate_fn

    async def start(self) -> None:
        if self.runner is not None:
            return
        app = web.Application()
        app.router.add_get("/healthz", self._handle_health_request)
        app.router.add_get("/v1/models", self._handle_models_request)
        app.router.add_post("/inference/v1/generate", self._handle_generate_request)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.requested_port)
        await site.start()
        sockets = getattr(site._server, "sockets", None)
        if not sockets:
            await runner.cleanup()
            raise RuntimeError("generation server did not bind a listening socket")
        self.runner = runner
        self.bound_port = int(sockets[0].getsockname()[1])
        logger.info(
            "Verifiers generation server listening on http://%s:%d",
            self.host,
            self.bound_port,
        )

    async def close(self) -> None:
        runner = self.runner
        if runner is not None:
            await runner.cleanup()
        self.runner = None
        self.bound_port = None
        self.request_counts.clear()
        self.generation_metadata.clear()

    def pop_generation_metadata(
        self, session_id: str
    ) -> dict[str, VerifiersGenerationMetadata]:
        """Detach one rollout's metadata for `trace_to_rollout_turns`."""
        self.request_counts.pop(session_id, None)
        return self.generation_metadata.pop(session_id, {})

    async def _handle_health_request(self, request: web.Request) -> web.Response:
        del request
        return web.json_response({"status": "ok"})

    async def _handle_models_request(self, request: web.Request) -> web.Response:
        del request
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.model_id,
                        "object": "model",
                        "owned_by": "torchtitan",
                        "max_model_len": self.max_model_len,
                    }
                ],
            }
        )

    async def _handle_generate_request(self, request: web.Request) -> web.Response:
        if self.generate_fn is None:
            return web.json_response(
                {"error": "TorchTitan GenerateFn is not ready"}, status=503
            )
        session_id = request.headers.get(_SESSION_ID_HEADER)
        if not session_id:
            return web.json_response(
                {"error": f"missing {_SESSION_ID_HEADER} header"}, status=400
            )

        try:
            body = await request.json()
            prompt_token_ids = _validate_token_ids(
                body.get("token_ids"), field_name="token_ids"
            )
            sampling = _parse_sampling_config(body.get("sampling_params"))
            if body.get("features") is not None:
                raise ValueError("multimodal features are not supported")
        except (TypeError, ValueError) as error:
            return web.json_response({"error": str(error)}, status=400)

        request_index = self.request_counts.get(session_id, 0)
        self.request_counts[session_id] = request_index + 1
        request_id = f"{session_id}/request={request_index}"
        try:
            completion = await self.generate_fn(
                prompt_token_ids,
                request_id=request_id,
                routing_session_id=session_id,
                sampling_config=sampling,
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
        if completion.finish_reason not in ("stop", "length"):
            return web.json_response(
                {
                    "error": "generation finished without a usable completion: "
                    f"{completion.finish_reason}"
                },
                status=502,
            )

        self.generation_metadata.setdefault(session_id, {})[
            completion.request_id
        ] = VerifiersGenerationMetadata(
            min_policy_version=completion.min_policy_version,
            max_policy_version=completion.max_policy_version,
            metrics=list(completion.metrics),
        )
        return web.json_response(
            {
                "request_id": completion.request_id,
                "choices": [
                    {
                        "index": 0,
                        "token_ids": completion.token_ids,
                        "logprobs": {
                            "content": [
                                {
                                    "token": f"token_id:{token_id}",
                                    "logprob": logprob,
                                }
                                for token_id, logprob in zip(
                                    completion.token_ids,
                                    completion.token_logprobs,
                                    strict=True,
                                )
                            ]
                        },
                        "finish_reason": completion.finish_reason,
                    }
                ],
                "prompt_logprobs": None,
                "kv_transfer_params": None,
            }
        )


def _validate_token_ids(value: object, *, field_name: str) -> list[int]:
    """Validate an untyped JSON value as integer token IDs and return a copy."""
    if not isinstance(value, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in value
    ):
        raise ValueError(f"{field_name} must be a list of integer token IDs")
    return list(value)


def _parse_sampling_config(value: object):
    """Convert Verifiers' vLLM sampling payload to TorchTitan config."""
    from torchtitan.experiments.rl.actors.generator import SamplingConfig

    if not isinstance(value, dict):
        raise ValueError("sampling_params must be an object")
    supported = {
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "stop_token_ids",
    }
    protocol_fields = {
        "logprobs",
        "skip_special_tokens",
        "routed_experts_prompt_start",
    }
    unsupported = set(value) - supported - protocol_fields
    if unsupported:
        raise ValueError(f"unsupported sampling parameters: {sorted(unsupported)}")

    stop_token_ids = value.get("stop_token_ids")
    if stop_token_ids is not None:
        stop_token_ids = _validate_token_ids(
            stop_token_ids,
            field_name="stop_token_ids",
        )
    return SamplingConfig(
        temperature=float(value.get("temperature", 0.8)),
        top_p=float(value.get("top_p", 0.95)),
        max_tokens=int(value.get("max_tokens", 100)),
        seed=value.get("seed"),
        stop_token_ids=stop_token_ids,
    )
