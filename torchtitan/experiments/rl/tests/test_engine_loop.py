# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the generator engine loop's decision logic (`_decide_next_action`).

Built on a bare `VLLMGenerator` (no vLLM engine) + a fake engine, so the loop's
admit/pull/shutdown branching is tested without a GPU.
"""

from __future__ import annotations

import asyncio

from torchtitan.experiments.rl.actors.generator import (
    CloseRequest,
    GenerationRequest,
    LoopAction,
    ModelStateDictPullRequest,
    SamplingConfig,
    VLLMGenerator,
)


class _FakeEngine:
    def __init__(self, *, unfinished: bool = False) -> None:
        self._unfinished = unfinished

    def has_unfinished_requests(self) -> bool:
        return self._unfinished


def _bare_generator(
    *,
    close_requested: bool = False,
    model_state_dict_pull_request: ModelStateDictPullRequest | None = None,
    pending: list[GenerationRequest] | None = None,
    unfinished: bool = False,
) -> VLLMGenerator:
    # Bypass __init__ (which builds the vLLM engine); set only the loop's state.
    generator = object.__new__(VLLMGenerator)
    generator._condition = asyncio.Condition()
    generator._close_request = CloseRequest() if close_requested else None
    generator._model_state_dict_pull_request = model_state_dict_pull_request
    generator._queued_generation_requests = pending or []
    generator._engine = _FakeEngine(unfinished=unfinished)
    return generator


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="r0",
        prompt_token_ids=[1, 2],
        sampling=SamplingConfig(),
    )


def test_closing_returns_close() -> None:
    decision = asyncio.run(_bare_generator(close_requested=True)._decide_next_action())
    assert decision.action is LoopAction.CLOSE


def test_pull_takes_precedence_over_queued_requests() -> None:
    request = _request()
    pull = ModelStateDictPullRequest(version=5)
    generator = _bare_generator(model_state_dict_pull_request=pull, pending=[request])
    decision = asyncio.run(generator._decide_next_action())
    assert (
        decision.action is LoopAction.PULL_MODEL_STATE_DICT
        and decision.pull_version == 5
    )
    # `_model_state_dict_pull_request` is NOT cleared at decide — the PULL_MODEL_STATE_DICT branch clears it after
    # applying; the single-threaded loop can't re-decide before then, so the predicate won't re-fire.
    assert generator._model_state_dict_pull_request is pull
    assert generator._queued_generation_requests == [
        request
    ]  # NOT consumed — pull runs first


def test_step_drains_the_queue() -> None:
    request = _request()
    generator = _bare_generator(pending=[request])
    decision = asyncio.run(generator._decide_next_action())
    assert decision.action is LoopAction.STEP and decision.requests == [request]
    assert generator._queued_generation_requests == []  # drained into the decision


def test_step_with_empty_queue_when_only_in_flight_work_remains() -> None:
    # No queue, no pull, but the engine still has in-flight requests to step.
    decision = asyncio.run(_bare_generator(unfinished=True)._decide_next_action())
    assert decision.action is LoopAction.STEP and decision.requests == []
