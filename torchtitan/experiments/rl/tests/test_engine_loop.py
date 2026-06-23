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


def _bare_generator(
    *,
    close_requested: bool = False,
    model_state_dict_pull_request: ModelStateDictPullRequest | None = None,
    pending: list[GenerationRequest] | None = None,
    inflight: bool = False,
    dp_size: int = 1,
) -> VLLMGenerator:
    # Bypass __init__ (which builds the vLLM engine); set only the loop's state.
    # _decide_next_action keys on rank 0's own _generation_futures (no engine), so
    # no fake engine is needed here.
    generator = object.__new__(VLLMGenerator)
    generator._engine_loop_condition = asyncio.Condition()
    generator._close_request = CloseRequest() if close_requested else None
    generator._model_state_dict_pull_request = model_state_dict_pull_request
    generator._queued_generation_requests = pending or []
    # A registered-but-unresolved future models in-flight work (possibly in a peer DP rank).
    generator._generation_futures = {"inflight": object()} if inflight else {}
    generator._dp_degree = dp_size
    return generator


def _request(request_id: str = "r0") -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
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
    # DP=1: a single DP rank holds the whole batch.
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [
        [request]
    ]
    assert generator._queued_generation_requests == []  # drained into the decision


def test_step_with_empty_queue_when_only_in_flight_work_remains() -> None:
    # No queue, no pull, but a registered future means a request is still in flight
    # (possibly in a peer DP rank), so rank 0 must keep issuing STEP.
    decision = asyncio.run(_bare_generator(inflight=True)._decide_next_action())
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [[]]


def test_step_routes_all_requests_to_highest_dp_for_now() -> None:
    # Hardcoded routing: every queued request lands in the highest DP rank.
    requests = [_request("r0"), _request("r1")]
    generator = _bare_generator(pending=requests, dp_size=3)
    decision = asyncio.run(generator._decide_next_action())
    assert decision.action is LoopAction.STEP
    assert decision.requests_per_dp_rank == [[], [], requests]
