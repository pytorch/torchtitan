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
from types import SimpleNamespace

from torchtitan.rl.distributed.routing.intra_generator import IntraGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoutingStrategy,
    StickySessionRoutingStrategy,
)

from torchtitan.rl.generator import (
    CloseRequest,
    GenerationRequest,
    LoopAction,
    ModelStateDictPullRequest,
    RequestDispatcher,
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
    dp_routing_strategy: RoutingStrategy.Config | None = None,
    reset_kv_cache_on_weight_sync: bool = False,
) -> VLLMGenerator:
    # Bypass __init__ (which builds the vLLM engine); set only the loop's state.
    # _decide_next_action delegates the in-flight check and routing to the
    # dispatcher, so wire up a bare one (no engine / GPU needed here).
    generator = object.__new__(VLLMGenerator)
    generator.config = SimpleNamespace(
        reset_kv_cache_on_weight_sync=reset_kv_cache_on_weight_sync
    )
    generator._engine_loop_condition = asyncio.Condition()
    generator._close_request = CloseRequest() if close_requested else None
    generator._model_state_dict_pull_request = model_state_dict_pull_request
    generator._queued_generation_requests = pending or []
    generator.policy_version = 0
    generator._group_min_policy_versions = {}
    generator._request_dispatcher = RequestDispatcher(
        rank=0,
        dp_rank=0,
        tp_rank=0,
        dp_degree=dp_size,
        broadcast_group=None,
        open_result_channel=None,
        intra_generator_router=IntraGeneratorRouter.Config(
            strategy=dp_routing_strategy or LeastLoadedRoutingStrategy.Config()
        ),
    )
    # A registered-but-unresolved future models in-flight work (possibly in a peer DP rank).
    if inflight:
        generator._request_dispatcher._rank0_generation_futures = {"inflight": object()}
    return generator


def _request(
    request_id: str = "r0",
    *,
    group_id: int = 0,
    routing_session_id: str | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2],
        sampling=SamplingConfig(),
        group_id=group_id,
        routing_session_id=routing_session_id or request_id,
    )


def _admit(generator: VLLMGenerator, request: GenerationRequest) -> int:
    """Queue ``request``, run one STEP decision, and return its min policy version."""
    generator._queued_generation_requests.append(request)
    decision = asyncio.run(generator._decide_next_action())
    assert decision.action is LoopAction.STEP
    return request.min_policy_version


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
    assert generator._request_dispatcher._rank0_dp_router is None


def test_step_with_empty_queue_when_only_in_flight_work_remains() -> None:
    # No queue, no pull, but a registered future means a request is still in flight
    # (possibly in a peer DP rank), so rank 0 must keep issuing STEP.
    decision = asyncio.run(_bare_generator(inflight=True)._decide_next_action())
    assert decision.action is LoopAction.STEP and decision.requests_per_dp_rank == [[]]


def test_step_routes_requests_across_dp_ranks() -> None:
    # Least-loaded over 3 idle DP ranks: r0 -> rank 0, r1 -> rank 1 (rank 0 now loaded).
    requests = [_request("r0"), _request("r1")]
    generator = _bare_generator(pending=requests, dp_size=3)
    decision = asyncio.run(generator._decide_next_action())
    assert decision.action is LoopAction.STEP
    assert decision.requests_per_dp_rank == [[requests[0]], [requests[1]], []]
    # Each request reserves one load unit on its chosen DP rank.
    dp_router = generator._request_dispatcher._rank0_dp_router
    assert dp_router._reservations == {"r0": 0, "r1": 1}
    assert [h.reserved_load for h in dp_router._handles] == [1, 1, 0]


def test_step_sticky_session_reuses_dp_rank() -> None:
    first = _request("r0", routing_session_id="s0")
    generator = _bare_generator(
        pending=[first],
        dp_size=3,
        dp_routing_strategy=StickySessionRoutingStrategy.Config(),
    )

    first_decision = asyncio.run(generator._decide_next_action())
    assert first_decision.action is LoopAction.STEP
    assert first_decision.requests_per_dp_rank == [[first], [], []]

    same_session = _request("r1", routing_session_id="s0")
    new_session = _request("r2", routing_session_id="s1")
    generator._queued_generation_requests = [same_session, new_session]

    second_decision = asyncio.run(generator._decide_next_action())
    assert second_decision.action is LoopAction.STEP
    assert second_decision.requests_per_dp_rank == [
        [same_session],
        [new_session],
        [],
    ]
    # r0 and r1 share session s0 -> same DP rank; r2's new session falls back.
    assert generator._request_dispatcher._rank0_dp_router._reservations == {
        "r0": 0,
        "r1": 0,
        "r2": 1,
    }


def test_step_pins_min_policy_version_per_group() -> None:
    generator = _bare_generator()
    generator.policy_version = 3
    assert _admit(generator, _request("t0", group_id=1, routing_session_id="s0")) == 3

    generator.policy_version = 4
    # Later turns and new rollouts of the group keep the salt its first admission
    # pinned, so they can reuse the group's KV; a new group pins the current version.
    assert _admit(generator, _request("t1", group_id=1, routing_session_id="s0")) == 3
    assert _admit(generator, _request("t2", group_id=1, routing_session_id="s1")) == 3
    assert _admit(generator, _request("t3", group_id=2, routing_session_id="s2")) == 4


def test_step_with_kv_reset_uses_current_version_without_pins() -> None:
    generator = _bare_generator(reset_kv_cache_on_weight_sync=True)
    generator.policy_version = 3
    assert _admit(generator, _request("t0", group_id=1, routing_session_id="s0")) == 3

    generator.policy_version = 4
    assert _admit(generator, _request("t1", group_id=1, routing_session_id="s0")) == 4
    assert generator._group_min_policy_versions == {}


def test_release_groups_drops_pins() -> None:
    generator = _bare_generator()
    generator.policy_version = 3
    _admit(generator, _request("t0", group_id=1, routing_session_id="s0"))
    _admit(generator, _request("t1", group_id=2, routing_session_id="s1"))

    # Releasing a group this generator never served is a no-op.
    asyncio.run(generator.release_groups([1, 9]))

    assert generator._group_min_policy_versions == {2: 3}
    generator.policy_version = 4
    assert _admit(generator, _request("t2", group_id=1, routing_session_id="s0")) == 4
