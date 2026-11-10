# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`KVGrowthEstimateAdmission` on a fake clock. 10-token blocks and no fixed blocks: an N-token
session holds N / 10 blocks, and a max-length context (200 tokens) 20."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import torchtitan.rl.routing.admission as admission_module
from torchtitan.rl.generator.engine import KVCacheBudget
from torchtitan.rl.routing.admission import KVGrowthEstimateAdmission

_BUDGET = KVCacheBudget(
    num_blocks=1000,
    block_size=10,
    num_growing_kv_cache_groups=1,
    fixed_blocks_per_session=0,
    max_model_len=200,
)


class _Clock:
    """Stands in for `time.monotonic` in the admission module only."""

    def __init__(self, monkeypatch):
        self.now = 1000.0
        monkeypatch.setattr(
            admission_module, "time", SimpleNamespace(monotonic=lambda: self.now)
        )


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    return _Clock(monkeypatch)


def _policy(limit=1.0, num_generators=2) -> KVGrowthEstimateAdmission:
    return KVGrowthEstimateAdmission.Config(limit=limit).build(
        budgets=[_BUDGET] * num_generators
    )


def _place(policy, group_id: int, group_size: int = 1) -> int | None:
    """Place one waiting group; return its generator, or None if it waits."""
    return policy.place({group_id: group_size}).get(group_id)


def _growth_per_session(policy) -> float | None:
    policy.charges()  # recomputes R
    return policy.growth_per_session


def _play(policy, group_id: int, session: str, *contexts: int | None) -> None:
    """Make a session's calls: each context is its tokens at a call start or end; None ends it."""
    for num_tokens in contexts:
        if num_tokens is None:
            policy.release_session(session)
        else:
            policy.set_session_tokens(
                group_id, session_id=session, num_tokens=num_tokens
            )


def test_r_is_the_growth_per_ended_session(clock):
    policy = _policy()
    assert _place(policy, 0) == 0
    # 49 sessions grow from 1 to 7 blocks, then end: still the cold start.
    for i in range(49):
        _play(policy, 0, f"s{i}", 10, 70, None)
    assert _growth_per_session(policy) is None
    # The 50th ended session gives R = growth / ends: (49 x 6 + 12) / 50 = 6.12 blocks.
    _play(policy, 0, "s49", 10, 130, None)
    assert _growth_per_session(policy) == pytest.approx(6.12)
    # 15 min with no session ending: R keeps its last estimate; the cold start does not come back.
    clock.now += 901.0
    assert _growth_per_session(policy) == pytest.approx(6.12)
    # Then only the sessions that ended in the last 15 min count: 2 that grew 3 blocks.
    for i in range(2):
        _play(policy, 0, f"late{i}", 10, 40, None)
    assert _growth_per_session(policy) == pytest.approx(3.0)


def test_cold_start_ends_after_50_sessions_however_slowly_they_end(clock):
    policy = _policy()
    assert _place(policy, 0) == 0
    # Long sessions: 30 end, then 20 more 20 minutes later, so the 15-min window never holds 50.
    for i in range(30):
        _play(policy, 0, f"early{i}", 10, 40, None)
    clock.now += 1200.0
    for i in range(20):
        _play(policy, 0, f"late{i}", 10, 50, None)
    # R comes from the 20 in the window, which grew 4 blocks each.
    assert _growth_per_session(policy) == pytest.approx(4.0)


def test_every_session_counts_as_a_max_length_context_until_50_end():
    policy = _policy()
    assert _place(policy, 0) == 0
    # 49 sessions end having grown G = 10 (25 of them) or 2 (24) blocks; a live one grew g = 4.
    for i in range(49):
        _play(policy, 0, f"ended{i}", 10, 110 if i < 25 else 30, None)
    _play(policy, 0, "live", 10, 50)
    assert _place(policy, 1) == 1
    # Fewer than 50 ended: the live session, and group 1's session not started yet, count as
    # max-length contexts.
    assert policy.charges() == [20.0, 20.0]
    # The 50th ends: R = (25 x 10 + 25 x 2 + 4) / 50 = 6.08. The live session expects
    # max(10 - 4, R - 4) = 6 more; group 1's charges the mean first call, 1, plus R.
    _play(policy, 0, "ended49", 10, 30, None)
    assert policy.charges() == pytest.approx([5 + 6.0, 1 + 6.08])


def test_live_session_expects_its_remaining_growth():
    policy = _policy()
    assert _place(policy, 0) == 0
    # 50 sessions end having grown G = 4 (26 of them) or 16 (24) blocks.
    for i in range(50):
        _play(policy, 0, f"ended{i}", 10, 50 if i < 26 else 170, None)
    # Two live sessions grew g = 0 and g = 12: R = (26 x 4 + 24 x 16 + 12) / 50 = 10.
    _play(policy, 0, "young", 10)
    _play(policy, 0, "old", 10, 130)
    assert _growth_per_session(policy) == pytest.approx(10.0)
    # Each charges its blocks plus max(mean(G - g | G > g), R - g): young 1 + max(9.76, 10),
    # old 13 + max(16 - 12, -2).
    assert policy.charges()[0] == pytest.approx(1 + 10.0 + 13 + 4.0)


def test_remaining_growth_averages_the_ended_sessions_that_grew_more():
    remaining = admission_module._remaining_growth([4, 10, 16])
    # A session that grew 4 counts only 10 and 16 as above it; past 16, none is.
    assert [remaining(grown) for grown in (0, 4, 12, 16, 20)] == [10, 9, 4, 0, 0]


def test_session_not_started_charges_the_mean_first_call_plus_r():
    policy = _policy()
    assert _place(policy, 0) == 0
    # 50 sessions end having grown 5 blocks: R = 5.
    for i in range(50):
        _play(policy, 0, f"ended{i}", 10, 60, None)
    policy.release_group(0)
    # Group 1's first session starts at 3 blocks; its other session has not started.
    assert _place(policy, 1, group_size=2) == 0
    _play(policy, 1, "a", 30)
    assert policy.charges() == [(3 + 5.0) + (3 + 5.0), 0.0]
    # With no session live, a new session still charges the last mean first call, 3:
    # "a" ended having grown 0, so R = 250 / 51.
    _play(policy, 1, "a", None)
    policy.release_group(1)
    assert _place(policy, 2, group_size=2) == 0
    assert policy.charges()[0] == pytest.approx(2 * (3 + 250 / 51))


def test_no_session_charges_more_than_a_max_length_context():
    policy = _policy()
    assert _place(policy, 0) == 0
    # 200 live sessions grow 5 blocks each while only 50 end, having grown 1: just after the cold
    # start, R = (200 x 5 + 50 x 1) / 50 = 21 blocks, past a max-length context (20).
    for i in range(200):
        _play(policy, 0, f"live{i}", 10, 60)
    for i in range(50):
        _play(policy, 0, f"ended{i}", 10, 20, None)
    assert _place(policy, 1) == 1
    assert _growth_per_session(policy) == pytest.approx(21.0)
    # A live session: 6 + max(0, 21 - 5) = 22 -> 20. Group 1's session not started: 1 + 21 -> 20.
    assert policy.charges() == [200 * 20.0, 20.0]


def test_sessions_live_at_group_release_count_as_ended():
    policy = _policy()
    # Sessions that end only with their group, e.g. after their rollout raised: 25 groups of 2
    # sessions that grew 3 blocks.
    for group_id in range(25):
        _place(policy, group_id, group_size=2)
        for session in ("s0", "s1"):
            _play(policy, group_id, f"g{group_id}/{session}", 10, 40)
        policy.release_group(group_id)
    # 50 sessions ended, so R is learned: 50 x 3 / 50.
    assert _growth_per_session(policy) == pytest.approx(3.0)


def test_waiting_groups_go_lowest_id_first_to_the_generator_with_the_most_room():
    # 70 blocks per generator; each group counts one 20-block context. Groups 6 and 7 wait.
    policy = _policy(limit=0.07)
    group_sizes = {group_id: 1 for group_id in (7, 6, 5, 4, 3, 2, 1, 0)}
    assert policy.place(group_sizes) == {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}
    assert policy.charges() == [60.0, 60.0]


def test_generator_charging_nothing_goes_first_even_with_less_room():
    # Limits 1,000 and 980 blocks; a group of 60 sessions counts 1,200, over both.
    policy = KVGrowthEstimateAdmission.Config(limit=1.0).build(
        budgets=[_BUDGET, replace(_BUDGET, num_blocks=980)]
    )
    # A group of one session on generator 0 leaves both with 980 blocks of room.
    assert _place(policy, 0) == 0
    assert _place(policy, 1, group_size=60) == 1


def test_generator_charging_nothing_takes_a_group_larger_than_its_limit():
    # 10 blocks per generator; a group of 2 sessions counts 40.
    policy = _policy(limit=0.01)
    assert [_place(policy, group_id, group_size=2) for group_id in range(3)] == [
        0,
        1,
        None,
    ]
    policy.release_group(1)
    # Group 0's sessions both end; the group stays placed, but charges nothing.
    for session in ("a", "b"):
        _play(policy, 0, session, 10, None)
    assert _place(policy, 2, group_size=2) == 0
    assert _place(policy, 3, group_size=2) == 1


def test_call_returning_after_its_session_ended_charges_nothing():
    policy = _policy()
    assert _place(policy, 0) == 0
    # The rollout releases its session while a call is still running (e.g. an agent timeout); the
    # call returns afterwards.
    _play(policy, 0, "t", 50, None, 120)
    assert policy.charges()[0] == 0.0
