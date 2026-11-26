# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.rl.generator.engine import KVCacheCapacity
from torchtitan.rl.routing.admission import KVRoomAdmission


def _capacity(*, num_blocks: int = 100, num_state_slots: int | None = None):
    """10-token blocks; a GDN state is 2 blocks, 1 per idle session, 3 per running one."""
    return KVCacheCapacity(
        num_blocks=num_blocks,
        attention_block_sizes=(10,),
        blocks_per_state=2,
        num_idle_states=1,
        num_running_states=3,
        num_state_slots=num_state_slots,
    )


def _admission(*capacities: KVCacheCapacity) -> KVRoomAdmission:
    return KVRoomAdmission.Config().build(capacities=capacities)


def _charged_blocks(admission: KVRoomAdmission, generator: int = 0) -> int:
    return admission._charges()[generator].num_blocks


def test_before_any_first_prompt_each_generator_takes_one_group():
    admission = _admission(_capacity(), _capacity())

    assert admission.place(0, num_sessions=1) == 0
    assert admission.place(1, num_sessions=1) == 1
    # Neither group has called: their charge is unknown, so no third group yet.
    assert admission.place(2, num_sessions=1) is None
    # Before any call, a waiting session charges 0 tokens: its states only.
    assert _charged_blocks(admission) == 3 * 2

    # Both charge ceil(100 / 10) + 6 = 16 once a first prompt is known; a tie goes to 0.
    admission.start_call("r0", session_id="0/0", group_id=0, num_prompt_tokens=100)
    assert admission.place(2, num_sessions=1) == 0


def test_places_a_group_where_blocks_and_states_fit():
    admission = _admission(_capacity(), _capacity())
    admission.place(0, num_sessions=1)
    admission.place(1, num_sessions=1)
    # Running: ceil(400 / 10) + 3 x 2 = 46 blocks on 0; ceil(100 / 10) + 6 = 16 on 1.
    admission.start_call("r0", session_id="0/0", group_id=0, num_prompt_tokens=400)
    admission.start_call("r1", session_id="1/0", group_id=1, num_prompt_tokens=100)

    # A new session: the mean first prompt, 250 tokens, = 31 blocks. Room left after it:
    # 95 - 46 - 31 = 18 on 0, 95 - 16 - 31 = 48 on 1. The most room wins.
    assert admission.place(2, num_sessions=1) == 1
    # 2 sessions fit neither: 95 - 46 - 62 < 0, and 95 - 16 - 31 - 62 < 0.
    assert admission.place(3, num_sessions=2) is None
    # A release frees room.
    admission.release_groups([0])
    assert admission.place(3, num_sessions=2) == 0

    # With 10 state slots, 3 running states per session: 3 + 3 x 3 = 12 > 9.5 slots.
    admission = _admission(_capacity(num_blocks=1000, num_state_slots=10))
    admission.place(0, num_sessions=1)
    admission.start_call("r0", session_id="0/0", group_id=0, num_prompt_tokens=10)
    assert admission.place(1, num_sessions=3) is None
    assert admission.place(1, num_sessions=2) == 0
    # The gauge reports the fuller of blocks and slots: 9 of 10 slots.
    assert admission.charged_fractions() == [pytest.approx(0.9)]


def test_charges_prompt_and_reply_and_counts_overlapping_calls():
    admission = _admission(_capacity())
    admission.place(0, num_sessions=2)

    # s0 runs: ceil(40 / 10) + 6 = 10. s1 has not called: the mean first prompt, 40 -> 10.
    admission.start_call("r0", session_id="s0", group_id=0, num_prompt_tokens=40)
    assert _charged_blocks(admission) == 10 + 10
    # Idle: its prompt + reply, ceil(60 / 10) + 1 x 2 = 8. s1: 40 + mean reply 20 -> 12.
    admission.end_call("r0", num_reply_tokens=20)
    assert _charged_blocks(admission) == 8 + 12

    # A retry overlaps the call it repeats: the newest prompt + the mean reply,
    # 75 + 20 -> 16. The older call ending late leaves it running.
    admission.start_call("r1", session_id="s0", group_id=0, num_prompt_tokens=70)
    admission.start_call("r2", session_id="s0", group_id=0, num_prompt_tokens=75)
    assert _charged_blocks(admission) == 16 + 12
    admission.end_call("r1", num_reply_tokens=30)
    # The mean reply is now 25: s0 runs at 75 + 25 = 100 -> 16, s1 at 40 + 25 -> 13.
    assert _charged_blocks(admission) == 16 + 13
    admission.end_call("r2", num_reply_tokens=5)
    # s0 idle at 75 + 5 = 80 -> 10; the mean reply (20 + 30 + 5) // 3 = 18 -> s1 at 58, 12.
    assert _charged_blocks(admission) == 10 + 12

    # A retried rollout opens a third session in a group of 2: nothing waits any more.
    admission.start_call("r3", session_id="s1", group_id=0, num_prompt_tokens=40)
    admission.start_call("r4", session_id="s2", group_id=0, num_prompt_tokens=40)
    # s1 and s2 run at 40 + 18 -> 12 each.
    assert _charged_blocks(admission) == 10 + 12 + 12
    assert admission.release_session("s0") == 0
    assert _charged_blocks(admission) == 12 + 12


def test_late_releases_are_no_ops():
    admission = _admission(_capacity())
    admission.place(0, num_sessions=1)
    admission.start_call("r0", session_id="s0", group_id=0, num_prompt_tokens=40)

    admission.release_groups([0])

    # The call's end and the session's release arrive after the group's release.
    admission.end_call("r0", num_reply_tokens=20)
    assert admission.release_session("s0") is None
    admission.end_call("unknown", num_reply_tokens=1)
    admission.release_groups([0, 5])
    assert admission.charged_fractions() == [0.0]
    # A call of a released group raises to its caller.
    with pytest.raises(ValueError, match="group 0"):
        admission.start_call("r1", session_id="s0", group_id=0, num_prompt_tokens=40)
