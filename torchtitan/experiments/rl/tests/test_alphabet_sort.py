# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the `examples.alphabet_sort` task (dataset, env, rubric, rollouter)."""

from __future__ import annotations

import asyncio

import pytest

from torchtitan.experiments.rl.examples.alphabet_sort import (
    AlphabetSortDataset,
    AlphabetSortRollouter,
    AlphabetSortSample,
    data as alphabet_data,
    RewardAlphabetSort,
)
from torchtitan.experiments.rl.examples.alphabet_sort.env import AlphabetSortEnv
from torchtitan.experiments.rl.examples.alphabet_sort.rubric import score_sorted_list
from torchtitan.experiments.rl.rollout import Rollout, RolloutStatus, RolloutTurn
from torchtitan.experiments.rl.types import RolloutTurnID


_Author = alphabet_data._Author
_AUTHORS = (
    _Author("AliceZephyr", "alice", "zephyr"),
    _Author("AliceYoung", "alice", "young"),  # ties on first name
    _Author("BobYang", "bob", "yang"),
    _Author("CarolYang", "carol", "yang"),  # ties on last name
    _Author("DanWalsh", "dan", "walsh"),
    _Author("EveVance", "eve", "vance"),
    _Author("FrankStone", "frank", "stone"),
    _Author("GraceReyes", "grace", "reyes"),
)
_AUTHOR_BY_DISPLAY = {author.display: author for author in _AUTHORS}


def _patch_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alphabet_data, "_load_authors", lambda *a, **k: _AUTHORS)


def _assistant_turn(content: str) -> RolloutTurn:
    return RolloutTurn(
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
        prompt_token_ids=[],
        completion_token_ids=[],
        completion_logprobs=[],
        min_policy_version=0,
        completion_message={"role": "assistant", "content": content},
    )


def _rollout(turns: list[RolloutTurn]) -> Rollout:
    return Rollout(
        group_id=0,
        rollout_id=0,
        status=RolloutStatus.COMPLETED,
        turns=turns,
    )


# --- Dataset ---


def test_dataset_rejects_invalid_max_turns() -> None:
    with pytest.raises(ValueError, match="max_turns must be >= 1"):
        AlphabetSortDataset.Config(max_turns=0)
    with pytest.raises(ValueError, match="max_names_per_turn must be >= 1"):
        AlphabetSortDataset.Config(max_names_per_turn=0)


def test_dataset_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_names(monkeypatch)
    a = AlphabetSortDataset(AlphabetSortDataset.Config(seed=7))
    b = AlphabetSortDataset(AlphabetSortDataset.Config(seed=7))
    assert next(a) == next(b)


def test_state_dict_resumes_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_names(monkeypatch)
    # Pin small bounds so the 8-author fixture can supply every turn (drawn without replacement).
    config = AlphabetSortDataset.Config(seed=3, max_turns=2, max_names_per_turn=2)
    dataset = AlphabetSortDataset(config)
    next(dataset)
    next(dataset)
    checkpoint = dataset.state_dict()
    expected_after = [next(dataset), next(dataset)]

    resumed = AlphabetSortDataset(config)
    resumed.load_state_dict(checkpoint)
    assert [next(resumed), next(resumed)] == expected_after


def test_single_turn_sorted_by_first_or_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_names(monkeypatch)
    dataset = AlphabetSortDataset(AlphabetSortDataset.Config(seed=0, max_turns=1))
    seen_first = seen_last = False
    for _ in range(50):
        sample = next(dataset)
        assert len(sample.expected_names) == 1  # max_turns=1 -> always one turn

        def key(display: str) -> tuple[str, str]:
            author = _AUTHOR_BY_DISPLAY[display]
            return (
                (author.first, author.last)
                if sample.sort_by_first_name
                else (author.last, author.first)
            )

        expected = sorted(sample.new_names_per_turn[0], key=key)
        assert list(sample.expected_names[0]) == expected
        seen_first |= sample.sort_by_first_name
        seen_last |= not sample.sort_by_first_name
    assert seen_first and seen_last


def test_turns_and_names_are_sampled_within_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_names(monkeypatch)
    # 3 turns x 2 names <= the 8 patched authors (names are drawn without replacement).
    dataset = AlphabetSortDataset(
        AlphabetSortDataset.Config(seed=0, max_turns=3, max_names_per_turn=2)
    )
    seen_turn_counts: set[int] = set()
    for _ in range(100):
        sample = next(dataset)
        seen_turn_counts.add(len(sample.new_names_per_turn))
        assert 1 <= len(sample.new_names_per_turn) <= 3
        assert all(1 <= len(names) <= 2 for names in sample.new_names_per_turn)
    assert seen_turn_counts == {1, 2, 3}  # turns drawn uniformly across the range


def test_names_never_repeat_across_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    # Names are drawn without replacement, so slicing them into per-turn chunks never repeats a
    # name within a sample (across turns) — the cumulative answer has no duplicates to mis-tag.
    _patch_names(monkeypatch)
    dataset = AlphabetSortDataset(
        AlphabetSortDataset.Config(seed=0, max_turns=3, max_names_per_turn=2)
    )
    for _ in range(100):
        sample = next(dataset)
        shown = [
            name for turn_names in sample.new_names_per_turn for name in turn_names
        ]
        assert len(shown) == len(set(shown))  # no name appears in two turns


def test_multiturn_targets_are_cumulative_and_tagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_names(monkeypatch)
    dataset = AlphabetSortDataset(
        AlphabetSortDataset.Config(seed=11, max_turns=3, max_names_per_turn=2)
    )
    # Find a genuinely multi-turn sample (turns are sampled, so not every draw is one).
    sample = next(
        s for s in (next(dataset) for _ in range(100)) if len(s.new_names_per_turn) >= 2
    )

    names_shown = [len(names) for names in sample.new_names_per_turn]
    cumulative = [sum(names_shown[: i + 1]) for i in range(len(names_shown))]
    # The cumulative answer grows by exactly the names introduced each turn.
    assert [len(lines) for lines in sample.expected_names] == cumulative
    # Turn 0 marks nothing; each later turn tags exactly the names new that turn.
    assert all("// new name!" not in line for line in sample.expected_names[0])
    for turn_idx in range(1, len(sample.expected_names)):
        marked = sum("// new name!" in line for line in sample.expected_names[turn_idx])
        assert marked == names_shown[turn_idx]


def test_ties_broken_by_other_name_part(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two authors share a first name. Whenever both are shown, the order must be reproducible
    # from the names alone (broken by the other part), not the dataset's sampling order.
    pair = (
        _Author("AliceZephyr", "alice", "zephyr"),
        _Author("AliceYoung", "alice", "young"),
    )
    monkeypatch.setattr(alphabet_data, "_load_authors", lambda *a, **k: pair)
    dataset = AlphabetSortDataset(
        AlphabetSortDataset.Config(seed=0, max_turns=1, max_names_per_turn=2)
    )
    checked_both = False
    for _ in range(50):
        sample = next(dataset)
        if len(sample.new_names_per_turn[0]) == 2:  # both authors drawn this turn
            # "young" < "zephyr" breaks the "alice" tie either way (by-first or by-last).
            assert sample.expected_names[0] == ("AliceYoung", "AliceZephyr")
            checked_both = True
    assert checked_both


# --- Grader ---


def test_score_exact_swapped_missing_and_garbage() -> None:
    correct = ("AliceZephyr", "BobYang", "CarolXu")
    exact = (
        "<alphabetical_sorted>\nAliceZephyr\nBobYang\nCarolXu\n</alphabetical_sorted>"
    )
    swapped = (
        "<alphabetical_sorted>\nAliceZephyr\nCarolXu\nBobYang\n</alphabetical_sorted>"
    )
    garbage = "<alphabetical_sorted>\nName1\nName2\n</alphabetical_sorted>"

    def score(text: str) -> float:
        return score_sorted_list(
            text,
            expected_names=correct,
            xml_tag="alphabetical_sorted",
            similarity_power=4,
        )

    assert score(exact) == 1.0
    assert score(exact) > score(swapped)
    assert score("no block here") == 0.0
    assert score(garbage) < 0.05  # echoed template names are nothing like the answer


def test_score_uses_the_last_block() -> None:
    correct = ("AliceZephyr", "BobYang")
    text = (
        "<alphabetical_sorted>\nBobYang\nAliceZephyr\n</alphabetical_sorted>\n"
        "<alphabetical_sorted>\nAliceZephyr\nBobYang\n</alphabetical_sorted>"
    )
    assert (
        score_sorted_list(
            text,
            expected_names=correct,
            xml_tag="alphabetical_sorted",
            similarity_power=4,
        )
        == 1.0
    )


def test_followup_turns_use_the_combined_tag() -> None:
    correct = ("AliceZephyr", "BobYang", "CarolXu // new name!")
    text = (
        "<combined_alphabetical_sorted>\n"
        "AliceZephyr\nBobYang\nCarolXu // new name!\n"
        "</combined_alphabetical_sorted>"
    )
    # Read with the combined tag -> match; with the turn-0 tag -> block not found.
    assert (
        score_sorted_list(
            text,
            expected_names=correct,
            xml_tag="combined_alphabetical_sorted",
            similarity_power=4,
        )
        == 1.0
    )
    assert (
        score_sorted_list(
            text,
            expected_names=correct,
            xml_tag="alphabetical_sorted",
            similarity_power=4,
        )
        == 0.0
    )


def test_reward_averages_every_turn() -> None:
    rollout = _rollout(
        [
            _assistant_turn(
                "<alphabetical_sorted>\nAliceZephyr\nBobYang\n</alphabetical_sorted>"
            ),
            _assistant_turn(
                "<combined_alphabetical_sorted>\n"
                "AliceZephyr\nBobYang\nCarolXu // new name!\n"
                "</combined_alphabetical_sorted>"
            ),
        ]
    )
    env_input = AlphabetSortSample(
        new_names_per_turn=(("BobYang", "AliceZephyr"), ("CarolXu",)),
        expected_names=(
            ("AliceZephyr", "BobYang"),
            ("AliceZephyr", "BobYang", "CarolXu // new name!"),
        ),
        sort_by_first_name=True,
    )
    reward = RewardAlphabetSort(RewardAlphabetSort.Config(similarity_power=4))
    assert asyncio.run(reward(rollout, env_input)) == pytest.approx(1.0)


def test_reward_penalizes_incomplete_rollout() -> None:
    # Two expected turns, but the rollout only answered turn 0 (perfectly). Dividing by
    # EXPECTED turns (not answered) halves the reward — an early exit is not full credit.
    rollout = _rollout(
        [
            _assistant_turn(
                "<alphabetical_sorted>\nAliceZephyr\nBobYang\n</alphabetical_sorted>"
            )
        ]
    )
    env_input = AlphabetSortSample(
        new_names_per_turn=(("BobYang", "AliceZephyr"), ("CarolXu",)),
        expected_names=(
            ("AliceZephyr", "BobYang"),
            ("AliceZephyr", "BobYang", "CarolXu // new name!"),
        ),
        sort_by_first_name=True,
    )
    reward = RewardAlphabetSort(RewardAlphabetSort.Config(similarity_power=4))
    assert asyncio.run(reward(rollout, env_input)) == pytest.approx(0.5)


# --- Env ---


def test_env_single_turn_finishes_after_first_answer() -> None:
    env = AlphabetSortEnv(
        AlphabetSortEnv.Config(),
        env_input=AlphabetSortSample(
            new_names_per_turn=(("BobYang", "AliceZephyr"),),
            expected_names=(("AliceZephyr", "BobYang"),),
            sort_by_first_name=True,
        ),
    )
    init = asyncio.run(env.init())
    prompt = init.init_prompt_messages[0]["content"]
    assert "BobYang" in prompt and "AliceZephyr" in prompt
    assert "<alphabetical_sorted>" in prompt

    step = asyncio.run(env.step({"role": "assistant", "content": "x"}))
    assert step.done is True


def test_env_walks_through_follow_up_turns() -> None:
    env = AlphabetSortEnv(
        AlphabetSortEnv.Config(),
        env_input=AlphabetSortSample(
            new_names_per_turn=(("CarolXu", "AliceZephyr"), ("BobYang",)),
            expected_names=(
                ("AliceZephyr", "CarolXu"),
                ("AliceZephyr", "BobYang // new name!", "CarolXu"),
            ),
            sort_by_first_name=True,
        ),
    )
    init = asyncio.run(env.init())
    assert "<alphabetical_sorted>" in init.init_prompt_messages[0]["content"]

    follow_up = asyncio.run(env.step({"role": "assistant", "content": "x"}))
    assert follow_up.done is False
    second_prompt = follow_up.env_messages[0]["content"]
    assert "<combined_alphabetical_sorted>" in second_prompt
    assert "// new name!" in second_prompt and "BobYang" in second_prompt

    done = asyncio.run(env.step({"role": "assistant", "content": "y"}))
    assert done.done is True
    assert done.env_messages == []


# --- Rollouter ---


def test_rollouter_builds_one_env_per_group_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_names(monkeypatch)
    rollouter = AlphabetSortRollouter(AlphabetSortRollouter.Config())
    sample = rollouter.get_training_sample()
    envs = rollouter.make_env_group(sample=sample, num_samples_per_prompt=3, renderer=None)
    assert len(envs) == 3


def test_rollouter_config_wires_alphabet_sort() -> None:
    config = AlphabetSortRollouter.Config()
    assert isinstance(config.train_dataset, AlphabetSortDataset.Config)
    assert isinstance(config.validation_dataset, AlphabetSortDataset.Config)
    assert isinstance(config.message_env, AlphabetSortEnv.Config)
    assert len(config.rubric.reward_fns) == 1
    assert isinstance(config.rubric.reward_fns[0], RewardAlphabetSort.Config)
