# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass

from torchtitan.experiments.rl.examples.alphabet_sort.data import AlphabetSortSample
from torchtitan.experiments.rl.rollout import Rollout
from torchtitan.experiments.rl.rubrics import RewardFn


class RewardAlphabetSort(RewardFn):
    """Average over the rollout's turns of how well each turn's list was sorted, each scored by
    `score_sorted_list` against that turn's expected names. Partial credit by order similarity
    enables rewards to gradually rise.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        similarity_power: int = 4
        """Exponent on the model-vs-expected order similarity; higher = stricter scoring."""

        def __post_init__(self) -> None:
            if self.similarity_power <= 0:
                raise ValueError(
                    f"similarity_power must be positive; got {self.similarity_power}"
                )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.similarity_power = config.similarity_power

    async def __call__(self, rollout: Rollout, env_input: AlphabetSortSample) -> float:
        turn_scores: list[float] = []
        for turn_idx, (rollout_turn, expected_names) in enumerate(
            zip(rollout.turns, env_input.expected_names)
        ):
            message = rollout_turn.completion_message
            response_text = (message.get("content") or "") if message else ""
            # turn 0 uses the plain-sort tag; later turns the combined re-sort tag.
            xml_tag = (
                "alphabetical_sorted"
                if turn_idx == 0
                else "combined_alphabetical_sorted"
            )
            turn_scores.append(
                score_sorted_list(
                    response_text,
                    expected_names=expected_names,
                    xml_tag=xml_tag,
                    similarity_power=self.similarity_power,
                )
            )
        # Divide by the number of EXPECTED turns, not the answered ones, so a rollout
        # that ends early (a perfect turn 0 but a missing follow-up) is penalized.
        num_expected_turns = len(env_input.expected_names)
        return sum(turn_scores) / num_expected_turns if num_expected_turns else 0.0


def score_sorted_list(
    response_text: str,
    *,
    expected_names: Sequence[str],
    xml_tag: str,
    similarity_power: int,
) -> float:
    """Reward in [0, 1] for how closely the model's `<xml_tag>` block matches `expected_names`.

    A difflib similarity ratio over the lowercased lines, raised to `similarity_power` so only
    near-perfect orderings score near 1.0; an absent block scores 0.0.

    Example — the model returns:

        <alphabetical_sorted>
        AnaChardin
        MarcChardin
        </alphabetical_sorted>

        score_sorted_list(response, expected_names=("AnaChardin", "MarcChardin"),
                          xml_tag="alphabetical_sorted", similarity_power=4)  # -> 1.0

    A swapped order scores below 1.0.
    """
    predicted = _answer_lines(response_text, xml_tag=xml_tag)
    if not predicted:
        return 0.0
    predicted_text = "\n".join(line.lower() for line in predicted)
    expected_text = "\n".join(line.lower() for line in expected_names)
    similarity = difflib.SequenceMatcher(None, predicted_text, expected_text).ratio()
    return similarity**similarity_power


def _answer_lines(text: str, *, xml_tag: str) -> list[str]:
    r"""The non-empty lines inside the model's last `<xml_tag>` block, or [] if absent.

    Example:

        text = "<alphabetical_sorted>\nAnaChardin\nMarcChardin\n</alphabetical_sorted>"
        _answer_lines(text, xml_tag="alphabetical_sorted")  # -> ["AnaChardin", "MarcChardin"]
    """
    blocks = re.findall(
        rf"<\s*{xml_tag}\s*>(.*?)</\s*{xml_tag}\s*>", text, re.DOTALL | re.IGNORECASE
    )
    if not blocks:
        return []
    return [line.strip() for line in blocks[-1].splitlines() if line.strip()]


__all__ = ["RewardAlphabetSort", "score_sorted_list"]
