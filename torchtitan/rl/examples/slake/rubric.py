# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from dataclasses import dataclass

from torchtitan.rl.examples.slake.data import SlakeSample
from torchtitan.rl.rollout import Rollout
from torchtitan.rl.rollout.rubric import RewardFn

# Frozen before training: model phrasings that mean the same SLAKE answer.
# Keys and values are already normalized (see `_normalize_answer`).
_SYNONYMS = {
    "axial": "transverse plane",
    "axial plane": "transverse plane",
    "transverse": "transverse plane",
    "coronal": "coronal plane",
    "sagittal": "sagittal plane",
    "xray": "x-ray",
    "x ray": "x-ray",
    "computed tomography": "ct",
    "ct scan": "ct",
    "magnetic resonance imaging": "mri",
    "t1 weighted": "t1",
    "t1-weighted": "t1",
    "t2 weighted": "t2",
    "t2-weighted": "t2",
    "small intestine": "small bowel",
    "large intestine": "colon",
    "large bowel": "colon",
}


class RewardSlakeAnswer(RewardFn):
    """1.0 when the final answer matches the SLAKE answer, else 0.0.

    Both sides go through `_normalize_answer`. For yes/no questions a reply
    whose first word is the expected yes/no also counts ("Yes, it does").
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: SlakeSample) -> float:
        message = rollout.turns[-1].completion_message if rollout.turns else None
        response_text = (message.get("content") or "") if message else ""
        return float(_is_correct(response_text, env_input.answer))


def _is_correct(response_text: str, expected: str) -> bool:
    """Whether the response's answer matches ``expected``.

    The answer is the last ``<answer>`` block if the response has one, else the whole
    response (the final message, without the reasoning channel).

    Example:

        _is_correct("Axial", "Transverse Plane")  # -> True
        _is_correct("<answer>yes, it does</answer>", "Yes")  # -> True
        _is_correct("Left lung", "Lung")  # -> False
    """
    blocks = re.findall(
        r"<\s*answer\s*>(.*?)</\s*answer\s*>", response_text, re.DOTALL | re.IGNORECASE
    )
    predicted = _normalize_answer(blocks[-1] if blocks else response_text)
    gold = _normalize_answer(expected)
    if predicted == gold:
        return True
    return gold in ("yes", "no") and predicted.split(" ", 1)[0] == gold


def _normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation other than hyphens and a leading article, map synonyms.

    Example:

        _normalize_answer(" The Axial plane. ")  # -> "transverse plane"
    """
    text = re.sub(r"[^\w\s-]", " ", text.lower())
    text = " ".join(text.split())
    text = re.sub(r"^(the|a|an) ", "", text)
    return _SYNONYMS.get(text, text)


__all__ = ["RewardSlakeAnswer"]
