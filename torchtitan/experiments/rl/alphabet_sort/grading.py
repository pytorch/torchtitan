# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AlphabetSort grading helpers.

Port of verifiers' ``score_response`` and ``eval_turn``
(``frameworks/verifiers/environments/alphabet_sort/alphabet_sort.py``,
2026-05). Similarity is ``difflib.SequenceMatcher.ratio()`` on
``"\\n".join(lower(stripped(names)))`` — Ratcliff/Obershelp, **not**
Levenshtein.

The model emits a name list inside an XML tag
(``<alphabetical_sorted>`` on turn 1, ``<combined_alphabetical_sorted>``
on later turns). If multiple tag blocks appear in one response, all
subsequent ones must strictly improve, else the turn scores 0; only
the last attempt's score counts. This matches verifiers verbatim.
"""

from __future__ import annotations

import difflib
import re

__all__ = ["score_completion_text"]


def _similarity(predicted: list[str], expected: list[str]) -> float:
    """Raw Ratcliff/Obershelp ratio over ``\\n``-joined, lowercased lists."""
    if not predicted or not expected:
        return 0.0
    pred = "\n".join(s.strip().lower() for s in predicted)
    exp = "\n".join(s.strip().lower() for s in expected)
    return difflib.SequenceMatcher(None, pred, exp).ratio()


def score_completion_text(
    completion_text: str, expected: list[str], *, turn_idx: int
) -> float:
    """Score one assistant turn's text against the cumulative expected list.

    Extracts the XML tag (``<alphabetical_sorted>`` on turn 0, else
    ``<combined_alphabetical_sorted>``), splits on newline, returns the
    Ratcliff/Obershelp similarity. The env applies the
    ``similarity_power`` aggregation; this helper only emits the raw
    ratio.

    Multiple tag blocks in one response: each successive attempt must
    strictly improve, otherwise the turn scores ``0.0``; only the LAST
    attempt's score is returned when monotonic. Strips ``// new name!``
    markers before scoring.
    """
    tag = "alphabetical_sorted" if turn_idx == 0 else "combined_alphabetical_sorted"
    pattern = f"<{tag}>(.*?)</{tag}>"
    contents = re.findall(pattern, completion_text, re.DOTALL)
    if not contents:
        return 0.0

    attempts: list[float] = []
    for content in contents:
        if not content.strip():
            attempts.append(0.0)
            continue
        predicted = [
            _strip_new_name_marker(line).strip()
            for line in content.strip().split("\n")
            if line.strip()
        ]
        attempts.append(_similarity(predicted, expected))

    if len(attempts) == 1:
        return attempts[0]
    # Adjacent-pair walk: ``attempts[1:]`` is intentionally one shorter,
    # so the pair-iterator yields ``len(attempts) - 1`` pairs.
    for a, b in zip(attempts, attempts[1:], strict=False):
        if b <= a:
            return 0.0
    return attempts[-1]


def _strip_new_name_marker(line: str) -> str:
    """Strip the trailing ``// new name!`` marker (and any whitespace)."""
    return line.split("//", 1)[0]
