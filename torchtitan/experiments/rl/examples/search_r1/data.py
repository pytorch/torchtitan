# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from torchtitan.config import Configurable


@dataclass(frozen=True, kw_only=True, slots=True)
class SearchR1Example:
    """Typed payload for one open-domain QA problem (Search-R1)."""

    question: str
    """The natural-language question the model must answer."""

    golden_answers: list[str]  # [num_answers]
    """Accepted gold answer strings; a prediction matching any one is correct (EM)."""


class SearchR1Dataset(Configurable):
    """Endless, seeded stream of Search-R1 QA examples read from a parquet file.

    The parquet is the Search-R1 NQ/HotpotQA format with ``question`` and
    ``golden_answers`` columns (e.g. produced by Search-R1's
    ``scripts/data_process/qa_search_*_merge.py``). The dataset shuffles row
    order with the given seed and reshuffles each time it wraps around, so a run
    sees a fresh permutation every epoch.

    Example:

        dataset = SearchR1Dataset(SearchR1Dataset.Config(data_path=".../train.parquet"))
        example: SearchR1Example = next(iter(dataset))
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        data_path: str
        """Path to the Search-R1 QA parquet (train or test split)."""

        seed: int = 42
        """Seed for the row-order shuffle."""

        data_source: str | None = None
        """If set, keep only rows whose ``data_source`` equals this (e.g. ``"nq"``).
        The merged Search-R1 test split mixes 7 datasets; use this to evaluate on a
        single benchmark like slime. ``None`` keeps all rows."""

        shuffle: bool = True
        """Shuffle row order (with ``seed``) and reshuffle on each wrap. Set False
        for validation to read rows in file order, so the first N rows match slime's
        ``--eval-prompt-data nq_test test.parquet@[0:N]`` exactly (apples-to-apples
        NQ EM)."""

    def __init__(self, config: Config) -> None:
        columns = ["question", "golden_answers"]
        if config.data_source is not None:
            columns.append("data_source")
        df = pd.read_parquet(config.data_path, columns=columns)
        if config.data_source is not None:
            df = df[df["data_source"] == config.data_source].reset_index(drop=True)
            if df.empty:
                raise ValueError(
                    f"no rows with data_source={config.data_source!r} in {config.data_path}"
                )
        self._questions: list[str] = [str(q) for q in df["question"].tolist()]
        # golden_answers is a per-row array of strings; normalize to list[str].
        self._golden_answers: list[list[str]] = [
            [str(a) for a in answers] for answers in df["golden_answers"].tolist()
        ]
        if not self._questions:
            raise ValueError(f"no rows found in {config.data_path}")

        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        self._order = list(range(len(self._questions)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._pos = 0

    def __iter__(self) -> Iterator[SearchR1Example]:
        return self

    def __next__(self) -> SearchR1Example:
        if self._pos >= len(self._order):
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._pos = 0
        idx = self._order[self._pos]
        self._pos += 1
        return SearchR1Example(
            question=self._questions[idx],
            golden_answers=self._golden_answers[idx],
        )

    def state_dict(self) -> dict:
        """Snapshot the RNG + position so a run can resume mid-stream."""
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "pos": self._pos,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._pos = state_dict["pos"]
