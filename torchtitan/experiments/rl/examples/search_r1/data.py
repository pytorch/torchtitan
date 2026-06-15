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
from huggingface_hub import hf_hub_download

from torchtitan.config import Configurable


@dataclass(frozen=True, kw_only=True, slots=True)
class SearchR1Sample:
    """One Search-R1 open-domain QA sample: a question and its accepted answers."""

    question: str
    """The natural-language question the model must answer."""

    golden_answers: list[str]  # [num_answers]
    """Accepted gold answer strings; a prediction matching any one is correct (EM)."""


class SearchR1Dataset(Configurable):
    """Endless, seeded stream of Search-R1 QA samples.

    The NQ/HotpotQA parquet (columns ``question``, ``golden_answers``, ``data_source``)
    is downloaded from the HF Hub dataset ``PeterJinGo/nq_hotpotqa_train``. Row order
    is shuffled with ``seed`` and reshuffled on each wrap, so a run sees a fresh permutation every epoch.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        filename: str = "train.parquet"
        """Which split to load from the HF dataset repo: ``train.parquet`` (train) or
        ``test.parquet`` (validation)."""

        repo_id: str = "PeterJinGo/nq_hotpotqa_train"
        """HF Hub dataset repo holding the preprocessed Search-R1 NQ/HotpotQA parquets."""

        data_path: str | None = None
        """Local parquet path; overrides the HF download when set (offline use)."""

        seed: int = 42
        """Seed for the row-order shuffle."""

        data_source: str | None = None
        """If set, keep only rows whose ``data_source`` equals this (e.g. ``"nq"``) —
        the merged test split mixes several datasets. ``None`` keeps all rows."""

        shuffle: bool = True
        """Shuffle row order (with ``seed``), reshuffling on each wrap. Set False for
        validation so the first N rows are a fixed, file-order held-out set."""

    def __init__(self, config: Config) -> None:
        columns = ["question", "golden_answers"]
        if config.data_source is not None:
            columns.append("data_source")
        path = config.data_path or hf_hub_download(
            repo_id=config.repo_id, filename=config.filename, repo_type="dataset"
        )
        df = pd.read_parquet(path, columns=columns)
        if config.data_source is not None:
            df = df[df["data_source"] == config.data_source].reset_index(drop=True)
            if df.empty:
                raise ValueError(
                    f"no rows with data_source={config.data_source!r} in {path}"
                )
        self._questions: list[str] = [str(q) for q in df["question"].tolist()]
        # golden_answers is a per-row array of strings; normalize to list[str].
        self._golden_answers: list[list[str]] = [
            [str(a) for a in answers] for answers in df["golden_answers"].tolist()
        ]
        if not self._questions:
            raise ValueError(f"no rows found in {path}")

        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        self._order = list(range(len(self._questions)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._pos = 0

    def __iter__(self) -> Iterator[SearchR1Sample]:
        return self

    def __next__(self) -> SearchR1Sample:
        if self._pos >= len(self._order):
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._pos = 0
        idx = self._order[self._pos]
        self._pos += 1
        return SearchR1Sample(
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
