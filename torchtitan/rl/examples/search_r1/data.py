# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

from torchtitan.config import Configurable


@dataclass(frozen=True, kw_only=True, slots=True)
class SearchR1Sample:
    """One Search-R1 open-domain QA sample: a question and its accepted answers."""

    question: str
    """The natural-language question the model must answer."""

    golden_answers: list[str]  # [num_answers]
    """Accepted golden answer strings; a prediction matching any one is correct (EM)."""


class SearchR1Source(Configurable):
    """Random-access Search-R1 QA samples.

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

        data_source: str | None = None
        """If set, keep only rows whose ``data_source`` equals this (e.g. ``"nq"``) —
        the merged test split mixes several datasets. ``None`` keeps all rows."""

    def __init__(self, config: Config) -> None:
        source = config.data_path or f"{config.repo_id}/{config.filename}"
        # A local parquet overrides the HF download; `datasets` handles both.
        if config.data_path:
            dataset = load_dataset(
                "parquet", data_files=config.data_path, split="train"
            )
        else:
            dataset = load_dataset(
                config.repo_id, data_files=config.filename, split="train"
            )
        if config.data_source is not None:
            dataset = dataset.filter(lambda ex: ex["data_source"] == config.data_source)
            if len(dataset) == 0:
                raise ValueError(
                    f"no rows with data_source={config.data_source!r} in {source}"
                )
        self._questions: list[str] = [str(q) for q in dataset["question"]]
        # golden_answers is a per-row array of strings; normalize to list[str].
        self._golden_answers: list[list[str]] = [
            [str(a) for a in answers] for answers in dataset["golden_answers"]
        ]
        if not self._questions:
            raise ValueError(f"no rows found in {source}")

    def __len__(self) -> int:
        return len(self._questions)

    def __getitem__(self, index: int) -> SearchR1Sample:
        return SearchR1Sample(
            question=self._questions[index],
            golden_answers=self._golden_answers[index],
        )
