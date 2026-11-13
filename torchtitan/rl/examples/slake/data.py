# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass

from torchtitan.rl.rollout.data import RLDataset


@dataclass(frozen=True, kw_only=True, slots=True)
class SlakeSample:
    """One SLAKE question about one medical image.

    Example:

        SlakeSample(
            qid=11,
            image_path="/data/slake/imgs/xmlab1/source.jpg",
            question="Does the picture contain liver?",
            answer="Yes",
            content_type="Organ",
        )
    """

    qid: int
    image_path: str
    question: str
    answer: str
    content_type: str


class SlakeDataset(RLDataset):
    """English SLAKE medical VQA (CT / MRI / X-ray), streamed in reshuffled epochs, or one pass
    if not `repeat`.

    Expects the official release layout (Hugging Face ``BoKelvin/SLAKE``):
    ``{data_dir}/{split}.json`` and the images unzipped under
    ``{data_dir}/imgs/<img_name>``. It keeps the English closed questions (a yes/no
    or a choice named in the question) and open ones, and drops knowledge-graph
    questions, whose answers do not depend on the image.

    Example:

        dataset = SlakeDataset(SlakeDataset.Config(data_dir="/data/slake"))
        sample: SlakeSample = next(iter(dataset))
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        data_dir: str
        """Directory with ``{split}.json`` and ``imgs/``."""

        split: str = "train"
        """``train``, ``validation`` or ``test``."""

        seed: int = 0

    def __init__(self, config: Config) -> None:
        with open(os.path.join(config.data_dir, f"{config.split}.json")) as f:
            rows = json.load(f)
        image_dir = os.path.join(config.data_dir, "imgs")
        self._samples = tuple(
            SlakeSample(
                qid=row["qid"],
                image_path=os.path.join(image_dir, row["img_name"]),
                question=row["question"],
                answer=row["answer"],
                content_type=row["content_type"],
            )
            for row in rows
            if row["q_lang"] == "en" and row["content_type"] != "KG"
        )
        if not self._samples:
            raise ValueError(f"No SLAKE samples selected from {config.data_dir}")
        self._rng = random.Random(config.seed)
        self._repeat = config.repeat
        self._order: list[int] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[SlakeSample]:
        return self

    def __next__(self) -> SlakeSample:
        if self._cursor == len(self._order):
            # The first call starts the first epoch; later ones start another only if `repeat`.
            if self._order and not self._repeat:
                raise StopIteration
            self._order = list(range(len(self._samples)))
            self._rng.shuffle(self._order)
            self._cursor = 0
        sample = self._samples[self._order[self._cursor]]
        self._cursor += 1
        return sample

    def state_dict(self) -> dict:
        """Snapshot the epoch order and position so a run resumes at the same sample."""
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "cursor": self._cursor,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._cursor = state_dict["cursor"]
