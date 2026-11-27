# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass

import datasets
from PIL import Image

from torchtitan.rl.rollout.data import RLDataset

IMAGE_MARKER = "<image>"
"""Where the problem text shows each of its images, in order."""


@dataclass(frozen=True, kw_only=True, slots=True)
class Geometry3kSample:
    """One Geometry3K problem: its diagram files, text and expected answer.

    Example:

        Geometry3kSample(
            index=0,
            image_paths=("./assets/datasets/geometry3k/images/train/00000_0.png",),
            problem="<image>Find x.",
            ground_truth="3",
        )
    """

    index: int
    image_paths: tuple[str, ...]
    """The problem's image files, in the order of its ``<image>`` markers."""
    problem: str
    """The problem text, with one ``<image>`` marker per image."""
    ground_truth: str
    """The expected final answer (LaTeX), which `RewardMathVerify` compares against."""


class Geometry3kDataset(RLDataset):
    """Geometry3K (Hugging Face ``hiyouga/geometry3k``), streamed in reshuffled epochs, or one
    pass if not `repeat`.

    The release stores each problem's images inside its parquet files. Rendering reads
    image files (the renderer computes each image's grid from the file header), so the
    dataset writes every image to ``image_dir`` once, then passes file paths.

    Example:

        dataset = Geometry3kDataset(
            Geometry3kDataset.Config(image_dir="./assets/datasets/geometry3k/images")
        )
        sample: Geometry3kSample = next(iter(dataset))
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        path: str = "hiyouga/geometry3k"
        """Hugging Face dataset id, or a local copy of it (a directory with ``data/*.parquet``)."""

        split: str = "train"
        """``train`` (2,101 problems), ``validation`` (300) or ``test`` (601)."""

        image_dir: str
        """Directory the images are written to, as ``{split}/{index:05d}_{n}.{format}`` (e.g.
        ``.png``). It must be readable where prompts are rendered (the rollout workers)."""

        seed: int = 0

    def __init__(self, config: Config) -> None:
        rows = datasets.load_dataset(config.path, split=config.split).cast_column(
            "images", datasets.Sequence(datasets.Image(decode=False))
        )
        split_dir = os.path.join(config.image_dir, config.split)
        os.makedirs(split_dir, exist_ok=True)
        samples = []
        for index, row in enumerate(rows):
            if row["problem"].count(IMAGE_MARKER) != len(row["images"]):
                raise ValueError(
                    f"Geometry3K {config.split} problem {index} has "
                    f"{row['problem'].count(IMAGE_MARKER)} {IMAGE_MARKER} markers but "
                    f"{len(row['images'])} images."
                )
            image_paths = tuple(
                _write_image(
                    image["bytes"], os.path.join(split_dir, f"{index:05d}_{number}")
                )
                for number, image in enumerate(row["images"])
            )
            samples.append(
                Geometry3kSample(
                    index=index,
                    image_paths=image_paths,
                    problem=row["problem"],
                    ground_truth=row["answer"],
                )
            )
        if not samples:
            raise ValueError(f"No Geometry3K samples in {config.path} ({config.split})")
        self._samples = tuple(samples)
        self._rng = random.Random(config.seed)
        self._repeat = config.repeat
        self._order: list[int] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Geometry3kSample]:
        return self

    def __next__(self) -> Geometry3kSample:
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


def _write_image(data: bytes, stem: str) -> str:
    """Write an image file's bytes to ``stem`` + its format's suffix, unless already there."""
    image_format = Image.open(io.BytesIO(data)).format
    path = f"{stem}.{(image_format or 'png').lower()}"
    if not os.path.exists(path):
        # Write then rename, so an interrupted write never leaves a partial file behind.
        with open(f"{path}.tmp", "wb") as f:
            f.write(data)
        os.replace(f"{path}.tmp", path)
    return path
