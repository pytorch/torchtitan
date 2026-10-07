# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from datasets import concatenate_datasets, load_dataset

from torchtitan.config import Configurable

_MATH_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: \\boxed{{$Answer}}, where $Answer is the answer "
    "to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line as "Answer: \\boxed{{...}}".'
)


@dataclass(frozen=True, kw_only=True, slots=True)
class DapoMathSample:
    """A math prompt paired with its expected final answer."""

    prompt: str
    ground_truth: str


class _MathSource(Configurable):
    def __init__(self, samples: list[DapoMathSample]) -> None:
        if not samples:
            raise ValueError("math dataset must contain at least one sample")
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> DapoMathSample:
        return self._samples[index]


class DapoMathSource(_MathSource):
    """Provides filtered DAPO-Math problems in the original `Answer:` format."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        repo_id: str = "hamishivi/DAPO-Math-17k-Processed_filtered"
        split: str = "train"

    def __init__(self, config: Config) -> None:
        dataset = load_dataset(config.repo_id, split=config.split)
        samples: list[DapoMathSample] = []
        for row in dataset:
            prompt_messages = row["source_prompt"]
            if len(prompt_messages) != 1 or prompt_messages[0]["role"] != "user":
                raise ValueError("DAPO-Math rows must contain exactly one user prompt")
            samples.append(
                DapoMathSample(
                    # `prompt` is the raw question without answer-format instructions.
                    prompt=_MATH_PROMPT_TEMPLATE.format(problem=row["prompt"]),
                    ground_truth=str(row["ground_truth"]),
                )
            )
        super().__init__(samples)


class AIME2025Source(_MathSource):
    """Provides AIME 2025 I+II problems using the DAPO answer format."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        repo_id: str = "opencompass/AIME2025"
        subsets: tuple[str, ...] = ("AIME2025-I", "AIME2025-II")
        split: str = "test"

    def __init__(self, config: Config) -> None:
        dataset = concatenate_datasets(
            [
                load_dataset(config.repo_id, subset, split=config.split)
                for subset in config.subsets
            ]
        )
        samples = [
            DapoMathSample(
                prompt=_MATH_PROMPT_TEMPLATE.format(problem=row["question"]),
                ground_truth=str(row["answer"]),
            )
            for row in dataset
        ]
        super().__init__(samples)
