# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

from datasets import load_dataset

from torchtitan.config import Configurable


class _Author(NamedTuple):
    # CamelCase name shown to the model + its lowercased (first, last) sort keys.
    display: str  # "MarcChardin"
    first: str  # "marc"
    last: str  # "chardin"


@dataclass(frozen=True, kw_only=True, slots=True)
class AlphabetSortSample:
    """The names shown and the expected sorted answer for each turn of one AlphabetSort problem.

    Example (2 turns, sorted by first name):

        AlphabetSortSample(
            new_names_per_turn=(("MarcChardin", "AnaChardin"), ("BobBeck",)),
            expected_names=(
                ("AnaChardin", "MarcChardin"),
                ("AnaChardin", "BobBeck // new name!", "MarcChardin"),
            ),
            sort_by_first_name=True,
        )
    """

    new_names_per_turn: tuple[tuple[str, ...], ...]  # [num_turns][new_names]
    expected_names: tuple[tuple[str, ...], ...]  # [num_turns][sorted_names]
    sort_by_first_name: bool


class AlphabetSortDataset(Configurable):
    """Provides lists of researchers' names to sort alphabetically, over one or more turns.

    Each sample shows CamelCase arXiv author names (e.g. "MarcChardin", from a Hugging Face
    dataset); a multi-turn sample adds names each turn and asks for the whole list re-sorted.

    Example:

        dataset = AlphabetSortDataset(AlphabetSortDataset.Config(seed=42))
        sample: AlphabetSortSample = next(iter(dataset))
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 1337420

        max_turns: int = 3
        """Maximum turns per sample; each sample draws uniformly from [1, max_turns]."""

        max_names_per_turn: int = 5
        """Names introduced per turn are drawn uniformly from [1, max_names_per_turn]."""

        hf_dataset: str = "kalomaze/alphabetic-arxiv-authors-it1"
        hf_split: str = "train"

        def __post_init__(self) -> None:
            if self.max_turns < 1:
                raise ValueError(f"max_turns must be >= 1; got {self.max_turns}")
            if self.max_names_per_turn < 1:
                raise ValueError(
                    f"max_names_per_turn must be >= 1; got {self.max_names_per_turn}"
                )

    def __init__(self, config: Config) -> None:
        self._config = config
        self._author_pool = _load_authors(config.hf_dataset, config.hf_split)
        self._rng = random.Random(config.seed)

    def __iter__(self) -> Iterator[AlphabetSortSample]:
        return self

    def __next__(self) -> AlphabetSortSample:
        return self._sample()

    def state_dict(self) -> dict:
        """Snapshot the RNG so a run can resume at the same point in the stream."""
        return {"rng_state": self._rng.getstate()}

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])

    def _sample(self) -> AlphabetSortSample:
        """Build one sample: pick the turn count and names, then for each turn record the names
        shown and the expected sorted answer over everything seen so far.

        Example (2 turns, sorted by first name, drawn without replacement):

            turn 0: show [MarcChardin, AnaChardin]  -> expect [AnaChardin, MarcChardin]
            turn 1: show [BobBeck]                  -> expect [AnaChardin, BobBeck // new name!, MarcChardin]
        """
        config = self._config
        num_turns = self._rng.randint(1, config.max_turns)

        # Draw how many names each turn introduces, then draw them all at once (without
        # replacement) so a name never repeats within or across turns.
        name_counts_per_turn = [
            self._rng.randint(1, config.max_names_per_turn) for _ in range(num_turns)
        ]
        sampled_authors = self._rng.sample(
            self._author_pool, k=sum(name_counts_per_turn)
        )
        sort_by_first_name = self._rng.choice([True, False])

        def name_sort_key(author: _Author) -> tuple[str, str]:
            # Sort by the chosen part, then the other part — so names sharing the chosen part
            # (e.g. two "Ana"s sorted by first name) land in a fixed order, not the draw order.
            return (
                (author.first, author.last)
                if sort_by_first_name
                else (author.last, author.first)
            )

        new_names_per_turn: list[tuple[str, ...]] = []
        expected_names: list[tuple[str, ...]] = []
        seen: list[_Author] = []
        start = 0
        for turn_idx, count in enumerate(name_counts_per_turn):
            new_authors = sampled_authors[start : start + count]
            start += count
            seen.extend(new_authors)

            # Show this turn's new names in random order.
            shown_names = [author.display for author in new_authors]
            self._rng.shuffle(shown_names)
            new_names_per_turn.append(tuple(shown_names))

            # Expected answer: every name seen so far, sorted; tag the names new this turn with
            # "// new name!", except turn 0 where every name is new so none is tagged.
            new_this_turn: set[_Author] = set(new_authors) if turn_idx > 0 else set()
            expected_names.append(
                tuple(
                    (
                        f"{author.display} // new name!"
                        if author in new_this_turn
                        else author.display
                    )
                    for author in sorted(seen, key=name_sort_key)
                )
            )

        return AlphabetSortSample(
            new_names_per_turn=tuple(new_names_per_turn),
            expected_names=tuple(expected_names),
            sort_by_first_name=sort_by_first_name,
        )


def _load_authors(hf_dataset: str, hf_split: str) -> tuple[_Author, ...]:
    """Load + dedupe arXiv authors, splitting each `"First Last"` name at its space.

    Example:

        "Marc Chardin"  -> _Author("MarcChardin", "marc", "chardin")
        "Yan Hong Yao"  -> skipped (not exactly two space-separated parts)
    """
    dataset = load_dataset(hf_dataset, split=hf_split)
    seen: set[str] = set()
    authors: list[_Author] = []
    for row in dataset:
        for raw_name in row["names"]:
            parts = raw_name.split()
            if len(parts) != 2:
                continue
            display = "".join(parts)
            if display in seen:
                continue
            seen.add(display)
            authors.append(_Author(display, parts[0].lower(), parts[1].lower()))
    return tuple(authors)
