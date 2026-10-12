# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import grain.python as grain
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.data.sources import RandomAccessDataSource
from torchtitan.config import Configurable
from torchtitan.rl.observability import metrics as m


class RLDataset(Configurable, ABC):
    """Base class for datasets that emit RL environment inputs."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def sample_key(self, sample: object) -> Hashable | None:
        """Stable identity of the task a sample poses: every draw of the same task, in any epoch
        or after a resume, returns the same key. None means the dataset has no task identity, so
        the loader can't track tasks. Keys are taken from the sample as the dataset yielded it.

        Example:
            VerifiersTaskDataset().sample_key(sample)  # -> "3f2a9c..." for every draw of this task
        """
        return None


class RLDataLoader(Stateful, ABC, Configurable):
    """Checkpointable iterator of globally identified RL inputs.

    Subclasses may generate inputs, stream them, or mix sources. They must
    restore every yielded but unacknowledged input after a checkpoint load.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dataset: RLDataset.Config
        """Builds the dataset consumed by this dataloader."""

    @abstractmethod
    def __iter__(self) -> RLDataLoader:
        ...

    @abstractmethod
    def __next__(self) -> tuple[int, object]:
        ...

    @abstractmethod
    def acknowledge(self, indices: Iterable[int]) -> None:
        ...

    def record_group_outcomes(self, outcomes: Mapping[int, GroupOutcome]) -> None:
        """Hear how yielded, not yet acknowledged samples went, right before they are
        acknowledged. Ignored by default.

        Args:
            outcomes: Sample index -> its group's outcome. Indices without a usable outcome
                (an errored rollout, a failed group) are left out.

        Example:
            loader.record_group_outcomes({
                7: GroupOutcome(mean_reward=1.0, zero_spread=True),   # all 16 rollouts passed
                8: GroupOutcome(mean_reward=0.25, zero_spread=False), # mixed: trains
            })                                                        # 9 errored: absent
            loader.acknowledge([7, 8, 9])
        """

    def metrics(self) -> list[m.Metric]:
        """Metrics since the last call; none by default."""
        return []


@dataclass(frozen=True, kw_only=True, slots=True)
class GroupOutcome:
    """How a training group's rollouts scored, for loaders that track tasks."""

    mean_reward: float
    zero_spread: bool
    """Every rollout got the same reward, so the group trained nothing."""


@dataclass(kw_only=True, slots=True)
class ZeroSpreadSkip:
    """Skip tasks whose last training group had zero reward spread: every rollout got the same
    reward, so the group trained nothing. On a cycling dataset such a task tends to come back
    the same, so skipping it saves its whole group of rollouts. By default only all-pass groups
    (the same positive reward) are skipped; all-fail tasks keep coming back.

    Example:
        IterableRLDataLoader.Config(
            dataset=VerifiersTaskDataset.Config(...),
            zero_spread_skip=ZeroSpreadSkip(),  # 1,600 tasks: a skipped task comes back ~4 epochs later
        )
    """

    skip_all_fail: bool = False
    """Also skip tasks whose group got the same non-positive reward on every rollout. Off:
    a task the policy fails today may be learned from its neighbors, so it stays in the mix."""

    recheck_after_skips: int = 4
    """Yield a skipped task again once it has been skipped this many times (about this many
    epochs on a cycling dataset), in case the policy changed."""

    min_active_frac: float = 0.25
    """Keep at least this fraction of the tasks seen so far unskipped; past it, the tasks
    skipped longest ago come back first."""

    def __post_init__(self) -> None:
        if self.recheck_after_skips < 1:
            raise ValueError(
                f"recheck_after_skips must be >= 1, got {self.recheck_after_skips}"
            )
        if not 0.0 < self.min_active_frac <= 1.0:
            raise ValueError(
                f"min_active_frac must be in (0, 1], got {self.min_active_frac}"
            )


class IterableRLDataLoader(RLDataLoader):
    """Checkpointable RL loader backed by a stateful iterable dataset.

    Unacknowledged samples are stored in the checkpoint and replayed before new
    samples after resume. The wrapped dataset remains responsible for its own
    cursor, RNG, and any external source state.
    """

    _STATE_VERSION = 1
    """Increment when the serialized fields or their interpretation changes."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataLoader.Config):
        zero_spread_skip: ZeroSpreadSkip | None = None
        """Skip tasks whose last group trained nothing (see `ZeroSpreadSkip`); off by default.
        Needs a dataset with `sample_key`."""

    def __init__(self, config: Config) -> None:
        dataset = config.dataset.build()
        if not isinstance(dataset, Iterable) or not isinstance(dataset, Stateful):
            raise ValueError("RL dataset must be a stateful iterable")
        # Fail at construction for a dataset without keys; `_sample_key` still catches a None key.
        if (
            config.zero_spread_skip is not None
            and type(dataset).sample_key is RLDataset.sample_key
        ):
            raise ValueError(
                f"zero_spread_skip needs task identities, but {type(dataset).__name__} "
                "does not implement sample_key"
            )
        self._dataset: RLDataset = dataset
        self._iterator: Iterator[object] = iter(dataset)
        self._next_index = 0
        self._pending: dict[int, object] = {}
        self._replay: deque[tuple[int, object]] = deque()

        self._zero_spread_skip = config.zero_spread_skip
        # Task key -> times skipped since its last zero-spread group; insertion order is the
        # order tasks were marked, so the floor brings back the oldest first.
        self._skipped: dict[Hashable, int] = {}
        self._seen_keys: set[Hashable] = set()
        self._step_counts = dict.fromkeys(
            ("skips", "rechecks", "floor_admits", "marked_all_pass", "marked_all_fail"), 0
        )

    def __iter__(self) -> IterableRLDataLoader:
        return self

    def __next__(self) -> tuple[int, object]:
        if self._replay:
            # Already yielded before the checkpoint: replay it as-is, never skip it.
            index, sample = self._replay.popleft()
        else:
            index = self._next_index
            sample = self._next_unskipped_sample()
            self._next_index += 1
        self._pending[index] = sample
        return index, sample

    def _next_unskipped_sample(self) -> object:
        """Next dataset sample whose task is not skipped, counting a skip for each one passed over."""
        if self._zero_spread_skip is None:
            return next(self._iterator)
        # Every skip moves a task toward its re-check, so this ends; the cap bounds one call.
        for _ in range(len(self._skipped) + 1):
            sample = next(self._iterator)
            key = self._sample_key(sample)
            self._seen_keys.add(key)
            num_skips = self._skipped.get(key)
            if num_skips is None:
                return sample
            if num_skips >= self._zero_spread_skip.recheck_after_skips:
                del self._skipped[key]
                self._step_counts["rechecks"] += 1
                return sample
            self._skipped[key] = num_skips + 1
            self._step_counts["skips"] += 1
        # Only skipped tasks came up: yield the last one as a re-check.
        del self._skipped[key]
        self._step_counts["rechecks"] += 1
        return sample

    def _sample_key(self, sample: object) -> Hashable:
        key = self._dataset.sample_key(sample)
        if key is None:
            raise ValueError(
                f"{type(self._dataset).__name__}.sample_key returned None; "
                "zero_spread_skip needs a key for every sample"
            )
        return key

    def record_group_outcomes(self, outcomes: Mapping[int, GroupOutcome]) -> None:
        """Mark the tasks of skippable zero-spread groups as skipped and unmark the others."""
        if self._zero_spread_skip is None:
            return
        for index, outcome in outcomes.items():
            if index not in self._pending:
                raise RuntimeError(f"cannot record an outcome for an unyielded index {index}")
            key = self._sample_key(self._pending[index])
            self._skipped.pop(key, None)
            if not outcome.zero_spread:
                continue
            all_pass = outcome.mean_reward > 0
            if not all_pass and not self._zero_spread_skip.skip_all_fail:
                continue
            self._skipped[key] = 0  # re-inserted last: the newest skip
            direction = "all_pass" if all_pass else "all_fail"
            self._step_counts[f"marked_{direction}"] += 1
        max_skipped = int(
            (1.0 - self._zero_spread_skip.min_active_frac) * len(self._seen_keys)
        )
        while len(self._skipped) > max_skipped:
            del self._skipped[next(iter(self._skipped))]
            self._step_counts["floor_admits"] += 1

    def acknowledge(self, indices: Iterable[int]) -> None:
        for index in indices:
            if index not in self._pending:
                raise RuntimeError(f"cannot acknowledge an unyielded index {index}")
            del self._pending[index]

    def metrics(self) -> list[m.Metric]:
        """`data/zero_spread_skip/`: skipped_tasks (now), and per step skips, rechecks,
        floor_admits, marked_all_pass, marked_all_fail."""
        if self._zero_spread_skip is None:
            return []
        metrics = [
            m.Metric("data/zero_spread_skip/skipped_tasks", m.NoReduce(len(self._skipped))),
            *[
                m.Metric(f"data/zero_spread_skip/{name}", m.Sum(float(count)))
                for name, count in self._step_counts.items()
            ],
        ]
        self._step_counts = dict.fromkeys(self._step_counts, 0)
        return metrics

    def state_dict(self) -> dict[str, Any]:
        """Save all unacknowledged samples as one ordered replay set.

        Pending and replay samples are intentionally merged because both must
        be yielded again after a restart.
        """
        replay = dict(self._replay)
        replay.update(self._pending)
        state = {
            "version": self._STATE_VERSION,
            "dataset": self._dataset.state_dict(),
            "next_index": self._next_index,
            "pending_samples": sorted(replay.items()),
        }
        if self._zero_spread_skip is not None:
            state["zero_spread_skip"] = {
                "skipped": list(self._skipped.items()),
                "seen_keys": list(self._seen_keys),
            }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore every unacknowledged sample into the replay queue.

        The pre-checkpoint pending state is not restored directly: those
        samples must be yielded again before they can be acknowledged.
        """
        if state_dict.get("version") != self._STATE_VERSION:
            raise ValueError(
                f"unsupported RL data state version {state_dict.get('version')!r}"
            )
        next_index = state_dict["next_index"]
        pending = list(state_dict["pending_samples"])
        pending_indices = [index for index, _ in pending]
        if next_index < 0 or any(
            index < 0 or index >= next_index for index in pending_indices
        ):
            raise ValueError("RL data state contains an invalid index")
        if len(pending_indices) != len(set(pending_indices)):
            raise ValueError("RL data state contains duplicate pending indices")

        self._dataset.load_state_dict(state_dict["dataset"])
        self._iterator = iter(self._dataset)
        self._next_index = next_index
        self._pending = {}
        self._replay = deque(sorted(pending))
        # Absent in checkpoints saved with the skip off: start with nothing skipped.
        skip_state = state_dict.get("zero_spread_skip", {})
        self._skipped = dict(skip_state.get("skipped", []))
        self._seen_keys = set(skip_state.get("seen_keys", []))


class MapStyleRLDataLoader(RLDataLoader):
    """Checkpointable RL input iterator backed by a random-access source.

    This class is not thread-safe. Call next, acknowledge, state_dict, and
    load_state_dict serially from the controller data-input thread.
    """

    _STATE_VERSION = 1
    """Increment when the serialized fields or their interpretation changes."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataLoader.Config):
        seed: int = 42
        shuffle: bool = True

    def __init__(self, config: Config) -> None:
        dataset_source = config.dataset.build()
        if not isinstance(dataset_source, RandomAccessDataSource):
            raise ValueError("RL dataset must support random access")
        if len(dataset_source) == 0:
            raise ValueError("RL dataset must not be empty")

        dataset = grain.MapDataset.source(dataset_source)
        if config.shuffle:
            dataset = dataset.shuffle(seed=config.seed)
        self._dataset = dataset.repeat()
        self._seed = config.seed
        self._shuffle = config.shuffle
        self._dataset_length = len(dataset_source)
        self._next_index = 0
        self._pending: set[int] = set()
        self._replay: deque[int] = deque()

    def __iter__(self) -> MapStyleRLDataLoader:
        return self

    def __next__(self) -> tuple[int, object]:
        if self._replay:
            index = self._replay.popleft()
        else:
            index = self._next_index
            self._next_index += 1
        self._pending.add(index)
        return index, self._dataset[index]

    def acknowledge(self, indices: Iterable[int]) -> None:
        for index in indices:
            if index not in self._pending:
                raise RuntimeError(f"cannot acknowledge an unyielded index {index}")
            self._pending.remove(index)

    def state_dict(self) -> dict[str, Any]:
        """Save the cursor and all unacknowledged indices for replay."""
        return {
            "version": self._STATE_VERSION,
            "seed": self._seed,
            "shuffle": self._shuffle,
            "dataset_length": self._dataset_length,
            "next_index": self._next_index,
            "pending_indices": sorted(self._pending | set(self._replay)),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore every unacknowledged index into the replay queue.

        The pre-checkpoint pending state is not restored directly: those
        indices must be yielded again before they can be acknowledged.
        """
        expected = {
            "version": self._STATE_VERSION,
            "seed": self._seed,
            "shuffle": self._shuffle,
            "dataset_length": self._dataset_length,
        }
        for key, value in expected.items():
            if state_dict.get(key) != value:
                raise ValueError(
                    f"RL data state {key}={state_dict.get(key)!r} does not match "
                    f"the configured value {value!r}"
                )

        next_index = state_dict["next_index"]
        pending = list(state_dict["pending_indices"])
        if next_index < 0 or any(index < 0 or index >= next_index for index in pending):
            raise ValueError("RL data state contains an invalid index")
        if len(pending) != len(set(pending)):
            raise ValueError("RL data state contains duplicate pending indices")

        self._next_index = next_index
        self._pending = set()
        self._replay = deque(sorted(pending))
        # TODO: Replay finished rollouts from a cache instead of regenerating
        # every pending prompt after resume.
