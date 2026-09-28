# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Input coordination for optimizer-step load balancing."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from torch.distributed.tensor import DeviceMesh

from torchtitan.components.data.load_balancing.planner import (
    BinAssignment,
    PackableItem,
    PackingBin,
    StableId,
)
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import OptimizerStepLayout, TrainingMicrobatch
from torchtitan.config import Configurable


ChildLoaderFactory = Callable[[int], BaseDataLoader]
Itemize = Callable[[StableId, TrainingMicrobatch], PackableItem]


@dataclass(frozen=True, kw_only=True, slots=True)
class CoordinatedWindow:
    """Planner inputs and locally available payloads for one optimizer step."""

    items: tuple[PackableItem, ...]
    bins: tuple[PackingBin, ...]
    local_payloads: dict[StableId, TrainingMicrobatch]
    local_bin_ids: tuple[StableId, ...]

    @property
    def local_payload_bytes(self) -> int:
        """Return bytes held locally for payloads represented in this window."""
        items_by_id = {item.stable_id: item for item in self.items}
        return sum(
            items_by_id[item_id].payload_bytes for item_id in self.local_payloads
        )


class InputCoordinator(Configurable, ABC):
    """Collect planner inputs and deliver assignments to one physical DP rank."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def collect(
        self,
        *,
        layout: OptimizerStepLayout,
        itemize: Itemize,
    ) -> CoordinatedWindow:
        """Collect one optimizer-step candidate window."""
        raise NotImplementedError

    @abstractmethod
    def distribute(
        self,
        window: CoordinatedWindow,
        assignments: Sequence[BinAssignment],
    ) -> list[TrainingMicrobatch]:
        """Return assigned payloads in local execution order."""
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return coordinator and source-reader state."""
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore coordinator and source-reader state."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release source-reader and communication resources."""
        raise NotImplementedError


class ReplicatedInputCoordinator(InputCoordinator):
    """Reconstruct every logical input stream on every balancing-group rank."""

    @dataclass(kw_only=True, slots=True)
    class Config(InputCoordinator.Config):
        group_size: int = 1

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        child_loader_factory: ChildLoaderFactory,
        document_capacity: int | None,
        dp_mesh: DeviceMesh | None = None,
    ) -> None:
        del dp_mesh
        if dp_world_size <= 0:
            raise ValueError("dp_world_size must be greater than 0")
        if dp_rank < 0 or dp_rank >= dp_world_size:
            raise ValueError("dp_rank must be within the effective DP world")
        if config.group_size <= 0:
            raise ValueError("replicated input group_size must be greater than 0")
        if dp_world_size % config.group_size != 0:
            raise ValueError("dp_world_size must be divisible by group_size")

        group_start = (dp_rank // config.group_size) * config.group_size
        self._logical_dp_ranks = tuple(
            range(group_start, group_start + config.group_size)
        )
        self._physical_logical_dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        self._document_capacity = document_capacity
        self._children: dict[int, BaseDataLoader] = {}
        self._child_iterators: dict[int, Iterator[TrainingMicrobatch]] = {}
        self._closed = False

        try:
            for logical_dp_rank in self._logical_dp_ranks:
                child = child_loader_factory(logical_dp_rank)
                self._children[logical_dp_rank] = child
                self._child_iterators[logical_dp_rank] = iter(child)
        except Exception as error:
            try:
                self.close()
            except Exception as close_error:
                error.add_note(f"child cleanup also failed: {close_error}")
            raise

    def collect(
        self,
        *,
        layout: OptimizerStepLayout,
        itemize: Itemize,
    ) -> CoordinatedWindow:
        if self._closed:
            raise RuntimeError("cannot collect from a closed input coordinator")

        items = []
        bins = []
        payloads = {}
        for logical_dp_rank in self._logical_dp_ranks:
            child_iterator = self._child_iterators[logical_dp_rank]
            for window_microbatch_index in range(layout.num_microbatches):
                stable_id = (logical_dp_rank, window_microbatch_index)
                microbatch = next(child_iterator)
                item = itemize(stable_id, microbatch)
                accumulation_index, pp_microbatch_index = divmod(
                    window_microbatch_index, layout.num_pp_microbatches
                )
                items.append(item)
                bins.append(
                    PackingBin(
                        stable_id=stable_id,
                        token_capacity=item.num_tokens,
                        document_capacity=self._document_capacity,
                        logical_dp_rank=logical_dp_rank,
                        accumulation_index=accumulation_index,
                        pp_microbatch_index=pp_microbatch_index,
                    )
                )
                payloads[stable_id] = microbatch

        return CoordinatedWindow(
            items=tuple(items),
            bins=tuple(bins),
            local_payloads=payloads,
            local_bin_ids=tuple(
                (self._physical_logical_dp_rank, window_microbatch_index)
                for window_microbatch_index in range(layout.num_microbatches)
            ),
        )

    def distribute(
        self,
        window: CoordinatedWindow,
        assignments: Sequence[BinAssignment],
    ) -> list[TrainingMicrobatch]:
        item_ids_by_bin = {
            assignment.bin_id: assignment.item_ids for assignment in assignments
        }
        output = []
        for bin_id in window.local_bin_ids:
            item_ids = item_ids_by_bin[bin_id]
            assert len(item_ids) == 1
            output.append(window.local_payloads[item_ids[0]])
        return output

    def state_dict(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("cannot checkpoint a closed input coordinator")
        physical_rank_state = {
            "balance_group_coordinates": list(self._logical_dp_ranks),
            "physical_logical_dp_rank": self._physical_logical_dp_rank,
            "children": {
                self._child_key(logical_dp_rank): child.state_dict()
                for logical_dp_rank, child in self._children.items()
            },
        }
        return {
            "schema_version": 1,
            "effective_dp_degree": self._dp_world_size,
            self._physical_rank_key(): physical_rank_state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if self._closed:
            raise RuntimeError("cannot restore a closed input coordinator")

        if state_dict.get("schema_version") != 1:
            raise ValueError(
                "unsupported load-balancing checkpoint schema version "
                f"{state_dict.get('schema_version')}"
            )
        if state_dict.get("effective_dp_degree") != self._dp_world_size:
            raise ValueError("checkpoint effective DP degree does not match")

        physical_rank_key = self._physical_rank_key()
        physical_rank_state = state_dict.get(physical_rank_key)
        if not isinstance(physical_rank_state, dict):
            raise ValueError(f"checkpoint is missing state for {physical_rank_key}")
        if physical_rank_state.get("balance_group_coordinates") != list(
            self._logical_dp_ranks
        ):
            raise ValueError("checkpoint balance group coordinates do not match")
        if physical_rank_state.get("physical_logical_dp_rank") != (
            self._physical_logical_dp_rank
        ):
            raise ValueError("checkpoint physical logical DP rank does not match")

        child_states = physical_rank_state.get("children")
        if not isinstance(child_states, dict):
            raise ValueError("checkpoint children must be a dictionary")
        expected_child_keys = {
            self._child_key(logical_dp_rank) for logical_dp_rank in self._children
        }
        if set(child_states) != expected_child_keys:
            raise ValueError("checkpoint logical child set does not match")
        try:
            for logical_dp_rank, child in self._children.items():
                child.load_state_dict(child_states[self._child_key(logical_dp_rank)])
        except Exception as error:
            try:
                self.close()
            except Exception as close_error:
                error.add_note(f"child cleanup also failed: {close_error}")
            raise

    @staticmethod
    def _child_key(logical_dp_rank: int) -> str:
        return f"logical_dp_rank_{logical_dp_rank}"

    def _physical_rank_key(self) -> str:
        return f"physical_dp_rank_{self._physical_logical_dp_rank}"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        for child in self._children.values():
            try:
                child.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
