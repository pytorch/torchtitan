# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer-step-aware load-balancing dataloader."""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Literal

import tyro

from torchtitan.components.data.collators import HAS_PIN_MEMORY, TextCollator
from torchtitan.components.data.load_balancing.planner import (
    LoadBalancePlan,
    PackableItem,
    PackingBin,
    StableId,
    WholeMicrobatchBalancer,
)
from torchtitan.components.data.load_balancing.text import (
    QuadraticAttentionCost,
    TokenizedTextPackingAdapter,
)
from torchtitan.components.data.loader import (
    BaseDataLoader,
    DataloaderExhaustedError,
    GrainDataLoader,
)
from torchtitan.components.data.types import (
    OptimizerStepBatch,
    OptimizerStepLayout,
    TrainingMicrobatch,
)
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Configurable


class ReplicatedInputCoordinator(Configurable):
    """Identify original logical DP streams reconstructed on each process."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        group_size: int = 1

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
    ) -> None:
        if dp_world_size <= 0:
            raise ValueError("dp_world_size must be greater than 0")
        if dp_rank < 0 or dp_rank >= dp_world_size:
            raise ValueError("dp_rank must be within the effective DP world")
        if config.group_size <= 0:
            raise ValueError("replicated input group_size must be greater than 0")
        if dp_world_size % config.group_size != 0:
            raise ValueError("dp_world_size must be divisible by group_size")

        group_start = (dp_rank // config.group_size) * config.group_size
        self.logical_dp_ranks = tuple(
            range(group_start, group_start + config.group_size)
        )
        self.physical_logical_dp_rank = dp_rank
        self.group_size = config.group_size


class LoadBalancingDataLoader(BaseDataLoader):
    """Balance complete optimizer-step windows from replicated Grain streams."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        max_num_documents: Annotated[int | None, tyro.conf.Suppress] = field(
            init=False, default=None
        )
        dataloader: Annotated[GrainDataLoader.Config, tyro.conf.Suppress]
        mode: Annotated[Literal["shadow", "balance"], tyro.conf.Suppress] = "shadow"
        adapter: Annotated[
            TokenizedTextPackingAdapter.Config, tyro.conf.Suppress
        ] = field(default_factory=TokenizedTextPackingAdapter.Config)
        cost_model: Annotated[
            QuadraticAttentionCost.Config, tyro.conf.Suppress
        ] = field(default_factory=QuadraticAttentionCost.Config)
        balancer: Annotated[WholeMicrobatchBalancer.Config, tyro.conf.Suppress] = field(
            default_factory=WholeMicrobatchBalancer.Config
        )
        coordinator: Annotated[
            ReplicatedInputCoordinator.Config, tyro.conf.Suppress
        ] = field(default_factory=ReplicatedInputCoordinator.Config)

        def __post_init__(self) -> None:
            if self.mode not in ("shadow", "balance"):
                raise ValueError("load-balancing mode must be 'shadow' or 'balance'")
            self.max_num_documents = self.dataloader.max_num_documents

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        max_context_length: int,
        num_tokens_per_microbatch: int,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if not isinstance(config.dataloader, GrainDataLoader.Config):
            raise ValueError(
                "load balancing currently requires a GrainDataLoader.Config child"
            )
        if not isinstance(config.dataloader.collator, TextCollator.Config):
            raise ValueError(
                "load balancing currently requires the built-in TextCollator"
            )

        coordinator = config.coordinator.build(
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )

        self._mode = config.mode
        self._coordinator = coordinator
        self._dp_world_size = dp_world_size
        self.max_num_documents = config.dataloader.max_num_documents
        self._adapter = config.adapter.build(
            num_tokens_per_microbatch=num_tokens_per_microbatch,
            max_context_length=max_context_length,
            max_num_documents=self.max_num_documents,
            expect_pinned_memory=HAS_PIN_MEMORY,
        )
        self._cost_model = config.cost_model.build()
        self._balancer = config.balancer.build()
        self._children: dict[int, BaseDataLoader] = {}
        self._child_iterators: dict[int, Iterator[TrainingMicrobatch]] = {}
        self._metrics: dict[str, float] = {}
        self._closed = False

        child_config = replace(
            config.dataloader,
            num_prefetch_microbatches=min(
                config.dataloader.num_prefetch_microbatches,
                1,
            ),
        )
        try:
            for logical_dp_rank in coordinator.logical_dp_ranks:
                child = child_config.build(
                    dp_world_size=dp_world_size,
                    dp_rank=logical_dp_rank,
                    tokenizer=tokenizer,
                    max_context_length=max_context_length,
                    num_tokens_per_microbatch=num_tokens_per_microbatch,
                )
                self._children[logical_dp_rank] = child
                self._child_iterators[logical_dp_rank] = iter(child)
        except Exception as error:
            try:
                self.close()
            except Exception as close_error:
                error.add_note(f"child cleanup also failed: {close_error}")
            raise

    def __iter__(self) -> Iterator[TrainingMicrobatch]:
        raise RuntimeError(
            "LoadBalancingDataLoader is optimizer-step-aware; "
            "use iter_optimizer_steps()"
        )

    def iter_optimizer_steps(
        self, layout: OptimizerStepLayout
    ) -> Iterator[OptimizerStepBatch]:
        """Yield fully planned optimizer steps for this physical DP rank."""
        while True:
            yield self._build_step(layout)

    def _build_step(self, layout: OptimizerStepLayout) -> OptimizerStepBatch:
        if self._closed:
            raise RuntimeError("cannot iterate a closed load-balancing dataloader")
        try:
            return self._prepare_step(layout)
        except StopIteration as error:
            raise DataloaderExhaustedError() from error

    def _prepare_step(self, layout: OptimizerStepLayout) -> OptimizerStepBatch:
        fetch_start = time.perf_counter()
        candidate_payloads: dict[StableId, TrainingMicrobatch] = {}
        for logical_dp_rank in self._coordinator.logical_dp_ranks:
            child_iterator = self._child_iterators[logical_dp_rank]
            for window_microbatch_index in range(layout.num_microbatches):
                microbatch = next(child_iterator)
                candidate_payloads[
                    (logical_dp_rank, window_microbatch_index)
                ] = microbatch
        self._record_metric(
            "data_load/source_fetch_ms",
            (time.perf_counter() - fetch_start) * 1_000,
        )

        items: list[PackableItem] = []
        bins: list[PackingBin] = []
        candidate_host_bytes = 0
        inspect_start = time.perf_counter()
        for logical_dp_rank in self._coordinator.logical_dp_ranks:
            for window_microbatch_index in range(layout.num_microbatches):
                stable_id = (logical_dp_rank, window_microbatch_index)
                accumulation_index, pp_microbatch_index = divmod(
                    window_microbatch_index, layout.num_pp_microbatches
                )
                metadata = self._adapter.inspect_microbatch(
                    candidate_payloads[stable_id]
                )
                items.append(
                    PackableItem(
                        stable_id=stable_id,
                        num_tokens=metadata.num_tokens,
                        num_documents=metadata.num_documents,
                        cost=self._cost_model.estimate(metadata),
                        payload_bytes=metadata.payload_bytes,
                        original_bin_id=stable_id,
                    )
                )
                bins.append(
                    PackingBin(
                        stable_id=stable_id,
                        token_capacity=metadata.num_tokens,
                        document_capacity=self.max_num_documents,
                        logical_dp_rank=logical_dp_rank,
                        accumulation_index=accumulation_index,
                        pp_microbatch_index=pp_microbatch_index,
                    )
                )
                candidate_host_bytes += metadata.payload_bytes
        self._record_metric(
            "data_load/inspection_ms",
            (time.perf_counter() - inspect_start) * 1_000,
        )

        planner_start = time.perf_counter()
        plan = self._balancer.plan(items, bins)
        self._record_metric(
            "data_load/planner_ms",
            (time.perf_counter() - planner_start) * 1_000,
        )
        self._record_plan_metrics(plan, candidate_host_bytes)

        assignments = (
            plan.baseline_assignments if self._mode == "shadow" else plan.assignments
        )
        item_ids_by_bin = {
            assignment.bin_id: assignment.item_ids for assignment in assignments
        }
        output_microbatches = []
        physical_rank = self._coordinator.physical_logical_dp_rank
        for window_microbatch_index in range(layout.num_microbatches):
            bin_id = (physical_rank, window_microbatch_index)
            item_ids = item_ids_by_bin[bin_id]
            assert len(item_ids) == 1
            output_microbatches.append(candidate_payloads[item_ids[0]])
        return layout.group_microbatches(output_microbatches)

    def _record_plan_metrics(
        self, plan: LoadBalancePlan, candidate_host_bytes: int
    ) -> None:
        self._record_metric(
            "data_load/baseline_predicted_cost",
            plan.baseline_objective.synchronized_cost,
        )
        self._record_metric(
            "data_load/balanced_predicted_cost",
            plan.objective.synchronized_cost,
        )
        self._record_metric(
            "data_load/moved_payload_bytes",
            plan.objective.moved_payload_bytes,
        )
        self._record_metric("data_load/candidate_host_bytes", candidate_host_bytes)
        self._record_metric(
            "data_load/replicated_read_amplification",
            self._coordinator.group_size,
        )
        self._record_metric("data_load/unchanged_plans", int(plan.is_unchanged))

    def _record_metric(self, name: str, value: int | float) -> None:
        self._metrics[name] = self._metrics.get(name, 0.0) + float(value)

    def drain_metrics(self) -> dict[str, float]:
        """Return and clear the bounded metrics accumulated since the last call."""
        metrics = self._metrics
        self._metrics = {}
        return metrics

    def state_dict(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("cannot checkpoint a closed load-balancing dataloader")
        physical_rank_state = {
            "balance_group_coordinates": list(self._coordinator.logical_dp_ranks),
            "physical_logical_dp_rank": (self._coordinator.physical_logical_dp_rank),
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
            raise RuntimeError("cannot restore a closed load-balancing dataloader")

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
            self._coordinator.logical_dp_ranks
        ):
            raise ValueError("checkpoint balance group coordinates do not match")
        if physical_rank_state.get("physical_logical_dp_rank") != (
            self._coordinator.physical_logical_dp_rank
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
        return f"physical_dp_rank_{self._coordinator.physical_logical_dp_rank}"

    def close(self) -> None:
        """Close every constructed child, raising only after all are attempted."""
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
