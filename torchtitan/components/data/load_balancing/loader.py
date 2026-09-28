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
from torch.distributed.tensor import DeviceMesh

from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.load_balancing.coordinator import (
    InputCoordinator,
    ReplicatedInputCoordinator,
)
from torchtitan.components.data.load_balancing.planner import (
    LoadBalancePlan,
    PackableItem,
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
from torchtitan.components.data.types import OptimizerStepLayout, TrainingMicrobatch
from torchtitan.components.tokenizer import BaseTokenizer


class LoadBalancingDataLoader(BaseDataLoader):
    """Balance complete optimizer-step windows through an input coordinator."""

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
        coordinator: Annotated[InputCoordinator.Config, tyro.conf.Suppress] = field(
            default_factory=ReplicatedInputCoordinator.Config
        )

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
        optimizer_step_layout: OptimizerStepLayout,
        dp_mesh: DeviceMesh | None = None,
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

        self._mode = config.mode
        self._optimizer_step_layout = optimizer_step_layout
        self.max_num_documents = config.dataloader.max_num_documents
        self._adapter = config.adapter.build()
        self._cost_model = config.cost_model.build()
        self._balancer = config.balancer.build()
        self._metrics: dict[str, float] = {}

        child_config = replace(
            config.dataloader,
            num_prefetch_microbatches=min(
                config.dataloader.num_prefetch_microbatches,
                1,
            ),
        )

        def build_child(logical_dp_rank: int) -> BaseDataLoader:
            return child_config.build(
                dp_world_size=dp_world_size,
                dp_rank=logical_dp_rank,
                tokenizer=tokenizer,
                max_context_length=max_context_length,
                num_tokens_per_microbatch=num_tokens_per_microbatch,
            )

        self._coordinator: InputCoordinator = config.coordinator.build(
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            dp_mesh=dp_mesh,
            child_loader_factory=build_child,
            document_capacity=self.max_num_documents,
        )

    def __iter__(self) -> Iterator[TrainingMicrobatch]:
        while True:
            yield from self._build_step()

    def _build_step(self) -> list[TrainingMicrobatch]:
        try:
            return self._prepare_step()
        except StopIteration as error:
            raise DataloaderExhaustedError() from error

    def _prepare_step(self) -> list[TrainingMicrobatch]:
        layout = self._optimizer_step_layout
        inspection_seconds = 0.0

        def timed_itemize(
            stable_id: StableId, microbatch: TrainingMicrobatch
        ) -> PackableItem:
            nonlocal inspection_seconds
            inspect_start = time.perf_counter()
            item = self._itemize(stable_id, microbatch)
            inspection_seconds += time.perf_counter() - inspect_start
            return item

        # The coordinator owns input topology and reads the complete candidate
        # window. It calls itemize for each payload so tensor-specific inspection
        # remains outside the coordinator.
        collect_start = time.perf_counter()
        window = self._coordinator.collect(layout=layout, itemize=timed_itemize)
        collect_seconds = time.perf_counter() - collect_start
        self._record_metric(
            "data_load/inspection_ms",
            inspection_seconds * 1_000,
        )
        # collect() includes itemization time, so subtract it to avoid counting
        # adapter work in both metrics.
        self._record_metric(
            "data_load/source_fetch_ms",
            (collect_seconds - inspection_seconds) * 1_000,
        )

        # Planning uses only abstract item and bin metadata. Coordinators that
        # expose the same candidate window therefore produce the same plan.
        planner_start = time.perf_counter()
        plan = self._balancer.plan(window.items, window.bins)
        self._record_metric(
            "data_load/planner_ms",
            (time.perf_counter() - planner_start) * 1_000,
        )
        self._record_plan_metrics(plan, window.local_payload_bytes)

        assignments = (
            plan.baseline_assignments if self._mode == "shadow" else plan.assignments
        )
        # The coordinator realizes the abstract assignment. Replicated input can
        # select local payloads directly; a communication-backed coordinator can
        # exchange reassigned payloads here.
        return self._coordinator.distribute(window, assignments)

    def _itemize(
        self, stable_id: StableId, microbatch: TrainingMicrobatch
    ) -> PackableItem:
        """Describe a concrete payload without exposing it to the planner."""
        metadata = self._adapter.inspect_microbatch(microbatch)
        return PackableItem(
            stable_id=stable_id,
            num_tokens=metadata.num_tokens,
            num_documents=metadata.num_documents,
            cost=self._cost_model.estimate(metadata),
            payload_bytes=metadata.payload_bytes,
            original_bin_id=stable_id,
        )

    def _record_plan_metrics(
        self, plan: LoadBalancePlan, candidate_host_bytes: int
    ) -> None:
        self._record_metric(
            "data_load/baseline_predicted_cost",
            plan.baseline_predicted_cost,
        )
        self._record_metric(
            "data_load/balanced_predicted_cost",
            plan.predicted_cost,
        )
        self._record_metric(
            "data_load/moved_payload_bytes",
            plan.moved_payload_bytes,
        )
        self._record_metric("data_load/candidate_host_bytes", candidate_host_bytes)
        self._record_metric("data_load/unchanged_plans", int(plan.is_unchanged))

    def _record_metric(self, name: str, value: int | float) -> None:
        self._metrics[name] = self._metrics.get(name, 0.0) + float(value)

    def drain_metrics(self) -> dict[str, float]:
        """Return and clear the bounded metrics accumulated since the last call."""
        metrics = self._metrics
        self._metrics = {}
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return self._coordinator.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._coordinator.load_state_dict(state_dict)

    def close(self) -> None:
        self._coordinator.close()
