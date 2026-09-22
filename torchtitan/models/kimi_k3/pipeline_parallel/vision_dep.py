# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The vision tower on a pipeline stage of its own, with its encodes moved off the
stage's own forward and into the intervals the rank would otherwise spend waiting."""

from __future__ import annotations

import logging
from typing import Any, cast, TYPE_CHECKING

import torch
from torch.distributed.fsdp import FSDPModule

from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    _get_pipeline_metadata,
)
from torchtitan.protocols.model import BaseModel

from .dep_backward import cut_for_deferred_backward, GradQueue
from .dep_bubble_plan import BubblePlan, plan_for_rank
from .stage import AttnResPipelineStage

if TYPE_CHECKING:
    from ..model import KimiK3Model

logger = logging.getLogger(__name__)

VIT_DEP_STAGE_FQNS = ("tok_embeddings", "vision_encoder")


def vit_dep_split(
    model: BaseModel, *, parallel_dims, parallelism, model_config
) -> list[list[str]]:
    """The split with the tower and the embedding on the first stage and the layers on the rest."""
    if getattr(model, "vision_encoder", None) is None:
        raise ValueError(
            "vit_dep gives the vision tower a pipeline stage of its own, and "
            "this model has no vision encoder."
        )
    num_stages, num_layers, _, output_weight = _get_pipeline_metadata(
        parallel_dims, parallelism, model_config
    )
    if num_stages < 2:
        raise ValueError(
            "vit_dep needs at least two pipeline stages: one for the tower and "
            "the embedding, one for the layers."
        )
    # The embedding rides with the tower: splicing the features needs the token
    # ids, which only the first stage receives.
    text = _generate_llm_fqn_per_model_part(
        num_stages - 1, num_layers, 0, output_weight
    )
    text[-1].extend(
        name
        for name in model.pipeline_last_stage_module_fqns
        if getattr(model, name, None) is not None
    )
    return [list(VIT_DEP_STAGE_FQNS)] + [
        [name for name in stage if name != "tok_embeddings"] for stage in text
    ]


class VisionFeatureCache:
    """The tower's features for the micro-batches of one step, keyed by micro-batch."""

    def __init__(self, owner: KimiK3Model) -> None:
        self._owner = owner
        self._fsdp = [m for m in owner.modules() if isinstance(m, FSDPModule)]
        self._features: dict[int, torch.Tensor] = {}
        self._kwargs: list[dict] | None = None
        self._num_mbs = 0
        self._hits = 0
        self._misses = 0

    def begin_step(self, kwarg_mbs) -> None:
        """Take the step's per-micro-batch kwargs and drop the previous step's features."""
        if self._hits or self._misses:
            logger.info(
                "DEP vision encode: %d served from the cache, %d encoded inline",
                self._hits,
                self._misses,
            )
            self._hits = self._misses = 0
        self._features.clear()
        if kwarg_mbs is None:
            self._kwargs, self._num_mbs = None, 0
            return
        self._kwargs = list(kwarg_mbs)
        self._num_mbs = len(self._kwargs)

    def count(self) -> int:
        """How many micro-batches this step carries; the same on every rank."""
        return self._num_mbs

    def _inputs_for(self, mb: int):
        if self._kwargs is None or not 0 <= mb < self._num_mbs:
            return None
        kw = self._kwargs[mb] or {}
        pixel_values, grid_thw = kw.get("pixel_values"), kw.get("grid_thw")
        if pixel_values is None or grid_thw is None:
            return None
        return pixel_values, grid_thw

    def encode(self, mb: int) -> None:
        """Encode ``mb`` now on the current stream if it is not cached yet."""
        if mb in self._features:
            return
        inputs = self._inputs_for(mb)
        if inputs is None:
            return
        # FSDP2 all-gathers in the pre-forward hook of the module the pipeline
        # calls, and this runs ahead of that call, so the gather happens here;
        # the stage's own forward reshards as its policy says.
        for module in self._fsdp:
            module.unshard()
        self._features[mb] = self._owner.encode_images(*inputs)

    def take(self, mb: int) -> torch.Tensor | None:
        """The features for ``mb`` if they were encoded ahead, else None; removes the entry."""
        feats = self._features.pop(mb, None)
        if feats is None:
            self._misses += 1
        else:
            self._hits += 1
        return feats


def _anchor_kind(computation_type: str) -> str:
    """F, B or W for the stage method that runs this action; empty when none does."""
    if "BACKWARD_WEIGHT" in computation_type:
        return "W"
    if "BACKWARD" in computation_type:
        return "B"
    if "FORWARD" in computation_type:
        return "F"
    return ""


class VisionDepRuntime:
    """What the rank runs at each of its schedule's actions: the planned encodes, the
    deferred tower backwards, and the features the tower's own stage reads."""

    def __init__(
        self,
        cache: VisionFeatureCache,
        *,
        tower_stage_index: int,
        rank: int,
        pp_size: int,
        prefetch: int,
        cost_ratio: float,
        pipeline_order: dict | None,
        queue: GradQueue | None,
    ) -> None:
        self._cache = cache
        self._tower = tower_stage_index
        self._rank = rank
        self._pp_size = pp_size
        self._prefetch = prefetch
        self._cost_ratio = cost_ratio
        self._pipeline_order = pipeline_order
        self._queue = queue
        self._by_anchor: dict[tuple[str, int, int], list[int]] = {}
        self._plan: BubblePlan | None = None
        self._fired = 0
        # FSDP2 builds its state in the root module's first forward, and an encode
        # issued before that makes the tower a root of its own, after which the
        # stage's forward refuses. So the first step encodes inline and the
        # run-ahead starts with the second.
        self._warm = False

    def begin_step(self, kwarg_mbs) -> None:
        self._cache.begin_step(kwarg_mbs)
        self._by_anchor = {}
        self._plan = None
        self._fired = 0
        if self._pipeline_order is None or not self._warm:
            return
        n = self._cache.count()
        if n == 0:
            return
        self._plan = plan_for_rank(
            self._pipeline_order[self._rank],
            rank=self._rank,
            vision_microbatches=n,
            cost_ratio=self._cost_ratio,
            upfront=min(self._pp_size, n),
            vision_stage=self._tower,
        )
        unfirable = 0
        for placement in self._plan.placed:
            kind, stage_index, mb_index = placement.anchor
            bucket = _anchor_kind(kind)
            if not bucket:
                unfirable += 1
                continue
            self._by_anchor.setdefault((bucket, stage_index, mb_index), []).append(
                placement.microbatch
            )
        if unfirable:
            logger.warning(
                "DEP bubble: %d placement(s) anchored on an action this rank does "
                "not run; they stay inline.",
                unfirable,
            )
        for mb in self._plan.upfront:
            self._cache.encode(mb)

    def forward_kwargs(
        self, stage_index: int, mb: int, kwargs: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """The tower stage's forward reads the features this step already encoded."""
        if stage_index != self._tower:
            return kwargs
        feats = self._cache.take(mb)
        if feats is None:
            return kwargs
        if self._queue is not None:
            feats = cut_for_deferred_backward(feats, self._queue, mb)
        return {**(kwargs or {}), "vision_embeds": feats}

    def after_forward(self, stage_index: int, mb: int) -> None:
        self._warm = True
        if stage_index == self._tower and self._prefetch:
            for ahead in range(1, self._prefetch + 1):
                self._cache.encode(mb + ahead)
        self._fire("F", stage_index, mb)

    def after_backward(self, stage_index: int, mb: int, *, weight_only: bool) -> None:
        self._fire("W" if weight_only else "B", stage_index, mb)
        if self._queue is not None and not weight_only:
            self._queue.run_next()

    def end_step(self) -> None:
        if self._queue is not None:
            self._queue.drain()
            self._queue.report()
        if self._plan is None:
            return
        placed = len(self._plan.placed)
        # Placed but never fired means the plan and the schedule disagree; the
        # encode then falls back to its inline path and still looks correct.
        level = logger.info if self._fired == placed else logger.warning
        level(
            "DEP bubble: %d/%d planned encode(s) ran in a bubble, %d upfront, "
            "%d left inline, %d idle slot(s) (%d starved, %d exhausted)",
            self._fired,
            placed,
            len(self._plan.upfront),
            len(self._plan.synchronous),
            self._plan.idle_slots,
            self._plan.slots_starved,
            self._plan.slots_exhausted,
        )

    def _fire(self, bucket: str, stage_index: int, mb: int) -> None:
        queued = self._by_anchor.pop((bucket, stage_index, mb), None)
        if not queued:
            return
        for vision_mb in queued:
            self._cache.encode(vision_mb)
        self._fired += len(queued)


class _NoVisionDep:
    """The runtime a rank holding no tower gets: every hook is a no-op."""

    def forward_kwargs(
        self, stage_index: int, mb: int, kwargs: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        return kwargs

    def after_forward(self, stage_index: int, mb: int) -> None:
        pass

    def after_backward(self, stage_index: int, mb: int, *, weight_only: bool) -> None:
        pass


class VisionDepPipelineStage(AttnResPipelineStage):
    """AttnRes stage that hands the runtime each of its actions as it completes."""

    # Every rank builds this stage class so the schedule is the same shape on all
    # of them; only the rank holding the tower is given a runtime.
    _dep: VisionDepRuntime | _NoVisionDep = _NoVisionDep()

    def set_vision_dep(self, runtime: VisionDepRuntime) -> None:
        self._dep = runtime

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        save_forward_output: bool = True,
    ):
        kwargs = self._dep.forward_kwargs(self.stage_index, int(fwd_chunk_id), kwargs)
        output = super().forward_one_chunk(
            fwd_chunk_id, args, kwargs, save_forward_output
        )
        self._dep.after_forward(self.stage_index, int(fwd_chunk_id))
        return output

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
    ):
        output = super().backward_one_chunk(
            bwd_chunk_id,
            loss=loss,
            full_backward=full_backward,
            last_backward=last_backward,
        )
        self._dep.after_backward(self.stage_index, int(bwd_chunk_id), weight_only=False)
        return output

    def backward_weight_one_chunk(self, bwd_chunk_id: int, last_backward=False):
        output = super().backward_weight_one_chunk(
            bwd_chunk_id, last_backward=last_backward
        )
        self._dep.after_backward(self.stage_index, int(bwd_chunk_id), weight_only=True)
        return output


def install_vision_dep(
    pp_schedule,
    stages: list[VisionDepPipelineStage],
    *,
    rank: int,
    pp_size: int,
    prefetch: int,
    bubble: bool,
    cost_ratio: float,
    max_pending: int,
) -> None:
    """Give this rank's stages the vision runtime, on the rank that holds the tower."""
    tower = [s for s in stages if getattr(s.submod, "vision_encoder", None) is not None]
    if not tower:
        return
    if len(tower) != 1:
        raise RuntimeError(
            f"rank {rank} holds {len(tower)} stages with a vision tower; vit_dep "
            "places it on one."
        )
    pipeline_order = None
    if bubble:
        pipeline_order = getattr(pp_schedule, "pipeline_order", None)
        if not pipeline_order:
            raise ValueError(
                "vit_bubble reads the schedule's action order, which only the "
                "looped schedules expose; use vit_prefetch with a single-stage "
                "schedule."
            )
    runtime = VisionDepRuntime(
        VisionFeatureCache(cast("KimiK3Model", tower[0].submod)),
        tower_stage_index=tower[0].stage_index,
        rank=rank,
        pp_size=pp_size,
        prefetch=prefetch,
        cost_ratio=cost_ratio,
        pipeline_order=pipeline_order,
        queue=GradQueue(max_pending=max_pending) if bubble else None,
    )
    for stage in stages:
        stage.set_vision_dep(runtime)

    # The step is where the micro-batch kwargs arrive, and the schedule class is
    # core's choice, so the instance is wrapped once here rather than subclassed.
    original_step = pp_schedule.step

    def step(*args, **kwargs):
        runtime.begin_step(kwargs.get("kwarg_mbs"))
        try:
            return original_step(*args, **kwargs)
        finally:
            runtime.end_step()

    pp_schedule.step = step
    logger.info(
        "DEP vision %s installed on stage %d",
        f"bubble (cost ratio {cost_ratio:.2f}, max pending {max_pending})"
        if bubble
        else f"run-ahead of {prefetch} micro-batch(es)",
        tower[0].stage_index,
    )
