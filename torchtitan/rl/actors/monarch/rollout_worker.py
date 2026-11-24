# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from monarch.actor import concurrent_endpoint, current_rank

# TODO: load the `grain` and `datasets` packages only where they are used, to save ~450 MiB per rollout worker.
#   This import runs `torchtitan/components/data/__init__.py`, which loads both. The renderer and
#   `rl/rollout/data.py` import them too. First check that a worker doesn't need them.
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.monarch.actor import Actor
from torchtitan.rl.actors.monarch.rdma_object import MonarchRDMAObject
from torchtitan.rl.rollout.finished_group import finish_rollout_group
from torchtitan.rl.rollout.packed_group import pack_rollout_group, PackedRolloutGroup
from torchtitan.rl.rollout.packed_group_handle import (
    PackedRolloutGroupHandle,
    RolloutGroupBuffers,
)
from torchtitan.rl.rollout.recorder import RolloutSampleEncoder, RolloutSampleRecorder
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker
from torchtitan.rl.rollout.types import RolloutGroupSummary
from torchtitan.rl.torch_transport_rdma import (
    make_rdma_probe,
    RDMAByteBuffers,
    RDMAProbe,
)


class RolloutWorkerActor(Actor):
    """Monarch adapter that owns a configurable standalone rollout worker."""

    def __init__(
        self,
        *,
        worker_config: BaseRolloutWorker.Config,
        worker_kwargs: dict[str, Any],
        num_threads: int,
        structured_logger_init_fn: functools.partial[None] | None,
    ) -> None:
        if structured_logger_init_fn is not None:
            structured_logger_init_fn(
                source="rl_rollout_worker", rank=current_rank().rank
            )
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=num_threads)
        )
        if worker_config.rdma_min_image_bytes is not None:
            worker_kwargs = {
                **worker_kwargs,
                "register_rdma_object": MonarchRDMAObject.register,
            }
        self._worker = worker_config.build(**worker_kwargs)
        # Large training groups stay here, packed, until the controller releases them.
        self._group_buffers: RolloutGroupBuffers | None = None
        self._rdma_byte_buffers: RDMAByteBuffers | None = None
        self._rdma_probe: RDMAProbe | None = None
        if worker_config.rdma_min_group_bytes is not None:
            self._rdma_byte_buffers = RDMAByteBuffers()
            self._group_buffers = RolloutGroupBuffers(
                min_bytes=worker_config.rdma_min_group_bytes,
                buffers=self._rdma_byte_buffers,
            )

    @concurrent_endpoint
    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        rollout_recorder_config: RolloutSampleRecorder.Config,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        self._encoder = RolloutSampleEncoder(rollout_recorder_config)
        await self._worker.setup_async(
            tokenizer_config=tokenizer_config,
            renderer_config=renderer_config,
            hf_assets_path=hf_assets_path,
            vision_processor_config=vision_processor_config,
        )

    @concurrent_endpoint
    async def run_group(
        self,
        *,
        generate_fn: Any,
        release_session_fn: Any,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: Any,
        is_validation: bool,
    ) -> tuple[RolloutGroupSummary, PackedRolloutGroup | PackedRolloutGroupHandle]:
        """Returns the scored group, with its rollout metrics and recorder lines
        (`finish_rollout_group`), packed: its summary for the batcher and its bytes for the
        trainer, or a handle to them when a training group stays here in a registered buffer
        (`BaseRolloutWorker.Config.rdma_min_group_bytes`)."""
        # The worker's time in the call. The controller's `rollout_group` span times the whole
        # call, so the difference is the hop's transfer, serialization and dispatch time.
        with sl.log_trace_span("run_group"):
            group = await self._worker.run_group(
                generate_fn=generate_fn,
                release_session_fn=release_session_fn,
                sample=sample,
                group_id=group_id,
                group_size=group_size,
                sampling=sampling,
            )
            group = finish_rollout_group(
                group, is_validation=is_validation, encoder=self._encoder
            )
            # Validation groups go by value: the controller reads only their summaries.
            if is_validation or self._group_buffers is None:
                return pack_rollout_group(group)
            return self._group_buffers.pack(group)

    @concurrent_endpoint
    async def release_groups(self, group_ids: list[int]) -> None:
        """Return the buffers of the released groups kept here to the pool."""
        assert self._group_buffers is not None
        self._group_buffers.release(group_ids)

    @concurrent_endpoint
    async def make_rdma_probe(self) -> RDMAProbe:
        """A small buffer for every trainer DP replica leader to read at setup."""
        assert self._rdma_byte_buffers is not None and self._rdma_probe is None
        self._rdma_probe = make_rdma_probe(self._rdma_byte_buffers)
        return self._rdma_probe

    @concurrent_endpoint
    async def drop_rdma_probe(self) -> None:
        assert self._rdma_byte_buffers is not None and self._rdma_probe is not None
        self._rdma_byte_buffers.release(self._rdma_probe.buffer)
        self._rdma_probe = None

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        sl.set_step(step)
