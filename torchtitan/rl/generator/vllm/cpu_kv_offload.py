# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM's native CPU KV offload, set up for the generator's in-process engine."""

from __future__ import annotations

import itertools
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import torch.distributed as dist
from vllm.config import KVTransferConfig
from vllm.v1.metrics.loggers import StatLoggerBase

from torchtitan.config import Configurable


class CPUKVOffload(Configurable):
    """vLLM's native CPU KV offload: vLLM copies every computed KV block to host RAM, and a request
    whose prefix left the GPU reloads it from there instead of recomputing it.

    On top of vLLM's own offload, for the generator's in-process engine:
    - completions are stored too: the next turn's prompt holds this turn's completion;
    - with TP > 1 or DP > 1, each scheduler waits for its own TP ranks' transfers (`attach`);
    - ended sessions' CPU chunks are evicted first (`evict_first`, called by `VLLMSessionKVHolder`).

    Example:

        offload = CPUKVOffload.Config(size_gib=96).build()
        engine_kwargs.update(offload.engine_kwargs())  # before the engine build
        engine = LLMEngine.from_engine_args(
            EngineArgs(**engine_kwargs), stat_loggers=[offload.stat_logger]
        )
        offload.attach(engine)
        offload.evict_first(ended_block_hashes)  # a session ended
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        size_gib: float
        """Host RAM (GiB) per generator, split across its DP x TP ranks. vLLM pins it in
        ``/dev/shm``, so the generators on one host must fit there together. Generator procs need
        ``expandable_segments:False`` and ``PYTHONHASHSEED=0`` (``train.py`` sets both)."""

        def __post_init__(self):
            if self.size_gib <= 0:
                raise ValueError(f"size_gib must be > 0, got {self.size_gib}")

    def __init__(self, config: Config) -> None:
        self.config = config
        # Set by `attach`: vLLM's CPU LRU, and the KV cache groups each block has a chunk in.
        self._lru: Any = None
        self._num_kv_cache_groups = 0
        # CPU store bytes, added by `stat_logger` (engine thread); a plain int any thread can read.
        self.store_bytes = 0

    def engine_kwargs(self) -> dict[str, Any]:
        """vLLM engine args that turn the offload on."""
        return {
            # Named here, not through kv_offloading_size, so vLLM's in-process config checks see the
            # connector: vLLM rejects router replay (enable_return_routed_experts) with it.
            "kv_transfer_config": KVTransferConfig(
                kv_connector="OffloadingConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "cpu_bytes_to_use": int(self.config.size_gib * (1 << 30)),
                    # Store completions too: vLLM's default stores prompt blocks only.
                    "offload_prompt_only": False,
                },
            )
        }

    def attach(self, engine: Any) -> None:
        """Keep vLLM's CPU LRU for `evict_first`, and make the offload work with TP > 1 or DP > 1.

        vLLM marks a CPU transfer done once every worker of its scheduler finished it, and only its
        multiprocess executor merges their results. Under external_launcher, each rank runs its own
        scheduler with one worker:
        1. Each scheduler waits for its own TP group only (vLLM's world size also counts DP).
        2. With TP > 1, every engine step all-gathers and merges the TP ranks' finished transfers:

               step s    rank 0: load 7 done   rank 1: -            both schedulers: 1 of 2
               step s+1  rank 0: -             rank 1: load 7 done  both schedulers: done

        3. The TP ranks' schedulers must stay identical: request ids get a counter suffix, not vLLM's
           random one, and procs run with PYTHONHASHSEED=0 (vLLM iterates sets). Each gather raises if
           the ranks' scheduler fingerprints differ.
        """
        # Imported here: runs without CPU offload never load vLLM's offload code.
        from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
        from vllm.distributed.parallel_state import get_tp_group
        from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

        scheduler = engine.engine_core.engine_core.scheduler
        connector_scheduler = scheduler.connector.connector_scheduler
        # `evict_first` reorders the LRU's private `evictable_chunks` (pinned vLLM version).
        self._lru = connector_scheduler.manager._policy
        if not isinstance(self._lru, LRUCachePolicy):
            raise RuntimeError(
                "CPU KV offload needs vLLM's LRUCachePolicy, got "
                f"{type(self._lru).__name__}"
            )
        self._num_kv_cache_groups = scheduler.kv_cache_manager.num_kv_cache_groups

        tp_group = get_tp_group()
        connector_scheduler.config = connector_scheduler.config._replace(
            num_workers=tp_group.world_size
        )
        if tp_group.world_size == 1:
            return
        if sys.flags.hash_randomization:
            raise ValueError(
                "session_kv_holder.cpu_kv_offload with tensor_parallel_degree > 1 needs "
                "PYTHONHASHSEED=0 in generator procs (train.py sets it): TP ranks iterate the same "
                "sets and must agree on the order"
            )

        request_counter = itertools.count()

        def assign_request_id(request) -> None:
            # Replaces vLLM's "<request_id>-<8 random chars>", which differs per rank.
            request.external_req_id = request.request_id
            request.request_id = f"{request.request_id}-{next(request_counter)}"

        if not hasattr(engine.input_processor, "assign_request_id"):
            raise RuntimeError(
                "vLLM's InputProcessor has no assign_request_id to replace; update "
                "CPUKVOffload.attach for this vLLM version"
            )
        engine.input_processor.assign_request_id = assign_request_id

        aggregator = KVOutputAggregator(expected_finished_count=tp_group.world_size)
        update_from_output = scheduler.update_from_output

        def update_from_merged_output(scheduler_output, model_runner_output):
            fingerprint = (
                hash(tuple(scheduler_output.num_scheduled_tokens.items())),
                connector_scheduler._job_counter,
            )
            gathered: list[Any] = [None] * tp_group.world_size
            # Every rank gathers on every step, even with nothing finished, or the ranks deadlock.
            dist.all_gather_object(
                gathered,
                (fingerprint, model_runner_output.kv_connector_output),
                group=tp_group.cpu_group,
            )
            fingerprints, kv_connector_outputs = zip(*gathered)
            if any(
                rank_fingerprint != fingerprint for rank_fingerprint in fingerprints
            ):
                raise RuntimeError(
                    "TP ranks' vLLM schedulers diverged; (scheduled requests hash, offload jobs) "
                    f"per rank: {fingerprints}"
                )
            # A copy per rank: aggregate() writes into its output_rank entry, and steps without work
            # share vLLM's EMPTY_MODEL_RUNNER_OUTPUT.
            merged = aggregator.aggregate(
                [
                    replace(
                        model_runner_output, kv_connector_output=kv_connector_output
                    )
                    for kv_connector_output in kv_connector_outputs
                ],
                output_rank=tp_group.rank_in_group,
            )
            return update_from_output(scheduler_output, merged)

        scheduler.update_from_output = update_from_merged_output

    def evict_first(self, block_hashes: Iterable[bytes]) -> None:
        """Make these blocks' CPU chunks (one per KV cache group) the first evicted in vLLM's LRU.
        Skips chunks vLLM never stored, or is still storing or loading."""
        from vllm.v1.kv_offload.base import make_offload_key

        # TODO: a chunk whose store finishes after the release (the last turn's final blocks,
        # stored on the next engine step) keeps its LRU place; evict it first once stored.
        evictable_chunks = self._lru.evictable_chunks
        for block_hash in block_hashes:
            for group_idx in range(self._num_kv_cache_groups):
                key = make_offload_key(block_hash, group_idx)
                if key in evictable_chunks:
                    evictable_chunks.move_to_end(key, last=False)

    def used_fraction(self) -> float:
        """Fraction of the CPU tier's chunks that hold KV (stored or being stored); vLLM's usage
        gauge counts only the chunks a transfer pins."""
        return len(self._lru.chunks) / self._lru.cache_capacity

    def stat_logger(self, vllm_config: Any, engine_index: int) -> StatLoggerBase:
        """A vLLM stat logger factory: adds each engine step's CPU store bytes to `store_bytes`."""
        return _StoreBytesStatLogger(self)


class _StoreBytesStatLogger(StatLoggerBase):
    def __init__(self, offload: CPUKVOffload) -> None:
        self._offload = offload

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ) -> None:
        if scheduler_stats is None or scheduler_stats.kv_connector_stats is None:
            return
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
            OffloadingConnectorStats,
        )

        counters = OffloadingConnectorStats(
            data=scheduler_stats.kv_connector_stats
        ).reduce()
        self._offload.store_bytes += counters.get("vllm:kv_offload_store_bytes", 0)

    def log_engine_initialized(self) -> None:
        pass
