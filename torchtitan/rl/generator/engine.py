# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The inference engine interface the generator's engine loop drives."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch

from torchtitan.config import Configurable
from torchtitan.rl.generator.types import EngineRequest, FinishedRequest


class GenerationEngine(Configurable, abc.ABC):
    """One inference engine replica that runs a TorchTitan model, driven by the engine loop.

    The generator builds the engine with ``config.build(...)`` on its engine loop thread,
    and every method runs there: implementations need not be thread-safe. Every rank of a
    replica receives the same calls in the same order, so collectives inside them (e.g. a
    TP engine step) line up across ranks.

    ``build`` passes the engine's own ``config`` and these keyword arguments, which every
    engine accepts:

        model: The generator's model config.
        parallelism: The generator's parallelism; each rank's DP / TP rank derives from
            ``rank`` and this layout.
        model_path: Path to the HF model checkpoint (tokenizer assets and, with a
            ``checkpointer``, initial weights).
        max_num_seqs: Upper bound on concurrently scheduled sequences.
        model_dtype: Data type for model weights (auto, float16, bfloat16, float32).
        gpu_memory_limit: Fraction of GPU memory the engine may use.
        max_num_batched_tokens: Max tokens scheduled per engine step, or None for the
            engine's default.
        checkpointer: Optional initial-weight loader; None when weights arrive through
            model state dict loads.
        dist_moe_runtime: Dist-MoE runtime policy, when the model has Dist-MoE experts.
        debug: Debug and determinism settings.
        reset_kv_cache_on_model_state_dict_load: Whether each model state dict load resets
            all KV. When False, a request may reuse only KV computed under its
            ``min_policy_version``.
        rank: Global rank of this process.
        generator_name: Name of the generator, for engine metrics.
        output_dir: Output directory, for engine metrics.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @property
    @abc.abstractmethod
    def data_parallel_size(self) -> int:
        """The data-parallel size the engine derived."""

    @property
    @abc.abstractmethod
    def data_parallel_rank(self) -> int:
        """The data-parallel rank the engine derived for this process."""

    @property
    @abc.abstractmethod
    def prefetched_model_state_dict(self) -> dict[str, torch.Tensor]:
        """The pinned CPU buffers a model state dict is prefetched into, and loaded from by
        ``load_prefetched_model_state_dict``. Safe to fill from any thread while no load runs."""

    @abc.abstractmethod
    def add_requests(
        self, requests: list[EngineRequest]
    ) -> list[tuple[str, Exception]]:
        """Admit ``requests`` in order and return the ``(request_id, error)`` of each request
        the engine rejected. A rejected request must leave no engine state."""

    @abc.abstractmethod
    def has_unfinished_requests(self) -> bool:
        """Whether any admitted request is still running."""

    @abc.abstractmethod
    def step(self) -> list[FinishedRequest]:
        """Run one engine step and return the requests it finished."""

    @abc.abstractmethod
    def load_prefetched_model_state_dict(self) -> None:
        """Load the prefetched model state dict into the model, and reset the KV cache when
        configured."""

    @property
    def holds_session_kv(self) -> bool:
        """Whether the engine holds multi-turn sessions' KV between turns, so callers must
        release sessions and groups when their rollouts end."""
        return False

    def release_sessions(self, session_ids: list[str], group_ids: list[int]) -> None:
        """Release the held KV of ``session_ids`` and of every session of ``group_ids``."""

    def log_stats(self, step: int) -> None:
        """Log the engine's own per-step stats, if any."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release engine-local resources; process teardown is the actor framework's."""
