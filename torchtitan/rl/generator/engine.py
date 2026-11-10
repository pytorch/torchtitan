# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The inference engine interface the generator's engine loop drives."""

from __future__ import annotations

import abc
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import torch

from torchtitan.config import Configurable
from torchtitan.rl.generator.types import EngineRequest, FinishedRequest


ModelStateDictLoadMode = Literal["cpu_prefetch", "gpu_fetch"]


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
        precision: The trainer's parameter and compute dtypes, which the engine's model
            mirrors: it computes in ``param_dtype`` and receives the trainer's weights
            in the dtypes the trainer sends them.
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
        model_state_dict_load_mode: ``"cpu_prefetch"`` allocates the pinned CPU buffers of
            ``prefetched_model_state_dict``; ``"gpu_fetch"`` does not.
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
    def prefetched_model_state_dict(self) -> dict[str, torch.Tensor] | None:
        """The pinned CPU buffers a model state dict is prefetched into, and loaded from by
        a model state dict load. Safe to fill from any thread while no load runs. None with
        ``model_state_dict_load_mode="gpu_fetch"``."""

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
    def load_model_state_dict(
        self, fetch: Callable[[dict[str, torch.Tensor]], Iterable[str]]
    ) -> None:
        """Load a model state dict by having ``fetch`` fill the model's weights in place, and
        reset the KV cache when configured.

        Args:
            fetch: Fills each tensor of the given subset of the model state dict in place
                with the new weights and returns the keys it filled.
        """

    def release_sessions(self, session_ids: list[str], group_ids: list[int]) -> None:
        """Drop the cached KV of ended rollouts (``session_ids``) and of every rollout of
        ended groups (``group_ids``), so live rollouts keep theirs longer. A released session
        may still have a request running (the engine loop admits a decision's requests
        first): keep that request's KV until it finishes."""

    def log_stats(self, step: int) -> None:
        """Log the engine's own per-step stats, if any."""

    def kv_cache_budget(self) -> KVCacheBudget:
        """Return the KV cache size and the blocks one session holds, for the inter-generator
        router's admission (``InterGeneratorRouter.Config.admission``). Called from the actor's
        thread, so it reads only what the build fixed."""
        raise NotImplementedError(
            f"{type(self).__name__} does not report its KV cache size, which admission needs"
        )

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release engine-local resources; process teardown is the actor framework's."""


@dataclass(frozen=True, kw_only=True, slots=True)
class KVCacheBudget:
    """A generator's KV cache size and the blocks one session holds in it.

    Example (Qwen3.5-35B-A3B on one GB300: 1 attention group, 3 Gated-DeltaNet groups)::

        budget = KVCacheBudget(
            num_blocks=8169, block_size=1152, num_growing_kv_cache_groups=1,
            fixed_blocks_per_session=6, max_model_len=131072,
        )
        budget.session_blocks(5000)  # 1 * ceil(5000 / 1152) + 6 = 11
    """

    num_blocks: int
    """KV cache blocks on the generator, summed over its data-parallel replicas."""

    block_size: int
    """Tokens per block."""

    num_growing_kv_cache_groups: int
    """vLLM KV cache groups where a session holds one block per ``block_size`` tokens of
    context: full attention, and Mamba in vLLM's "all" mode."""

    fixed_blocks_per_session: int
    """Blocks a session holds in the other groups whatever its length, e.g. 2 per Mamba group in
    vLLM's "align" mode, or a sliding window."""

    max_model_len: int
    """Longest context the engine accepts, in tokens."""

    def session_blocks(self, num_tokens: int) -> int:
        """Return the blocks one session of ``num_tokens`` tokens holds."""
        return (
            self.num_growing_kv_cache_groups * math.ceil(num_tokens / self.block_size)
            + self.fixed_blocks_per_session
        )
