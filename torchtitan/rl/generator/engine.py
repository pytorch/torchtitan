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

# Released groups an engine remembers, so that a late request of one starts no session. A late
# request arrives seconds after its group's release; 4,096 groups span far longer.
NUM_RELEASED_GROUP_IDS_KEPT = 4096


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
            all KV (and vLLM's encoder cache for image prompts). When False, a request may
            reuse only KV and image embeddings salted with its ``min_policy_version``.
        model_state_dict_load_mode: ``"cpu_prefetch"`` allocates the pinned CPU buffers of
            ``prefetched_model_state_dict``; ``"gpu_fetch"`` does not.
        vision_processor: Preprocesses image prompts' files (`PromptImage.data`). None:
            text only; requests with images are rejected.
        mooncake_master_address: ``host:port`` of the Mooncake Store master the
            controller started when ``config.uses_mooncake_store``, else None.
        rank: Global rank of this process.
        generator_name: Name of the generator, unique within a run and shared by its
            ranks; names engine metrics and per-generator connector resources.
        output_dir: Output directory, for engine metrics.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        @property
        def uses_mooncake_store(self) -> bool:
            """Whether the generators share a Mooncake Store. The controller then
            starts its master and builds the engines with its address."""
            return False

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
        """Whether requests or their remaining engine work need more steps.

        All ranks must call this together before stepping: implementations may
        synchronize local work so every rank makes the same continue/idle decision.
        """

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
        """Release the cached KV of ``session_ids`` and of every session of ``group_ids``."""

    def log_stats(self, step: int) -> None:
        """Log the engine's own per-step stats, if any."""

    def kv_cache_capacity(self) -> KVCacheCapacity:
        """Return the KV cache size and what one session takes in it, for the router's
        admission (``InterGeneratorRouter.Config.admission``). It reads only what the build
        fixed, so the actor's thread may call it."""
        raise NotImplementedError(
            f"{type(self).__name__} does not report its KV cache capacity, which admission needs"
        )

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release engine-local resources; process teardown is the actor framework's."""


@dataclass(frozen=True, kw_only=True, slots=True)
class KVCacheCapacity:
    """A generator's KV cache size, and the blocks one session takes in it.

    A session takes a block per that many tokens in each attention KV cache group, plus a fixed
    number of GDN (linear attention) states, more while a call runs:

        capacity = KVCacheCapacity(
            num_blocks=119, attention_block_sizes=(16,), blocks_per_state=3,
            num_idle_states=1, num_running_states=3, num_state_slots=None,
        )
        capacity.session_blocks(40, running=True)   # ceil(40 / 16) + 3 x 3 = 12
        capacity.session_blocks(50, running=False)  # ceil(50 / 16) + 3 x 1 = 7
    """

    num_blocks: int
    """KV cache blocks admission may fill."""

    attention_block_sizes: tuple[int, ...]
    """Tokens per block of each KV cache group that takes a block per that many tokens."""

    blocks_per_state: int
    """Blocks one GDN state takes, over every GDN KV cache group; 0 without GDN layers."""

    num_idle_states: int
    """GDN states a session keeps between its calls."""

    num_running_states: int
    """GDN states a session takes while a call of it runs."""

    num_state_slots: int | None
    """A cap on GDN states apart from blocks (SGLang's state pool), or None."""

    def session_states(self, *, running: bool) -> int:
        """GDN states a session takes: while a call runs, or between calls."""
        return self.num_running_states if running else self.num_idle_states

    def session_blocks(self, num_tokens: int, *, running: bool) -> int:
        """Blocks a session of ``num_tokens`` tokens takes: while a call runs, or between calls."""
        return self.blocks_per_state * self.session_states(running=running) + sum(
            math.ceil(num_tokens / block_size)
            for block_size in self.attention_block_sizes
        )
