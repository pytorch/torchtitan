# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference-engine-agnostic data types shared by the generator's service, engine
loop, request dispatcher and inference engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from torchtitan.rl.rollout.types import KeptSets, Prompt


@dataclass(kw_only=True, slots=True)
class SamplingConfig:
    """Sampling parameters for one generation request."""

    temperature: float = 0.8
    """Sampling temperature. 0.0 = greedy, higher = more random."""

    top_p: float = 1.0
    """Nucleus sampling threshold: sample from the most likely tokens that hold ``top_p`` of the
    probability. Below 1.0 it needs ``top_k``, which bounds the kept set."""

    top_k: int | None = None
    """Sample from the ``top_k`` most likely tokens (with ``top_p``, from those of them that hold
    ``top_p`` of their probability); None keeps every token.

    When ``top_k`` or ``top_p`` filters the vocabulary, the generator returns each sampled token's
    kept set (``FinishedRequest.kept_sets``) and the trainer replays it: it renormalizes its
    logprobs over the same tokens, so it trains the policy the generator sampled from."""

    max_tokens: int = 100
    """Maximum number of tokens to generate per completion."""

    num_topk_logprobs: int = 0
    """Per generated token, also return the generator's k most likely token ids and their
    logprobs; 0 returns only the sampled token's logprob. Filled by the controller from the
    policy loss's ``ScoreCentering.Config`` on training requests; leave 0 in recipes."""

    seed: int | None = None
    """Per-request RNG seed. The rollouter offsets this per sample so a group's
    n=1 requests stay diverse while remaining reproducible (None = nondeterministic)."""

    stop_token_ids: list[int] | None = None
    """Renderer role-boundary stop tokens; filled by the controller. Required at
    generation time: these are the only ids that end a request (the engine's EOS stops
    are off)."""

    def __post_init__(self) -> None:
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be None or >= 1, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_p < 1.0 and self.top_k is None:
            raise ValueError(
                f"top_p < 1 (got {self.top_p}) needs top_k: the generator sends each sampled "
                "token's kept set to the trainer, and top_k bounds its size."
            )

    @property
    def filters_vocab(self) -> bool:
        """Whether ``top_k`` or ``top_p`` samples from fewer tokens than the full vocabulary."""
        return self.top_k is not None or self.top_p < 1.0


@dataclass(kw_only=True, slots=True)
class EngineRequest:
    """One `generate` call's request for the engine, awaiting admission; what a `LoopDecision` broadcasts."""

    request_id: str
    prompt: Prompt
    """The prompt's tokens and images, with the image files' bytes (not pixels)."""
    sampling: SamplingConfig
    group_id: int
    """Rollout group id. Sets the vLLM priority: lower is admitted first and preempted last.
    Training ids grow as groups start. Validation ids count down (-1, -2, ...),
    so all validation gets priority -1: it runs in arrival order (not newest first), before training.

        waiting, in arrival order:  g7 turn 1   g9 turn 1   g4 turn 2   val -1      val -2
        priority:                   7           9           4           -1          -1
        vLLM admits:                val -1      val -2      g4 turn 2   g7 turn 1   g9 turn 1
                                                            ^ g4 started before g7
    """
    routing_session_id: str
    arrival_time: float
    """Rank 0's `time.time()` in `generate`; vLLM breaks priority ties by it and measures TTFT from it.
    All TP ranks pass this one value, so they order ties the same way even if their clocks differ."""
    min_policy_version: int = field(init=False)
    """Oldest policy version this request's KV can come from; rank 0 sets it at admission.
    Without a KV reset on model state dict load it is the group's pinned version and salts the
    prefix cache."""


@dataclass(kw_only=True, slots=True)
class RequestMetricsInputs:
    """Raw inputs needed to build a request's metrics. Used to pass
    metric related information when fan-in from DPs to rank 0.
    """

    num_cached_tokens: int | None
    has_stats: bool
    queued_ts: float = 0.0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0
    first_token_latency: float = 0.0
    num_generation_tokens: int = 0
    num_preemptions: int = 0


@dataclass(kw_only=True, slots=True)
class FinishedRequest:
    """A request the engine finished: its one sample and the raw inputs of its metrics."""

    request_id: str
    token_ids: list[int]  # [generated_tokens]
    token_logprobs: list[float]  # [generated_tokens], of each sampled token
    """With vocabulary filtering, renormalized over the token's kept set: the distribution it was
    sampled from."""
    finish_reason: str | None
    metrics_inputs: RequestMetricsInputs
    topk_token_ids: np.ndarray | None = None
    """[generated_tokens, k] int32; None unless `SamplingConfig.num_topk_logprobs` > 0."""
    topk_logprobs: np.ndarray | None = None
    """[generated_tokens, k] float32 logprobs of `topk_token_ids`."""
    kept_sets: KeptSets | None = None
    """Each generated token's kept set; None unless `SamplingConfig.filters_vocab`."""
