# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference-engine-agnostic data types shared by the generator's service, engine
loop, request dispatcher and inference engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(kw_only=True, slots=True)
class SamplingConfig:
    """Sampling parameters for one generation request."""

    temperature: float = 0.8
    """Sampling temperature. 0.0 = greedy, higher = more random."""

    top_p: float = 1.0
    """Nucleus sampling threshold. Must be 1.0: the trainer scores tokens over the full vocabulary,
    so sampling from a truncated nucleus would bias the gradient."""

    max_tokens: int = 100
    """Maximum number of tokens to generate per completion."""

    seed: int | None = None
    """Per-request RNG seed. The rollouter offsets this per sample so a group's
    n=1 requests stay diverse while remaining reproducible (None = nondeterministic)."""

    stop_token_ids: list[int] | None = None
    """Renderer role-boundary stop tokens; filled by the controller. Required at
    generation time: these are the only ids that end a request (the engine's EOS stops
    are off)."""

    def __post_init__(self) -> None:
        # TODO(mask-replay): to allow top_p < 1, turn on vLLM's `return_sampling_mask` (needs a vLLM
        # upgrade, the V2 model runner, top_k > 0 and processed logprobs), carry each token's kept
        # ids next to generator_logprobs, and take the trainer's logsumexp of logits / T over them.
        if self.top_p < 1.0:
            raise ValueError(
                f"top_p must be 1.0, got {self.top_p}: the trainer computes logprobs "
                "over the full vocabulary."
            )


@dataclass(kw_only=True, slots=True)
class EngineRequest:
    """One `generate` call's request for the engine, awaiting admission; what a `LoopDecision` broadcasts."""

    request_id: str
    prompt_token_ids: list[int]  # [prompt_tokens]
    sampling: SamplingConfig
    group_id: int
    """Rollout group id. Sets the vLLM priority: lower is admitted first and preempted last.
    Training ids grow as groups start. Validation ids count down (-1, -2, ...) and restart every run,
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
    finish_reason: str | None
    metrics_inputs: RequestMetricsInputs
