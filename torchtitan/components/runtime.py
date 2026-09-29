# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lifecycle contract for optional training runtimes."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, ExitStack, nullcontext
from dataclasses import dataclass

from torch.distributed.pipelining import PipelineStageInfo

from torchtitan.config import Configurable


class TrainingRuntime(Configurable):
    """Optional runtime state shared across model modules on one rank.

    Training engines construct configured runtimes after model parallelization
    and pipeline schedule creation, initialize them after model state is
    materialized, and close them in reverse construction order. A runtime may
    also provide a context around each pipeline-stage forward.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configuration owned by a :class:`TrainingRuntime` implementation."""

        def validate(self, training_config: Configurable.Config) -> None:
            """Validate this runtime against its containing training config."""
            del training_config

    def initialize(self) -> None:
        """Materialize resources that require initialized model state."""

    def forward_context(self, info: PipelineStageInfo) -> AbstractContextManager[None]:
        """Return a context entered around one pipeline-stage forward."""
        del info
        return nullcontext()

    def forward_context_key(self, info: PipelineStageInfo) -> object | None:
        """Return replay-stable state that specializes a traced stage forward.

        GraphPP may reuse one stage graph for invocations with equal keys. A
        runtime that binds different graph-visible state across invocations
        returns a hashable key describing that state. ``None`` means the
        runtime does not require graph specialization.
        """
        del info
        return None

    def close(self) -> None:
        """Release runtime-owned resources; repeated calls must be safe."""


class _TrainingRuntimeForwardContext:
    """Compose the pipeline contexts exposed by configured runtimes."""

    def __init__(self, runtimes: Sequence[TrainingRuntime]) -> None:
        self._runtimes = tuple(runtimes)

    @contextmanager
    def __call__(self, info: PipelineStageInfo) -> Iterator[None]:
        with ExitStack() as stack:
            for runtime in self._runtimes:
                stack.enter_context(runtime.forward_context(info))
            yield

    def graph_cache_key(self, info: PipelineStageInfo) -> tuple[object, ...] | None:
        """Return the runtime state that must remain fixed in one graph."""
        keys = tuple(runtime.forward_context_key(info) for runtime in self._runtimes)
        if all(key is None for key in keys):
            return None
        try:
            hash(keys)
        except TypeError as error:
            raise TypeError(
                "training runtime forward-context keys must be hashable"
            ) from error
        return keys
