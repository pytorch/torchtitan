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

import torch
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

    def graph_forward_inputs(
        self,
    ) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
        """Return static GraphPP inputs indexed by ``(stage, microbatch)``."""
        return {}

    def close(self) -> None:
        """Release runtime-owned resources; repeated calls must be safe."""


@contextmanager
def _training_runtime_forward_context(
    runtimes: Sequence[TrainingRuntime],
    info: PipelineStageInfo,
) -> Iterator[None]:
    """Enter every configured runtime's eager pipeline context."""
    with ExitStack() as stack:
        for runtime in runtimes:
            stack.enter_context(runtime.forward_context(info))
        yield
