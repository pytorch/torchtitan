# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One bounded startup capture, owned by the training entry point."""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .capture import MemoryCapture

if TYPE_CHECKING:
    from torchtitan.observability.profiler import CudaMemoryProfilerConfig

logger = logging.getLogger(__name__)


class CudaMemorySession:
    def __init__(self, config: "CudaMemoryProfilerConfig", trace_folder: str):
        self.config = config
        self.destination = (
            Path(trace_folder)
            / "cuda_memory"
            / f"rank{int(os.environ.get('RANK', '0'))}"
        )
        self.capture: MemoryCapture | None = MemoryCapture(
            stacks=config.stacks, max_records=config.max_records
        )
        self.completed_steps = 0
        self.closed = False

    def start(self) -> None:
        # Refuse to replace evidence from an earlier run in the same output folder.
        self.destination.mkdir(parents=True, exist_ok=False)
        assert self.capture is not None
        try:
            self.capture.start()
        except BaseException as error:
            try:
                self.destination.rmdir()
            except OSError as cleanup:
                error.add_note(f"Could not remove empty capture directory: {cleanup!r}")
            raise

    def step(self) -> None:
        if self.closed:
            return
        self.completed_steps += 1
        if self.completed_steps >= self.config.capture_steps:
            self.close(export=True)

    def close(self, *, export: bool) -> None:
        if self.closed:
            return
        self.closed = True
        # Do not retain native stack strings for the rest of a long training run.
        capture, self.capture = self.capture, None
        assert capture is not None
        capture.stop()
        if export:
            capture.coverage.update(
                window="startup",
                completed_steps=self.completed_steps,
                requested_steps=self.config.capture_steps,
                stop_reason="step_limit"
                if self.completed_steps >= self.config.capture_steps
                else "scope_exit",
            )
            capture.export(self.destination / "memory.json.gz")
            capture.export_chrome_trace(self.destination / "memory-trace.json.gz")
            logger.info("CUDA memory capture saved to %s", self.destination)
