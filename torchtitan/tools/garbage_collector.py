# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging
import time
from dataclasses import dataclass

import torch

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl


logger = logging.getLogger(__name__)


# used to avoid stragglers in garbage collection
class GarbageCollector(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        gc_freq: int = 50
        """Python garbage control scheduling interval, in steps"""

        debug: bool = False
        """
        Enable GC debugging mode. This will perform gc.collect() at every step to
        detect if there is a reference cycle that includes a CUDA Tensor.
        Note that you may want to lower the training steps to avoid generating too
        many temporary files.
        """

        def __post_init__(self) -> None:
            if self.gc_freq <= 0:
                raise ValueError(
                    "garbage_collector.gc_freq must be a positive integer, "
                    f"got {self.gc_freq}"
                )

    def __init__(self, config: Config) -> None:
        self.gc_freq = config.gc_freq
        self.debug = config.debug
        gc.disable()
        self.collect("Initial GC collection")
        if self.debug:
            from torch.utils.viz._cycles import warn_tensor_cycles

            if torch.distributed.get_rank() == 0:
                warn_tensor_cycles()

    @sl.log_trace_span("gc_collect")
    def run(self, step_count: int) -> bool:
        """Run a GC cycle if this step should collect. Returns True when a
        collection actually ran, False otherwise."""
        if self.debug:
            self.collect(
                "Force GC to perform collection to obtain debug information",
                generation=2,
            )
            sl.add_step_tag("gc")
            return True
        if step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Performing periodic GC collection")
            sl.add_step_tag("gc")
            return True
        return False

    @staticmethod
    def collect(reason: str, generation: int = 1):
        begin = time.monotonic()
        gc.collect(generation)
        logger.info("[GC] %s took %.2f seconds", reason, time.monotonic() - begin)
