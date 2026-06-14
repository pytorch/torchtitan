# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, List

from torchtitan.components.metrics import MetricsProcessor
from torchtitan.tools.logging import logger


@dataclass(frozen=True)
class ReportSpec:
    output_names: tuple[str, ...]
    output_types: tuple[str, ...]
    func: Callable[..., tuple[Any, ...]]
    steps: List[int]
    arguments: list[Any] = field(default_factory=list)


def run_and_save_report(
    report_spec: ReportSpec,
    step: int,
    metrics_processor: MetricsProcessor,
) -> None:
    output_names = report_spec.output_names
    output_types = report_spec.output_types
    try:
        results = report_spec.func(*report_spec.arguments)
    except Exception:
        results = ('<plaintext>' + traceback.format_exc(),) * len(output_names)
        output_types = ('html',) * len(output_names)

    assert len(results) == len(output_types) == len(output_names)
    for result, output_name, output_type in zip(
        results, output_names, output_types, strict=True
    ):
        metrics_processor.write_report(result, step, output_name, output_type)


class ReportRunner:
    def __init__(
        self,
        *,
        metrics_processor: MetricsProcessor,
        enabled: bool,
        thread_name_prefix: str = "sample-report",
    ) -> None:

        self.metrics_processor = metrics_processor
        self.enabled = enabled
        self._report_executor = ThreadPoolExecutor(
            thread_name_prefix=thread_name_prefix,
        )
        self._report_futures: list[Future[None]] = []

    def submit_due(self, *, step: int, report_specs: Mapping[str, ReportSpec]) -> None:
        self._prune_futures()
        if not self.enabled:
            return

        for report_name, report_spec in report_specs.items():
            if step not in report_spec.steps:
                continue
            future = self._report_executor.submit(
                run_and_save_report,
                report_specs[report_name],
                step,
                self.metrics_processor,
            )
            self._report_futures.append(future)

    def close(self) -> None:
        try:
            for future in self._report_futures:
                self._log_future_error(future, wait=True)
            self._report_futures.clear()
        finally:
            self._report_executor.shutdown(wait=True)

    def _prune_futures(self) -> None:
        pending = []
        for future in self._report_futures:
            if future.done():
                self._log_future_error(future)
            else:
                pending.append(future)
        self._report_futures = pending

    @staticmethod
    def _log_future_error(future: Future[None], wait: bool = False) -> None:
        try:
            future.result() if wait else future.result(timeout=0)
        except Exception:
            logger.exception("Report future failed")
