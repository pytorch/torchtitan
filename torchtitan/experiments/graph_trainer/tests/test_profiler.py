# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from typing import Any
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN
from torchtitan.observability.profiler import Profiler


class TestTraceAnnotationExport(TestCase):
    """Verify CUDA graph annotations reach every profiler trace the Profiler writes."""

    ANNOTATIONS = {42: [{_MODULE_FQN: "layers.0.attention.wq"}]}

    def _run_profiler(self, supports_annotations: bool) -> list[tuple[str, Any]]:
        """Drive one profile cycle, returning (path, cuda_graph_annotations) per export."""
        calls: list[tuple[str, Any]] = []

        def record_export(self_prof, path, *args, **kwargs):
            calls.append((path, kwargs.get("cuda_graph_annotations")))

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("torch.distributed.get_rank", return_value=0),
            patch(
                "torchtitan.observability.profiler.get_cuda_graph_annotations",
                return_value=self.ANNOTATIONS,
            ),
            patch(
                "torchtitan.observability.profiler._EXPORT_SUPPORTS_ANNOTATIONS",
                supports_annotations,
            ),
            patch.object(
                torch.profiler.profile,
                "export_chrome_trace",
                autospec=True,
                side_effect=record_export,
            ),
        ):
            config = Profiler.Config(
                enable_profiling=True,
                save_traces_folder="traces",
                profile_freq=4,
                profiler_warmup=1,
                profiler_active=1,
            )
            profiler = config.build(global_step=0, base_folder=tmp)

            with profiler:
                for _ in range(4):
                    profiler.step()

        self.assertEqual(len(calls), 1, f"Expected 1 export, got {calls}")
        return calls

    def test_annotations_baked_into_export(self):
        """The trace handler hands the captured annotations to the export rather than
        joining them onto the written file afterwards."""
        path, passed = self._run_profiler(supports_annotations=True)[0]
        # Profiler exports gzip-compressed traces (.json.gz) since #3483; the exporter
        # keys compression off that suffix and bakes the annotations in as it writes.
        self.assertTrue(path.endswith("rank0_trace.json.gz"))
        self.assertEqual(passed, self.ANNOTATIONS)

    def test_export_still_runs_without_annotation_support(self):
        """On a torch whose export_chrome_trace predates cuda_graph_annotations the
        trace is still written, just without them."""
        path, passed = self._run_profiler(supports_annotations=False)[0]
        self.assertTrue(path.endswith("rank0_trace.json.gz"))
        self.assertIsNone(passed)


if __name__ == "__main__":
    run_tests()
