# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

import torch
from torch.cuda._graph_annotations import _is_tools_id_unavailable
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.distributed.cudagraph import (
    cudagraph_teardown,
    get_cudagraph_annotations,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    annotate_module_fqns,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    construct_default_graph_passes,
)
from torchtitan.tools.profiler import _EXPORT_SUPPORTS_ANNOTATIONS, Profiler


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestKernelAnnotationsE2E(TestCase):
    """E2E test: trace fwd+bwd → insert annotations → cudagraph → profile → check trace."""

    def test_profiler_trace_has_module_fqn_annotations(self):
        """After the full pipeline (minimal_fx_tracer → insert_kernel_annotations
        → cudagraph → profile), the profiler trace should contain
        ``module_fqn`` fields on graphed kernel events."""
        if _is_tools_id_unavailable():
            self.skipTest("cudaGraphNodeGetToolsId not available")
        if not _EXPORT_SUPPORTS_ANNOTATIONS:
            self.skipTest("export_chrome_trace has no cuda_graph_annotations argument")

        # Simple model with annotated submodules.
        class FFN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                return torch.relu(self.linear(x))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(16)
                self.ffn = FFN()

            def forward(self, x):
                return self.ffn(self.norm(x))

        model = Model().cuda()
        annotate_module_fqns(model)

        x = torch.randn(4, 16, device="cuda")
        labels = torch.randn(4, 16, device="cuda")

        # Trace fwd + loss + bwd via minimal_fx_tracer.
        def fwd_bwd_step(inputs, labels):
            pred = model(inputs)
            loss = torch.nn.functional.mse_loss(pred, labels)
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params)
            return [loss] + list(grads)

        traced = minimal_fx_tracer(fwd_bwd_step, module=model)(x, labels)

        # Verify module_fqn metadata survived tracing.
        fqns_in_graph = set()
        for node in traced.gm.graph.nodes:
            fqn = (node.meta.get("custom") or {}).get(_MODULE_FQN)
            if fqn:
                fqns_in_graph.add(fqn)
        self.assertIn("norm", fqns_in_graph)
        self.assertIn("ffn", fqns_in_graph)

        # Apply passes (annotation + cudagraph).
        passes = construct_default_graph_passes(traced)
        traced.gm = apply_graph_passes(traced.gm, traced.example_inputs, passes)

        # Run: warmup + capture + replay.
        run_traced(traced, module=model)(x, labels)  # warmup + capture
        run_traced(traced, module=model)(x, labels)  # replay

        # Check annotations were captured.
        annotations = get_cudagraph_annotations()
        self.assertGreater(len(annotations), 0, "No annotations captured")

        all_fqns = set()
        for ann_list in annotations.values():
            for ann in ann_list:
                if isinstance(ann, dict) and _MODULE_FQN in ann:
                    all_fqns.add(ann[_MODULE_FQN])
        self.assertIn("norm", all_fqns)
        self.assertIn("ffn", all_fqns)

        # Profile and check the trace.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            run_traced(traced, module=model)(x, labels)
            torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            trace_path = f.name
        prof.export_chrome_trace(trace_path, cuda_graph_annotations=annotations)

        with open(trace_path) as f:
            trace = json.load(f)

        # Verify module_fqn fields appear on graphed kernel events.
        # Since minimal_fx_tracer traces fwd+bwd into a single graph,
        # backward kernels (e.g. layer_norm_backward) should also carry
        # annotations from _copy_fwd_metadata_to_bw_nodes.
        fqns_in_trace = set()
        for e in trace["traceEvents"]:
            args = e.get("args", {})
            if args.get("graph node id", 0) != 0 and _MODULE_FQN in args:
                fqns_in_trace.add(args[_MODULE_FQN])

        self.assertIn("norm", fqns_in_trace)
        self.assertIn("ffn", fqns_in_trace)

        # Backward kernels should also be annotated (via
        # _copy_fwd_metadata_to_bw_nodes).  Verify by checking the order:
        # forward annotations appear first (norm → ffn), then backward
        # annotations appear in reverse order (ffn → norm).
        ordered_fqns = []
        for tid in sorted(annotations.keys(), key=lambda t: t & 0xFFFFFFFF):
            for ann in annotations[tid]:
                if isinstance(ann, dict) and _MODULE_FQN in ann:
                    fqn = ann[_MODULE_FQN]
                    if not ordered_fqns or ordered_fqns[-1] != fqn:
                        ordered_fqns.append(fqn)

        # Forward: norm → ffn.linear → ffn; backward (reverse): ffn.linear
        # → norm.  All three fqns should appear, and norm/ffn.linear should
        # appear in both fwd and bwd (at least twice each).
        for expected_fqn in ("norm", "ffn", "ffn.linear"):
            self.assertIn(expected_fqn, ordered_fqns, f"Missing '{expected_fqn}'")
        for expected_fqn in ("norm", "ffn.linear"):
            positions = [i for i, f in enumerate(ordered_fqns) if f == expected_fqn]
            self.assertGreaterEqual(
                len(positions),
                2,
                f"Expected '{expected_fqn}' in both fwd and bwd, "
                f"got positions {positions} in {ordered_fqns}",
            )

        # Cleanup.
        os.unlink(trace_path)
        cudagraph_teardown()


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
                "torchtitan.tools.profiler.get_cudagraph_annotations",
                return_value=self.ANNOTATIONS,
            ),
            patch(
                "torchtitan.tools.profiler._EXPORT_SUPPORTS_ANNOTATIONS",
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
