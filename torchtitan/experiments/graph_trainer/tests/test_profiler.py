# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.cuda._annotate_cuda_graph_trace import (  # pyrefly: ignore[missing-import]
    annotate_trace,
)
from torch.cuda._graph_annotations import _is_tools_id_unavailable
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.config.function import Function
from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    annotate_module_fqns,
)
from torchtitan.experiments.graph_trainer.cudagraph import (
    cudagraph_teardown,
    get_cudagraph_annotations,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    run_traced_train_step,
    trace_train_step,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    construct_default_graph_passes,
)
from torchtitan.tools.profiler import Profiler


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestKernelAnnotationsE2E(TestCase):
    """E2E test: trace fwd+bwd → insert annotations → cudagraph → profile → check trace."""

    def test_profiler_trace_has_module_fqn_annotations(self):
        """After the full pipeline (trace_train_step → insert_kernel_annotations
        → cudagraph → profile), the profiler trace should contain
        ``module_fqn`` fields on graphed kernel events."""
        if _is_tools_id_unavailable():
            self.skipTest("cudaGraphNodeGetToolsId not available")

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

        # Trace fwd + loss + bwd via trace_train_step.
        def fwd_bwd_step(model, inputs, labels):
            pred = model(inputs)
            loss = torch.nn.functional.mse_loss(pred, labels)
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params)
            return [loss] + list(grads)

        traced = trace_train_step(fwd_bwd_step)(model, x, labels)

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
        run_traced_train_step(traced, model, x, labels)  # warmup + capture
        run_traced_train_step(traced, model, x, labels)  # replay

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
            run_traced_train_step(traced, model, x, labels)
            torch.cuda.synchronize()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            trace_path = f.name
        prof.export_chrome_trace(trace_path)

        with open(trace_path) as f:
            trace = json.load(f)

        count = annotate_trace(trace, annotations)
        self.assertGreater(count, 0, "annotate_trace matched 0 events")

        # Verify module_fqn fields appear on graphed kernel events.
        # Since trace_train_step traces fwd+bwd into a single graph,
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


class TestTracePostProcessorConfig(TestCase):
    """Verify Profiler.Config.trace_post_processor runs on every trace export."""

    def test_post_processor_called_with_trace_path(self):
        """Function.Config in trace_post_processor is invoked with the
        exported trace file path."""

        calls: list[tuple[str, bool]] = []

        def record_call(trace_path: str) -> None:
            calls.append((trace_path, os.path.exists(trace_path)))

        with tempfile.TemporaryDirectory() as tmp, patch(
            "torch.distributed.get_rank", return_value=0
        ):
            config = Profiler.Config(
                enable_profiling=True,
                save_traces_folder="traces",
                profile_freq=4,
                profiler_warmup=1,
                profiler_active=1,
                trace_post_processor=Function.Config(fn=record_call),
            )
            profiler = config.build(global_step=0, base_folder=tmp)

            with profiler:
                for _ in range(4):
                    profiler.step()

        self.assertEqual(len(calls), 1, f"Expected 1 call, got {calls}")
        path, existed = calls[0]
        self.assertTrue(path.endswith("rank0_trace.json"))
        self.assertTrue(existed, f"Trace file {path} did not exist when callback ran")


if __name__ == "__main__":
    run_tests()
