# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.deepseek_v3 import model_registry as deepseek_v3_model_registry

from torchtitan.tools import module_profiler
from torchtitan.tools.module_profiler import apply_module_profiler


class _Block(TransformerBlock):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _Block()})
        self.extra = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers["0"](x)
        return self.extra(x)


class _RaisingBlock(TransformerBlock):
    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom")


class _RaisingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _RaisingBlock()})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x)


class _PassthroughTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


class TestModuleProfiler(unittest.TestCase):
    def test_real_debug_model_modules_are_wrapped(self):
        model = deepseek_v3_model_registry("debugmodel").model.build()
        wrapped: dict[str, str] = {}

        def fake_wrap_module_forward(
            module: nn.Module,
            *,
            module_fqn: str,
            model_part_idx: int,
        ) -> bool:
            self.assertEqual(model_part_idx, 0)
            wrapped[module_fqn] = type(module).__name__
            return True

        with mock.patch(
            "torchtitan.tools.module_profiler.wrap_module_forward",
            side_effect=fake_wrap_module_forward,
        ):
            num_wrapped = apply_module_profiler([model])

        self.assertEqual(num_wrapped, len(wrapped))

        expected_wrapped_types = {
            "layers.0": "DeepSeekV3TransformerBlock",
            "layers.0.attention": "Attention",
            "layers.0.feed_forward": "FeedForward",
            "layers.1": "DeepSeekV3TransformerBlock",
            "layers.1.attention": "Attention",
            "layers.1.moe": "MoE",
            "layers.1.moe.shared_experts": "FeedForward",
        }
        self.assertEqual(
            {fqn: wrapped[fqn] for fqn in expected_wrapped_types},
            expected_wrapped_types,
        )

    def test_record_function_contexts_are_emitted_for_coarse_modules(self):
        model = _Model()
        num_wrapped = apply_module_profiler([model])

        self.assertEqual(num_wrapped, 1)

        x = torch.randn(2, 4)
        expected = model(x)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            actual = model(x)

        self.assertTrue(torch.allclose(actual, expected))
        keys = {event.key for event in prof.key_averages()}
        self.assertIn("layers.0", keys)

    def test_record_function_contexts_nest_under_torch_dispatch_mode(self):
        model = _Model()
        apply_module_profiler([model])

        # Public torch.profiler.record_function regressed here because its
        # dispatcher ops are intercepted by TorchDispatchMode and create crossing
        # PythonDispatchMode slices in the raw trace.
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.json"
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
            ) as prof:
                with _PassthroughTorchDispatchMode():
                    model(torch.randn(2, 4))
            prof.export_chrome_trace(str(trace_path))

            trace = json.loads(trace_path.read_text())

        complete_events = [
            event
            for event in trace["traceEvents"]
            if event.get("ph") == "X" and "ts" in event and "dur" in event
        ]
        expected_names = {
            "layers.0",
        }
        module_events = [
            event for event in complete_events if event.get("name") in expected_names
        ]
        self.assertTrue(module_events)
        self.assertTrue(
            any(event.get("name") == "PythonDispatchMode" for event in complete_events)
        )

        def contains(outer, inner):
            outer_start = outer["ts"]
            outer_end = outer_start + outer["dur"]
            inner_start = inner["ts"]
            inner_end = inner_start + inner["dur"]
            return outer_start <= inner_start and inner_end <= outer_end

        def crosses(first, second):
            first_start = first["ts"]
            first_end = first_start + first["dur"]
            second_start = second["ts"]
            second_end = second_start + second["dur"]
            if first_end <= second_start or second_end <= first_start:
                return False
            return not contains(first, second) and not contains(second, first)

        for module_event in module_events:
            crossing_events = [
                event
                for event in complete_events
                if event is not module_event
                and event.get("pid") == module_event.get("pid")
                and event.get("tid") == module_event.get("tid")
                and event.get("cat") != "kernel"
                and crosses(event, module_event)
            ]
            self.assertEqual(crossing_events, [], module_event["name"])

    def test_wrapping_is_idempotent(self):
        model = _Model()
        first = apply_module_profiler([model])
        second = apply_module_profiler([model])

        self.assertEqual(first, 1)
        self.assertEqual(second, 0)

    def test_mark_kernels_context_uses_same_annotation_metadata(self):
        model = _Model()
        annotations: list[dict] = []

        @contextlib.contextmanager
        def fake_mark_kernels(annotation):
            annotations.append(annotation)
            yield

        with mock.patch(
            "torchtitan.tools.module_profiler.get_mark_kernels",
            return_value=fake_mark_kernels,
        ):
            apply_module_profiler([model])
            model(torch.randn(2, 4))

        self.assertEqual(
            {
                annotation["module_fqn"]: annotation["module_type"]
                for annotation in annotations
            },
            {
                "layers.0": "_Block",
            },
        )
        self.assertTrue(
            all(annotation["model_part_idx"] == 0 for annotation in annotations)
        )
        self.assertEqual(
            {annotation["name"] for annotation in annotations},
            {
                "layers.0",
            },
        )

    def test_mark_kernels_context_closes_when_forward_raises(self):
        model = _RaisingModel()
        events: list[tuple[str, str]] = []

        @contextlib.contextmanager
        def fake_mark_kernels(annotation):
            name = annotation["name"]
            events.append(("enter", name))
            try:
                yield
            finally:
                events.append(("exit", name))

        with mock.patch(
            "torchtitan.tools.module_profiler.get_mark_kernels",
            return_value=fake_mark_kernels,
        ):
            self.assertEqual(apply_module_profiler([model]), 1)
            with self.assertRaisesRegex(RuntimeError, "boom"):
                model(torch.randn(2, 4))

        self.assertEqual(events, [("enter", "layers.0"), ("exit", "layers.0")])

    def test_mark_kernels_unavailable_warns_once(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch.cuda._graph_annotations":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        module_profiler.get_mark_kernels.cache_clear()
        try:
            with (
                mock.patch("builtins.__import__", side_effect=fake_import),
                mock.patch.object(module_profiler.logger, "warning") as warning,
            ):
                self.assertIsNone(module_profiler.get_mark_kernels())
                self.assertIsNone(module_profiler.get_mark_kernels())

            self.assertEqual(warning.call_count, 1)
        finally:
            module_profiler.get_mark_kernels.cache_clear()


if __name__ == "__main__":
    unittest.main()
