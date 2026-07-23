# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import unittest
from unittest import mock

import torch
from torch import nn

from torchtitan.tools import module_profiler
from torchtitan.tools.module_profiler import apply_module_profiler


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Linear(4, 4)
        self.feed_forward = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        return x + self.feed_forward(x)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _Block()})
        self.extra = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers["0"](x)
        return self.extra(x)


class TestModuleProfiler(unittest.TestCase):
    def test_record_function_contexts_are_emitted_for_coarse_modules(self):
        model = _Model()
        num_wrapped = apply_module_profiler([model])

        self.assertEqual(num_wrapped, 3)

        x = torch.randn(2, 4)
        expected = model(x)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            actual = model(x)

        self.assertTrue(torch.allclose(actual, expected))
        keys = {event.key for event in prof.key_averages()}
        self.assertIn("block::layers.0", keys)
        self.assertIn("attention::layers.0.attention", keys)
        self.assertIn("ffn::layers.0.feed_forward", keys)

    def test_wrapping_is_idempotent(self):
        model = _Model()
        first = apply_module_profiler([model])
        second = apply_module_profiler([model])

        self.assertEqual(first, 3)
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
                (annotation["module_kind"], annotation["module_fqn"])
                for annotation in annotations
            },
            {
                ("block", "layers.0"),
                ("attention", "layers.0.attention"),
                ("ffn", "layers.0.feed_forward"),
            },
        )
        self.assertTrue(
            all(annotation["model_part_idx"] == 0 for annotation in annotations)
        )
        self.assertEqual(
            {annotation["name"] for annotation in annotations},
            {
                "block::layers.0",
                "attention::layers.0.attention",
                "ffn::layers.0.feed_forward",
            },
        )

    def test_mark_kernels_unavailable_warns_once(self):
        original_warned = module_profiler._mark_kernels_warned
        module_profiler._mark_kernels_warned = False
        try:
            with mock.patch.object(module_profiler.logger, "warning") as warning:
                module_profiler.warn_mark_kernels_unavailable(ImportError("missing"))
                module_profiler.warn_mark_kernels_unavailable(ImportError("missing"))

            self.assertEqual(warning.call_count, 1)
        finally:
            module_profiler._mark_kernels_warned = original_warned


if __name__ == "__main__":
    unittest.main()
