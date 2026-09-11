# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.runner import GraphRunner


class TestGraphRunner(unittest.TestCase):
    def test_observes_in_place_updates_without_resampling_state(self):
        class CountingLinear(nn.Linear):
            def __init__(self):
                super().__init__(3, 2, dtype=torch.float64)
                self.named_parameters_calls = 0

            def named_parameters(self, *args, **kwargs):
                self.named_parameters_calls += 1
                return super().named_parameters(*args, **kwargs)

        torch.manual_seed(42)
        model = CountingLinear()
        inputs = torch.randn(4, 3, dtype=torch.float64)

        def forward(value):
            return model(value)

        traced = minimal_fx_tracer(forward, module=model)(inputs)
        runner = GraphRunner(traced, module=model)
        calls_after_binding = model.named_parameters_calls
        initial_output = runner(inputs)

        with torch.no_grad():
            model.weight.add_(0.25)
            model.bias.sub_(0.5)

        expected = model(inputs)
        actual = runner(inputs)

        self.assertFalse(torch.equal(initial_output, expected))
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(calls_after_binding, model.named_parameters_calls)

    def test_rejects_incompatible_module_at_bind_time(self):
        model = nn.Linear(3, 2)
        inputs = torch.randn(4, 3)

        def forward(value):
            return model(value)

        traced = minimal_fx_tracer(forward, module=model)(inputs)
        incompatible_model = nn.Sequential(nn.Linear(3, 2))

        with self.assertRaisesRegex(ValueError, "parameter/buffer names"):
            GraphRunner(traced, module=incompatible_model)

    def test_validation_rejects_parameter_and_storage_replacement(self):
        model = nn.Linear(3, 2)
        inputs = torch.randn(4, 3)

        def forward(value):
            return model(value)

        traced = minimal_fx_tracer(forward, module=model)(inputs)
        runner = GraphRunner(traced, module=model)
        original_weight = model.weight
        model.weight = nn.Parameter(model.weight.detach().clone())
        with self.assertRaisesRegex(RuntimeError, "state objects changed"):
            runner.validate_state()

        model.weight = original_weight
        runner = GraphRunner(traced, module=model)
        with torch.no_grad():
            model.weight.set_(model.weight.detach().clone())
        with self.assertRaisesRegex(RuntimeError, "state storage changed"):
            runner.validate_state()

    def test_validation_allows_in_place_buffer_updates(self):
        class BufferedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.ones(3))

            def forward(self, value):
                return value + self.offset

        model = BufferedModule()
        inputs = torch.randn(4, 3)
        traced = minimal_fx_tracer(model.forward, module=model)(inputs)
        runner = GraphRunner(traced, module=model)

        model.offset.add_(2)
        runner.validate_state()
        self.assertTrue(torch.equal(model(inputs), runner(inputs)))

        model.offset = model.offset.clone()
        with self.assertRaisesRegex(RuntimeError, "state objects changed"):
            runner.validate_state()

    def test_validation_rejects_graph_state_replacement(self):
        graph_state = {"accumulator": torch.zeros(3)}
        inputs = torch.randn(4, 3)
        traced = minimal_fx_tracer(
            lambda _state, value: value.sin(),
            graph_state=graph_state,
        )(inputs)
        runner = GraphRunner(traced, graph_state=graph_state)
        graph_state["accumulator"] = torch.ones(3)

        with self.assertRaisesRegex(RuntimeError, "state objects changed"):
            runner.validate_state()

    def test_rejects_traced_optimizer_state(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        inputs = torch.randn(4, 3)
        model(inputs).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)

        def forward(value):
            return model(value)

        traced = minimal_fx_tracer(
            forward,
            module=model,
            optimizer=optimizer,
        )(inputs)

        with self.assertRaisesRegex(ValueError, "optimizer state"):
            GraphRunner(traced, module=model)

    def test_rejects_runtime_mesh_count_mismatch(self):
        traced = minimal_fx_tracer(lambda value: value.sin())(torch.ones(3))
        traced.num_runtime_mesh_inputs = 1

        with self.assertRaisesRegex(ValueError, "runtime meshes"):
            GraphRunner(traced)

    def test_dtensor_state_with_runtime_mesh(self):
        import torch.distributed as dist
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor, Replicate

        dist.init_process_group(
            "gloo",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )
        try:
            mesh = DeviceMesh("cpu", [0])
            model = nn.Linear(3, 2)
            model.weight = nn.Parameter(
                DTensor.from_local(
                    model.weight.detach(), mesh, [Replicate()], run_check=False
                )
            )
            model.bias = nn.Parameter(
                DTensor.from_local(
                    model.bias.detach(), mesh, [Replicate()], run_check=False
                )
            )
            inputs = DTensor.from_local(
                torch.randn(4, 3), mesh, [Replicate()], run_check=False
            )

            def forward(value):
                return model(value)

            traced = minimal_fx_tracer(
                forward,
                module=model,
                precompile_meshes=[mesh],
            )(inputs)
            runner = GraphRunner(traced, module=model, runtime_meshes=[mesh])

            expected = model(inputs)
            actual = runner(inputs)

            self.assertIsInstance(actual, DTensor)
            self.assertTrue(torch.equal(expected.to_local(), actual.to_local()))
        finally:
            dist.destroy_process_group()

    def test_optional_user_input_validation(self):
        traced = minimal_fx_tracer(lambda value: value.sin())(torch.ones(3))
        runner = GraphRunner(traced, validate_user_inputs=True)

        with self.assertRaisesRegex(ValueError, "input spec mismatch"):
            runner(value=torch.ones(3))


if __name__ == "__main__":
    unittest.main()
