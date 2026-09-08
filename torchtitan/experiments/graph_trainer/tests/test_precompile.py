# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import tempfile
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from torchtitan.experiments.graph_trainer.configs import EpOverlapConfig
from graph_trainer.storage import DiskStorageAdapter




@dataclass
class _StubCompileConfig:
    mode: str = "aot_fx_trace"
    backend: str = "aot_eager"
    passes: list = field(default_factory=list)
    memory_policy: str = "default"
    full_recompute_save_ops: str = ""
    ep_overlap: EpOverlapConfig = field(default_factory=EpOverlapConfig)


@dataclass
class _StubParallelDims:
    world_size: int = 8
    dp_replicate: int = 1
    dp_shard: int = 2
    cp: int = 1
    tp: int = 2
    pp: int = 2
    ep: int = 1


def _make_stub_model(params=None, buffers=None):
    """
    Build a mock model with controlled named_parameters() and
    named_buffers() for deterministic fingerprint testing.
    """
    if params is None:
        params = [
            ("layer.weight", torch.zeros(4, 4)),
            ("layer.bias", torch.zeros(4)),
        ]
    if buffers is None:
        buffers = [("running_mean", torch.zeros(4))]

    model = MagicMock()
    # Use side_effect (not return_value) so each call produces a
    # fresh iterator — just like real nn.Module methods. A single
    # return_value=iter(...) would be exhausted after the first call.
    model.named_parameters.side_effect = lambda: iter(params)
    model.named_buffers.side_effect = lambda: iter(buffers)
    return model


class TestPrecompileMain(unittest.TestCase):
    def test_validates_memory_policy_after_model_setup(self):
        from torchtitan.experiments.graph_trainer import precompile_main

        events = []
        compile_config = SimpleNamespace(mode="aot_fx_trace")
        config = SimpleNamespace(compile=compile_config)
        config_manager = MagicMock()
        config_manager.parse_args.return_value = config
        setup_result = (
            object(),
            object(),
            object(),
            compile_config,
            object(),
            object(),
            object(),
        )

        def common_setup(_config):
            events.append("setup")
            return setup_result

        def validate(actual_compile_config):
            self.assertIs(actual_compile_config, compile_config)
            self.assertEqual(events, ["setup"])
            events.append("validate")

        def precompile(*_args):
            self.assertEqual(events, ["setup", "validate"])
            events.append("precompile")

        with (
            patch.object(precompile_main, "ConfigManager", return_value=config_manager),
            patch.object(precompile_main, "_common_setup", side_effect=common_setup),
            patch.object(
                precompile_main,
                "validate_memory_policy_config",
                side_effect=validate,
            ),
            patch.object(
                precompile_main,
                "_precompile_aot_fx_trace",
                side_effect=precompile,
            ),
            patch.object(precompile_main.dist, "destroy_process_group"),
        ):
            precompile_main.main()

        self.assertEqual(events, ["setup", "validate", "precompile"])




class TestPrecompileLossSetup(unittest.TestCase):
    def test_chunked_loss_setup_matches_trainer_boundary(self):
        from torchtitan.experiments.graph_trainer.chunked_loss import (
            ChunkedLossWrapperWithParamGrads,
        )
        from torchtitan.experiments.graph_trainer.precompile_main import (
            _prepare_loss_for_precompile,
        )

        lm_head = torch.nn.Linear(2, 3)
        model = SimpleNamespace(lm_head=lm_head, _skip_lm_head=False)
        loss_fn = ChunkedLossWrapperWithParamGrads.Config().build()

        _prepare_loss_for_precompile(model, loss_fn)

        self.assertIs(loss_fn.lm_head, lm_head)
        self.assertTrue(model._skip_lm_head)


class TestPrecompiledFxTraceArtifact(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_standalone_inductor_precompile(self):
        from graph_trainer.inductor_passes import (
            standalone_inductor_compilation_pass,
        )
        from graph_trainer.make_fx_tracer import (
            minimal_fx_tracer,
            run_traced,
        )
        from graph_trainer.precompile import (
            flatten_runtime_inputs,
            precompile_fx_trace_load,
            precompile_fx_trace_save,
        )

        model = torch.nn.Sequential(
            torch.nn.RMSNorm(8),
            torch.nn.Linear(8, 4),
        ).cuda()

        def train_step(x, unused):
            loss = model(x).square().sum()
            return loss, *torch.autograd.grad(loss, tuple(model.parameters()))

        x = torch.randn(2, 8, device="cuda")
        unused = torch.randn(1, device="cuda")
        expected = train_step(x, unused)
        traced = minimal_fx_tracer(train_step, module=model)(x, unused)
        traced.gm = standalone_inductor_compilation_pass(
            traced.gm, traced.example_inputs
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskStorageAdapter(tmpdir)
            precompile_fx_trace_save(traced, storage)
            example_inputs = flatten_runtime_inputs(model, (x, unused), {})
            loaded = precompile_fx_trace_load(
                storage,
                expected_fingerprint="",
                example_inputs=example_inputs,
            )

        self.assertEqual(len(loaded.example_inputs), len(example_inputs))
        self.assertTrue(
            all(
                loaded_input is example_input
                for loaded_input, example_input in zip(
                    loaded.example_inputs, example_inputs, strict=True
                )
            )
        )

        with patch(
            "torch._inductor.standalone_compile",
            side_effect=AssertionError("precompiled graph must not compile again"),
        ):
            for _ in range(2):
                actual = run_traced(loaded, module=model)(x, unused)
                for actual_tensor, expected_tensor in zip(
                    actual, expected, strict=True
                ):
                    torch.testing.assert_close(actual_tensor, expected_tensor)





class TestCudagraphPass(unittest.TestCase):
    """Test cudagraph_pass behavior."""



    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_minimal_async_ep_custom_ops_are_wrapped_by_cudagraph_pass(self):
        """MinimalAsyncEP custom ops should not force cudagraph_pass fallback."""
        import torchtitan.distributed.minimal_async_ep  # noqa: F401
        from graph_trainer.passes import cudagraph_pass

        def cuda_i64(*shape):
            return torch.empty(*shape, device="cuda", dtype=torch.int64)

        def cuda_f32(*shape):
            return torch.empty(*shape, device="cuda", dtype=torch.float32)

        graph = torch.fx.Graph()
        op_outputs = [
            (
                torch.ops.minimal_async_ep.dispatch.default,
                (
                    cuda_f32(16, 8),
                    cuda_i64(12),
                    cuda_i64(12),
                    cuda_i64(16),
                    cuda_i64(16),
                    cuda_i64(1),
                    cuda_i64(12),
                    cuda_i64(12),
                    cuda_i64(4),
                ),
            ),
            (
                torch.ops.minimal_async_ep.combine.default,
                (cuda_f32(4, 8), cuda_f32(12, 8)),
            ),
            (
                torch.ops.minimal_async_ep.dispatch_backward.default,
                cuda_f32(4, 8),
            ),
            (
                torch.ops.minimal_async_ep.combine_backward.default,
                (cuda_f32(16, 8), cuda_f32(12)),
            ),
        ]
        nodes = []
        for target, meta_val in op_outputs:
            node = graph.call_function(target, args=())
            node.meta["val"] = meta_val
            nodes.append(node)
        graph.output(tuple(nodes))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        with patch(
            "graph_trainer.cudagraph.CUDAGraphWrapper"
        ) as MockWrapper:
            mock_instance = MagicMock()
            MockWrapper.return_value = mock_instance
            result = cudagraph_pass(gm, (), static_input_indices=[])

            self.assertIs(result, gm)
            self.assertIs(gm.forward, mock_instance)
            MockWrapper.assert_called_once()
            _, example_inputs, static_input_indices = MockWrapper.call_args.args
            self.assertEqual(example_inputs, ())
            self.assertEqual(static_input_indices, [])




if __name__ == "__main__":
    unittest.main()
