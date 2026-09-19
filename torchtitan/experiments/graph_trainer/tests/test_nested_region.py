# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bitwise correctness tests for aot_nested_region."""

import unittest

import torch
import torch.nn as nn
from torch._higher_order_ops.invoke_subgraph import invoke_subgraph

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    run_traced_module,
    trace_module,
)
from torchtitan.experiments.graph_trainer.nested_region import (
    _subgraph_cache,
    aot_nested_region,
)


def _get_params_and_buffers(mod):
    return {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }


def create_model(config_cls, model_config, device="cuda", dtype=torch.float32):
    model = config_cls(model_config)
    model.to(device=device, dtype=dtype)
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device(device))
    return model


CONST_HASH = lambda *args: "block"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestNestedRegion(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32
    BATCH_SIZE = 2
    SEQ_LEN = 128

    def setUp(self):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        _subgraph_cache.clear()

    def tearDown(self):
        torch.use_deterministic_algorithms(False)
        _subgraph_cache.clear()

    def _assert_invoke_subgraph_count(self, traced_result, expected):
        invoke_count = sum(
            1
            for n in traced_result.gm.graph.nodes
            if n.op == "call_function" and "invoke_subgraph" in str(n.target)
        )
        self.assertEqual(invoke_count, expected)

    def _run_forward_bitwise_test(self, model_ref, model_test, fwd_args):
        """Compare eager forward vs traced-with-nested-region forward."""
        traced_result = trace_module(model_test, fwd_args)

        num_layers = len(list(model_ref.layers.children()))
        self._assert_invoke_subgraph_count(traced_result, num_layers)

        self.assertEqual(len(_subgraph_cache), 1)

        out_eager = model_ref(*fwd_args)
        params_and_buffers = _get_params_and_buffers(model_test)
        out_traced = run_traced_module(traced_result, params_and_buffers, fwd_args)
        self.assertTrue(torch.equal(out_eager, out_traced[0]))

    def _make_model_pair(self, config_cls, config):
        model_ref = create_model(config_cls, config, self.DEVICE, self.DTYPE)
        model_test = create_model(config_cls, config, self.DEVICE, self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())
        tokens = torch.randint(
            0, config.vocab_size, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        return model_ref, model_test, tokens

    def test_class_decorator(self):
        # Expected outer graph (4-layer MLP, dim=64):
        #
        # graph():
        #     %arg0_1 .. %arg15_1 = placeholder  (4 params × 4 layers)
        #     %arg16_1 = placeholder              (input x)
        #     %repeated_subgraph0 = get_attr[target=repeated_subgraph0]
        #     %invoke_subgraph = invoke_subgraph(%repeated_subgraph0, block, %arg0_1, %arg1_1, %arg2_1, %arg3_1, %arg16_1)
        #     %getitem = invoke_subgraph[0]
        #     %invoke_subgraph_1 = invoke_subgraph(%repeated_subgraph0, block, %arg4_1, %arg5_1, %arg6_1, %arg7_1, %getitem)
        #     %getitem_1 = invoke_subgraph_1[0]
        #     %invoke_subgraph_2 = invoke_subgraph(%repeated_subgraph0, block, %arg8_1, %arg9_1, %arg10_1, %arg11_1, %getitem_1)
        #     %getitem_2 = invoke_subgraph_2[0]
        #     %invoke_subgraph_3 = invoke_subgraph(%repeated_subgraph0, block, %arg12_1, %arg13_1, %arg14_1, %arg15_1, %getitem_2)
        #     %getitem_3 = invoke_subgraph_3[0]
        #     return [getitem_3]
        #
        # repeated_subgraph0:
        #     %arg0_1 .. %arg3_1 = placeholder  (fc1.weight, fc1.bias, fc2.weight, fc2.bias)
        #     %arg4_1 = placeholder              (x)
        #     %view = aten.view(%arg4_1, [16, 64])
        #     %t = aten.t(%arg0_1)
        #     %addmm = aten.addmm(%arg1_1, %view, %t)
        #     %view_1 = aten.view(%addmm, [2, 8, 128])
        #     %relu = aten.relu(%view_1)
        #     %detach = aten.detach(%relu)
        #     %view_2 = aten.view(%relu, [16, 128])
        #     %t_1 = aten.t(%arg2_1)
        #     %addmm_1 = aten.addmm(%arg3_1, %view_2, %t_1)
        #     %view_3 = aten.view(%addmm_1, [2, 8, 64])
        #     %add = aten.add(%arg4_1, %view_3)
        #     return (add,)

        @aot_nested_region(hash_fn=CONST_HASH)
        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc1 = nn.Linear(dim, dim * 2)
                self.fc2 = nn.Linear(dim * 2, dim)

            def forward(self, x):
                return x + self.fc2(torch.relu(self.fc1(x)))

        class Model(nn.Module):
            def __init__(self, dim, n_layers):
                super().__init__()
                self.layers = nn.ModuleDict(
                    {str(i): Block(dim) for i in range(n_layers)}
                )

            def forward(self, x):
                for layer in self.layers.values():
                    x = layer(x)
                return x

        dim, n_layers = 64, 4
        model = Model(dim, n_layers).to(device=self.DEVICE)
        x = torch.randn(2, 8, dim, device=self.DEVICE)

        out_eager = model(x)
        traced_result = trace_module(model, (x,))
        gm = traced_result.gm

        # Outer graph: only invoke_subgraph calls, no inlined aten ops
        invoke_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is invoke_subgraph
        ]
        self.assertEqual(len(invoke_nodes), n_layers)

        aten_ops_in_outer = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function"
            and "aten" in str(n.target)
            and "empty_strided" not in str(n.target)
        ]
        self.assertEqual(aten_ops_in_outer, [])

        # All invoke_subgraph nodes reference the same subgraph
        subgraph_targets = {n.args[0].target for n in invoke_nodes}
        self.assertEqual(len(subgraph_targets), 1)

        # invoke_subgraph calls are chained: output of N feeds input of N+1
        for i in range(1, len(invoke_nodes)):
            prev_out = invoke_nodes[i].args[-1]
            self.assertIn("getitem", prev_out.name)

        # Single cache entry
        self.assertEqual(len(_subgraph_cache), 1)

        # Bitwise correctness
        params_and_buffers = _get_params_and_buffers(model)
        out_traced = run_traced_module(traced_result, params_and_buffers, (x,))
        self.assertTrue(torch.equal(out_eager, out_traced[0]))

    def test_llama3_forward(self):
        from torchtitan.models.llama3 import llama3_configs, Llama3Model

        model_ref, model_test, tokens = self._make_model_pair(
            Llama3Model, llama3_configs["debugmodel"]
        )
        for layer in model_test.layers.values():
            aot_nested_region(layer, hash_fn=CONST_HASH)
        self._run_forward_bitwise_test(model_ref, model_test, (tokens,))

    def test_qwen3_forward(self):
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.model import Qwen3Model

        model_ref, model_test, tokens = self._make_model_pair(
            Qwen3Model, qwen3_configs["debugmodel"]
        )
        for layer in model_test.layers.values():
            aot_nested_region(layer, hash_fn=CONST_HASH)
        self._run_forward_bitwise_test(model_ref, model_test, (tokens,))

    def test_forward_hooks_preserved(self):
        """Verify that forward hooks fire and their ops appear in the outer graph."""

        @aot_nested_region(hash_fn=CONST_HASH)
        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc = nn.Linear(dim, dim)

            def forward(self, x):
                return self.fc(x)

        class Model(nn.Module):
            def __init__(self, dim, n_layers):
                super().__init__()
                self.layers = nn.ModuleDict(
                    {str(i): Block(dim) for i in range(n_layers)}
                )

            def forward(self, x):
                for layer in self.layers.values():
                    x = layer(x)
                return x

        dim, n_layers = 64, 3
        model = Model(dim, n_layers).to(device=self.DEVICE)

        # Register a forward hook that multiplies output by 2
        for layer in model.layers.values():
            layer.register_forward_hook(lambda mod, inp, out: out * 2)

        x = torch.randn(2, 8, dim, device=self.DEVICE)

        out_eager = model(x)
        traced_result = trace_module(model, (x,))
        self._assert_invoke_subgraph_count(traced_result, n_layers)

        # The hook's mul op should be in the outer graph, not inside invoke_subgraph
        mul_count = sum(
            1
            for n in traced_result.gm.graph.nodes
            if n.op == "call_function" and "mul" in str(n.target)
        )
        self.assertEqual(mul_count, n_layers, "hook mul ops should be in outer graph")

        params_and_buffers = _get_params_and_buffers(model)
        out_traced = run_traced_module(traced_result, params_and_buffers, (x,))
        self.assertTrue(torch.equal(out_eager, out_traced[0]))

    # NOTE: MoE models (qwen3_moe, deepseek_v3) are not tested here because:
    # - MoE layers have in-place mutations that invoke_subgraph doesn't support
    # - DeepSeek V3 has heterogeneous layers (dense vs MoE) with different param counts


if __name__ == "__main__":
    unittest.main()
