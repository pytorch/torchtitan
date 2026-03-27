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
from torchtitan.experiments.graph_trainer.nested_region import aot_nested_region


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


CONST_HASH = lambda *args, **kwargs: "block"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestNestedRegion(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32
    BATCH_SIZE = 2
    SEQ_LEN = 128

    def setUp(self):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        torch.use_deterministic_algorithms(False)

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

    def test_module_instance(self):
        # Verify graph structure (4-layer MLP, dim=64):
        #
        # outer graph: only invoke_subgraph nodes, no inlined aten ops
        #   %invoke_subgraph   = invoke_subgraph(subgraph, block, w1, b1, w2, b2, x)
        #   %invoke_subgraph_1 = invoke_subgraph(subgraph, block, w3, b3, w4, b4, x1)
        #   ...
        #
        # subgraph (repeated_subgraph0):
        #   placeholders: fc1.weight, fc1.bias, fc2.weight, fc2.bias, x
        #   aten ops: t, addmm, relu, t, addmm, add
        #   return (add,)

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
        for layer in model.layers.values():
            aot_nested_region(layer, hash_fn=CONST_HASH)

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

        # Bitwise correctness
        params_and_buffers = _get_params_and_buffers(model)
        out_traced = run_traced_module(traced_result, params_and_buffers, (x,))
        self.assertTrue(torch.equal(out_eager, out_traced[0]))

    def test_module_instance_with_nontensor_args(self):
        """Non-tensor forward args (None, int) are captured via closure, not passed
        as invoke_subgraph operands."""

        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc = nn.Linear(dim, dim)

            def forward(self, x, mask=None, scale: int = 1):
                out = self.fc(x)
                if mask is not None:
                    out = out * mask
                return out * scale

        class Model(nn.Module):
            def __init__(self, dim, n_layers):
                super().__init__()
                self.layers = nn.ModuleDict(
                    {str(i): Block(dim) for i in range(n_layers)}
                )

            def forward(self, x):
                for layer in self.layers.values():
                    # None and int are non-tensor args
                    x = layer(x, None, 1)
                return x

        dim, n_layers = 16, 3
        model = Model(dim, n_layers).to(device=self.DEVICE)
        for layer in model.layers.values():
            aot_nested_region(layer, hash_fn=CONST_HASH)

        x = torch.randn(2, 4, dim, device=self.DEVICE)
        out_eager = model(x)
        traced_result = trace_module(model, (x,))

        self._assert_invoke_subgraph_count(traced_result, n_layers)

        params_and_buffers = _get_params_and_buffers(model)
        out_traced = run_traced_module(traced_result, params_and_buffers, (x,))
        self.assertTrue(torch.equal(out_eager, out_traced[0]))

    def test_module_instance_with_nontensor_args_interleaved(self):
        """Tensor arg in the middle: forward(mask, x, scale) exercises that
        tensor_arg_indices correctly identifies non-first tensor positions."""

        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc = nn.Linear(dim, dim)

            def forward(self, mask, x, scale: int = 1):
                out = self.fc(x)
                if mask is not None:
                    out = out * mask
                return out * scale

        class Model(nn.Module):
            def __init__(self, dim, n_layers):
                super().__init__()
                self.layers = nn.ModuleDict(
                    {str(i): Block(dim) for i in range(n_layers)}
                )

            def forward(self, x):
                for layer in self.layers.values():
                    x = layer(None, x, 2)
                return x

        dim, n_layers = 16, 3
        model = Model(dim, n_layers).to(device=self.DEVICE)
        for layer in model.layers.values():
            aot_nested_region(layer, hash_fn=CONST_HASH)

        x = torch.randn(2, 4, dim, device=self.DEVICE)
        out_eager = model(x)
        traced_result = trace_module(model, (x,))

        self._assert_invoke_subgraph_count(traced_result, n_layers)

        params_and_buffers = _get_params_and_buffers(model)
        out_traced = run_traced_module(traced_result, params_and_buffers, (x,))
        self.assertTrue(torch.equal(out_eager, out_traced[0]))

    def test_free_function_decorator(self):
        """@aot_nested_region on a free function with explicit param args."""
        import torch.nn.functional as F

        @aot_nested_region(hash_fn=CONST_HASH)
        def block_fwd(x, w1, b1, w2, b2):
            h = F.relu(F.linear(x, w1, b1))
            return x + F.linear(h, w2, b2)

        class Model(nn.Module):
            def __init__(self, dim, n_layers):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Sequential(nn.Linear(dim, dim * 2), nn.Linear(dim * 2, dim))
                     for _ in range(n_layers)]
                )

            def forward(self, x):
                for blk in self.layers:
                    fc1, fc2 = blk[0], blk[1]
                    x = block_fwd(x, fc1.weight, fc1.bias, fc2.weight, fc2.bias)
                return x

        dim, n_layers = 16, 3
        model = Model(dim, n_layers).to(device=self.DEVICE)
        x = torch.randn(2, 4, dim, device=self.DEVICE)

        out_eager = model(x)
        traced_result = trace_module(model, (x,))

        self._assert_invoke_subgraph_count(traced_result, n_layers)

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
        """Forward hooks fire in the outer graph, not inside invoke_subgraph."""

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
        for layer in model.layers.values():
            aot_nested_region(layer, hash_fn=CONST_HASH)

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

    def test_error_on_class(self):
        with self.assertRaisesRegex(ValueError, "does not support classes"):
            @aot_nested_region(hash_fn=CONST_HASH)
            class Block(nn.Module):
                pass

    def test_module_kwargs(self):
        """Tensor kwargs become operands; non-tensor kwargs are constants."""

        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc = nn.Linear(dim, dim)

            def forward(self, x, *, mask=None, scale: int = 1):
                out = self.fc(x)
                if mask is not None:
                    out = out * mask
                return out * scale

        class Model(nn.Module):
            def __init__(self, dim, n_layers):
                super().__init__()
                self.layers = nn.ModuleDict(
                    {str(i): Block(dim) for i in range(n_layers)}
                )

            def forward(self, x, mask):
                for layer in self.layers.values():
                    x = layer(x, mask=mask, scale=2)
                return x

        dim, n_layers = 16, 3
        model = Model(dim, n_layers).to(device=self.DEVICE)
        for layer in model.layers.values():
            aot_nested_region(layer, hash_fn=CONST_HASH)

        x = torch.randn(2, 4, dim, device=self.DEVICE)
        mask = torch.ones(2, 4, dim, device=self.DEVICE)
        out_eager = model(x, mask)
        traced_result = trace_module(model, (x, mask))

        self._assert_invoke_subgraph_count(traced_result, n_layers)

        params_and_buffers = _get_params_and_buffers(model)
        out_traced = run_traced_module(traced_result, params_and_buffers, (x, mask))
        self.assertTrue(torch.equal(out_eager, out_traced[0]))

    def test_free_function_kwargs(self):
        """@aot_nested_region on a free function with tensor and non-tensor kwargs."""
        import torch.nn.functional as F

        @aot_nested_region(hash_fn=CONST_HASH)
        def block_fwd(w1, b1, w2, b2, *, x, scale: int = 1):
            h = F.relu(F.linear(x, w1, b1))
            return F.linear(h, w2, b2) * scale

        class Model(nn.Module):
            def __init__(self, dim, n_layers):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Sequential(nn.Linear(dim, dim * 2), nn.Linear(dim * 2, dim))
                     for _ in range(n_layers)]
                )

            def forward(self, x):
                for blk in self.layers:
                    fc1, fc2 = blk[0], blk[1]
                    x = block_fwd(fc1.weight, fc1.bias, fc2.weight, fc2.bias, x=x, scale=2)
                return x

        dim, n_layers = 16, 3
        model = Model(dim, n_layers).to(device=self.DEVICE)
        x = torch.randn(2, 4, dim, device=self.DEVICE)

        out_eager = model(x)
        traced_result = trace_module(model, (x,))

        self._assert_invoke_subgraph_count(traced_result, n_layers)

        params_and_buffers = _get_params_and_buffers(model)
        out_traced = run_traced_module(traced_result, params_and_buffers, (x,))
        self.assertTrue(torch.equal(out_eager, out_traced[0]))

    def test_unique_hash_no_dedup(self):
        """Each unique hash key traces a separate subgraph — no deduplication.
        With hash_fn returning the layer index, n_layers invoke_subgraph nodes
        must reference n_layers distinct subgraphs (one per unique key)."""

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

        dim, n_layers = 16, 3
        model = Model(dim, n_layers).to(device=self.DEVICE)
        for i, layer in enumerate(model.layers.values()):
            aot_nested_region(layer, hash_fn=lambda *args, _i=i: f"block_{_i}")

        x = torch.randn(2, 4, dim, device=self.DEVICE)
        out_eager = model(x)
        traced_result = trace_module(model, (x,))

        # One invoke_subgraph per layer, each with a distinct subgraph target
        invoke_nodes = [
            n
            for n in traced_result.gm.graph.nodes
            if n.op == "call_function" and "invoke_subgraph" in str(n.target)
        ]
        self.assertEqual(len(invoke_nodes), n_layers)
        subgraph_targets = {n.args[0].target for n in invoke_nodes}
        self.assertEqual(len(subgraph_targets), n_layers)

        params_and_buffers = _get_params_and_buffers(model)
        out_traced = run_traced_module(traced_result, params_and_buffers, (x,))
        self.assertTrue(torch.equal(out_eager, out_traced[0]))

    def test_full_train_step(self):
        """Trace a full forward+backward train step and verify both forward and
        backward invoke_subgraph HOPs appear in the outer graph."""

        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc = nn.Linear(dim, dim)

            def forward(self, x):
                return torch.relu(self.fc(x))

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

        class TrainStep(nn.Module):
            """Wraps model so that trace_module captures fwd+bwd in one graph."""

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                out = self.model(x)
                loss = out.sum()
                loss.backward()
                return loss.detach()

        dim, n_layers = 16, 3
        model = Model(dim, n_layers).to(device=self.DEVICE)
        for layer in model.layers.values():
            aot_nested_region(layer, hash_fn=CONST_HASH)
        train_step = TrainStep(model)

        x = torch.randn(2, 4, dim, device=self.DEVICE)
        traced_result = trace_module(train_step, (x,))

        invoke_nodes = [
            n
            for n in traced_result.gm.graph.nodes
            if n.op == "call_function" and "invoke_subgraph" in str(n.target)
        ]
        # Forward: n_layers HOPs; backward: n_layers HOPs
        self.assertEqual(len(invoke_nodes), n_layers * 2)

        # Forward HOPs use one identifier, backward HOPs use another
        identifiers = {n.args[1] for n in invoke_nodes}
        self.assertEqual(len(identifiers), 2)
        fwd_ids = {id_ for id_ in identifiers if id_.startswith("fw_")}
        bwd_ids = {id_ for id_ in identifiers if id_.startswith("bw_")}
        self.assertEqual(len(fwd_ids), 1)
        self.assertEqual(len(bwd_ids), 1)

    # NOTE: MoE models (qwen3_moe, deepseek_v3) are not tested here because:
    # - MoE layers have in-place mutations that invoke_subgraph doesn't support
    # - DeepSeek V3 has heterogeneous layers (dense vs MoE) with different param counts


if __name__ == "__main__":
    unittest.main()
