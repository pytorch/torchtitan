# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import Counter
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import swap_in_optimizer_params_and_state
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    _maybe_materialize_grad_for_param_layout,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _copy_fwd_metadata_to_bw_nodes,
    extract_module_state,
    minimal_fx_tracer,
    run_traced,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    annotate_flex_attention_for_regional_inductor_pass,
)


def get_loss(logits, labels):
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
    )


def make_train_step(model, loss_fn):
    """Return a plain function that closes over ``model`` for module-based tracing."""

    def train_step(*args):
        *fwd_args, labels = args
        logits = model(*fwd_args)
        loss = loss_fn(logits, labels)
        params = list(model.parameters())
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)

    return train_step


def create_model(config_cls, model_config, device="cuda", dtype=torch.float32):
    model = config_cls(model_config)
    model.to(device=device, dtype=dtype)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device(device))
    return model


def _apply_regional_inductor(traced_result):
    """Apply regional_inductor to compile annotated HOP regions in the traced graph."""
    from torch.fx.graph import CodeGen
    from torch.fx.passes.regional_inductor import regional_inductor

    from torchtitan.models.common.attention import FlexAttention

    annotate_flex_attention_for_regional_inductor_pass(
        traced_result.gm,
        flex_compile_config=FlexAttention.inductor_configs,
    )

    fake_inputs = _graph_placeholder_fake_inputs(traced_result.gm)
    fake_mode = _graph_fake_mode(fake_inputs)
    with torch._guards.tracing(torch._guards.TracingContext(fake_mode)):
        traced_result.gm = regional_inductor(traced_result.gm)

    traced_result.gm.graph.set_codegen(CodeGen())
    traced_result.gm.recompile()


def _graph_placeholder_fake_inputs(gm):
    fake_inputs = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val")
        if val is None:
            raise RuntimeError(f"Missing placeholder meta val for {node}")
        fake_inputs.append(val)
    return fake_inputs


def _graph_fake_mode(fake_inputs):
    return next(
        (
            val.fake_mode
            for val in fake_inputs
            if isinstance(val, torch.Tensor) and hasattr(val, "fake_mode")
        ),
        None,
    )


class SimpleMLP(nn.Module):
    def __init__(self, dim=64, hidden=128, vocab_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(self.embed(x))))


class _TraceableWrapper(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem):
        wrapper = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
        )
        wrapper.elem = elem
        return wrapper

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise RuntimeError("Test wrapper should be rejected before tracing")

    def __tensor_flatten__(self):
        return ["elem"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return _TraceableWrapper(inner_tensors["elem"])


class TestMinimalFXTracerDynamicShapes(unittest.TestCase):
    def _trace_mark_dynamic_value_range(self, *, min_value=None, max_value=None):
        from torch._dynamo import mark_dynamic

        def forward(x):
            return x.sin()

        kwargs = {}
        if min_value is not None:
            kwargs["min"] = min_value
        if max_value is not None:
            kwargs["max"] = max_value

        x = torch.randn(4, 4)
        mark_dynamic(x, 0, **kwargs)

        traced = minimal_fx_tracer(forward)(x)
        fake_x = next(
            node.meta["val"]
            for node in traced.gm.graph.nodes
            if node.op == "placeholder"
        )
        sym = fake_x.shape[0].node.expr
        return fake_x.shape[0].node.shape_env.var_to_range[sym]

    def test_fakeify_input_copies_only_shape_annotations(self):
        from torch._dynamo import mark_dynamic
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        from torchtitan.experiments.graph_trainer.dynamic_shapes import _fakeify_input

        x = torch.randn(2, 4)
        mark_dynamic(x, 0)
        x._graph_trainer_unrelated_state = "must not be copied"

        fake_mode = FakeTensorMode(shape_env=ShapeEnv(), static_shapes=False)
        with fake_mode:
            fake_x = _fakeify_input(fake_mode, x, input_name="x")

        self.assertTrue(hasattr(fake_x, "_dynamo_dynamic_indices"))
        self.assertTrue(hasattr(fake_x, "_dynamo_dynamic_range"))
        self.assertFalse(hasattr(fake_x, "_graph_trainer_unrelated_state"))

    def test_mark_dynamic_min_max_preserves_range(self):
        value_range = self._trace_mark_dynamic_value_range(min_value=2, max_value=8)

        self.assertEqual(value_range.lower, 2)
        self.assertEqual(value_range.upper, 8)

    def test_mark_dynamic_one_sided_ranges_preserve_bounds(self):
        from torch.utils._sympy.numbers import int_oo

        # ShapeEnv tightens dynamic tensor sizes to exclude 0/1, so a max-only
        # user range is observed as [2, max] after fakeification.
        cases = (
            ("min_only", {"min_value": 2}, 2, int_oo),
            ("max_only", {"max_value": 8}, 2, 8),
        )
        for name, kwargs, expected_lower, expected_upper in cases:
            with self.subTest(name=name):
                value_range = self._trace_mark_dynamic_value_range(**kwargs)

                self.assertEqual(value_range.lower, expected_lower)
                self.assertEqual(value_range.upper, expected_upper)

    def test_mark_dynamic_wrapper_subclass_rejected(self):
        from torch._dynamo import mark_dynamic

        wrapper = _TraceableWrapper(torch.randn(2, 4))
        mark_dynamic(wrapper, 0)

        def forward(x):
            return x

        with self.assertRaisesRegex(
            ValueError,
            "only supports marked dynamic dims on plain tensor inputs",
        ):
            minimal_fx_tracer(forward)(wrapper)

    def test_nested_wrapper_subclass_marked_inner_rejected(self):
        from torch._dynamo import mark_dynamic
        from torch._dynamo.decorators import mark_unbacked

        def forward(x):
            return x

        for marker in (mark_dynamic, mark_unbacked):
            with self.subTest(marker=marker.__name__):
                inner = torch.randn(2, 4)
                marker(inner, 0)
                wrapper = _TraceableWrapper(_TraceableWrapper(inner))

                with self.assertRaisesRegex(
                    ValueError,
                    "only supports marked dynamic dims on plain tensor inputs",
                ):
                    minimal_fx_tracer(forward)(wrapper)

    def test_mark_dynamic_batch_and_seq_dims_with_rope(self):
        from torch._dynamo import mark_dynamic

        from torchtitan.models.common.rope import (
            _reshape_for_broadcast,
            ComplexRoPE,
            CosSinRoPE,
        )

        def forward(x, xq, xk, freqs_cis, rope_cache, positions):
            complex_cache = _reshape_for_broadcast(
                freqs_cis, (*x.shape[:-1], x.shape[-1] // 2), positions
            )
            single, _ = ComplexRoPE.apply_rotary_emb(x, x, complex_cache)
            cos_sin_cache = _reshape_for_broadcast(rope_cache, xq.shape, positions)
            q, k = CosSinRoPE.apply_rotary_emb(xq, xk, cos_sin_cache)
            return single + q + k

        batch, seq, heads, head_dim = 2, 4, 1, 8
        position_cases = {
            "none": None,
            "single": torch.arange(seq).unsqueeze(0),
            "batched": torch.arange(seq).repeat(batch, 1),
        }

        for name, positions in position_cases.items():
            with self.subTest(positions=name):
                x = torch.randn(batch, seq, heads, head_dim)
                xq = torch.randn(batch, seq, heads, head_dim)
                xk = torch.randn(batch, seq, heads, head_dim)
                freqs = torch.randn(seq * 2, head_dim // 2)
                freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
                rope_cache = torch.randn(seq * 2, head_dim * 2)

                for tensor in (x, xq, xk):
                    mark_dynamic(tensor, 0)
                    mark_dynamic(tensor, 1)
                if positions is not None:
                    mark_dynamic(positions, 1)
                    if positions.shape[0] > 1:
                        mark_dynamic(positions, 0)

                traced = minimal_fx_tracer(forward)(
                    x, xq, xk, freqs_cis, rope_cache, positions
                )
                self.assertTrue(
                    torch.equal(
                        forward(x, xq, xk, freqs_cis, rope_cache, positions),
                        run_traced(traced)(x, xq, xk, freqs_cis, rope_cache, positions),
                    )
                )

    def test_mark_unbacked_positions_batch_dim_with_rope(self):
        from torch._dynamo.decorators import mark_unbacked

        from torchtitan.models.common.rope import _reshape_for_broadcast, CosSinRoPE

        def forward(xq, xk, rope_cache, positions):
            cos_sin_cache = _reshape_for_broadcast(rope_cache, xq.shape, positions)
            q, k = CosSinRoPE.apply_rotary_emb(xq, xk, cos_sin_cache)
            return q + k

        batch, seq, heads, head_dim = 4, 4, 1, 8
        xq = torch.randn(batch, seq, heads, head_dim)
        xk = torch.randn(batch, seq, heads, head_dim)
        rope_cache = torch.randn(seq * 2, head_dim * 2)
        positions = torch.arange(seq).repeat(batch, 1)
        for tensor in (xq, xk, positions):
            mark_unbacked(
                tensor,
                0,
                hint_override=batch,
                min=1,
                max=batch,
                shape_id="batch",
            )

        traced = minimal_fx_tracer(forward)(xq, xk, rope_cache, positions)

        self.assertTrue(
            torch.equal(
                forward(xq, xk, rope_cache, positions),
                run_traced(traced)(xq, xk, rope_cache, positions),
            )
        )

    def test_rope_symbolic_positions_compile_fullgraph(self):
        from torchtitan.models.common.rope import _reshape_for_broadcast

        @torch.compile(backend="eager", fullgraph=True, dynamic=True)
        def forward(xq, rope_cache, positions):
            return _reshape_for_broadcast(rope_cache, xq.shape, positions)

        seq, head_dim = 5, 8
        xq = torch.randn(4, seq, 1, head_dim)
        rope_cache = torch.randn(seq * 2, head_dim * 2)
        positions = torch.arange(seq).repeat(xq.shape[0], 1)

        self.assertTrue(
            torch.equal(
                forward(xq, rope_cache, positions),
                _reshape_for_broadcast(rope_cache, xq.shape, positions),
            )
        )

    def test_maybe_materialize_grad_for_param_layout_restores_param_strides(self):
        param = torch.empty_strided((2, 3), (1, 2))
        grad = torch.arange(6.0).reshape(2, 3)

        materialized = _maybe_materialize_grad_for_param_layout(param, grad)

        self.assertEqual(materialized.stride(), param.stride())
        self.assertTrue(torch.equal(materialized, grad))
        self.assertIs(
            _maybe_materialize_grad_for_param_layout(param, materialized),
            materialized,
        )

    def test_mark_unbacked_mixed_with_static_input_replay(self):
        from torch._dynamo.decorators import mark_unbacked

        def forward(dynamic_x, static_y):
            return dynamic_x.cos() + static_y.sin()

        dynamic_x = torch.randn(2, 4)
        static_y = torch.randn(2, 4)
        mark_unbacked(dynamic_x, 0)

        traced = minimal_fx_tracer(forward)(dynamic_x, static_y)
        dynamic_x_other = torch.randn(3, 4)
        static_y_other = torch.randn(3, 4)

        self.assertTrue(
            torch.equal(
                forward(dynamic_x, static_y),
                run_traced(traced)(dynamic_x, static_y),
            )
        )
        self.assertTrue(
            torch.equal(
                forward(dynamic_x_other, static_y_other),
                run_traced(traced)(dynamic_x_other, static_y_other),
            )
        )

    def test_mark_unbacked_shape_branch_rejected(self):
        from torch._dynamo.decorators import mark_unbacked
        from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

        def forward(x):
            if x.shape[0] > 100:
                return x.cos()
            return x.sin()

        x = torch.randn(4, 4)
        mark_unbacked(x, 0)

        with self.assertRaisesRegex(
            GuardOnDataDependentSymNode,
            "Could not guard on data-dependent expression",
        ):
            minimal_fx_tracer(forward)(x)

    def test_mark_unbacked_min_max_preserves_unbacked_placeholder_dim(self):
        from torch._dynamo.decorators import mark_unbacked
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

        def forward(x):
            if x.size(0) >= 2 and x.size(0) <= 5:
                return x.sin()
            return x.cos()

        x = torch.randn(3, 4)
        mark_unbacked(x, 0, min=2, max=5)

        traced = minimal_fx_tracer(forward)(x)
        fake_x = next(
            node.meta["val"]
            for node in traced.gm.graph.nodes
            if node.op == "placeholder"
        )
        x_min = torch.randn(2, 4)
        x_max = torch.randn(5, 4)

        self.assertIsInstance(fake_x.size(0), torch.SymInt)
        self.assertTrue(free_unbacked_symbols(fake_x.size(0)))
        self.assertEqual(fake_x.size(1), 4)
        self.assertTrue(torch.equal(forward(x_min), run_traced(traced)(x_min)))
        self.assertTrue(torch.equal(forward(x_max), run_traced(traced)(x_max)))

    def test_mark_unbacked_input_symbol_is_not_pending_fresh(self):
        from torch._dynamo.decorators import mark_unbacked

        def forward(x):
            return x.sin()

        x = torch.randn(3, 4)
        mark_unbacked(x, 0, min=2, max=5)

        traced = minimal_fx_tracer(forward)(x)
        fake_x = next(
            node.meta["val"]
            for node in traced.gm.graph.nodes
            if node.op == "placeholder"
        )
        shape_env = fake_x.shape[0].node.shape_env

        self.assertEqual(shape_env.pending_fresh_unbacked_symbols, [])
        self.assertEqual(shape_env.ignorable_fresh_unbacked_symbols, [])

    def test_mark_unbacked_preserves_unbacked_placeholder_dim(self):
        from torch._dynamo.decorators import mark_unbacked
        from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

        def forward(x):
            return x.sin()

        x = torch.randn(2, 4)
        mark_unbacked(x, 0)

        traced = minimal_fx_tracer(forward)(x)
        fake_x = next(
            node.meta["val"]
            for node in traced.gm.graph.nodes
            if node.op == "placeholder"
        )
        x_other = torch.randn(3, 4)

        self.assertIsInstance(fake_x.size(0), torch.SymInt)
        self.assertTrue(free_unbacked_symbols(fake_x.size(0)))
        self.assertEqual(fake_x.size(1), 4)
        self.assertTrue(torch.equal(forward(x), run_traced(traced)(x)))
        self.assertTrue(
            torch.equal(
                forward(x_other),
                run_traced(traced)(x_other),
            )
        )

    def test_mark_unbacked_multiple_inputs_replay(self):
        from torch._dynamo.decorators import mark_unbacked

        def forward(x, y):
            return x.sin() + y.cos()

        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        mark_unbacked(x, 0)
        mark_unbacked(y, 0)

        traced = minimal_fx_tracer(forward)(x, y)
        x_other = torch.randn(3, 4)
        y_other = torch.randn(3, 4)

        self.assertTrue(torch.equal(forward(x, y), run_traced(traced)(x, y)))
        self.assertTrue(
            torch.equal(
                forward(x_other, y_other),
                run_traced(traced)(x_other, y_other),
            )
        )

    def test_data_dependent_check_emits_runtime_asserts(self):
        """torch._check on a data-dependent .item() symbol becomes _assert_scalar nodes."""

        def forward(x, n):
            v = n.item()
            torch._check(v > 0)
            torch._check(v < 100)
            return x.sin().sum() + v

        x = torch.randn(8, 4)
        n = torch.tensor([5])

        traced = minimal_fx_tracer(forward, _insert_runtime_asserts=True)(x, n)
        assert_count = sum(
            1
            for node in traced.gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.aten._assert_scalar.default
        )
        # Both inline (gt/lt) and bound-style (>= 1, <= 99) asserts are emitted.
        self.assertEqual(assert_count, 4)

    def test_no_runtime_asserts_when_no_constraints(self):
        """Tracing without data-dependent _check produces no _assert_scalar nodes."""
        from torch._dynamo.decorators import mark_unbacked

        def forward(x):
            return x.sin()

        x = torch.randn(4, 4)
        mark_unbacked(x, 0)

        traced = minimal_fx_tracer(forward)(x)
        assert_count = sum(
            1
            for node in traced.gm.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.aten._assert_scalar.default
        )
        self.assertEqual(assert_count, 0)

    def test_mark_unbacked_shape_id_multiple_inputs_replay(self):
        from torch._dynamo.decorators import mark_unbacked

        def forward(x, y):
            if x.size(0) == y.size(0):
                return x.sin() + y.cos()
            return x.cos() + y.sin()

        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        mark_unbacked(x, 0, shape_id="batch")
        mark_unbacked(y, 0, shape_id="batch")

        traced = minimal_fx_tracer(forward)(x, y)
        x_other = torch.randn(3, 4)
        y_other = torch.randn(3, 4)

        self.assertTrue(
            torch.equal(
                forward(x_other, y_other),
                run_traced(traced)(x_other, y_other),
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTraceModule(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32
    BATCH_SIZE = 2
    SEQ_LEN = 128
    NUM_STEPS = 5
    LR = 1e-3

    def setUp(self):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def _make_mlp(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(
            0, 256, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        labels = torch.randint(
            0, 256, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        return model, tokens, labels, get_loss

    def test_mlp_forward(self):
        model, tokens, labels, loss_fn = self._make_mlp()

        def forward(tokens):
            return model(tokens)

        traced = minimal_fx_tracer(forward, module=model)(tokens)
        out_eager = model(tokens)
        wrapped = run_traced(traced, module=model)(tokens)
        self.assertTrue(torch.equal(out_eager, wrapped))

    def test_mlp_train_step(self):
        model_ref, tokens, labels, loss_fn = self._make_mlp()
        model_test = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())

        train_step = make_train_step(model_ref, loss_fn)
        traced = minimal_fx_tracer(train_step, module=model_ref)(tokens, labels)

        logits_ref = model_ref(tokens)
        loss_ref = loss_fn(logits_ref, labels)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]

        wrapped = run_traced(traced, module=model_test)(tokens, labels)
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]

        self.assertTrue(torch.equal(loss_ref, loss_tr))
        for gr, gt in zip(grads_ref, grads_tr, strict=True):
            self.assertTrue(torch.equal(gr, gt))

    def test_chunked_loss_train_step(self):
        D, V, num_chunks = 32, 257, 4
        lm_head_ref = nn.Linear(D, V, bias=False).to(
            device=self.DEVICE, dtype=self.DTYPE
        )
        lm_head_test = nn.Linear(D, V, bias=False).to(
            device=self.DEVICE, dtype=self.DTYPE
        )
        lm_head_test.load_state_dict(lm_head_ref.state_dict())
        hidden_states = torch.randn(
            self.BATCH_SIZE,
            self.SEQ_LEN,
            D,
            device=self.DEVICE,
            dtype=self.DTYPE,
            requires_grad=True,
        )
        labels = torch.randint(
            0, V, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )

        def train_step(lm_head, hidden_states, labels):
            loss_fn = ChunkedLossWrapperWithParamGrads(
                ChunkedLossWrapperWithParamGrads.Config(num_chunks=num_chunks)
            )
            loss_fn.set_lm_head(lm_head)
            loss, _ = loss_fn(hidden_states, labels)
            grads = torch.autograd.grad(loss, [hidden_states, *lm_head.parameters()])
            return [loss, *grads]

        eager_out = train_step(lm_head_ref, hidden_states, labels)

        def train_step_closure(hidden_states, labels):
            return train_step(lm_head_test, hidden_states, labels)

        traced = minimal_fx_tracer(train_step_closure, module=lm_head_test)(
            hidden_states, labels
        )
        replay_out = run_traced(traced, module=lm_head_test)(hidden_states, labels)

        for ref, tr in zip(eager_out, replay_out, strict=True):
            self.assertTrue(torch.equal(ref, tr))

    def test_mlp_multistep_bitwise(self):
        model_ref, tokens, labels, loss_fn = self._make_mlp()
        model_test = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())

        train_step = make_train_step(model_ref, loss_fn)
        traced = minimal_fx_tracer(train_step, module=model_ref)(tokens, labels)

        opt_ref = torch.optim.Adam(model_ref.parameters(), lr=self.LR)
        opt_copy = torch.optim.Adam(model_test.parameters(), lr=self.LR)

        for step in range(1, self.NUM_STEPS + 1):
            logits_ref = model_ref(tokens)
            loss_ref = loss_fn(logits_ref, labels)
            loss_ref.backward()
            grads_ref = [p.grad.clone() for p in model_ref.parameters()]
            opt_ref.step()
            opt_ref.zero_grad()

            wrapped = run_traced(traced, module=model_test)(tokens, labels)
            loss_tr = wrapped[0]
            grads_tr = wrapped[1:]
            for p, g in zip(model_test.parameters(), grads_tr, strict=True):
                p.grad = g
            opt_copy.step()
            opt_copy.zero_grad()

            self.assertTrue(
                torch.equal(loss_ref, loss_tr), f"Step {step}: loss mismatch"
            )
            for gr, gt in zip(grads_ref, grads_tr, strict=True):
                self.assertTrue(torch.equal(gr, gt), f"Step {step}: grad mismatch")

    def test_non_tensor_leaf_raises(self):
        """Passing a callable leaf in args raises (should be in closure instead)."""

        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        def fn(x, loss_fn):
            return loss_fn(model(x))

        with self.assertRaises(ValueError, msg="all pytree leaves"):
            minimal_fx_tracer(fn, module=model)(tokens, lambda x: x.sum())

    def test_mismatched_module_raises_when_validation_enabled(self):
        """Opt-in module FQN validation catches execution with the wrong module."""
        model, tokens, labels, loss_fn = self._make_mlp()

        def forward(tokens):
            return model(tokens)

        traced = minimal_fx_tracer(forward, module=model)(tokens)

        different_model = nn.Sequential(
            nn.Embedding(256, 64),
            nn.Linear(64, 256),
        ).to(device=self.DEVICE, dtype=self.DTYPE)

        with self.assertRaises(ValueError, msg="different parameter/buffer names"):
            run_traced(traced, module=different_model, _validate_runtime=True)(tokens)

    def test_optimizer_passed_without_module_raises(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        opt = torch.optim.Adam(model.parameters())
        with self.assertRaises(ValueError, msg="optimizer"):
            minimal_fx_tracer(lambda: None, optimizer=opt)

    def test_kwargs_roundtrip(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        scale = torch.tensor(2.0, device=self.DEVICE)

        def forward(state, tokens, *, scale):
            with torch.nn.utils.stateless._reparametrize_module(model, state):
                return model(tokens) * scale

        state = extract_module_state(model)
        traced = minimal_fx_tracer(forward)(state, tokens, scale=scale)
        out_ref = forward(state, tokens, scale=scale)
        out_traced = run_traced(traced)(state, tokens, scale=scale)
        self.assertTrue(torch.equal(out_ref, out_traced))

    def test_kwargs_runtime_reorder_raises(self):
        """Runtime kwargs in different order produce a different spec; with
        ``_validate_runtime=True``, this raises."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        a = torch.tensor(2.0, device=self.DEVICE)
        b = torch.tensor(3.0, device=self.DEVICE)

        def forward(state, tokens, *, a, b):
            with torch.nn.utils.stateless._reparametrize_module(model, state):
                return model(tokens) * a + b

        state = extract_module_state(model)
        traced = minimal_fx_tracer(forward)(state, tokens, a=a, b=b)
        with self.assertRaisesRegex(ValueError, "input spec mismatch"):
            run_traced(traced, _validate_runtime=True)(state, tokens, b=b, a=a)

    def test_kwargs_unknown_kwarg_raises(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        scale = torch.tensor(2.0, device=self.DEVICE)

        def forward(state, tokens, *, scale):
            with torch.nn.utils.stateless._reparametrize_module(model, state):
                return model(tokens) * scale

        state = extract_module_state(model)
        traced = minimal_fx_tracer(forward)(state, tokens, scale=scale)
        with self.assertRaisesRegex(ValueError, "input spec mismatch"):
            run_traced(traced, _validate_runtime=True)(state, tokens, factor=scale)

    def test_kwargs_default_omitted_bakes_constant(self):
        """fn with a default kwarg, not passed at trace: default is baked in.
        Runtime must also omit it (passing it would change the spec)."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        scale = torch.tensor(3.0, device=self.DEVICE)

        def forward(state, tokens, *, scale=2.0):
            with torch.nn.utils.stateless._reparametrize_module(model, state):
                return model(tokens) * scale

        state = extract_module_state(model)
        traced = minimal_fx_tracer(forward)(state, tokens)
        out_default = run_traced(traced)(state, tokens)
        out_ref = forward(state, tokens)
        self.assertTrue(torch.equal(out_ref, out_default))

        with self.assertRaisesRegex(ValueError, "input spec mismatch"):
            run_traced(traced, _validate_runtime=True)(state, tokens, scale=scale)

    def test_kwargs_var_keyword_missing_key_raises(self):
        """fn with **opts: missing a kwarg at runtime changes the kwargs spec."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        a = torch.tensor(2.0, device=self.DEVICE)
        b = torch.tensor(5.0, device=self.DEVICE)

        def forward(state, tokens, **opts):
            with torch.nn.utils.stateless._reparametrize_module(model, state):
                return model(tokens) * opts["a"] + opts["b"]

        state = extract_module_state(model)
        traced = minimal_fx_tracer(forward)(state, tokens, a=a, b=b)
        out_ref = forward(state, tokens, a=a, b=b)
        out_traced = run_traced(traced)(state, tokens, a=a, b=b)
        self.assertTrue(torch.equal(out_ref, out_traced))

        with self.assertRaisesRegex(ValueError, "input spec mismatch"):
            run_traced(traced, _validate_runtime=True)(state, tokens, a=a)

    def test_flex_attention_block_mask_mark_unbacked(self):
        from torch._dynamo.decorators import mark_unbacked
        from torch.nn.attention.flex_attention import (
            AuxRequest,
            BlockMask,
            flex_attention,
        )

        maybe_register_blockmask_pytree_node()

        def make_mask_fn(attn_regions, document_ids):
            def mask_mod(b, h, q_idx, kv_idx):
                return (
                    (q_idx >= kv_idx)
                    & (attn_regions[q_idx] == attn_regions[kv_idx])
                    & (document_ids[q_idx] == document_ids[kv_idx])
                )

            return mask_mod

        def make_mask(ntoks=256, block_size=128):
            nblocks = (ntoks + block_size - 1) // block_size
            width = 2

            kv_indices = (
                torch.arange(width, dtype=torch.int32, device=self.DEVICE)
                .expand(nblocks, width)
                .clone()
            )
            full_kv_indices = kv_indices.clone()
            q_indices = kv_indices.clone()
            full_q_indices = kv_indices.clone()
            kv_num_blocks = torch.full(
                (nblocks,), width, dtype=torch.int32, device=self.DEVICE
            )
            full_kv_num_blocks = kv_num_blocks.clone()
            q_num_blocks = kv_num_blocks.clone()
            full_q_num_blocks = kv_num_blocks.clone()

            mark_unbacked(kv_indices, 1)
            mark_unbacked(full_kv_indices, 1)
            mark_unbacked(q_indices, 1)
            mark_unbacked(full_q_indices, 1)

            attn_regions = torch.arange(ntoks, dtype=torch.int32, device=self.DEVICE)
            document_ids = torch.zeros(ntoks, dtype=torch.int32, device=self.DEVICE)
            return BlockMask(
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                full_kv_num_blocks=full_kv_num_blocks,
                full_kv_indices=full_kv_indices,
                q_num_blocks=q_num_blocks,
                q_indices=q_indices,
                full_q_num_blocks=full_q_num_blocks,
                full_q_indices=full_q_indices,
                BLOCK_SIZE=(block_size, block_size),
                mask_mod=make_mask_fn(attn_regions, document_ids),
                seq_lengths=(ntoks, ntoks),
            )

        q = torch.randn(1, 2, 256, 32, device=self.DEVICE)
        k = torch.randn(1, 2, 256, 32, device=self.DEVICE)
        v = torch.randn(1, 2, 256, 32, device=self.DEVICE)
        mask = make_mask()
        cflex = torch.compile(flex_attention, dynamic=False, fullgraph=True)

        def forward(q, k, v, block_mask):
            out, aux = cflex(
                q,
                k,
                v,
                block_mask=block_mask,
                return_aux=AuxRequest(max_scores=True),
            )
            return out.sum().detach(), aux.max_scores.max().detach()

        minimal_fx_tracer(forward)(q, k, v, mask)

    def test_module_in_args_raises(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        other_model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        def forward(other, tokens):
            return other(tokens)

        with self.assertRaises(ValueError, msg="nn.Module"):
            minimal_fx_tracer(forward, module=model)(other_model, tokens)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestReparametrizeOptimizer(unittest.TestCase):
    """Verify swap_in_optimizer_params_and_state works with a torchtitan OptimizersContainer.

    OptimizersContainer is itself an Optimizer subclass, but it delegates
    ``step``/``state``/``state_dict`` to inner ``torch.optim.Adam``/``AdamW``
    instances. ``OptimizersContainer.state_dict()`` returns a DCP-flattened
    (FQN-keyed) dict, so the reparametrize helper consumes the inner
    optimizer's raw ``state_dict()`` (packed-int-id format) instead.
    """

    DEVICE = "cuda"
    DTYPE = torch.float32

    def test_titan_optimizers_container(self):
        from torchtitan.components.optimizer import (
            OptimizersContainer,
            ParamGroupConfig,
        )

        torch.manual_seed(0)
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        container = OptimizersContainer(
            OptimizersContainer.Config(
                param_groups=[
                    ParamGroupConfig(
                        pattern=r".*",
                        optimizer_name="AdamW",
                        optimizer_kwargs={"lr": 1e-3},
                    )
                ],
                implementation="for-loop",
            ),
            model_parts=[model],
        )
        inner = container.optimizers[0]

        # Initialize Adam's lazy per-parameter state.
        x = torch.randint(0, 256, (2, 16), device=self.DEVICE)
        loss = model(x).sum()
        loss.backward()
        container.step()
        container.zero_grad(set_to_none=True)

        # Snapshot the originals so we can verify perfect restoration.
        optim_state_dict = inner.state_dict()
        original_param_ids = [[id(p) for p in g["params"]] for g in inner.param_groups]
        original_state_keys = list(inner.state.keys())
        original_state_id = id(inner.state)

        # Rebind to fake parameter tensors (zeros to make the swap obvious).
        params_dict = dict(model.named_parameters(remove_duplicate=False))
        fake_params = {name: torch.zeros_like(p) for name, p in params_dict.items()}

        with swap_in_optimizer_params_and_state(inner, fake_params, optim_state_dict):
            # The live optimizer now points at the fake tensors.
            rebound = [p for g in inner.param_groups for p in g["params"]]
            for fake, rebound_p in zip(fake_params.values(), rebound, strict=True):
                self.assertIs(fake, rebound_p)

            # Per-param state is keyed by the rebound tensors and shares the
            # original tensor values (so in-place ops would propagate).
            for fake in fake_params.values():
                self.assertIn(fake, inner.state)
            for name, fake in fake_params.items():
                # Match against the original state_dict via positional
                # alignment in the optimizer's first (only) param group.
                idx = list(fake_params).index(name)
                packed_id = optim_state_dict["param_groups"][0]["params"][idx]
                expected_state = optim_state_dict["state"][packed_id]
                self.assertEqual(
                    set(inner.state[fake].keys()), set(expected_state.keys())
                )
                for k, v in expected_state.items():
                    if isinstance(v, torch.Tensor):
                        self.assertIs(inner.state[fake][k], v)

        # After the context the live optimizer is fully restored.
        self.assertEqual(id(inner.state), original_state_id)
        self.assertEqual(list(inner.state.keys()), original_state_keys)
        for orig_ids, group in zip(original_param_ids, inner.param_groups, strict=True):
            self.assertEqual([id(p) for p in group["params"]], orig_ids)

    def test_minimal_fx_tracer_with_bucketed_optimizer(self):
        torch.manual_seed(0)
        module = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 7))
        weights = [p for n, p in module.named_parameters() if n.endswith("weight")]
        biases = [p for n, p in module.named_parameters() if n.endswith("bias")]
        optimizer = torch.optim.AdamW(
            [{"params": weights, "lr": 0.1}, {"params": biases, "lr": 0.01}]
        )

        x = torch.randn(2, 3)
        module(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        def train_step(x):
            optimizer.zero_grad(set_to_none=True)
            module(x).sum().backward()
            optimizer.step()
            return torch.stack([p.detach().sum() for p in module.parameters()])

        traced = minimal_fx_tracer(train_step, module=module, optimizer=optimizer)(x)

        model_sd = deepcopy(module.state_dict())
        optim_sd = deepcopy(optimizer.state_dict())

        eager_out = train_step(x)
        eager_params = [p.detach().clone() for p in module.parameters()]

        module.load_state_dict(model_sd)
        optimizer.load_state_dict(optim_sd)
        traced_out = run_traced(traced, module=module, optimizer=optimizer)(x)
        traced_params = [p.detach().clone() for p in module.parameters()]

        self.assertTrue(torch.equal(eager_out, traced_out))
        for ep, tp in zip(eager_params, traced_params, strict=True):
            self.assertTrue(torch.equal(ep, tp))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTraceDTensor(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32

    def setUp(self):
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:12357",
                world_size=1,
                rank=0,
            )
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

    def tearDown(self):
        import torch.distributed as dist

        torch.use_deterministic_algorithms(False)
        if dist.is_initialized():
            dist.destroy_process_group()

    def _distribute_params(self, model, mesh):
        from torch.distributed._tensor import distribute_tensor, Replicate

        for name, param in list(model.named_parameters()):
            dt = distribute_tensor(param, mesh, [Replicate()])
            param_parts = name.split(".")
            mod = model
            for part in param_parts[:-1]:
                mod = getattr(mod, part)
            setattr(mod, param_parts[-1], nn.Parameter(dt))

    def test_dtensor_forward(self):
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(self.DEVICE, (1,))

        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        self._distribute_params(model, mesh)

        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        tokens_dt = DTensor.from_local(tokens, mesh, [Replicate()])

        def forward(tokens):
            return model(tokens)

        traced = minimal_fx_tracer(forward, module=model)(tokens_dt)
        has_subclass = any(
            layout.meta is not None for layout in traced.input_subclass_layouts.values()
        )
        self.assertTrue(has_subclass)

        out_eager = model(tokens_dt)
        wrapped = run_traced(traced, module=model)(tokens_dt)
        self.assertTrue(torch.equal(out_eager.full_tensor(), wrapped.full_tensor()))

    def test_dtensor_mark_unbacked_rejected(self):
        from torch._dynamo.decorators import mark_unbacked
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(self.DEVICE, (1,))
        tokens = torch.randn(2, 32, device=self.DEVICE)
        tokens_dt = DTensor.from_local(tokens, mesh, [Replicate()])
        mark_unbacked(tokens_dt, 0)

        def forward(tokens):
            return tokens

        with self.assertRaisesRegex(
            ValueError,
            "only supports marked dynamic dims on plain tensor inputs",
        ):
            minimal_fx_tracer(forward)(tokens_dt)

    def test_dtensor_train_step(self):
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(self.DEVICE, (1,))

        model_ref = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_test = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())

        self._distribute_params(model_ref, mesh)
        self._distribute_params(model_test, mesh)

        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        tokens_dt = DTensor.from_local(tokens, mesh, [Replicate()])
        labels_dt = DTensor.from_local(labels, mesh, [Replicate()])

        train_step = make_train_step(model_ref, get_loss)
        traced = minimal_fx_tracer(train_step, module=model_ref)(tokens_dt, labels_dt)

        logits_ref = model_ref(tokens_dt)
        loss_ref = get_loss(logits_ref, labels_dt)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]

        wrapped = run_traced(traced, module=model_test)(tokens_dt, labels_dt)
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]

        self.assertTrue(torch.equal(loss_ref.full_tensor(), loss_tr.full_tensor()))
        for gr, gt in zip(grads_ref, grads_tr, strict=True):
            self.assertTrue(torch.equal(gr.full_tensor(), gt.full_tensor()))

    def test_full_inductor_pass_on_collective(self):
        # ``make_fx`` traces ``dist.*`` collectives as raw ``c10d.{op}_``
        # inplace ops with a torchbind ``ProcessGroup`` baked in as a graph
        # attr. ``full_inductor_compilation_pass`` must functionalize the
        # collective and unbox the PG before ``compile_fx_inner`` — otherwise
        # the cache key calls ``__eq__`` on the torchbind and crashes.
        import torch.distributed as dist

        from torchtitan.experiments.graph_trainer.inductor_passes import (
            full_inductor_compilation_pass,
        )

        def f(_state, t):
            t = t.clone()
            dist.all_reduce(t)
            return t + 1

        traced = minimal_fx_tracer(f)({}, torch.ones(4, device=self.DEVICE))
        compiled_gm = full_inductor_compilation_pass(traced.gm, traced.example_inputs)

        real_input = torch.ones(4, device=self.DEVICE)
        expected = f({}, real_input.clone())
        actual = compiled_gm(real_input.clone())
        if isinstance(actual, (list, tuple)):
            actual = actual[0]
        torch.testing.assert_close(actual, expected)

    def test_full_inductor_pass_migrates_cpu_attrs(self):
        from torchtitan.experiments.graph_trainer.cudagraph import cudagraph_pass
        from torchtitan.experiments.graph_trainer.inductor_passes import (
            full_inductor_compilation_pass,
        )

        def f(_state, x):
            pad = torch.tensor(-1, dtype=torch.int64)
            fill = torch.tensor(0, dtype=torch.bfloat16)
            scale = torch.tensor(1.0, dtype=torch.float32)
            return x + pad.to(x.dtype) + fill.to(x.dtype) + scale.to(x.dtype)

        traced = minimal_fx_tracer(f)(
            {}, torch.zeros(4, dtype=torch.float32, device=self.DEVICE)
        )

        cpu_attr_names = [
            n.target
            for n in traced.gm.graph.find_nodes(op="get_attr")
            if isinstance(getattr(traced.gm, n.target, None), torch.Tensor)
            and getattr(traced.gm, n.target).device.type == "cpu"
        ]

        gm = full_inductor_compilation_pass(traced.gm, traced.example_inputs)

        for name in cpu_attr_names:
            attr = getattr(traced.gm, name, None)
            self.assertIsInstance(attr, torch.Tensor)
            self.assertEqual(
                attr.device.type,
                "cuda",
                f"{name} should have been migrated to CUDA",
            )

        gm = cudagraph_pass(gm, traced.example_inputs)
        real_x = torch.zeros(4, dtype=torch.float32, device=self.DEVICE)
        expected = f({}, real_x.clone())
        for _ in range(3):
            actual = gm(real_x.clone())
            if isinstance(actual, (list, tuple)):
                actual = actual[0]
            torch.testing.assert_close(actual, expected)


class TestMetadataPropagationUnit(unittest.TestCase):
    """CPU-only unit tests for _copy_fwd_metadata_to_bw_nodes."""

    def test_bracketed_dangling_backward_seq_uses_next_forward_seq(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")

        fwd_prev = graph.call_function(torch.ops.aten.add.Tensor, args=(x, x))
        fwd_prev.meta["seq_nr"] = 10
        fwd_prev.meta["custom"] = {"module_fqn": "prev"}

        fwd_target = graph.call_function(torch.ops.aten.mul.Tensor, args=(fwd_prev, 2))
        fwd_target.meta["seq_nr"] = 12
        fwd_target.meta["custom"] = {"module_fqn": "loss"}
        fwd_target.meta["nn_module_stack"] = {"loss": ("loss", "Loss")}
        fwd_target.meta["stack_trace"] = "loss stack"

        fwd_after = graph.call_function(torch.ops.aten.sub.Tensor, args=(fwd_target, 1))
        fwd_after.meta["seq_nr"] = 13
        fwd_after.meta["custom"] = {"module_fqn": "after"}

        bwd_dangling = graph.call_function(torch.ops.aten.neg.default, args=(fwd_after,))
        bwd_dangling.meta["seq_nr"] = 11
        bwd_dangling.meta["autograd_backward"] = True

        bwd_unbracketed = graph.call_function(torch.ops.aten.relu.default, args=(fwd_after,))
        bwd_unbracketed.meta["seq_nr"] = 99
        bwd_unbracketed.meta["autograd_backward"] = True

        graph.output((bwd_dangling, bwd_unbracketed))
        gm = torch.fx.GraphModule(nn.Module(), graph)

        _copy_fwd_metadata_to_bw_nodes(gm)

        self.assertEqual(bwd_dangling.meta["custom"]["module_fqn"], "loss")
        self.assertEqual(
            bwd_dangling.meta["nn_module_stack"],
            {"loss": ("loss", "Loss")},
        )
        self.assertEqual(bwd_dangling.meta["stack_trace"], "loss stack")
        self.assertNotIn("custom", bwd_unbracketed.meta)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMetadataPropagation(unittest.TestCase):
    """Tests for _copy_fwd_metadata_to_bw_nodes."""

    DEVICE = "cuda"
    DTYPE = torch.float32

    def setUp(self):
        torch.manual_seed(42)

    def test_backward_nodes_have_seq_nr(self):
        """Verify that backward FX nodes get seq_nr metadata via patched autograd.grad."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        train_step = make_train_step(model, get_loss)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        traced = minimal_fx_tracer(train_step, module=model)(tokens, labels)

        # Collect seq_nr values from all call_function nodes
        seq_nrs = []
        for node in traced.gm.graph.nodes:
            if node.op == "call_function" and "seq_nr" in node.meta:
                seq_nrs.append(node.meta["seq_nr"])

        # There should be seq_nr values present (both fwd and bwd nodes)
        self.assertGreater(len(seq_nrs), 0, "No seq_nr metadata found on any node")

        # There should be duplicate seq_nrs (fwd and bwd nodes sharing seq_nr)
        counts = Counter(seq_nrs)
        shared = [nr for nr, cnt in counts.items() if cnt > 1]
        self.assertGreater(
            len(shared),
            0,
            "Expected some seq_nr values shared between fwd and bwd nodes",
        )

    def test_copy_fwd_metadata_propagates_custom(self):
        """Verify _copy_fwd_metadata_to_bw_nodes copies custom metadata to bwd nodes."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)

        train_step = make_train_step(model, get_loss)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        traced = minimal_fx_tracer(train_step, module=model)(tokens, labels)
        gm = traced.gm

        # Manually set custom metadata on the first fwd node for each seq_nr
        # to test that _copy_fwd_metadata_to_bw_nodes works
        seq_nr_first: dict[int, torch.fx.Node] = {}
        for node in gm.graph.nodes:
            if node.op == "call_function" and "seq_nr" in node.meta:
                seq_nr = node.meta["seq_nr"]
                if seq_nr not in seq_nr_first:
                    seq_nr_first[seq_nr] = node
                    node.meta["custom"] = {"test_key": "test_value"}

        # Run the copy pass again
        _copy_fwd_metadata_to_bw_nodes(gm)

        def is_backward(node: torch.fx.Node) -> bool:
            return node.meta.get("autograd_backward", False)

        # Check that bwd nodes with shared seq_nr got the custom metadata
        for node in gm.graph.nodes:
            if node.op != "call_function" or "seq_nr" not in node.meta:
                continue
            seq_nr = node.meta["seq_nr"]
            if node is not seq_nr_first.get(seq_nr) and is_backward(node):
                # This is a backward node
                custom = node.meta.get("custom")
                self.assertIsNotNone(
                    custom,
                    f"Backward node {node.name} with seq_nr={seq_nr} missing custom metadata",
                )
                self.assertEqual(custom.get("test_key"), "test_value")

    def test_copy_fwd_metadata_uses_backward_tagging(self):
        graph = torch.fx.Graph()
        fwd = graph.call_function(torch.ops.aten.add.Tensor, args=(1, 2))
        fwd.meta["seq_nr"] = 7
        fwd.meta["custom"] = {"test_key": "test_value"}
        bwd = graph.call_function(torch.ops.aten.mul.Tensor, args=(fwd, 3))
        bwd.meta["seq_nr"] = 7
        bwd.meta["autograd_backward"] = True
        graph.output(bwd)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        _copy_fwd_metadata_to_bw_nodes(gm)

        self.assertEqual(bwd.meta["custom"].get("test_key"), "test_value")

    def test_backward_nodes_have_stack_trace(self):
        """Verify that backward nodes get stack_trace from their forward counterpart."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        train_step = make_train_step(model, get_loss)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        traced = minimal_fx_tracer(train_step, module=model)(tokens, labels)

        # Find backward nodes: nodes sharing a seq_nr with an earlier (forward) node
        seq_nr_first: dict[int, torch.fx.Node] = {}
        bwd_nodes_missing_stack_trace = []
        num_checked = 0
        for node in traced.gm.graph.nodes:
            if node.op != "call_function" or "seq_nr" not in node.meta:
                continue
            seq_nr = node.meta["seq_nr"]
            if seq_nr not in seq_nr_first:
                seq_nr_first[seq_nr] = node
            else:
                # This is a backward node
                fwd_node = seq_nr_first[seq_nr]
                if not fwd_node.stack_trace:
                    continue
                num_checked += 1
                if not node.stack_trace:
                    bwd_nodes_missing_stack_trace.append((node.name, seq_nr))

        self.assertEqual(num_checked, 24)
        self.assertEqual(
            bwd_nodes_missing_stack_trace,
            [],
            f"Backward nodes missing stack_trace: {bwd_nodes_missing_stack_trace}",
        )


# Large head_dims (qwen3 head_dim=128, deepseek qk_head_dim=192) run in bf16:
# the FlexAttention Triton kernel's fp32 shared-memory footprint exceeds the
# H100 default limit (~99KB) -> "InductorError: out of resource:
# triton_tem_fused_flex_attention". bf16 halves the smem so the kernel fits.
# SDPA never hit this; it only surfaced once flex became the default LM backend.
# llama3 (head_dim 16) is small enough to stay in fp32.


def _disable_flex_autotune():
    """Disable FlexAttention max_autotune; returns the originals to restore.

    max_autotune searches flex block sizes that exceed the H100 shared-memory
    limit for larger head_dims (qwen3 head_dim=128, deepseek qk_head_dim=192),
    raising ``InductorError: out of resource: triton_tem_fused_flex_attention``.
    Disabling it falls back to the default block config (which fits) and keeps
    the eager vs regional-inductor kernels consistent. Mirrors
    ``test_bitwise_deterministic.setUp``.
    """
    from torch.nn.attention.flex_attention import flex_attention

    from torchtitan.models.common.attention import FlexAttention

    orig = (FlexAttention.inductor_configs, FlexAttention._compiled_flex_attn)
    FlexAttention.inductor_configs = {
        **FlexAttention.inductor_configs,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
    }
    FlexAttention._compiled_flex_attn = torch.compile(
        flex_attention, options=FlexAttention.inductor_configs
    )
    return orig


def _restore_flex_autotune(orig):
    from torchtitan.models.common.attention import FlexAttention

    FlexAttention.inductor_configs, FlexAttention._compiled_flex_attn = orig


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTraceModels(unittest.TestCase):
    DEVICE = "cuda"
    DTYPE = torch.float32
    BATCH_SIZE = 2
    SEQ_LEN = 128
    NUM_STEPS = 5
    LR = 1e-3

    def setUp(self):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        self._flex_orig = _disable_flex_autotune()

    def tearDown(self):
        _restore_flex_autotune(self._flex_orig)
        torch.use_deterministic_algorithms(False)

    def _run_bitwise_test(
        self,
        model_ref,
        model_test,
        fwd_args,
        labels,
        check_collective_ops=False,
        use_regional_inductor=False,
        num_steps=5,
        lr=1e-3,
    ):
        train_step = make_train_step(model_ref, get_loss)

        maybe_register_blockmask_pytree_node()
        traced: TracedResult = minimal_fx_tracer(train_step, module=model_ref)(
            *fwd_args, labels
        )

        if check_collective_ops:
            ag = sum(
                1
                for n in traced.gm.graph.nodes
                if "all_gather_into_tensor" in str(n.target)
            )
            rs = sum(
                1
                for n in traced.gm.graph.nodes
                if "reduce_scatter_tensor" in str(n.target)
            )
            self.assertTrue(
                ag > 0 and rs > 0,
                f"Expected collective ops in FSDP graph (ag={ag}, rs={rs})",
            )

        if use_regional_inductor:
            _apply_regional_inductor(traced)

        opt_ref = torch.optim.Adam(model_ref.parameters(), lr=lr)
        opt_copy = torch.optim.Adam(model_test.parameters(), lr=lr)

        for step in range(1, num_steps + 1):
            logits_ref = model_ref(*fwd_args)
            loss_ref = get_loss(logits_ref, labels)
            loss_ref.backward()
            grads_ref = [p.grad.clone() for p in model_ref.parameters()]
            opt_ref.step()
            opt_ref.zero_grad()

            wrapped = run_traced(traced, module=model_test)(*fwd_args, labels)
            loss_tr = wrapped[0]
            grads_tr = wrapped[1:]
            for p, g in zip(model_test.parameters(), grads_tr, strict=True):
                p.grad = g
            opt_copy.step()
            opt_copy.zero_grad()

            self.assertTrue(
                torch.equal(loss_ref, loss_tr), f"Step {step}: loss mismatch"
            )
            for gr, gt in zip(grads_ref, grads_tr, strict=True):
                self.assertTrue(torch.equal(gr, gt), f"Step {step}: grad mismatch")

    def _run_model_test(
        self,
        config_cls,
        model_config,
        use_attn_masks=False,
        use_regional_inductor=False,
        dtype=None,
    ):
        dtype = dtype or self.DTYPE
        vocab_size = model_config.vocab_size
        model_ref = create_model(config_cls, model_config, self.DEVICE, dtype)
        model_test = create_model(config_cls, model_config, self.DEVICE, dtype)
        model_test.load_state_dict(model_ref.state_dict())
        tokens = torch.randint(
            0, vocab_size, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        labels = torch.randint(
            0, vocab_size, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )

        fwd_args = (tokens,)
        if use_attn_masks:
            from torchtitan.models.common.attention import (
                create_attention_mask,
                get_causal_mask_mod,
            )

            attn_masks = create_attention_mask(
                get_causal_mask_mod(), 1, None, self.SEQ_LEN, self.SEQ_LEN
            )
            # Decoder.forward is (tokens, positions, attention_masks). Pass
            # explicit sequential positions (make_fx can't trace a None
            # placeholder) so the BlockMask lands in the attention_masks slot.
            positions = torch.arange(self.SEQ_LEN, device=self.DEVICE).repeat(
                self.BATCH_SIZE, 1
            )
            fwd_args = (tokens, positions, attn_masks)

        self._run_bitwise_test(
            model_ref,
            model_test,
            fwd_args,
            labels,
            use_regional_inductor=use_regional_inductor,
            num_steps=self.NUM_STEPS,
            lr=self.LR,
        )

    def test_llama3(self):
        from torchtitan.models.llama3 import llama3_configs, Llama3Model

        config = llama3_configs["debugmodel"](attn_backend="flex")
        self._run_model_test(
            Llama3Model, config, use_attn_masks=True, use_regional_inductor=True
        )

    def test_qwen3(self):
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.model import Qwen3Model

        config = qwen3_configs["debugmodel"](attn_backend="flex")
        self._run_model_test(
            Qwen3Model,
            config,
            use_attn_masks=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
        )

    def test_qwen3_moe(self):
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.model import Qwen3Model

        config = qwen3_configs["debugmodel_moe"](attn_backend="flex")
        self._run_model_test(
            Qwen3Model,
            config,
            use_attn_masks=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
        )

    def test_deepseek_v3(self):
        from torchtitan.models.deepseek_v3 import deepseekv3_configs
        from torchtitan.models.deepseek_v3.model import DeepSeekV3Model

        config = deepseekv3_configs["debugmodel"](
            attn_backend="flex", moe_comm_backend="standard"
        )
        self._run_model_test(
            DeepSeekV3Model,
            config,
            use_attn_masks=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
        )

    def test_deepseek_v3_flex_attention(self):
        """Tests if we can propagate fwd node metadata reliably through backward.
        Annotates FlexAttention.forward via annotate_fn before
        tracing so compile_with_inductor flows into the graph naturally.
        """
        from torch.fx.traceback import annotate_fn
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.models.common.attention import (
            create_attention_mask,
            FlexAttention,
            get_causal_mask_mod,
            get_document_mask_mod,
        )
        from torchtitan.models.common.nn_modules import Linear, RMSNorm
        from torchtitan.models.common.rope import ComplexRoPE
        from torchtitan.models.deepseek_v3.model import Attention as DSAttention

        dim = 64
        n_heads = 4
        rope_dim = 16
        seq_len = 64
        vocab_size = 128

        # Build a tiny model: embedding -> MLA flex attention -> projection
        class TinyFlexMLA(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, dim)
                kv_lora_rank = 32
                qk_nope_head_dim = 16
                v_head_dim = 16
                qk_head_dim = qk_nope_head_dim + rope_dim
                self.attn = DSAttention(
                    DSAttention.Config(
                        n_heads=n_heads,
                        dim=dim,
                        q_lora_rank=0,
                        kv_lora_rank=kv_lora_rank,
                        qk_nope_head_dim=qk_nope_head_dim,
                        qk_rope_head_dim=rope_dim,
                        v_head_dim=v_head_dim,
                        rope=ComplexRoPE.Config(
                            dim=rope_dim,
                            max_seq_len=seq_len,
                            scaling="none",
                        ),
                        q_norm=RMSNorm.Config(normalized_shape=1),
                        kv_norm=RMSNorm.Config(normalized_shape=kv_lora_rank),
                        inner_attention=FlexAttention.Config(),
                        wq=Linear.Config(
                            in_features=dim,
                            out_features=n_heads * qk_head_dim,
                        ),
                        wkv_a=Linear.Config(
                            in_features=dim,
                            out_features=kv_lora_rank + rope_dim,
                        ),
                        wkv_b=Linear.Config(
                            in_features=kv_lora_rank,
                            out_features=n_heads * (qk_nope_head_dim + v_head_dim),
                        ),
                        wo=Linear.Config(
                            in_features=n_heads * v_head_dim,
                            out_features=dim,
                        ),
                    ),
                )
                self.proj = nn.Linear(dim, vocab_size)

            def init_states(self, buffer_device=None):
                self.attn.rope._init_self_buffers(
                    buffer_device=buffer_device or torch.device("cuda")
                )

            def forward(self, tokens, block_mask):
                x = self.embed(tokens)
                x = self.attn(x, block_mask)
                return self.proj(x)

        model = TinyFlexMLA().to(device=self.DEVICE, dtype=self.DTYPE)
        with torch.no_grad():
            model.init_states(buffer_device=torch.device(self.DEVICE))

        tokens = torch.randint(0, vocab_size, (1, seq_len), device=self.DEVICE)
        labels = torch.randint(0, vocab_size, (1, seq_len), device=self.DEVICE)
        # Build positions that reset to 0 every 16 tokens (document boundaries)
        positions = torch.arange(seq_len, device=self.DEVICE) % 16
        positions = positions.unsqueeze(0)  # [1, seq_len]
        block_mask = create_attention_mask(
            and_masks(get_causal_mask_mod(), get_document_mask_mod(positions)),
            B=1,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
        )

        # Annotate FlexAttention.forward so compile_with_inductor flows into
        # the traced graph. Restore the original after tracing.
        orig_forward = FlexAttention.forward
        FlexAttention.forward = annotate_fn(
            {
                "compile_with_inductor": {
                    "inductor_configs": FlexAttention.inductor_configs
                }
            }
        )(FlexAttention.forward)
        try:
            train_step = make_train_step(model, get_loss)
            maybe_register_blockmask_pytree_node()
            traced = minimal_fx_tracer(train_step, module=model)(
                tokens, block_mask, labels
            )
        finally:
            FlexAttention.forward = orig_forward

        # Verify flex attention HOPs got the annotation
        for node in traced.gm.graph.nodes:
            if node.target in {
                torch.ops.higher_order.flex_attention,
                torch.ops.higher_order.flex_attention_backward,
            }:
                custom = node.meta.get("custom", {})
                self.assertIn(
                    "compile_with_inductor",
                    custom,
                    f"{node.name} missing compile_with_inductor annotation",
                )

    # TODO: Fix scatter() dtype mismatch — scatter_add expects self.dtype == src.dtype
    # but GptOss produces mismatched dtypes during tracing.
    @unittest.skip("scatter(): Expected self.dtype to be equal to src.dtype")
    def test_gpt_oss(self):
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.models.common.attention import (
            create_attention_mask,
            get_causal_mask_mod,
            get_sliding_window_mask_mod,
        )
        from torchtitan.models.gpt_oss import gptoss_configs
        from torchtitan.models.gpt_oss.model import GptOssModel

        config = gptoss_configs["debugmodel"](
            moe_comm_backend="standard", attn_backend="flex"
        )
        vocab_size = config.vocab_size
        model_ref = create_model(GptOssModel, config, self.DEVICE, self.DTYPE)
        model_test = create_model(GptOssModel, config, self.DEVICE, self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())
        tokens = torch.randint(
            0, vocab_size, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        labels = torch.randint(
            0, vocab_size, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        causal = get_causal_mask_mod()
        sw_size = config.layers[0].attention.sliding_window_size
        basic_mask = create_attention_mask(causal, 1, None, self.SEQ_LEN, self.SEQ_LEN)
        sliding_window_mask = create_attention_mask(
            and_masks(causal, get_sliding_window_mask_mod(sw_size)),
            1,
            None,
            self.SEQ_LEN,
            self.SEQ_LEN,
        )
        attn_masks = {
            "basic_mask": basic_mask,
            "sliding_window_mask": sliding_window_mask,
        }
        self._run_bitwise_test(
            model_ref,
            model_test,
            (tokens, attn_masks),
            labels,
            use_regional_inductor=True,
            num_steps=self.NUM_STEPS,
            lr=self.LR,
        )

    def test_flex_attention_annotations(self):
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.experiments.graph_trainer.common_utils import (
            annotate_module_fqns,
        )
        from torchtitan.models.common.attention import (
            create_attention_mask,
            get_causal_mask_mod,
            get_sliding_window_mask_mod,
        )
        from torchtitan.models.gpt_oss import gptoss_configs
        from torchtitan.models.gpt_oss.model import GptOssModel

        config = gptoss_configs["debugmodel"](
            moe_comm_backend="standard", attn_backend="flex"
        )
        model = create_model(GptOssModel, config, self.DEVICE, self.DTYPE)
        annotate_module_fqns(model)

        tokens = torch.randint(
            0, config.vocab_size, (self.BATCH_SIZE, self.SEQ_LEN), device=self.DEVICE
        )
        causal = get_causal_mask_mod()
        sw_size = config.layers[0].attention.sliding_window_size
        basic_mask = create_attention_mask(causal, 1, None, self.SEQ_LEN, self.SEQ_LEN)
        sliding_window_mask = create_attention_mask(
            and_masks(causal, get_sliding_window_mask_mod(sw_size)),
            1,
            None,
            self.SEQ_LEN,
            self.SEQ_LEN,
        )
        attn_masks = {
            "basic_mask": basic_mask,
            "sliding_window_mask": sliding_window_mask,
        }
        maybe_register_blockmask_pytree_node()

        def forward(tokens, attn_masks):
            return model(tokens, attention_masks=attn_masks)

        traced = minimal_fx_tracer(forward, module=model)(tokens, attn_masks)

        flex_nodes = [
            n
            for n in traced.gm.graph.nodes
            if "flex_attention" in str(n.target) and "backward" not in str(n.target)
        ]
        self.assertGreater(len(flex_nodes), 0, "No FlexAttentionHOP nodes found")

        from torchtitan.models.common.attention import FlexAttention

        annotate_flex_attention_for_regional_inductor_pass(
            traced.gm,
            flex_compile_config=FlexAttention.inductor_configs,
        )

        for node in flex_nodes:
            custom = node.meta.get("custom", {})
            self.assertIn(
                "compile_with_inductor",
                custom,
                f"{node.name} missing compile_with_inductor annotation",
            )


class TestTraceFSDP(FSDPTest):
    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 4)

    def _setup(self):
        from torchtitan.distributed import ParallelDims

        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )

    def _run_fsdp_model_test(
        self,
        config_cls,
        model_config,
        use_attn_masks=False,
        attn_masks=None,
        use_regional_inductor=False,
        dtype=torch.float32,
    ):
        from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        self._setup()
        # FSDPTest.setUp runs in the parent, so disable flex max_autotune here
        # (in the child process) to keep flex kernels within the H100 shared
        # memory limit. No restore needed: each rank is a fresh subprocess.
        _disable_flex_autotune()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")

        model_ref = create_model(config_cls, model_config, "cuda", dtype)
        model_test = create_model(config_cls, model_config, "cuda", dtype)
        model_test.load_state_dict(model_ref.state_dict())
        model_ref = data_parallel(model_ref, device_mesh=fsdp_mesh, mode="fully_shard")
        model_test = data_parallel(
            model_test, device_mesh=fsdp_mesh, mode="fully_shard"
        )

        vocab_size = model_config.vocab_size
        seq_len = 128
        tokens = torch.randint(0, vocab_size, (2, seq_len), device="cuda")
        labels = torch.randint(0, vocab_size, (2, seq_len), device="cuda")
        # Decoder.forward is (tokens, positions, attention_masks). Pass explicit
        # sequential positions (make_fx can't trace a None placeholder) so the
        # BlockMask lands in the attention_masks slot.
        positions = torch.arange(seq_len, device="cuda").repeat(2, 1)

        if attn_masks is not None:
            fwd_args = (tokens, positions, attn_masks)
        elif use_attn_masks:
            from torchtitan.models.common.attention import (
                create_attention_mask,
                get_causal_mask_mod,
            )

            attn_masks = create_attention_mask(
                get_causal_mask_mod(), 1, None, seq_len, seq_len
            )
            fwd_args = (tokens, positions, attn_masks)
        else:
            fwd_args = (tokens,)

        train_step = make_train_step(model_ref, get_loss)

        maybe_register_blockmask_pytree_node()
        traced = minimal_fx_tracer(train_step, module=model_ref)(*fwd_args, labels)

        ag = sum(
            1
            for n in traced.gm.graph.nodes
            if "all_gather_into_tensor" in str(n.target)
        )
        rs = sum(
            1 for n in traced.gm.graph.nodes if "reduce_scatter_tensor" in str(n.target)
        )
        self.assertTrue(
            ag > 0 and rs > 0,
            f"Expected collective ops in FSDP graph (ag={ag}, rs={rs})",
        )

        if use_regional_inductor:
            _apply_regional_inductor(traced)

        opt_ref = torch.optim.Adam(model_ref.parameters(), lr=1e-3)
        opt_copy = torch.optim.Adam(model_test.parameters(), lr=1e-3)

        for step in range(1, 6):
            logits_ref = model_ref(*fwd_args)
            loss_ref = get_loss(logits_ref, labels)
            loss_ref.backward()
            grads_ref = [p.grad.clone() for p in model_ref.parameters()]
            opt_ref.step()
            opt_ref.zero_grad()

            wrapped = run_traced(traced, module=model_test)(*fwd_args, labels)
            loss_tr = wrapped[0]
            grads_tr = wrapped[1:]
            for p, g in zip(model_test.parameters(), grads_tr, strict=True):
                p.grad = g
            opt_copy.step()
            opt_copy.zero_grad()

            self.assertTrue(
                torch.equal(loss_ref, loss_tr), f"Step {step}: loss mismatch"
            )
            for gr, gt in zip(grads_ref, grads_tr, strict=True):
                self.assertTrue(torch.equal(gr, gt), f"Step {step}: grad mismatch")

    def test_llama3_fsdp(self):
        from torchtitan.models.llama3 import llama3_configs, Llama3Model

        config = llama3_configs["debugmodel"](attn_backend="flex")
        self._run_fsdp_model_test(
            Llama3Model, config, use_attn_masks=True, use_regional_inductor=True
        )

    def test_qwen3_fsdp(self):
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.model import Qwen3Model

        config = qwen3_configs["debugmodel"](attn_backend="flex")
        self._run_fsdp_model_test(
            Qwen3Model,
            config,
            use_attn_masks=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
        )

    def test_deepseek_v3_fsdp(self):
        from torchtitan.models.deepseek_v3 import deepseekv3_configs
        from torchtitan.models.deepseek_v3.model import DeepSeekV3Model

        config = deepseekv3_configs["debugmodel"](
            attn_backend="flex", moe_comm_backend="standard"
        )
        self._run_fsdp_model_test(
            DeepSeekV3Model,
            config,
            use_attn_masks=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
        )

    # TODO: Fix scatter() dtype mismatch — same root cause as TestTraceModels.test_gpt_oss.
    @unittest.skip("scatter(): Expected self.dtype to be equal to src.dtype")
    def test_gpt_oss_fsdp(self):
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.models.common.attention import (
            create_attention_mask,
            get_causal_mask_mod,
            get_sliding_window_mask_mod,
        )
        from torchtitan.models.gpt_oss import gptoss_configs
        from torchtitan.models.gpt_oss.model import GptOssModel

        config = gptoss_configs["debugmodel"](
            moe_comm_backend="standard", attn_backend="flex"
        )
        seq_len = 128
        causal = get_causal_mask_mod()
        sw_size = config.layers[0].attention.sliding_window_size
        basic_mask = create_attention_mask(causal, 1, None, seq_len, seq_len)
        sliding_window_mask = create_attention_mask(
            and_masks(causal, get_sliding_window_mask_mod(sw_size)),
            1,
            None,
            seq_len,
            seq_len,
        )
        attn_masks = {
            "basic_mask": basic_mask,
            "sliding_window_mask": sliding_window_mask,
        }
        self._run_fsdp_model_test(
            GptOssModel,
            config,
            attn_masks=attn_masks,
            use_regional_inductor=True,
        )


@unittest.skipIf(torch.cuda.device_count() < 2, "CP trace test requires 2 GPUs")
class TestTraceContextParallel(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _trace_llama3_step_code(
        self,
        *,
        dp_shard_degree: int,
        context_parallel_degree: int,
    ) -> dict[str, object]:
        import os
        import tempfile

        import torch.distributed as dist

        from torchtitan.experiments.graph_trainer.llama3.config_registry import (
            graph_trainer_llama3_debugmodel_sdpa,
        )
        from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

        old_local_rank = os.environ.get("LOCAL_RANK")
        os.environ["LOCAL_RANK"] = str(dist.get_rank() % torch.cuda.device_count())

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        trainer = None
        try:
            with tempfile.TemporaryDirectory() as dump_folder:
                config = graph_trainer_llama3_debugmodel_sdpa()
                config.dump_folder = dump_folder
                config.training.local_batch_size = 2
                config.training.seq_len = 128
                config.training.steps = 1
                config.parallelism.data_parallel_replicate_degree = 1
                config.parallelism.data_parallel_shard_degree = dp_shard_degree
                config.parallelism.context_parallel_degree = context_parallel_degree
                config.parallelism.tensor_parallel_degree = 1
                config.activation_checkpoint = None
                config.compile.enable = False
                config.compile.enable_passes = False
                config.debug.enable_structured_logging = False
                config.model_spec.model.layers = config.model_spec.model.layers[:1]

                trainer = GraphTrainer(config)
                tokens = torch.randint(
                    0,
                    trainer.model_config.vocab_size,
                    (config.training.local_batch_size, config.training.seq_len),
                    device=trainer.device,
                )
                labels = torch.randint(
                    0,
                    trainer.model_config.vocab_size,
                    (config.training.local_batch_size, config.training.seq_len),
                    device=trainer.device,
                )
                # The dataloader always supplies per-document positions, which
                # drive RoPE (SDPA itself is maskless and uses is_causal).
                positions = (
                    torch.arange(
                        config.training.seq_len,
                        device=trainer.device,
                        dtype=torch.int32,
                    )
                    .unsqueeze(0)
                    .expand(config.training.local_batch_size, config.training.seq_len)
                )
                trainer.forward_backward_step(
                    input_dict={"input": tokens, "positions": positions},
                    labels=labels,
                    global_valid_tokens=torch.tensor(
                        labels.numel(), device=trainer.device
                    ),
                )
                assert trainer._traced_step is not None
                code_lines = trainer._traced_step.gm.graph.python_code(
                    "self"
                ).src.splitlines()
                sdpa_line = next(
                    (
                        idx
                        for idx, line in enumerate(code_lines)
                        if "scaled_dot_product" in line
                    ),
                    None,
                )
                self.assertIsNotNone(
                    sdpa_line,
                    "Expected SDPA in generated code:\n" + "\n".join(code_lines),
                )
                assert sdpa_line is not None
                all_gather_pg_names_before_sdpa = []
                for node in trainer._traced_step.gm.graph.nodes:
                    if "scaled_dot_product" in str(node.target):
                        break
                    if "all_gather_into_tensor" in str(node.target):
                        all_gather_pg_names_before_sdpa.append(node.args[2])

                cp_pg_name = (
                    trainer.parallel_dims.get_mesh("cp").get_group().group_name
                    if trainer.parallel_dims.cp_enabled
                    else None
                )
                fsdp_pg_name = (
                    trainer.parallel_dims.get_mesh("fsdp").get_group().group_name
                )
                code = trainer._traced_step.gm.graph.python_code("self").src
                trainer.close()
                trainer = None
                return {
                    "code": code,
                    "all_gather_pg_names_before_sdpa": (
                        all_gather_pg_names_before_sdpa
                    ),
                    "cp_pg_name": cp_pg_name,
                    "fsdp_pg_name": fsdp_pg_name,
                }
        finally:
            if trainer is not None:
                trainer.close()
            if old_local_rank is None:
                os.environ.pop("LOCAL_RANK", None)
            else:
                os.environ["LOCAL_RANK"] = old_local_rank

    # Pinned to the SDPA backend: this validates CP all_gather-before-SDPA
    # codegen, which requires the scaled_dot_product op. The default
    # FlexAttention backend has no SDPA op and flex + CP is unsupported anyway
    # (torch's _create_cp_block_mask requires seq_len divisible by 2 *
    # BLOCK_SIZE, here 128 < 256; see the aot_fx_trace_llama3_fsdp_tp_cp
    # integration flavor). SDPA has native CP support and emits the SDPA op.
    def test_llama3_cp_only_codegen_all_gather_before_sdpa(self):
        cp_trace = self._trace_llama3_step_code(
            dp_shard_degree=1,
            context_parallel_degree=2,
        )
        # Verify AG along CP PG exists before SDPA
        self.assertIn(
            cp_trace["cp_pg_name"],
            cp_trace["all_gather_pg_names_before_sdpa"],
            "Expected CP all_gather on the CP mesh before SDPA. "
            f"CP pg={cp_trace['cp_pg_name']}, "
            f"FSDP pg={cp_trace['fsdp_pg_name']}, "
            "pre-SDPA all_gather pgs="
            f"{cp_trace['all_gather_pg_names_before_sdpa']}.\n"
            f"Generated code:\n{cp_trace['code']}",
        )


class TestAutogradGradVsBackwardFSDP(FSDPTest):
    """Verify autograd.grad() and loss.backward() have identical peak memory with FSDP."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 4)

    def test_peak_memory_identical_fsdp(self):
        from torchtitan.distributed import ParallelDims
        from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
        from torchtitan.models.llama3 import llama3_configs, Llama3Model

        config = llama3_configs["debugmodel"](attn_backend="flex")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)

        try:
            parallel_dims = ParallelDims(
                dp_shard=-1,
                dp_replicate=1,
                cp=1,
                tp=1,
                pp=1,
                ep=1,
                world_size=self.world_size,
            )
            fsdp_mesh = parallel_dims.get_mesh("fsdp")

            model_backward = create_model(Llama3Model, config, "cuda", torch.bfloat16)
            model_grad = create_model(Llama3Model, config, "cuda", torch.bfloat16)
            model_grad.load_state_dict(model_backward.state_dict())
            model_backward = data_parallel(
                model_backward, device_mesh=fsdp_mesh, mode="fully_shard"
            )
            model_grad = data_parallel(
                model_grad, device_mesh=fsdp_mesh, mode="fully_shard"
            )

            tokens = torch.randint(0, config.vocab_size, (2, 128), device="cuda")
            labels = torch.randint(0, config.vocab_size, (2, 128), device="cuda")

            from torchtitan.models.common.attention import (
                create_attention_mask,
                get_causal_mask_mod,
            )

            attention_masks = create_attention_mask(
                get_causal_mask_mod(), 1, None, 128, 128
            )

            def run_backward(model):
                logits = model(tokens, attention_masks=attention_masks)
                loss = get_loss(logits, labels)
                loss.backward()

            def run_grad(model):
                logits = model(tokens, attention_masks=attention_masks)
                loss = get_loss(logits, labels)
                params = [p for p in model.parameters() if p.requires_grad]
                grads = torch.autograd.grad(loss, params)
                for p, g in zip(params, grads, strict=True):
                    p.grad = g

            # Warmup
            run_backward(model_backward)
            model_backward.zero_grad()
            run_grad(model_grad)
            model_grad.zero_grad()
            torch.cuda.empty_cache()

            # Measure backward()
            torch.cuda.reset_peak_memory_stats()
            run_backward(model_backward)
            peak_backward = torch.cuda.max_memory_allocated()
            model_backward.zero_grad()
            torch.cuda.empty_cache()

            # Measure autograd.grad()
            torch.cuda.reset_peak_memory_stats()
            run_grad(model_grad)
            peak_grad = torch.cuda.max_memory_allocated()
            model_grad.zero_grad()
            torch.cuda.empty_cache()

            self.assertEqual(
                peak_backward,
                peak_grad,
                f"Peak memory differs: backward()={peak_backward / 1e9:.2f} GB "
                f"vs autograd.grad()={peak_grad / 1e9:.2f} GB",
            )
        finally:
            torch.use_deterministic_algorithms(prev_deterministic)


if __name__ == "__main__":
    unittest.main()
