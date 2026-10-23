# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import unittest

import torch
from graph_trainer.make_fx_tracer import minimal_fx_tracer, run_traced, TracedResult
from torch import nn
from torch.optim import swap_in_optimizer_params_and_state
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    disable_inplace_wgrad_accum,
    get_simple_fsdp_mesh,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.passes import (
    annotate_flex_attention_for_regional_inductor_pass,
)
from torchtitan.observability.profiler import Profiler


def get_loss(logits, labels):
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        reduction="sum",
    )


def make_train_step(model, loss_fn, *, fwd_kwargs=None):
    """Return a plain function that closes over ``model`` for module-based tracing."""
    fwd_kwargs = fwd_kwargs or {}

    def train_step(*args):
        *fwd_args, labels = args
        logits = model(*fwd_args, **fwd_kwargs)
        loss = loss_fn(logits, labels)
        params = list(model.parameters())
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)

    return train_step


def create_model(config_cls, model_config, device="cuda", dtype=torch.float32):
    # make_fx traces the backward, which cannot add into weight.grad in place.
    disable_inplace_wgrad_accum(model_config)
    model = config_cls(model_config)
    model.to(device=device, dtype=dtype)
    with torch.no_grad():
        model.init_states(buffer_device=torch.device(device))
    return model


def _apply_regional_inductor(traced_result):
    """Apply regional_inductor to compile annotated HOP regions in the traced graph."""
    from torch.fx.graph import CodeGen
    from torch.fx.passes.regional_inductor import regional_inductor

    from torchtitan.models.common.attention import FlexInnerAttention

    annotate_flex_attention_for_regional_inductor_pass(
        traced_result.gm,
        flex_compile_config=FlexInnerAttention.inductor_configs,
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


class TestGraphGradientAccumulation(unittest.TestCase):
    def test_rejects_fsdp2_deferred_gradient_reduction(self):
        from types import SimpleNamespace

        from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine

        engine = object.__new__(GraphTrainingEngine)
        engine.config = SimpleNamespace(
            parallelism=SimpleNamespace(fsdp_defer_gradient_reduction=True)
        )

        with self.assertRaisesRegex(
            ValueError, "does not support fsdp_defer_gradient_reduction"
        ):
            engine._initialize_forward_backward()

    def test_initializes_one_stage_pipeline_runtime(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine

        engine = object.__new__(GraphTrainingEngine)
        engine.config = SimpleNamespace(
            compile=SimpleNamespace(memory_policy="none"),
            dist_moe_runtime=None,
            parallelism=SimpleNamespace(fsdp_defer_gradient_reduction=False),
            profiler=Profiler.Config(),
            sdc_replayer=None,
            training=SimpleNamespace(
                disable_cuda_graphs=False,
                num_tokens_per_microbatch_per_dp_rank=1,
                num_tokens_per_train_step=2,
            ),
        )
        engine.parallelism_context = SimpleNamespace(
            pp_enabled=False,
            dp_replicate=1,
            dp_shard=1,
        )
        engine.model_parts = [MagicMock()]
        engine.device = torch.device("cpu")
        engine.model_config = MagicMock()
        engine.loss_fn = MagicMock()
        stage = SimpleNamespace(is_first=True, is_last=True)
        graph_runtime = SimpleNamespace(
            schedule=SimpleNamespace(_stages=[stage]),
        )

        with (
            patch(
                "torchtitan.experiments.graph_trainer.trainer.make_spmd_graph_runtime",
                return_value=graph_runtime,
            ) as make_runtime,
            patch(
                "torchtitan.experiments.graph_trainer.trainer.cuda_graphs_supported",
                return_value=True,
            ),
            patch(
                "torchtitan.experiments.graph_trainer.trainer."
                "wrap_fwd_bwd_with_cuda_graph",
                return_value=MagicMock(),
            ) as wrap_cuda_graph,
        ):
            engine._initialize_forward_backward()

        self.assertIs(engine.pp_schedule, graph_runtime)
        wrap_cuda_graph.assert_called_once()
        self.assertIs(engine._run_forward_backward, wrap_cuda_graph.return_value)
        make_runtime.assert_called_once_with(
            engine.model_parts[0],
            gradient_accumulation_steps=2,
            parallelism_context=engine.parallelism_context,
            device=engine.device,
            loss_fn=engine.loss_fn,
            trainer_config=engine.config,
        )

    def test_aot_fx_runs_multi_microbatch_group_through_graph_runtime(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine

        model = MagicMock()
        model.preprocess_inputs.side_effect = lambda input_dict, **_: (
            input_dict["input"],
            input_dict["labels"],
            {"positions": input_dict["positions"]},
            {},
        )
        engine = object.__new__(GraphTrainingEngine)
        engine.config = SimpleNamespace(
            compile=SimpleNamespace(),
            parallelism="PARALLELISM",
            training=SimpleNamespace(
                max_context_length=8,
                num_tokens_per_microbatch_per_dp_rank=1,
            ),
        )
        engine.parallelism_context = SimpleNamespace(
            pp_enabled=False,
            cp=1,
            activate_spmd=contextlib.nullcontext,
        )
        engine.model_parts = [model]
        engine.device = torch.device("cpu")
        engine.max_num_documents = None
        engine.preprocess_inputs_kwargs = {}
        engine.ntokens_seen = 0
        microbatches = [
            TokenizedTrainingMicrobatch(
                input=torch.tensor([index]),
                labels=torch.tensor([index + 1]),
                positions=torch.tensor([index + 2]),
                padding_mask=torch.tensor([False]),
                loss_token_counts=torch.tensor(1),
                routing_token_counts=torch.tensor([1]),
            )
            for index in range(2)
        ]

        prepared_groups = engine._preprocess_microbatch_groups(
            [microbatches], global_routing_token_counts=torch.tensor([2])
        )
        engine._pp_forward_backward_microbatch_group = MagicMock(
            return_value=(torch.tensor(3.0), [{}, {}])
        )
        result = engine._forward_backward_body(
            prepared_groups,
            torch.tensor(2),
            defer_fsdp_gradient_reduction=False,
        )

        assert engine.ntokens_seen == 2
        torch.testing.assert_close(result.loss, torch.tensor(3.0))
        assert result.loss_metrics == [[{}, {}]]
        call = engine._pp_forward_backward_microbatch_group.call_args
        assert len(call.kwargs["inputs"]) == 2
        assert len(call.kwargs["model_kwargs"]) == 2
        assert len(call.kwargs["labels"]) == 2
        assert len(call.kwargs["loss_kwargs"]) == 2
        for mb_loss_kwargs in call.kwargs["loss_kwargs"]:
            torch.testing.assert_close(
                mb_loss_kwargs["global_loss_token_counts"], torch.tensor(2)
            )
        assert call.kwargs["finalize_gradients"]

    def test_graph_runtime_receives_shared_loss_kwargs(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine

        engine = object.__new__(GraphTrainingEngine)
        engine.config = SimpleNamespace(debug=SimpleNamespace(spmd_typechecking=False))
        engine.parallelism_context = SimpleNamespace(
            activate_spmd=lambda **kwargs: contextlib.nullcontext()
        )
        engine.pp_has_last_stage = True
        engine.device = torch.device("cpu")

        def step(**kwargs):
            kwargs["losses"].extend([torch.tensor(1.0), torch.tensor(2.0)])

        engine.pp_schedule = SimpleNamespace(step=MagicMock(side_effect=step))
        loss_kwargs = {"global_loss_token_counts": torch.tensor(2)}

        loss, loss_metrics = engine._pp_forward_backward_microbatch_group(
            inputs=[(torch.ones(1),), (torch.ones(1),)],
            labels=[torch.ones(1), torch.ones(1)],
            model_kwargs=[{}, {}],
            loss_kwargs=[loss_kwargs, loss_kwargs],
        )

        call = engine.pp_schedule.step.call_args
        assert call.kwargs["loss_kwargs"] is loss_kwargs
        assert "loss_kwarg_mbs" not in call.kwargs
        torch.testing.assert_close(loss, torch.tensor(3.0))
        assert loss_metrics == [{}, {}]

    def test_aot_fx_preserves_tuple_model_inputs(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from torchtitan.experiments.graph_trainer.trainer import GraphTrainingEngine

        model = MagicMock()
        main_tokens = torch.tensor([1, 2])
        mtp_tokens = torch.tensor([2, 0])
        main_labels = torch.tensor([2, 3])
        mtp_labels = torch.tensor([3, -100])
        model.preprocess_inputs.return_value = (
            (main_tokens, mtp_tokens),
            (main_labels, mtp_labels),
            {"mtp_input_valid_masks": (torch.tensor([True, False]),)},
            {},
        )
        engine = object.__new__(GraphTrainingEngine)
        engine.config = SimpleNamespace(
            parallelism="PARALLELISM",
            training=SimpleNamespace(
                max_context_length=8,
                num_tokens_per_microbatch_per_dp_rank=2,
            ),
        )
        engine.parallelism_context = SimpleNamespace(
            pp_enabled=False,
            cp=1,
            activate_spmd=contextlib.nullcontext,
        )
        engine.model_parts = [model]
        engine.device = torch.device("cpu")
        engine.max_num_documents = None
        engine.preprocess_inputs_kwargs = {}
        engine.ntokens_seen = 0
        microbatch = TokenizedTrainingMicrobatch(
            input=main_tokens,
            labels=main_labels,
            positions=torch.arange(2),
            padding_mask=torch.tensor([False, False]),
            loss_token_counts=torch.tensor([2, 1]),
            routing_token_counts=torch.tensor([2, 1]),
        )

        [(arg_mbs, kwarg_mbs, target_mbs)] = engine._preprocess_microbatch_groups(
            [[microbatch]], global_routing_token_counts=torch.tensor([2, 1])
        )

        assert len(arg_mbs) == 1 and len(arg_mbs[0]) == 1
        prepared_tokens = arg_mbs[0][0]
        assert isinstance(prepared_tokens, tuple)
        assert prepared_tokens[0] is main_tokens
        assert prepared_tokens[1] is mtp_tokens
        prepared_labels = target_mbs[0]
        assert isinstance(prepared_labels, tuple)
        assert prepared_labels[0] is main_labels
        assert prepared_labels[1] is mtp_labels
        (prepared_valid_mask,) = kwarg_mbs[0]["mtp_input_valid_masks"]
        torch.testing.assert_close(prepared_valid_mask, torch.tensor([True, False]))
        assert engine.ntokens_seen == 2


class TestMinimalFXTracerTorchTitanDynamicShapes(unittest.TestCase):
    def test_mark_dynamic_token_dim_with_rope(self):
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

        num_tokens, heads, head_dim = 8, 1, 8
        position_cases = {
            "none": None,
            "flat": torch.arange(num_tokens),
        }

        for name, positions in position_cases.items():
            with self.subTest(positions=name):
                x = torch.randn(num_tokens, heads, head_dim)
                xq = torch.randn(num_tokens, heads, head_dim)
                xk = torch.randn(num_tokens, heads, head_dim)
                freqs = torch.randn(num_tokens * 2, head_dim // 2)
                freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
                rope_cache = torch.randn(num_tokens * 2, head_dim * 2)

                for tensor in (x, xq, xk):
                    mark_dynamic(tensor, 0)
                if positions is not None:
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

    def test_mark_unbacked_positions_token_dim_with_rope(self):
        from torch._dynamo.decorators import mark_unbacked

        from torchtitan.models.common.rope import _reshape_for_broadcast, CosSinRoPE

        def forward(xq, xk, rope_cache, positions):
            cos_sin_cache = _reshape_for_broadcast(rope_cache, xq.shape, positions)
            q, k = CosSinRoPE.apply_rotary_emb(xq, xk, cos_sin_cache)
            return q + k

        num_tokens, heads, head_dim = 16, 1, 8
        xq = torch.randn(num_tokens, heads, head_dim)
        xk = torch.randn(num_tokens, heads, head_dim)
        rope_cache = torch.randn(num_tokens * 2, head_dim * 2)
        positions = torch.arange(num_tokens)
        for tensor in (xq, xk, positions):
            mark_unbacked(
                tensor,
                0,
                hint_override=num_tokens,
                min=1,
                max=num_tokens,
                shape_id="tokens",
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

        num_tokens, head_dim = 5, 8
        xq = torch.randn(num_tokens, 1, head_dim)
        rope_cache = torch.randn(num_tokens * 2, head_dim * 2)
        positions = torch.arange(num_tokens)

        self.assertTrue(
            torch.equal(
                forward(xq, rope_cache, positions),
                _reshape_for_broadcast(rope_cache, xq.shape, positions),
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

    def test_chunked_loss_train_step(self):
        D, V, num_chunks = 32, 257, 4
        lm_head_ref = nn.Linear(D, V, bias=False).to(
            device=self.DEVICE, dtype=self.DTYPE
        )
        lm_head_test = nn.Linear(D, V, bias=False).to(
            device=self.DEVICE, dtype=self.DTYPE
        )
        lm_head_test.load_state_dict(lm_head_ref.state_dict())
        num_tokens = self.BATCH_SIZE * self.SEQ_LEN
        hidden_states = torch.randn(
            num_tokens,
            D,
            device=self.DEVICE,
            dtype=self.DTYPE,
            requires_grad=True,
        )
        labels = torch.randint(0, V, (num_tokens,), device=self.DEVICE)

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
        from torchtitan.components.optim import AdamW, OptimizersContainer

        torch.manual_seed(0)
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        container = OptimizersContainer(
            OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r".*",
                        lr=1e-3,
                        fused=False,
                    )
                ],
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


# Large head_dims (qwen3 head_dim=128, deepseek qk_head_dim=192) run in bf16:
# the FlexInnerAttention Triton kernel's fp32 shared-memory footprint exceeds the
# H100 default limit (~99KB) -> "InductorError: out of resource:
# triton_tem_fused_flex_attention". bf16 halves the smem so the kernel fits.
# SDPA never hit this; it only surfaced once flex became the default LM backend.
# llama3 (head_dim 16) is small enough to stay in fp32.


def _disable_flex_autotune():
    """Disable FlexInnerAttention max_autotune; returns the originals to restore.

    max_autotune searches flex block sizes that exceed the H100 shared-memory
    limit for larger head_dims (qwen3 head_dim=128, deepseek qk_head_dim=192),
    raising ``InductorError: out of resource: triton_tem_fused_flex_attention``.
    Disabling it falls back to the default block config (which fits) and keeps
    the eager vs regional-inductor kernels consistent. Mirrors
    ``test_bitwise_deterministic.setUp``.
    """
    from torch.nn.attention.flex_attention import flex_attention

    from torchtitan.models.common.attention import FlexInnerAttention

    orig = (FlexInnerAttention.inductor_configs, FlexInnerAttention._compiled_flex_attn)
    FlexInnerAttention.inductor_configs = {
        **FlexInnerAttention.inductor_configs,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
    }
    FlexInnerAttention._compiled_flex_attn = torch.compile(
        flex_attention, options=FlexInnerAttention.inductor_configs
    )
    return orig


def _restore_flex_autotune(orig):
    from torchtitan.models.common.attention import FlexInnerAttention

    FlexInnerAttention.inductor_configs, FlexInnerAttention._compiled_flex_attn = orig


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
        fwd_kwargs=None,
    ):
        fwd_kwargs = fwd_kwargs or {}
        train_step = make_train_step(model_ref, get_loss, fwd_kwargs=fwd_kwargs)

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
            logits_ref = model_ref(*fwd_args, **fwd_kwargs)
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
        use_attention_metadata=False,
        use_regional_inductor=False,
        dtype=None,
        fwd_kwargs=None,
    ):
        dtype = dtype or self.DTYPE
        vocab_size = model_config.vocab_size
        model_ref = create_model(config_cls, model_config, self.DEVICE, dtype)
        model_test = create_model(config_cls, model_config, self.DEVICE, dtype)
        model_test.load_state_dict(model_ref.state_dict())
        num_tokens = self.BATCH_SIZE * self.SEQ_LEN
        tokens = torch.randint(0, vocab_size, (num_tokens,), device=self.DEVICE)
        labels = torch.randint(0, vocab_size, (num_tokens,), device=self.DEVICE)

        fwd_args = (tokens,)
        if use_attention_metadata:
            positions = torch.arange(num_tokens, device=self.DEVICE)
            attention_metadata = model_ref._get_attention_metadata(positions)
            fwd_args = (tokens, positions, attention_metadata)

        self._run_bitwise_test(
            model_ref,
            model_test,
            fwd_args,
            labels,
            use_regional_inductor=use_regional_inductor,
            num_steps=self.NUM_STEPS,
            lr=self.LR,
            fwd_kwargs=fwd_kwargs,
        )

    def test_llama3(self):
        from torchtitan.models.llama3 import Llama3Model, MODEL_FLAVORS

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        self._run_model_test(
            Llama3Model,
            config,
            use_attention_metadata=True,
            use_regional_inductor=True,
        )

    def test_qwen3(self):
        from torchtitan.models.qwen3 import MODEL_FLAVORS
        from torchtitan.models.qwen3.model import Qwen3Model

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        self._run_model_test(
            Qwen3Model,
            config,
            use_attention_metadata=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
        )

    def test_qwen3_moe(self):
        from torchtitan.models.qwen3 import MODEL_FLAVORS
        from torchtitan.models.qwen3.model import Qwen3Model

        build_config, max_context_length = MODEL_FLAVORS["debugmodel_moe"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        self._run_model_test(
            Qwen3Model,
            config,
            use_attention_metadata=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
            fwd_kwargs={
                "aux_loss_denominators": torch.tensor(
                    [self.BATCH_SIZE * self.SEQ_LEN], device=self.DEVICE
                )
            },
        )

    def test_deepseek_v3(self):
        from torchtitan.models.deepseek_v3 import MODEL_FLAVORS
        from torchtitan.models.deepseek_v3.model import DeepSeekV3Model

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        self._run_model_test(
            DeepSeekV3Model,
            config,
            use_attention_metadata=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
            fwd_kwargs={
                "aux_loss_denominators": torch.tensor(
                    [self.BATCH_SIZE * self.SEQ_LEN], device=self.DEVICE
                )
            },
        )

    def test_deepseek_v3_flex_attention(self):
        """Tests if we can propagate fwd node metadata reliably through backward.
        Annotates FlexInnerAttention.forward via annotate_fn before
        tracing so compile_with_inductor flows into the graph naturally.
        """
        from torch.fx.traceback import annotate_fn
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.models.common.attention import (
            create_attention_mask,
            FlexInnerAttention,
            get_causal_mask_mod,
            get_document_mask_mod,
            MLAAttention,
        )
        from torchtitan.models.common.linear import Linear
        from torchtitan.models.common.nn_modules import RMSNorm
        from torchtitan.models.common.rope import ComplexRoPE
        from torchtitan.models.deepseek_v3.attention import DeepSeekMLAAttention

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
                attn_config = DeepSeekMLAAttention.Config(
                    n_heads=n_heads,
                    dim=dim,
                    q_lora_rank=0,
                    kv_lora_rank=kv_lora_rank,
                    qk_nope_head_dim=qk_nope_head_dim,
                    qk_rope_head_dim=rope_dim,
                    v_head_dim=v_head_dim,
                    rope=ComplexRoPE.Config(
                        dim=rope_dim,
                        max_context_length=seq_len,
                        scaling="none",
                    ),
                    q_norm=RMSNorm.Config(normalized_shape=1),
                    kv_norm=RMSNorm.Config(normalized_shape=kv_lora_rank),
                    wq=Linear.Config(
                        in_features=dim,
                        out_features=n_heads * qk_head_dim,
                    ),
                    wkv_a=Linear.Config(
                        in_features=dim,
                        out_features=kv_lora_rank + rope_dim,
                    ),
                    inner_attention=MLAAttention.Config(
                        wkv_b=Linear.Config(
                            in_features=kv_lora_rank,
                            out_features=n_heads * (qk_nope_head_dim + v_head_dim),
                        ),
                        packed_kv_head_dim=qk_nope_head_dim + v_head_dim,
                        inner_attention=FlexInnerAttention.Config(),
                    ),
                    wo=Linear.Config(
                        in_features=n_heads * v_head_dim,
                        out_features=dim,
                    ),
                )
                # make_fx traces the backward, which cannot add into weight.grad in place.
                disable_inplace_wgrad_accum(attn_config)
                self.attn = DeepSeekMLAAttention(attn_config)
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

        tokens = torch.randint(0, vocab_size, (seq_len,), device=self.DEVICE)
        labels = torch.randint(0, vocab_size, (seq_len,), device=self.DEVICE)
        # Build positions that reset to 0 every 16 tokens (document boundaries)
        positions = torch.arange(seq_len, device=self.DEVICE) % 16
        block_mask = create_attention_mask(
            and_masks(get_causal_mask_mod(), get_document_mask_mod(positions)),
            B=1,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
        )

        # Annotate FlexInnerAttention.forward so compile_with_inductor flows into
        # the traced graph. Restore the original after tracing.
        orig_forward = FlexInnerAttention.forward
        FlexInnerAttention.forward = annotate_fn(
            {
                "compile_with_inductor": {
                    "inductor_configs": FlexInnerAttention.inductor_configs
                }
            }
        )(FlexInnerAttention.forward)
        try:
            train_step = make_train_step(model, get_loss)
            maybe_register_blockmask_pytree_node()
            traced = minimal_fx_tracer(train_step, module=model)(
                tokens, block_mask, labels
            )
        finally:
            FlexInnerAttention.forward = orig_forward

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
        from torchtitan.models.gpt_oss import MODEL_FLAVORS
        from torchtitan.models.gpt_oss.model import GptOssModel

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        vocab_size = config.vocab_size
        model_ref = create_model(GptOssModel, config, self.DEVICE, self.DTYPE)
        model_test = create_model(GptOssModel, config, self.DEVICE, self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())
        num_tokens = self.BATCH_SIZE * self.SEQ_LEN
        tokens = torch.randint(0, vocab_size, (num_tokens,), device=self.DEVICE)
        labels = torch.randint(0, vocab_size, (num_tokens,), device=self.DEVICE)
        positions = torch.arange(num_tokens, device=self.DEVICE)
        attention_metadata = model_ref._get_attention_metadata(positions)
        self._run_bitwise_test(
            model_ref,
            model_test,
            (tokens, positions, attention_metadata),
            labels,
            use_regional_inductor=True,
            num_steps=self.NUM_STEPS,
            lr=self.LR,
        )

    def test_flex_attention_annotations(self):
        from torchtitan.experiments.graph_trainer.common_utils import (
            annotate_module_fqns,
        )
        from torchtitan.models.gpt_oss import MODEL_FLAVORS
        from torchtitan.models.gpt_oss.model import GptOssModel

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        model = create_model(GptOssModel, config, self.DEVICE, self.DTYPE)
        annotate_module_fqns(model)

        num_tokens = self.BATCH_SIZE * self.SEQ_LEN
        tokens = torch.randint(0, config.vocab_size, (num_tokens,), device=self.DEVICE)
        positions = torch.arange(num_tokens, device=self.DEVICE)
        attention_metadata = model._get_attention_metadata(positions)
        aux_loss_denominators = torch.tensor([num_tokens], device=self.DEVICE)
        maybe_register_blockmask_pytree_node()

        def forward(tokens, positions, attention_metadata, aux_loss_denominators):
            return model(
                tokens,
                positions=positions,
                attention_metadata=attention_metadata,
                aux_loss_denominators=aux_loss_denominators,
            )

        traced = minimal_fx_tracer(forward, module=model)(
            tokens, positions, attention_metadata, aux_loss_denominators
        )

        flex_nodes = [
            n
            for n in traced.gm.graph.nodes
            if "flex_attention" in str(n.target) and "backward" not in str(n.target)
        ]
        self.assertGreater(len(flex_nodes), 0, "No FlexInnerAttentionHOP nodes found")

        from torchtitan.models.common.attention import FlexInnerAttention

        annotate_flex_attention_for_regional_inductor_pass(
            traced.gm,
            flex_compile_config=FlexInnerAttention.inductor_configs,
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
        from torchtitan.distributed import ParallelismContext

        self.parallelism_context = ParallelismContext(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )

    def _run_fsdp_model_test(
        self,
        config_cls,
        model_config,
        use_attention_metadata=False,
        attention_metadata=None,
        use_regional_inductor=False,
        dtype=torch.float32,
        fwd_kwargs=None,
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
        from torchtitan.experiments.graph_trainer.common_utils import (
            get_simple_fsdp_mesh,
        )

        fsdp_mesh = get_simple_fsdp_mesh(self.parallelism_context)

        model_ref = create_model(config_cls, model_config, "cuda", dtype)
        model_test = create_model(config_cls, model_config, "cuda", dtype)
        model_test.load_state_dict(model_ref.state_dict())
        model_ref = data_parallel(model_ref, device_mesh=fsdp_mesh, mode="fully_shard")
        model_test = data_parallel(
            model_test, device_mesh=fsdp_mesh, mode="fully_shard"
        )

        vocab_size = model_config.vocab_size
        seq_len = 128
        num_tokens = 2 * seq_len
        tokens = torch.randint(0, vocab_size, (num_tokens,), device="cuda")
        labels = torch.randint(0, vocab_size, (num_tokens,), device="cuda")
        positions = torch.arange(num_tokens, device="cuda")

        if attention_metadata is not None:
            fwd_args = (tokens, positions, attention_metadata)
        elif use_attention_metadata:
            attention_metadata = model_ref._get_attention_metadata(positions)
            fwd_args = (tokens, positions, attention_metadata)
        else:
            fwd_args = (tokens,)

        fwd_kwargs = fwd_kwargs or {}
        train_step = make_train_step(model_ref, get_loss, fwd_kwargs=fwd_kwargs)

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
            logits_ref = model_ref(*fwd_args, **fwd_kwargs)
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
        from torchtitan.models.llama3 import Llama3Model, MODEL_FLAVORS

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        self._run_fsdp_model_test(
            Llama3Model,
            config,
            use_attention_metadata=True,
            use_regional_inductor=True,
        )

    def test_qwen3_fsdp(self):
        from torchtitan.models.qwen3 import MODEL_FLAVORS
        from torchtitan.models.qwen3.model import Qwen3Model

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        self._run_fsdp_model_test(
            Qwen3Model,
            config,
            use_attention_metadata=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
        )

    def test_deepseek_v3_fsdp(self):
        from torchtitan.models.deepseek_v3 import MODEL_FLAVORS
        from torchtitan.models.deepseek_v3.model import DeepSeekV3Model

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        self._run_fsdp_model_test(
            DeepSeekV3Model,
            config,
            use_attention_metadata=True,
            use_regional_inductor=True,
            dtype=torch.bfloat16,
            fwd_kwargs={"aux_loss_denominators": torch.tensor([256.0], device="cuda")},
        )

    # TODO: Fix scatter() dtype mismatch — same root cause as TestTraceModels.test_gpt_oss.
    @unittest.skip("scatter(): Expected self.dtype to be equal to src.dtype")
    def test_gpt_oss_fsdp(self):
        from torchtitan.models.gpt_oss import MODEL_FLAVORS
        from torchtitan.models.gpt_oss.model import GptOssModel

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        self._run_fsdp_model_test(
            GptOssModel,
            config,
            use_attention_metadata=True,
            use_regional_inductor=True,
        )


# TODO: Re-enable after graph_trainer supports context parallel tracing.
@unittest.skip("Context Parallel is not supported by graph_trainer")
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
        from torchtitan_recipes.tests.graph_trainer.llama3 import (
            graph_trainer_llama3_debugmodel_varlen,
        )

        from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

        old_local_rank = os.environ.get("LOCAL_RANK")
        os.environ["LOCAL_RANK"] = str(dist.get_rank() % torch.cuda.device_count())

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        trainer = None
        try:
            with tempfile.TemporaryDirectory() as dump_folder:
                config = graph_trainer_llama3_debugmodel_varlen()
                config.dump_folder = dump_folder
                config.training.max_context_length = 128
                config.training.num_tokens_per_microbatch_per_dp_rank = 2 * 128
                config.training.steps = 1
                config.parallelism.data_parallel_replicate_degree = 1
                config.parallelism.data_parallel_shard_degree = dp_shard_degree
                config.parallelism.context_parallel_degree = context_parallel_degree
                config.parallelism.tensor_parallel_degree = 1
                config.activation_remat = None
                config.compile.enable_passes = False
                config.debug.enable_structured_logging = False
                config.model.layers = config.model.layers[:1]

                trainer = GraphTrainer(config)
                num_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
                tokens = torch.randint(
                    0,
                    trainer.engine.model_config.vocab_size,
                    (num_tokens,),
                    device=trainer.engine.device,
                )
                labels = torch.randint(
                    0,
                    trainer.engine.model_config.vocab_size,
                    (num_tokens,),
                    device=trainer.engine.device,
                )
                # The dataloader always supplies per-document positions, which
                # drive RoPE and define the Varlen document boundaries.
                positions = (
                    torch.arange(
                        num_tokens,
                        device=trainer.engine.device,
                        dtype=torch.int32,
                    )
                    % config.training.max_context_length
                )
                trainer.engine.forward_backward_microbatch(
                    microbatch_group=[
                        TokenizedTrainingMicrobatch(
                            input=tokens,
                            positions=positions,
                            labels=labels,
                            padding_mask=torch.zeros_like(labels, dtype=torch.bool),
                            loss_token_counts=torch.tensor(labels.numel()),
                            routing_token_counts=torch.tensor([labels.numel()]),
                        )
                    ],
                    global_loss_token_counts=torch.tensor(
                        labels.numel(), device=trainer.engine.device
                    ),
                )
                assert trainer.engine._traced_step is not None
                code_lines = trainer.engine._traced_step.gm.graph.python_code(
                    "self"
                ).src.splitlines()
                attention_line = next(
                    (
                        idx
                        for idx, line in enumerate(code_lines)
                        if "flash_attention" in line or "flash_attn" in line
                    ),
                    None,
                )
                self.assertIsNotNone(
                    attention_line,
                    "Expected Varlen attention in generated code:\n"
                    + "\n".join(code_lines),
                )
                assert attention_line is not None
                all_gather_pg_names_before_attention = []
                for node in trainer.engine._traced_step.gm.graph.nodes:
                    if any(
                        name in str(node.target)
                        for name in ("flash_attention", "flash_attn")
                    ):
                        break
                    if "all_gather_into_tensor" in str(node.target):
                        all_gather_pg_names_before_attention.append(node.args[2])

                cp_pg_name = (
                    trainer.engine.parallelism_context.get_mesh("cp")
                    .get_group()
                    .group_name
                    if trainer.engine.parallelism_context.cp_enabled
                    else None
                )
                fsdp_pg_name = (
                    get_simple_fsdp_mesh(trainer.engine.parallelism_context)
                    .get_group()
                    .group_name
                )
                code = trainer.engine._traced_step.gm.graph.python_code("self").src
                trainer.close()
                trainer = None
                return {
                    "code": code,
                    "all_gather_pg_names_before_attention": (
                        all_gather_pg_names_before_attention
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

    def test_llama3_cp_only_codegen_all_gather_before_attention(self):
        cp_trace = self._trace_llama3_step_code(
            dp_shard_degree=1,
            context_parallel_degree=2,
        )
        # Verify AG along CP PG exists before attention
        self.assertIn(
            cp_trace["cp_pg_name"],
            cp_trace["all_gather_pg_names_before_attention"],
            "Expected CP all_gather on the CP mesh before attention. "
            f"CP pg={cp_trace['cp_pg_name']}, "
            f"FSDP pg={cp_trace['fsdp_pg_name']}, "
            "pre-attention all_gather pgs="
            f"{cp_trace['all_gather_pg_names_before_attention']}.\n"
            f"Generated code:\n{cp_trace['code']}",
        )


class TestAutogradGradVsBackwardFSDP(FSDPTest):
    """Verify autograd.grad() and loss.backward() have identical peak memory with FSDP."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 4)

    def test_peak_memory_identical_fsdp(self):
        from torchtitan.distributed import ParallelismContext
        from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
        from torchtitan.models.llama3 import Llama3Model, MODEL_FLAVORS

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
        config = build_config(
            attn_backend="flex",
            seq_len=max_context_length,
            parallelism=ParallelismConfig(),
        )
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)

        try:
            parallelism_context = ParallelismContext(
                dp_shard=-1,
                dp_replicate=1,
                cp=1,
                tp=1,
                pp=1,
                ep=1,
                world_size=self.world_size,
                enable_sequence_parallel=False,
            )
            from torchtitan.experiments.graph_trainer.common_utils import (
                get_simple_fsdp_mesh,
            )

            fsdp_mesh = get_simple_fsdp_mesh(parallelism_context)

            model_backward = create_model(Llama3Model, config, "cuda", torch.bfloat16)
            model_grad = create_model(Llama3Model, config, "cuda", torch.bfloat16)
            model_grad.load_state_dict(model_backward.state_dict())
            model_backward = data_parallel(
                model_backward, device_mesh=fsdp_mesh, mode="fully_shard"
            )
            model_grad = data_parallel(
                model_grad, device_mesh=fsdp_mesh, mode="fully_shard"
            )

            num_tokens = 2 * 128
            tokens = torch.randint(0, config.vocab_size, (num_tokens,), device="cuda")
            labels = torch.randint(0, config.vocab_size, (num_tokens,), device="cuda")
            positions = torch.arange(num_tokens, device="cuda")
            attention_metadata = model_backward._get_attention_metadata(positions)

            def run_backward(model):
                logits = model(
                    tokens,
                    positions=positions,
                    attention_metadata=attention_metadata,
                )
                loss = get_loss(logits, labels)
                loss.backward()

            def run_grad(model):
                logits = model(
                    tokens,
                    positions=positions,
                    attention_metadata=attention_metadata,
                )
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
