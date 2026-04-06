# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import unittest
from collections import Counter

import torch
import torch.nn as nn
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_flex_attention_for_regional_inductor,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _copy_fwd_metadata_to_bw_nodes,
    _patch_engine_run_backward,
    extract_module_state,
    minimal_fx_tracer,
    run_traced,
    run_traced_train_step,
    trace_train_step,
)


def get_loss(logits, labels):
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
    )


def make_train_step(loss_fn):
    """Return a plain function for module-first tracing. loss_fn is captured in closure."""

    def train_step(model, *args):
        *fwd_args, labels = args
        logits = model(*fwd_args)
        loss = loss_fn(logits, labels)
        params = list(model.parameters())
        grads = torch.autograd.grad(loss, params)
        return [loss] + list(grads)

    return train_step


def make_stateless_train_step(model, loss_fn):
    """Return a state-first function for the minimal_fx_tracer core API."""

    def train_step(state, *args):
        *fwd_args, labels = args
        with torch.nn.utils.stateless._reparametrize_module(model, state):
            logits = model(*fwd_args)
        loss = loss_fn(logits, labels)
        grads = torch.autograd.grad(loss, list(state.values()))
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

    fake_mode = None
    for node in traced_result.gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            val = node.meta["val"]
            if isinstance(val, torch.Tensor) and hasattr(val, "fake_mode"):
                fake_mode = val.fake_mode
                break

    context = torch._guards.TracingContext(fake_mode)
    with torch._guards.tracing(context):
        traced_result.gm = regional_inductor(traced_result.gm)

    traced_result.gm.graph.set_codegen(CodeGen())
    traced_result.gm.recompile()


class SimpleMLP(nn.Module):
    def __init__(self, dim=64, hidden=128, vocab_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(self.embed(x))))


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

        def forward(model, tokens):
            return model(tokens)

        traced = trace_train_step(forward)(model, tokens)
        out_eager = model(tokens)
        wrapped = run_traced_train_step(traced, model, tokens)
        self.assertTrue(torch.equal(out_eager, wrapped))

    def test_mlp_train_step(self):
        model_ref, tokens, labels, loss_fn = self._make_mlp()
        model_test = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())

        train_step = make_train_step(loss_fn)
        traced = trace_train_step(train_step)(model_ref, tokens, labels)

        logits_ref = model_ref(tokens)
        loss_ref = loss_fn(logits_ref, labels)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]

        wrapped = run_traced_train_step(traced, model_test, tokens, labels)
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]

        self.assertTrue(torch.equal(loss_ref, loss_tr))
        for gr, gt in zip(grads_ref, grads_tr, strict=True):
            self.assertTrue(torch.equal(gr, gt))

    def test_mlp_multistep_bitwise(self):
        model_ref, tokens, labels, loss_fn = self._make_mlp()
        model_test = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())

        train_step = make_train_step(loss_fn)
        traced = trace_train_step(train_step)(model_ref, tokens, labels)

        opt_ref = torch.optim.Adam(model_ref.parameters(), lr=self.LR)
        opt_copy = torch.optim.Adam(model_test.parameters(), lr=self.LR)

        for step in range(1, self.NUM_STEPS + 1):
            logits_ref = model_ref(tokens)
            loss_ref = loss_fn(logits_ref, labels)
            loss_ref.backward()
            grads_ref = [p.grad.clone() for p in model_ref.parameters()]
            opt_ref.step()
            opt_ref.zero_grad()

            wrapped = run_traced_train_step(traced, model_test, tokens, labels)
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

        def fn(model, x, loss_fn):
            return loss_fn(model(x))

        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        with self.assertRaises(ValueError, msg="all pytree leaves"):
            trace_train_step(fn)(model, tokens, lambda x: x.sum())

    def test_mismatched_module_raises_when_validation_enabled(self):
        """Opt-in module FQN validation catches execution with the wrong module."""
        model, tokens, labels, loss_fn = self._make_mlp()

        def forward(model, tokens):
            return model(tokens)

        traced = trace_train_step(forward)(model, tokens)

        different_model = nn.Sequential(
            nn.Embedding(256, 64),
            nn.Linear(64, 256),
        ).to(device=self.DEVICE, dtype=self.DTYPE)

        with self.assertRaises(ValueError, msg="different parameter/buffer names"):
            run_traced_train_step(
                traced, different_model, tokens, validate_module_fqns=True
            )

    def test_trace_train_step_requires_module_first_arg(self):
        def forward(model, tokens):
            return model(tokens)

        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        with self.assertRaises(ValueError, msg="args\\[0\\]"):
            trace_train_step(forward)(tokens)

    def test_core_explicit_state_executes(self):
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        def forward(state, tokens):
            with torch.nn.utils.stateless._reparametrize_module(model, state):
                return model(tokens)

        state = extract_module_state(model)
        traced = minimal_fx_tracer(forward)(state, tokens)
        out_ref = forward(state, tokens)
        out_traced = run_traced(traced, state, tokens)

        self.assertTrue(torch.equal(out_ref, out_traced))

    def test_core_explicit_state_train_step(self):
        model_ref, tokens, labels, loss_fn = self._make_mlp()
        model_test = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        model_test.load_state_dict(model_ref.state_dict())

        state_ref = extract_module_state(model_ref)
        state_test = extract_module_state(model_test)
        train_step = make_stateless_train_step(model_ref, loss_fn)
        traced = minimal_fx_tracer(train_step)(state_ref, tokens, labels)

        logits_ref = model_ref(tokens)
        loss_ref = loss_fn(logits_ref, labels)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]

        wrapped = run_traced(traced, state_test, tokens, labels)
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]

        self.assertTrue(torch.equal(loss_ref, loss_tr))
        for gr, gt in zip(grads_ref, grads_tr, strict=True):
            self.assertTrue(torch.equal(gr, gt))

    def test_additional_module_arg_raises(self):
        def forward(model, other_model, tokens):
            del other_model
            return model(tokens)

        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        other_model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        with self.assertRaises(ValueError, msg="Additional nn.Module"):
            trace_train_step(forward)(model, other_model, tokens)


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

        def forward(model, tokens):
            return model(tokens)

        traced = trace_train_step(forward)(model, tokens_dt)
        has_subclass = any(
            layout.meta is not None for layout in traced.input_subclass_layouts.values()
        )
        self.assertTrue(has_subclass)

        out_eager = model(tokens_dt)
        wrapped = run_traced_train_step(traced, model, tokens_dt)
        self.assertTrue(torch.equal(out_eager.full_tensor(), wrapped.full_tensor()))

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

        train_step = make_train_step(get_loss)
        traced = trace_train_step(train_step)(model_ref, tokens_dt, labels_dt)

        logits_ref = model_ref(tokens_dt)
        loss_ref = get_loss(logits_ref, labels_dt)
        loss_ref.backward()
        grads_ref = [p.grad.clone() for p in model_ref.parameters()]

        wrapped = run_traced_train_step(traced, model_test, tokens_dt, labels_dt)
        loss_tr = wrapped[0]
        grads_tr = wrapped[1:]

        self.assertTrue(torch.equal(loss_ref.full_tensor(), loss_tr.full_tensor()))
        for gr, gt in zip(grads_ref, grads_tr, strict=True):
            self.assertTrue(torch.equal(gr.full_tensor(), gt.full_tensor()))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMetadataPropagation(unittest.TestCase):
    """Tests for _patch_engine_run_backward and _copy_fwd_metadata_to_bw_nodes."""

    DEVICE = "cuda"
    DTYPE = torch.float32

    def setUp(self):
        torch.manual_seed(42)

    def test_backward_nodes_have_seq_nr(self):
        """Verify that backward FX nodes get seq_nr metadata via the patched engine."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        train_step = make_train_step(get_loss)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        traced = trace_train_step(train_step)(model, tokens, labels)

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

        train_step = make_train_step(get_loss)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        traced = trace_train_step(train_step)(model, tokens, labels)
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

        # Check that bwd nodes with shared seq_nr got the custom metadata
        for node in gm.graph.nodes:
            if node.op != "call_function" or "seq_nr" not in node.meta:
                continue
            seq_nr = node.meta["seq_nr"]
            if node is not seq_nr_first.get(seq_nr):
                # This is a backward node
                custom = node.meta.get("custom")
                self.assertIsNotNone(
                    custom,
                    f"Backward node {node.name} with seq_nr={seq_nr} missing custom metadata",
                )
                self.assertEqual(custom.get("test_key"), "test_value")

    def test_backward_nodes_have_stack_trace(self):
        """Verify that backward nodes get stack_trace from their forward counterpart."""
        model = SimpleMLP().to(device=self.DEVICE, dtype=self.DTYPE)
        train_step = make_train_step(get_loss)
        tokens = torch.randint(0, 256, (2, 32), device=self.DEVICE)
        labels = torch.randint(0, 256, (2, 32), device=self.DEVICE)

        traced = trace_train_step(train_step)(model, tokens, labels)

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

    def test_patch_engine_restores_original(self):
        """Verify that _patch_engine_run_backward restores the original function."""
        import torch.autograd
        import torch.autograd.graph

        orig_fn = torch.autograd.graph._engine_run_backward

        with _patch_engine_run_backward():
            # Inside the context, it should be patched
            self.assertIsNot(torch.autograd.graph._engine_run_backward, orig_fn)
            self.assertIsNot(torch.autograd._engine_run_backward, orig_fn)

        # After the context, it should be restored
        self.assertIs(torch.autograd.graph._engine_run_backward, orig_fn)
        self.assertIs(torch.autograd._engine_run_backward, orig_fn)


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

    def tearDown(self):
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
        train_step = make_train_step(get_loss)

        maybe_regional_inductor = (
            annotate_flex_attention_for_regional_inductor()
            if use_regional_inductor
            else contextlib.nullcontext()
        )
        maybe_register_blockmask_pytree_node()
        with maybe_regional_inductor:
            traced = trace_train_step(train_step)(model_ref, *fwd_args, labels)

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

            wrapped = run_traced_train_step(traced, model_test, *fwd_args, labels)
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
    ):
        vocab_size = model_config.vocab_size
        model_ref = create_model(config_cls, model_config, self.DEVICE, self.DTYPE)
        model_test = create_model(config_cls, model_config, self.DEVICE, self.DTYPE)
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
            fwd_args = (tokens, attn_masks)

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

        config = llama3_configs["debugmodel"]()
        self._run_model_test(Llama3Model, config)

    def test_qwen3(self):
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.model import Qwen3Model

        config = qwen3_configs["debugmodel"]()
        self._run_model_test(Qwen3Model, config)

    def test_qwen3_moe(self):
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.model import Qwen3Model

        config = qwen3_configs["debugmodel_moe"]()
        self._run_model_test(Qwen3Model, config)

    def test_deepseek_v3(self):
        from torchtitan.models.deepseek_v3 import deepseekv3_configs
        from torchtitan.models.deepseek_v3.model import DeepSeekV3Model

        config = deepseekv3_configs["debugmodel"]()
        self._run_model_test(DeepSeekV3Model, config)

    def test_llama4(self):
        from torchtitan.models.llama4 import llama4_configs
        from torchtitan.models.llama4.model import Llama4Model

        config = llama4_configs["debugmodel"]()
        self._run_model_test(
            Llama4Model,
            config,
            use_attn_masks=True,
            use_regional_inductor=True,
        )

    def test_gpt_oss(self):
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.models.common.attention import (
            create_attention_mask,
            get_causal_mask_mod,
            get_sliding_window_mask_mod,
        )
        from torchtitan.models.gpt_oss import gptoss_configs
        from torchtitan.models.gpt_oss.model import GptOssModel

        config = gptoss_configs["debugmodel"]()
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
            annotate_ac_regions,
        )
        from torchtitan.models.common.attention import (
            create_attention_mask,
            get_causal_mask_mod,
            get_sliding_window_mask_mod,
        )
        from torchtitan.models.gpt_oss import gptoss_configs
        from torchtitan.models.gpt_oss.model import GptOssModel

        config = gptoss_configs["debugmodel"]()
        model = create_model(GptOssModel, config, self.DEVICE, self.DTYPE)
        annotate_ac_regions(model)

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
        with annotate_flex_attention_for_regional_inductor():

            def forward(model, tokens, attn_masks):
                return model(tokens, attn_masks)

            traced = trace_train_step(forward)(model, tokens, attn_masks)

        flex_nodes = [
            n
            for n in traced.gm.graph.nodes
            if "flex_attention" in str(n.target) and "backward" not in str(n.target)
        ]
        self.assertGreater(len(flex_nodes), 0, "No FlexAttentionHOP nodes found")

        for node in flex_nodes:
            custom = node.meta.get("custom", {})
            self.assertIn(
                "compile_with_inductor",
                custom,
                f"{node.name} missing compile_with_inductor annotation",
            )
            self.assertIn(
                "ac_region_id",
                custom,
                f"{node.name} missing ac_region_id annotation",
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
            etp=1,
            world_size=self.world_size,
        )

    def _run_fsdp_model_test(
        self,
        config_cls,
        model_config,
        use_attn_masks=False,
        attn_masks=None,
        use_regional_inductor=False,
    ):
        from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        self._setup()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")

        model_ref = create_model(config_cls, model_config, "cuda", torch.float32)
        model_test = create_model(config_cls, model_config, "cuda", torch.float32)
        model_test.load_state_dict(model_ref.state_dict())
        model_ref = data_parallel(model_ref, device_mesh=fsdp_mesh, mode="fully_shard")
        model_test = data_parallel(
            model_test, device_mesh=fsdp_mesh, mode="fully_shard"
        )

        vocab_size = model_config.vocab_size
        seq_len = 128
        tokens = torch.randint(0, vocab_size, (2, seq_len), device="cuda")
        labels = torch.randint(0, vocab_size, (2, seq_len), device="cuda")

        if attn_masks is not None:
            fwd_args = (tokens, attn_masks)
        elif use_attn_masks:
            from torchtitan.models.common.attention import (
                create_attention_mask,
                get_causal_mask_mod,
            )

            attn_masks = create_attention_mask(
                get_causal_mask_mod(), 1, None, seq_len, seq_len
            )
            fwd_args = (tokens, attn_masks)
        else:
            fwd_args = (tokens,)

        train_step = make_train_step(get_loss)

        maybe_regional_inductor = (
            annotate_flex_attention_for_regional_inductor()
            if use_regional_inductor
            else contextlib.nullcontext()
        )
        maybe_register_blockmask_pytree_node()
        with maybe_regional_inductor:
            traced = trace_train_step(train_step)(model_ref, *fwd_args, labels)

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

            wrapped = run_traced_train_step(traced, model_test, *fwd_args, labels)
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

        config = llama3_configs["debugmodel"]()
        self._run_fsdp_model_test(Llama3Model, config)

    def test_qwen3_fsdp(self):
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.model import Qwen3Model

        config = qwen3_configs["debugmodel"]()
        self._run_fsdp_model_test(Qwen3Model, config)

    def test_deepseek_v3_fsdp(self):
        from torchtitan.models.deepseek_v3 import deepseekv3_configs
        from torchtitan.models.deepseek_v3.model import DeepSeekV3Model

        config = deepseekv3_configs["debugmodel"]()
        self._run_fsdp_model_test(DeepSeekV3Model, config)

    def test_llama4_fsdp(self):
        from torchtitan.models.llama4 import llama4_configs
        from torchtitan.models.llama4.model import Llama4Model

        config = llama4_configs["debugmodel"]()
        self._run_fsdp_model_test(
            Llama4Model,
            config,
            use_attn_masks=True,
            use_regional_inductor=True,
        )

    def test_gpt_oss_fsdp(self):
        from torch.nn.attention.flex_attention import and_masks

        from torchtitan.models.common.attention import (
            create_attention_mask,
            get_causal_mask_mod,
            get_sliding_window_mask_mod,
        )
        from torchtitan.models.gpt_oss import gptoss_configs
        from torchtitan.models.gpt_oss.model import GptOssModel

        config = gptoss_configs["debugmodel"]()
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


class TestAutogradGradVsBackwardFSDP(FSDPTest):
    """Verify autograd.grad() and loss.backward() have identical peak memory with FSDP."""

    @property
    def world_size(self):
        return min(torch.cuda.device_count(), 4)

    def test_peak_memory_identical_fsdp(self):
        from torchtitan.distributed import ParallelDims
        from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
        from torchtitan.models.llama3 import llama3_configs, Llama3Model

        config = llama3_configs["debugmodel"]
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
                etp=1,
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

            def run_backward(model):
                logits = model(tokens)
                loss = get_loss(logits, labels)
                loss.backward()

            def run_grad(model):
                logits = model(tokens)
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
