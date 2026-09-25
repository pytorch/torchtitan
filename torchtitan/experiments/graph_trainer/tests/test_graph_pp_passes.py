# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import functools
import operator
import unittest
import warnings
from dataclasses import dataclass, replace
from typing import Any
from unittest.mock import patch

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.attention.flex_attention import flex_attention
from torch.testing._internal.common_fsdp import FSDPTest
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    get_simple_fsdp_mesh,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    model_registry as dsv3_model_registry,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    deduplicate_fsdp_unshard_chains_pass,
    joint_transformer_block_bucketing_reordering_pass,
    merge_all_all_gathers,
    merge_all_all_reduces,
    merge_all_reduce_scatters,
)
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_reduce_grad_input,
    find_fsdp_unshard_output,
    find_fsdp_unshard_outputs_by_param,
    find_fsdp_unshard_save_node,
    find_fsdp_unshard_save_nodes,
)
from torchtitan.experiments.graph_trainer.graph_builder import (
    _defer_fsdp_action_bucketing,
)
from torchtitan.experiments.graph_trainer.graph_pp import (
    extract_fsdp_reduce_grad_graph,
    extract_fsdp_unshard_graph,
    partition_joint_graph,
    split_di_dw_graph,
)
from torchtitan.experiments.graph_trainer.graph_pp.partition import GraphMeta
from torchtitan.experiments.graph_trainer.graph_pp.utils import flatten_graph_values
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    FSDP_MESH_AXIS_NAMES_META,
    FSDP_PARAM_FQNS_META,
)
from torchtitan.experiments.graph_trainer.wgrad_accumulation import (
    insert_graph_gradient_accumulation,
)
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.trainer import Trainer


@dataclass(frozen=True, slots=True)
class _Dsv3MoeBlockTrace:
    traced: TracedResult
    flat_inputs: list[Any]
    output_grad: torch.Tensor
    num_param_grad_values: int
    num_flat_param_values: int


@contextlib.contextmanager
def _stable_flex_attention_compile_config():
    original_configs = FlexInnerAttention.inductor_configs
    original_compiled_flex_attn = FlexInnerAttention._compiled_flex_attn
    FlexInnerAttention.inductor_configs = {
        **original_configs,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
    }
    FlexInnerAttention._compiled_flex_attn = torch.compile(
        flex_attention,
        options=FlexInnerAttention.inductor_configs,
    )
    try:
        yield
    finally:
        FlexInnerAttention.inductor_configs = original_configs
        FlexInnerAttention._compiled_flex_attn = original_compiled_flex_attn


def _trace_dsv3_moe_block_stage(
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    include_input_grad: bool = True,
    fsdp_mesh: Any | None = None,
) -> _Dsv3MoeBlockTrace:
    """Trace a real DeepSeek V3 MoE decoder block as a GraphPP stage.

    This is intentionally CUDA-only: FlexInnerAttention backward is not supported on
    CPU, and these pass tests need to exercise the real BlockMask tracing path
    rather than a maskless SDPA shortcut. The unit test enables EP sharding
    metadata and traces the first MoE layer. When ``fsdp_mesh`` is supplied the
    same block is wrapped with the graph trainer's simple-FSDP path so the
    partition pass is tested against the collective shapes that later passes
    consume. True EP numerics are covered by the distributed GraphPP DSV3
    loss-compare tests.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("DeepSeek V3 MoE block tracing requires CUDA")

    maybe_register_blockmask_pytree_node()
    torch.manual_seed(0)

    with _stable_flex_attention_compile_config():
        model_config = dsv3_model_registry("debugmodel", attn_backend="flex")
        runtime_config = Trainer.Config(
            model=model_config,
            training=TrainingConfig(
                num_tokens_per_microbatch_per_dp_rank=batch_size * seq_len,
                max_context_length=seq_len,
                steps=1,
                disable_cuda_graphs=True,
            ),
            parallelism=ParallelismConfig(expert_parallel_degree=2),
            checkpointer=CheckpointManager.Config(initial_load_model_only=False),
            debug=DebugConfig(seed=0, deterministic=True),
        )
        model_config.update_from_config(config=runtime_config)
        moe_layer_config = model_config.layers[1]
        if moe_layer_config.moe is None:
            raise AssertionError("DeepSeek V3 MoE layer must contain an MoE block")

        with torch.device("meta"):
            model = model_config.build()
        model.to_empty(device="cuda")
        with torch.no_grad():
            model.init_states(buffer_device=None)
        model._apply(
            lambda tensor: tensor.to(dtype=torch.bfloat16)
            if tensor.is_floating_point()
            else tensor
        )
        model.train()

        block = model.layers["1"]
        if not block.moe_enabled:
            raise AssertionError("DeepSeek V3 debug layer 1 must be a MoE layer")
        if fsdp_mesh is not None:
            block = data_parallel(
                block,
                device_mesh=fsdp_mesh,
                mode="fully_shard",
            )

        num_tokens = batch_size * seq_len
        AuxLoss.set_step_denominator(torch.tensor(num_tokens, device="cuda"))
        x = torch.randn(
            num_tokens,
            model_config.dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=include_input_grad,
        )
        positions = torch.arange(seq_len, device="cuda").repeat(batch_size)
        attention_masks = model.get_attention_masks(positions)
        output_grad = torch.randn_like(x)

        def stage_step(
            x: torch.Tensor,
            positions: torch.Tensor,
            attention_masks: Any,
            output_grad: torch.Tensor,
        ):
            out = block(x, attention_masks, positions)
            params = [
                p
                for _, p in block.named_parameters(remove_duplicate=False)
                if p.requires_grad
            ]
            grad_targets = [*params, x] if include_input_grad else params
            grads = torch.autograd.grad(
                out,
                grad_targets,
                grad_outputs=output_grad,
            )
            return [out, *grads]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="flex_attention called without torch.compile",
                category=UserWarning,
            )
            traced = minimal_fx_tracer(stage_step, module=block)(
                x,
                positions,
                attention_masks,
                output_grad,
            )

        user_flat_inputs, _ = pytree.tree_flatten(
            ((x, positions, attention_masks, output_grad), {})
        )
        state_flat_inputs, _ = pytree.tree_flatten(extract_module_state(block))
        flat_inputs = flatten_graph_values([*state_flat_inputs, *user_flat_inputs])
        if len(flat_inputs) != len(traced.example_inputs):
            raise AssertionError(
                "Real flat inputs must match traced flat input count: "
                f"{len(flat_inputs)} != {len(traced.example_inputs)}"
            )

        state_params = [p for _, p in block.named_parameters(remove_duplicate=False)]
        grad_params = [p for p in state_params if p.requires_grad]
        return _Dsv3MoeBlockTrace(
            traced=traced,
            flat_inputs=flat_inputs,
            output_grad=output_grad,
            num_param_grad_values=len(flatten_graph_values(grad_params)),
            num_flat_param_values=len(flatten_graph_values(state_params)),
        )


def _boxed_run(gm: fx.GraphModule, args: list[Any]):
    return fx.Interpreter(gm).boxed_run(args)


def _backward_args_from_partition(
    meta: GraphMeta,
    fw_outputs: tuple[Any, ...],
    backward_grad_inputs: tuple[Any, ...],
) -> list[Any]:
    saved_by_name = dict(
        zip(
            meta.saved_for_backward_names,
            fw_outputs[
                meta.num_fwd_user_outputs : meta.num_fwd_user_outputs
                + meta.num_saved_for_backward
            ],
            strict=True,
        )
    )
    backward_grad_by_name = dict(
        zip(
            meta.backward_grad_input_names,
            (backward_grad_inputs[index] for index in meta.backward_grad_input_indices),
            strict=True,
        )
    )
    return [
        saved_by_name[name] if name in saved_by_name else backward_grad_by_name[name]
        for name in meta.bwd_input_names
    ]


def _assert_tensor_sequence_equal(
    test_case: unittest.TestCase,
    actual_values: tuple[Any, ...],
    expected_values: tuple[Any, ...],
) -> None:
    test_case.assertEqual(len(actual_values), len(expected_values))
    for actual, expected in zip(actual_values, expected_values, strict=True):
        if actual is None or expected is None:
            test_case.assertIs(actual, expected)
        elif not isinstance(actual, torch.Tensor) or not isinstance(
            expected, torch.Tensor
        ):
            test_case.assertEqual(actual, expected)
        else:
            test_case.assertTrue(torch.equal(actual, expected))


class GraphPPPartitionTest(unittest.TestCase):
    def test_partition_keeps_only_same_phase_effects(self) -> None:
        x = torch.randn(2, 4)
        fwd_state = torch.zeros_like(x)
        bwd_state = torch.zeros_like(x)
        base_traced = minimal_fx_tracer(lambda a, b, c: [a, b])(
            x,
            fwd_state,
            bwd_state,
        )

        graph = fx.Graph()
        graph_x = graph.placeholder("x")
        graph_fwd_state = graph.placeholder("fwd_state")
        graph_bwd_state = graph.placeholder("bwd_state")
        fwd_value = graph.call_function(
            torch.ops.aten.add.Tensor,
            args=(graph_x, 1.0),
        )
        fwd_copy = graph.call_function(
            torch.ops.aten.copy_.default,
            args=(graph_fwd_state, fwd_value),
        )
        graph.call_function(
            torch.ops.aten.rand.default,
            args=([2, 4],),
            kwargs={"device": "cpu"},
        )
        fwd_output = graph.call_function(torch.ops.aten.sin.default, args=(fwd_copy,))
        bwd_value = graph.call_function(
            torch.ops.aten.sub.Tensor,
            args=(graph_x, 1.0),
        )
        bwd_copy = graph.call_function(
            torch.ops.aten.copy_.default,
            args=(graph_bwd_state, bwd_value),
        )
        bwd_random = graph.call_function(
            torch.ops.aten.rand.default,
            args=([2, 4],),
            kwargs={"device": "cpu"},
        )
        bwd_output = graph.call_function(torch.ops.aten.cos.default, args=(bwd_copy,))
        for node in (bwd_value, bwd_copy, bwd_random, bwd_output):
            node.meta["autograd_backward"] = True
        graph.output((fwd_output, bwd_output))
        joint = _make_graph_module(graph)
        traced = replace(base_traced, gm=joint)

        fw_module, bw_module, meta = partition_joint_graph(
            traced,
            num_fwd_outputs=1,
        )

        for module, expected_backward in (
            (fw_module, False),
            (bw_module, True),
        ):
            effect_nodes = [
                node
                for node in module.graph.nodes
                if node.op == "call_function"
                and node.target
                in (torch.ops.aten.copy_.default, torch.ops.aten.rand.default)
            ]
            self.assertEqual(len(effect_nodes), 2)
            self.assertTrue(
                all(
                    bool(node.meta.get("autograd_backward", False)) == expected_backward
                    for node in effect_nodes
                )
            )

        joint_inputs = [x.clone(), fwd_state.clone(), bwd_state.clone()]
        torch.manual_seed(42)
        joint_outputs = _boxed_run(joint, list(joint_inputs))
        joint_rng_state = torch.get_rng_state()

        split_inputs = [x.clone(), fwd_state.clone(), bwd_state.clone()]
        torch.manual_seed(42)
        fw_args = [split_inputs[index] for index in meta.fwd_flat_input_indices]
        fw_outputs = _boxed_run(fw_module, list(fw_args))
        bw_args = _backward_args_from_partition(meta, fw_outputs, ())
        bw_outputs = _boxed_run(bw_module, bw_args)
        split_rng_state = torch.get_rng_state()

        _assert_tensor_sequence_equal(self, fw_outputs[:1], joint_outputs[:1])
        _assert_tensor_sequence_equal(self, bw_outputs, joint_outputs[1:])
        self.assertTrue(torch.equal(split_inputs[1], joint_inputs[1]))
        self.assertTrue(torch.equal(split_inputs[2], joint_inputs[2]))
        self.assertTrue(torch.equal(split_rng_state, joint_rng_state))

    def test_real_dsv3_moe_block_partition_matches_joint_graph(self) -> None:
        traced_block = _trace_dsv3_moe_block_stage()

        fw_module, bw_module, meta = partition_joint_graph(
            traced_block.traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced_block.traced.example_inputs) - 1,),
        )

        joint_outputs = traced_block.traced.gm(*traced_block.flat_inputs)
        fw_args = [
            traced_block.flat_inputs[index] for index in meta.fwd_flat_input_indices
        ]
        fw_outputs = _boxed_run(fw_module, list(fw_args))

        self.assertTrue(torch.equal(fw_outputs[0], joint_outputs[0]))
        self.assertEqual(meta.num_backward_grad_inputs, 1)
        self.assertEqual(
            meta.num_bwd_outputs,
            traced_block.num_param_grad_values + 1,
        )
        self.assertGreater(meta.num_saved_for_backward, 0)
        self.assertEqual(
            len(fw_outputs),
            meta.num_fwd_user_outputs
            + meta.num_saved_for_backward
            + meta.num_fwd_side_effect_outputs,
        )

        bw_args = _backward_args_from_partition(
            meta,
            fw_outputs,
            (traced_block.output_grad,),
        )
        bw_outputs = _boxed_run(bw_module, list(bw_args))
        _assert_tensor_sequence_equal(self, bw_outputs, joint_outputs[1:])

    def test_partition_saves_backward_passthrough_placeholders(self) -> None:
        def stage_step(
            x: torch.Tensor,
            dtensor_layout_metadata: torch.Tensor,
            output_grad: torch.Tensor,
        ):
            out = x.sin()
            (grad_x,) = torch.autograd.grad(
                out,
                x,
                grad_outputs=output_grad,
            )
            return [out, grad_x, dtensor_layout_metadata]

        x = torch.randn(2, 4, requires_grad=True)
        dtensor_layout_metadata = torch.arange(2)
        output_grad = torch.randn(2, 4)
        traced = minimal_fx_tracer(stage_step)(x, dtensor_layout_metadata, output_grad)

        fw_module, bw_module, meta = partition_joint_graph(
            traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced.example_inputs) - 1,),
        )

        flat_inputs = [x, dtensor_layout_metadata, output_grad]
        fw_args = [flat_inputs[index] for index in meta.fwd_flat_input_indices]
        fw_outputs = _boxed_run(fw_module, fw_args)

        self.assertIn("arg1_1", meta.saved_for_backward_names)
        self.assertNotIn("arg2_1", meta.fwd_input_names)
        self.assertEqual(
            meta.bwd_input_names,
            (*meta.saved_for_backward_names, *meta.backward_grad_input_names),
        )

        bw_args = _backward_args_from_partition(meta, fw_outputs, (output_grad,))
        bw_outputs = _boxed_run(bw_module, bw_args)
        joint_outputs = traced.gm(*flat_inputs)
        _assert_tensor_sequence_equal(self, bw_outputs, joint_outputs[1:])

    def test_partition_saves_forward_input_used_only_by_backward(self) -> None:
        def stage_step(
            x: torch.Tensor,
            backward_mask: torch.Tensor,
            output_grad: torch.Tensor,
        ):
            out = x.sin()
            grad_x = output_grad * backward_mask
            return [out, grad_x]

        x = torch.randn(2, 4)
        backward_mask = torch.randn(2, 4)
        output_grad = torch.randn(2, 4)
        traced = minimal_fx_tracer(stage_step)(x, backward_mask, output_grad)

        fw_module, bw_module, meta = partition_joint_graph(
            traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced.example_inputs) - 1,),
        )

        flat_inputs = [x, backward_mask, output_grad]
        fw_args = [flat_inputs[index] for index in meta.fwd_flat_input_indices]
        fw_outputs = _boxed_run(fw_module, fw_args)

        self.assertIn("arg1_1", meta.saved_for_backward_names)
        self.assertNotIn("arg2_1", meta.fwd_input_names)
        bw_args = _backward_args_from_partition(meta, fw_outputs, (output_grad,))
        bw_outputs = _boxed_run(bw_module, bw_args)
        joint_outputs = traced.gm(*flat_inputs)
        _assert_tensor_sequence_equal(self, bw_outputs, joint_outputs[1:])

    def test_invalid_backward_only_input_indices_raise(self) -> None:
        def stage_step(x: torch.Tensor, output_grad: torch.Tensor):
            out = x.sin()
            (grad_x,) = torch.autograd.grad(out, x, grad_outputs=output_grad)
            return [out, grad_x]

        x = torch.randn(2, 4, requires_grad=True)
        output_grad = torch.randn(2, 4)
        traced = minimal_fx_tracer(stage_step)(x, output_grad)

        with self.assertRaisesRegex(ValueError, "must be unique"):
            partition_joint_graph(
                traced,
                num_fwd_outputs=1,
                backward_only_input_indices=(1, 1),
            )

        with self.assertRaisesRegex(ValueError, "must reference traced graph"):
            partition_joint_graph(
                traced,
                num_fwd_outputs=1,
                backward_only_input_indices=(len(traced.example_inputs),),
            )

    def test_backward_only_input_required_by_forward_raises(self) -> None:
        def stage_step(x: torch.Tensor, output_grads_from_next: torch.Tensor):
            out = x + output_grads_from_next
            (grad_x,) = torch.autograd.grad(out.sum(), x)
            return [out, grad_x]

        x = torch.randn(2, 4, requires_grad=True)
        output_grads_from_next = torch.randn(2, 4)
        traced = minimal_fx_tracer(stage_step)(x, output_grads_from_next)

        with self.assertRaisesRegex(
            ValueError,
            "Forward graph outputs require backward-only inputs",
        ):
            partition_joint_graph(
                traced,
                num_fwd_outputs=1,
                backward_only_input_indices=(1,),
            )

    def test_forward_mutation_of_backward_only_input_raises(self) -> None:
        def stage_step(x: torch.Tensor, output_grads_from_next: torch.Tensor):
            output_grads_from_next.add_(1.0)
            out = x.sin()
            (grad_x,) = torch.autograd.grad(
                out,
                x,
                grad_outputs=torch.ones_like(out),
            )
            return [out, grad_x]

        x = torch.randn(2, 4, requires_grad=True)
        output_grads_from_next = torch.randn(2, 4)
        traced = minimal_fx_tracer(stage_step)(x, output_grads_from_next)

        with self.assertRaisesRegex(
            ValueError,
            "Forward mutation cannot target a backward-only input",
        ):
            partition_joint_graph(
                traced,
                num_fwd_outputs=1,
                backward_only_input_indices=(1,),
            )


class _GraphPPDsv3FSDPTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return max(1, min(torch.cuda.device_count(), 2))

    def _setup(self) -> None:
        self.parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )


class GraphPPPartitionFSDPTest(_GraphPPDsv3FSDPTest):
    def test_real_dsv3_moe_block_fsdp_partition_matches_joint_graph(self) -> None:
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("real FSDP collective trace requires 2 GPUs")

        self._setup()
        traced_block = _trace_dsv3_moe_block_stage(
            fsdp_mesh=get_simple_fsdp_mesh(self.parallel_dims)
        )
        all_gathers = traced_block.traced.gm.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        self.assertGreater(len(all_gathers), 0)
        for all_gather in all_gathers:
            param_fqns = all_gather.meta.get("custom", {}).get(FSDP_PARAM_FQNS_META)
            self.assertIsNotNone(param_fqns)
            self.assertEqual(len(param_fqns), 1)
            (wait,) = all_gather.users
            self.assertEqual(
                wait.meta.get("custom", {}).get(FSDP_PARAM_FQNS_META),
                param_fqns,
            )

        fw_module, bw_module, meta = partition_joint_graph(
            traced_block.traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced_block.traced.example_inputs) - 1,),
        )

        joint_outputs = traced_block.traced.gm(*traced_block.flat_inputs)
        fw_args = [
            traced_block.flat_inputs[index] for index in meta.fwd_flat_input_indices
        ]
        fw_outputs = _boxed_run(fw_module, list(fw_args))
        bw_args = _backward_args_from_partition(
            meta,
            fw_outputs,
            (traced_block.output_grad,),
        )
        bw_outputs = _boxed_run(bw_module, list(bw_args))

        self.assertTrue(torch.equal(fw_outputs[0], joint_outputs[0]))
        _assert_tensor_sequence_equal(self, bw_outputs, joint_outputs[1:])


class GraphPPSplitDiDwTest(unittest.TestCase):
    def test_real_dsv3_moe_block_split_reconstructs_backward(self) -> None:
        traced_block = _trace_dsv3_moe_block_stage()
        fw_module, bw_module, meta = partition_joint_graph(
            traced_block.traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced_block.traced.example_inputs) - 1,),
        )
        split = split_di_dw_graph(
            bw_module,
            num_param_grads=traced_block.num_param_grad_values,
        )

        self.assertIsNotNone(split)
        if split is None:
            self.fail("Expected dI/dW split for decoder block with input grad")
        self.assertEqual(split.num_input_grads, 1)
        self.assertGreater(len(split.bw_dw_input_names), 0)

        fw_args = [
            traced_block.flat_inputs[index] for index in meta.fwd_flat_input_indices
        ]
        fw_outputs = _boxed_run(fw_module, list(fw_args))
        bw_args = _backward_args_from_partition(
            meta,
            fw_outputs,
            (traced_block.output_grad,),
        )
        full_bw_outputs = _boxed_run(bw_module, list(bw_args))

        di_outputs = _boxed_run(split.bw_di_module, list(bw_args))
        input_grads_to_prev = di_outputs[: split.num_input_grads]
        dw_live_ins = di_outputs[split.num_input_grads :]
        dw_outputs = _boxed_run(split.bw_dw_module, list(dw_live_ins))

        _assert_tensor_sequence_equal(
            self,
            input_grads_to_prev,
            full_bw_outputs[traced_block.num_param_grad_values :],
        )
        _assert_tensor_sequence_equal(
            self,
            dw_outputs,
            full_bw_outputs[: traced_block.num_param_grad_values],
        )

    def test_real_dsv3_moe_block_without_input_grad_skips_split(self) -> None:
        traced_block = _trace_dsv3_moe_block_stage(include_input_grad=False)
        _, bw_module, _ = partition_joint_graph(
            traced_block.traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced_block.traced.example_inputs) - 1,),
        )

        split = split_di_dw_graph(
            bw_module,
            num_param_grads=traced_block.num_param_grad_values,
        )

        self.assertIsNone(split)


_FAKE_PG = "graph_pp_test_pg"
_FAKE_PG_2 = "graph_pp_test_pg_2"


def _call_targets(gm: fx.GraphModule) -> set[object]:
    return {node.target for node in gm.graph.nodes if node.op == "call_function"}


def _placeholder_names(gm: fx.GraphModule) -> tuple[str, ...]:
    return tuple(node.name for node in gm.graph.find_nodes(op="placeholder"))


def _make_graph_module(graph: fx.Graph) -> fx.GraphModule:
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


class _FakeCollectiveInterpreter(fx.Interpreter):
    def call_function(self, target, args, kwargs):
        if target == torch.ops._c10d_functional.all_gather_into_tensor.default:
            if args[1] != 1:
                raise ValueError("Test interpreter supports group size one only")
            return args[0]
        if target == torch.ops._c10d_functional.all_gather_into_tensor_out.default:
            if args[1] != 1:
                raise ValueError("Test interpreter supports group size one only")
            kwargs["out"].copy_(args[0])
            return kwargs["out"]
        if target == torch.ops._c10d_functional.reduce_scatter_tensor.default:
            return args[0].chunk(args[2], dim=0)[0]
        if target == torch.ops._c10d_functional.all_reduce.default:
            return args[0]
        if target == torch.ops._c10d_functional.wait_tensor.default:
            return args[0]
        return super().call_function(target, args, kwargs)


def _make_unbucketed_action_graph(collective: str) -> fx.GraphModule:
    graph = fx.Graph()
    first = graph.placeholder("first")
    second = graph.placeholder("second")
    with FakeTensorMode() as fake_mode:
        first_value = fake_mode.from_tensor(torch.empty(2, dtype=torch.float32))
        second_value = fake_mode.from_tensor(torch.empty(3, dtype=torch.float32))
    first.meta["val"] = first_value
    second.meta["val"] = second_value

    outputs = []
    for index, (value, fake_value) in enumerate(
        ((first, first_value), (second, second_value))
    ):
        if collective == "all_gather":
            start = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(value, 1, _FAKE_PG),
            )
        elif collective == "reduce_scatter":
            start = graph.call_function(
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                args=(value, "sum", 1, _FAKE_PG),
            )
        elif collective == "all_reduce":
            start = graph.call_function(
                torch.ops._c10d_functional.all_reduce.default,
                args=(value, "sum", _FAKE_PG),
            )
        else:
            raise ValueError(f"Unsupported test collective: {collective}")
        wait = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(start,),
        )
        for node in (start, wait):
            node.meta["val"] = fake_value
            node.meta["custom"] = {_MODULE_FQN: f"layers.0.part{index}"}
        outputs.append(wait)
    graph.output(tuple(outputs))
    return _make_graph_module(graph)


def _make_interleaved_reduce_grad_action_graph() -> fx.GraphModule:
    graph = fx.Graph()
    outputs = []
    with FakeTensorMode() as fake_mode:
        fake_values = tuple(
            fake_mode.from_tensor(torch.empty(size, dtype=torch.float32))
            for size in (2, 3)
        )
    for index, fake_value in enumerate(fake_values):
        gradient = graph.placeholder(f"gradient_{index}")
        gradient.meta["val"] = fake_value
        reduce_scatter = graph.call_function(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            args=(gradient, "sum", 1, _FAKE_PG),
        )
        reduce_scatter.meta["val"] = fake_value
        reduce_scatter_wait = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(reduce_scatter,),
        )
        reduce_scatter_wait.meta["val"] = fake_value
        cast = graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(reduce_scatter_wait,),
            kwargs={"dtype": torch.float32},
        )
        cast.meta["val"] = fake_value
        all_reduce = graph.call_function(
            torch.ops._c10d_functional.all_reduce.default,
            args=(cast, "sum", _FAKE_PG),
        )
        all_reduce.meta["val"] = fake_value
        all_reduce_wait = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(all_reduce,),
        )
        all_reduce_wait.meta["val"] = fake_value
        outputs.append(all_reduce_wait)
    graph.output(tuple(outputs))
    return _make_graph_module(graph)


def _quantize_weight_for_graph_test(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return weight, weight, weight


def _make_forward_graph_with_unshard_and_replicated_param() -> fx.GraphModule:
    graph = fx.Graph()
    sharded_param = graph.placeholder("sharded_param")
    replicated_param = graph.placeholder("replicated_param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(sharded_param, 1, _FAKE_PG),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    split = graph.call_function(torch.ops.aten.split.Tensor, args=(wait, 2, 0))
    left = graph.call_function(operator.getitem, args=(split, 0))
    right = graph.call_function(operator.getitem, args=(split, 1))
    unsharded_param = graph.call_function(
        torch.ops.aten.cat.default,
        args=([left, right], 0),
    )
    duplicate_all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(sharded_param, 1, _FAKE_PG),
    )
    duplicate_wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(duplicate_all_gather,),
    )
    duplicate_split = graph.call_function(
        torch.ops.aten.split.Tensor,
        args=(duplicate_wait, 2, 0),
    )
    duplicate_left = graph.call_function(operator.getitem, args=(duplicate_split, 0))
    duplicate_right = graph.call_function(operator.getitem, args=(duplicate_split, 1))
    duplicate_unsharded_param = graph.call_function(
        torch.ops.aten.cat.default,
        args=([duplicate_left, duplicate_right], 0),
    )
    fsdp_meta = {FSDP_PARAM_FQNS_META: ("linear.weight",)}
    for node in (
        all_gather,
        wait,
        split,
        left,
        right,
        unsharded_param,
        duplicate_all_gather,
        duplicate_wait,
        duplicate_split,
        duplicate_left,
        duplicate_right,
        duplicate_unsharded_param,
    ):
        node.meta["custom"] = fsdp_meta
    sharded_param_uses = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=(unsharded_param, duplicate_unsharded_param),
    )
    params = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=(sharded_param_uses, replicated_param),
    )
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(params, x))
    graph.output((out,))
    return _make_graph_module(graph)


def _make_forward_graph_with_quantized_unshard() -> fx.GraphModule:
    graph = fx.Graph()
    sharded_param = graph.placeholder("sharded_param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(sharded_param, 1, _FAKE_PG),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    unsharded_param = graph.call_function(
        torch.ops.aten.view.default,
        args=(wait, [4]),
    )
    quantized_operands = graph.call_function(
        _quantize_weight_for_graph_test,
        args=(unsharded_param,),
    )
    fsdp_meta = {FSDP_PARAM_FQNS_META: ("linear.weight",)}
    for node in (all_gather, wait, unsharded_param, quantized_operands):
        node.meta["custom"] = fsdp_meta
    qdata = graph.call_function(operator.getitem, args=(quantized_operands, 0))
    fwd_scale = graph.call_function(operator.getitem, args=(quantized_operands, 1))
    dgrad_scale = graph.call_function(operator.getitem, args=(quantized_operands, 2))
    operands = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=(qdata, fwd_scale),
    )
    operands = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=(operands, dgrad_scale),
    )
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(operands, x))
    graph.output((out,))
    return _make_graph_module(graph)


def _make_forward_graph_with_annotated_unshard() -> fx.GraphModule:
    graph = fx.Graph()
    sharded_param = graph.placeholder("sharded_param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(sharded_param, 1, _FAKE_PG),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    view = graph.call_function(torch.ops.aten.view.default, args=(wait, [4]))
    unsharded_param = graph.call_function(torch.ops.aten.clone.default, args=(view,))
    compute = graph.call_function(torch.ops.aten.relu.default, args=(unsharded_param,))
    backward = graph.call_function(torch.ops.aten.neg.default, args=(unsharded_param,))
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(compute, x))
    graph.output((out, backward))

    fsdp_meta = {FSDP_PARAM_FQNS_META: ("linear.weight",)}
    for node in (all_gather, wait, view, unsharded_param):
        node.meta["custom"] = fsdp_meta
    backward.meta["custom"] = fsdp_meta
    backward.meta["autograd_backward"] = True
    return _make_graph_module(graph)


def _make_forward_graph_with_direct_unshard() -> fx.GraphModule:
    graph = fx.Graph()
    sharded_param = graph.placeholder("sharded_param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(sharded_param, 1, _FAKE_PG),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    fsdp_meta = {FSDP_PARAM_FQNS_META: ("linear.weight",)}
    for node in (all_gather, wait):
        node.meta["custom"] = fsdp_meta
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(wait, x))
    graph.output((out,))
    return _make_graph_module(graph)


def _make_forward_graph_with_dense_and_expert_unshards() -> fx.GraphModule:
    graph = fx.Graph()
    outputs = []
    with FakeTensorMode() as fake_mode:
        specs = (
            ("dense_param_0", 2, 4, _FAKE_PG, "dp_shard"),
            ("dense_param_1", 3, 4, _FAKE_PG, "dp_shard"),
            ("expert_param", 4, 2, _FAKE_PG_2, "efsdp"),
        )
        for name, size, group_size, group_name, mesh_axis_name in specs:
            param = graph.placeholder(name)
            param.meta["val"] = fake_mode.from_tensor(torch.empty(size))
            all_gather = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(param, group_size, group_name),
            )
            wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(all_gather,),
            )
            gathered = fake_mode.from_tensor(torch.empty(size * group_size))
            fsdp_meta = {
                FSDP_PARAM_FQNS_META: (name,),
                FSDP_MESH_AXIS_NAMES_META: (mesh_axis_name,),
            }
            for node in (all_gather, wait):
                node.meta["val"] = gathered
                node.meta["custom"] = fsdp_meta
            outputs.append(wait)
    graph.output(tuple(outputs))
    return _make_graph_module(graph)


def _make_joint_graph_with_fsdp_boundaries() -> fx.GraphModule:
    graph = fx.Graph()
    sharded_param = graph.placeholder("sharded_param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(sharded_param, 1, _FAKE_PG),
    )
    wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_gather,),
    )
    fsdp_meta = {FSDP_PARAM_FQNS_META: ("linear.weight",)}
    for node in (all_gather, wait):
        node.meta["custom"] = fsdp_meta
    forward = graph.call_function(torch.ops.aten.mul.Tensor, args=(wait, x))
    loss = graph.call_function(torch.ops.aten.sum.default, args=(forward,))
    raw_grad = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=(sharded_param, x),
    )
    raw_grad.meta["autograd_backward"] = True
    cast = graph.call_function(
        torch.ops.aten._to_copy.default,
        args=(raw_grad,),
        kwargs={"dtype": torch.float32},
    )
    reduce_scatter = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        args=(cast, "sum", 1, _FAKE_PG),
    )
    reduce_scatter_wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(reduce_scatter,),
    )
    graph.output((loss, reduce_scatter_wait))
    return _make_graph_module(graph)


def _make_forward_graph_without_fsdp() -> fx.GraphModule:
    graph = fx.Graph()
    param = graph.placeholder("param")
    x = graph.placeholder("x")
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(param, x))
    graph.output((out,))
    return _make_graph_module(graph)


def _make_forward_graph_without_wait() -> fx.GraphModule:
    graph = fx.Graph()
    param = graph.placeholder("param")
    x = graph.placeholder("x")
    all_gather = graph.call_function(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        args=(param, 1, _FAKE_PG),
    )
    all_gather.meta["custom"] = {FSDP_PARAM_FQNS_META: ("linear.weight",)}
    out = graph.call_function(torch.ops.aten.add.Tensor, args=(all_gather, x))
    graph.output((out,))
    return _make_graph_module(graph)


def _make_backward_graph_with_reduce_grad_epilogues() -> fx.GraphModule:
    graph = fx.Graph()
    fsdp_grad = graph.placeholder("fsdp_grad")
    ddp_grad = graph.placeholder("ddp_grad")
    input_grad = graph.placeholder("input_grad")
    cast = graph.call_function(
        torch.ops.aten._to_copy.default,
        args=(fsdp_grad,),
        kwargs={"dtype": torch.float32},
    )
    reduce_scatter = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        args=(cast, "sum", 1, _FAKE_PG),
    )
    reduce_scatter_wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(reduce_scatter,),
    )
    all_reduce = graph.call_function(
        torch.ops._c10d_functional.all_reduce.default,
        args=(ddp_grad, "sum", _FAKE_PG),
    )
    all_reduce_wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(all_reduce,),
    )
    graph.output((reduce_scatter_wait, all_reduce_wait, None, input_grad))
    return _make_graph_module(graph)


def _make_non_fsdp_split_backward_graph() -> fx.GraphModule:
    graph = fx.Graph()
    grad = graph.placeholder("grad")
    split = graph.call_function(
        torch.ops.aten.split_with_sizes.default,
        args=(grad, [2, 3], 0),
    )
    first = graph.call_function(operator.getitem, args=(split, 0))
    graph.output((first,))
    return _make_graph_module(graph)


def _make_backward_graph_without_fsdp() -> fx.GraphModule:
    graph = fx.Graph()
    grad = graph.placeholder("grad")
    out = graph.call_function(torch.ops.aten.neg.default, args=(grad,))
    graph.output((out,))
    return _make_graph_module(graph)


class GraphPPActionBucketingTest(unittest.TestCase):
    def _assert_action_merge(
        self,
        collective: str,
        pass_fn,
        expected_collective_target,
        expected_pre_bucket_target,
    ) -> None:
        gm = _make_unbucketed_action_graph(collective)
        real_inputs = (torch.tensor([1.0, -2.0]), torch.tensor([3.0, 4.0, 5.0]))
        expected = _FakeCollectiveInterpreter(gm).run(*real_inputs)
        input_names = _placeholder_names(gm)
        num_outputs = len(gm.graph.find_nodes(op="output")[0].args[0])

        with (
            patch(
                "torch.distributed.distributed_c10d._resolve_process_group",
                return_value=object(),
            ),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            pass_fn(gm)

        actual = _FakeCollectiveInterpreter(gm).run(*real_inputs)
        _assert_tensor_sequence_equal(self, actual, expected)
        self.assertEqual(_placeholder_names(gm), input_names)
        self.assertEqual(len(gm.graph.find_nodes(op="output")[0].args[0]), num_outputs)
        self.assertEqual(
            sum(node.target == expected_collective_target for node in gm.graph.nodes),
            1,
        )
        self.assertEqual(
            sum(node.target == expected_pre_bucket_target for node in gm.graph.nodes),
            1,
        )

    def test_unshard_action_merges_all_gathers(self) -> None:
        self._assert_action_merge(
            "all_gather",
            merge_all_all_gathers,
            torch.ops._c10d_functional.all_gather_into_tensor_out.default,
            torch.ops.bucketing._pre_bucket_all_gather.default,
        )

    def test_reduce_grad_action_merges_reduce_scatters(self) -> None:
        self._assert_action_merge(
            "reduce_scatter",
            merge_all_reduce_scatters,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            torch.ops.bucketing._pre_bucket_reduce_scatter.default,
        )

    def test_reduce_grad_action_merges_all_reduces(self) -> None:
        self._assert_action_merge(
            "all_reduce",
            merge_all_all_reduces,
            torch.ops._c10d_functional.all_reduce.default,
            torch.ops.aten.cat.default,
        )

    def test_reduce_grad_action_sorts_interleaved_reduction_inputs(self) -> None:
        gm = _make_interleaved_reduce_grad_action_graph()
        real_inputs = (torch.tensor([1.0, -2.0]), torch.tensor([3.0, 4.0, 5.0]))
        expected = _FakeCollectiveInterpreter(gm).run(*real_inputs)

        with (
            patch(
                "torch.distributed.distributed_c10d._resolve_process_group",
                return_value=object(),
            ),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            merge_all_reduce_scatters(gm)
            merge_all_all_reduces(gm)

        gm.graph.lint()
        actual = _FakeCollectiveInterpreter(gm).run(*real_inputs)
        _assert_tensor_sequence_equal(self, actual, expected)
        self.assertEqual(
            sum(
                node.target == torch.ops._c10d_functional.reduce_scatter_tensor.default
                for node in gm.graph.nodes
            ),
            1,
        )
        self.assertEqual(
            sum(
                node.target == torch.ops._c10d_functional.all_reduce.default
                for node in gm.graph.nodes
            ),
            1,
        )

    def _assert_joint_bucketing_filter(
        self,
        collective: str,
        *,
        bucket_all_gathers: bool,
        bucket_reduce_scatters: bool,
        bucket_all_reduces: bool,
    ) -> None:
        gm = _make_unbucketed_action_graph(collective)
        real_inputs = (torch.tensor([1.0, -2.0]), torch.tensor([3.0, 4.0, 5.0]))
        expected = _FakeCollectiveInterpreter(gm).run(*real_inputs)
        with (
            patch(
                "torch.distributed.distributed_c10d._resolve_process_group",
                return_value=object(),
            ),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            joint_transformer_block_bucketing_reordering_pass(
                gm,
                module_bucket_plans=[["layers.0.part0", "layers.0.part1"]],
                bucket_mode="custom_ops",
                bucket_all_gathers=bucket_all_gathers,
                bucket_reduce_scatters=bucket_reduce_scatters,
                bucket_all_reduces=bucket_all_reduces,
            )

        actual = _FakeCollectiveInterpreter(gm).run(*real_inputs)
        _assert_tensor_sequence_equal(self, actual, expected)
        enabled = {
            "all_gather": bucket_all_gathers,
            "reduce_scatter": bucket_reduce_scatters,
            "all_reduce": bucket_all_reduces,
        }[collective]
        if collective == "all_gather":
            original_target = torch.ops._c10d_functional.all_gather_into_tensor.default
            merged_target = (
                torch.ops._c10d_functional.all_gather_into_tensor_out.default
            )
            pre_bucket_target = torch.ops.bucketing._pre_bucket_all_gather.default
            self.assertEqual(
                sum(node.target == original_target for node in gm.graph.nodes),
                0 if enabled else 2,
            )
            self.assertEqual(
                sum(node.target == merged_target for node in gm.graph.nodes),
                1 if enabled else 0,
            )
        else:
            merged_target = (
                torch.ops._c10d_functional.reduce_scatter_tensor.default
                if collective == "reduce_scatter"
                else torch.ops._c10d_functional.all_reduce.default
            )
            pre_bucket_target = (
                torch.ops.bucketing._pre_bucket_reduce_scatter.default
                if collective == "reduce_scatter"
                else torch.ops.aten.cat.default
            )
            self.assertEqual(
                sum(node.target == merged_target for node in gm.graph.nodes),
                1 if enabled else 2,
            )
        self.assertEqual(
            sum(node.target == pre_bucket_target for node in gm.graph.nodes),
            1 if enabled else 0,
        )

    def test_joint_bucketing_collective_type_filters(self) -> None:
        for collective in ("all_gather", "reduce_scatter", "all_reduce"):
            for (bucket_all_gathers, bucket_reduce_scatters, bucket_all_reduces,) in (
                (True, False, False),
                (False, True, False),
                (False, False, True),
            ):
                with self.subTest(
                    collective=collective,
                    bucket_all_gathers=bucket_all_gathers,
                    bucket_reduce_scatters=bucket_reduce_scatters,
                    bucket_all_reduces=bucket_all_reduces,
                ):
                    self._assert_joint_bucketing_filter(
                        collective,
                        bucket_all_gathers=bucket_all_gathers,
                        bucket_reduce_scatters=bucket_reduce_scatters,
                        bucket_all_reduces=bucket_all_reduces,
                    )

    def test_deferred_actions_configure_joint_bucketing(self) -> None:
        for extract_unshard, extract_reduce_grad in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                extract_unshard=extract_unshard,
                extract_reduce_grad=extract_reduce_grad,
            ):
                passes = [
                    functools.partial(
                        joint_transformer_block_bucketing_reordering_pass,
                        module_bucket_plans=[["layers.0.part0", "layers.0.part1"]],
                        bucket_mode="custom_ops",
                    )
                ]
                original_pass = passes[0]
                if extract_unshard or extract_reduce_grad:
                    configured_passes = _defer_fsdp_action_bucketing(
                        passes,
                        bucket_all_gathers=not extract_unshard,
                        bucket_reduce_scatters=not extract_reduce_grad,
                        bucket_all_reduces=not extract_reduce_grad,
                    )
                    if extract_unshard and extract_reduce_grad:
                        self.assertEqual(configured_passes, [])
                    else:
                        self.assertEqual(len(configured_passes), 1)
                        self.assertEqual(
                            configured_passes[0].keywords["bucket_all_gathers"],
                            not extract_unshard,
                        )
                        self.assertEqual(
                            configured_passes[0].keywords["bucket_reduce_scatters"],
                            not extract_reduce_grad,
                        )
                        self.assertEqual(
                            configured_passes[0].keywords["bucket_all_reduces"],
                            not extract_reduce_grad,
                        )
                else:
                    self.assertEqual(passes, [original_pass])


class GraphPPFSDPCollectiveSplitTest(unittest.TestCase):
    def test_forward_pattern_batch_lookup(self) -> None:
        gm = _make_forward_graph_with_unshard_and_replicated_param()
        deduplicate_fsdp_unshard_chains_pass(gm)
        sharded_param, replicated_param, _ = gm.graph.find_nodes(op="placeholder")

        outputs_by_param = find_fsdp_unshard_outputs_by_param(
            (sharded_param, replicated_param)
        )

        self.assertEqual(len(outputs_by_param[sharded_param]), 1)
        self.assertEqual(outputs_by_param[replicated_param], ())

    def test_forward_pattern_matches_reshard_force_save_pattern(self) -> None:
        gm = _make_forward_graph_with_unshard_and_replicated_param()
        deduplicate_fsdp_unshard_chains_pass(gm)
        sharded_param = gm.graph.find_nodes(op="placeholder")[0]
        save_nodes = find_fsdp_unshard_save_nodes(sharded_param)

        self.assertEqual(len(save_nodes), 1)
        self.assertIs(find_fsdp_unshard_output(sharded_param), save_nodes[0])
        self.assertIs(find_fsdp_unshard_save_node(sharded_param), save_nodes[0])

    def test_forward_split_extracts_unshard_and_replicated_params(self) -> None:
        gm = _make_forward_graph_with_unshard_and_replicated_param()
        deduplicate_fsdp_unshard_chains_pass(gm)

        split = extract_fsdp_unshard_graph(
            gm,
            num_params=2,
            input_names=("sharded_param", "replicated_param", "x"),
            flat_input_indices=(0, 1, 2),
        )

        self.assertIsNotNone(split.unshard_module)
        if split.unshard_module is None:
            self.fail("Expected forward FSDP split to extract an unshard graph")
        self.assertEqual(
            _placeholder_names(split.unshard_module),
            ("sharded_param", "replicated_param"),
        )
        self.assertEqual(split.unshard_flat_param_indices, (0, 1))
        self.assertEqual(split.num_compute_param_inputs, 2)
        self.assertEqual(split.compute_flat_input_indices, (2,))
        self.assertIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _call_targets(split.unshard_module),
        )
        self.assertIn(torch.ops.aten.cat.default, _call_targets(split.unshard_module))
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _call_targets(split.compute_module),
        )

    def test_forward_split_excludes_expert_fsdp_from_unshard_bucket(self) -> None:
        gm = _make_forward_graph_with_dense_and_expert_unshards()
        split = extract_fsdp_unshard_graph(
            gm,
            num_params=3,
            input_names=("dense_param_0", "dense_param_1", "expert_param"),
            flat_input_indices=(0, 1, 2),
        )

        self.assertIsNotNone(split.unshard_module)
        if split.unshard_module is None:
            self.fail("Expected dense FSDP all-gathers to be extracted")
        unshard_all_gathers = split.unshard_module.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        compute_all_gathers = split.compute_module.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )
        self.assertEqual(
            {node.args[2] for node in unshard_all_gathers},
            {_FAKE_PG},
        )
        self.assertEqual(
            {node.args[2] for node in compute_all_gathers},
            {_FAKE_PG_2},
        )

        with (
            patch(
                "torch.distributed.distributed_c10d._resolve_process_group",
                return_value=object(),
            ),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            merge_all_all_gathers(split.unshard_module)

        self.assertEqual(
            sum(
                node.target
                == torch.ops._c10d_functional.all_gather_into_tensor_out.default
                for node in split.unshard_module.graph.nodes
            ),
            1,
        )

    def test_forward_split_passes_through_shard_saved_for_backward(self) -> None:
        gm = _make_forward_graph_with_direct_unshard()
        sharded_param = gm.graph.find_nodes(op="placeholder")[0]
        output = gm.graph.find_nodes(op="output")[0]
        (forward_output,) = output.args[0]
        output.args = ((forward_output, sharded_param),)
        gm.graph.lint()
        gm.recompile()
        deduplicate_fsdp_unshard_chains_pass(gm)

        split = extract_fsdp_unshard_graph(
            gm,
            num_params=1,
            input_names=("sharded_param", "x"),
            flat_input_indices=(0, 1),
        )

        self.assertIsNotNone(split.unshard_module)
        if split.unshard_module is None:
            self.fail("Expected forward FSDP split to extract an unshard graph")
        self.assertEqual(split.num_compute_param_inputs, 2)
        self.assertEqual(len(_placeholder_names(split.compute_module)), 3)
        unshard_output = split.unshard_module.graph.find_nodes(op="output")[0]
        self.assertEqual(len(unshard_output.all_input_nodes), 2)

    def test_unshard_extraction_preserves_joint_backward_parameter_dependency(
        self,
    ) -> None:
        gm = _make_joint_graph_with_fsdp_boundaries()
        deduplicate_fsdp_unshard_chains_pass(gm)

        extraction = extract_fsdp_unshard_graph(
            gm,
            num_params=1,
            input_names=("sharded_param", "x"),
            flat_input_indices=(0, 1),
            extract_fsdp_param_unshard=True,
        )

        self.assertIsNotNone(extraction.unshard_module)
        if extraction.unshard_module is None:
            self.fail("Expected joint FSDP graph to extract an unshard graph")
        self.assertEqual(extraction.num_compute_param_inputs, 2)
        self.assertEqual(extraction.compute_flat_input_indices, (1,))

        sharded_param = torch.randn(4)
        x = torch.randn(4)
        expected = _FakeCollectiveInterpreter(gm).run(sharded_param, x)
        parameter_values = _FakeCollectiveInterpreter(extraction.unshard_module).run(
            sharded_param
        )
        actual = _FakeCollectiveInterpreter(extraction.compute_module).run(
            *parameter_values, x
        )
        _assert_tensor_sequence_equal(self, actual, expected)

    def test_joint_unshard_and_reduce_grad_extractions_compose(self) -> None:
        gm = _make_joint_graph_with_fsdp_boundaries()
        deduplicate_fsdp_unshard_chains_pass(gm)

        reduce_grad = extract_fsdp_reduce_grad_graph(
            gm,
            num_param_grads=1,
            param_grad_output_start=1,
            extract_grad_reduction=True,
        )
        unshard = extract_fsdp_unshard_graph(
            reduce_grad.compute_module,
            num_params=1,
            input_names=("sharded_param", "x"),
            flat_input_indices=(0, 1),
            extract_fsdp_param_unshard=True,
        )

        self.assertIsNotNone(unshard.unshard_module)
        self.assertIsNotNone(reduce_grad.reduce_grad_module)
        if unshard.unshard_module is None or reduce_grad.reduce_grad_module is None:
            self.fail("Expected both FSDP boundaries to be extracted")

        sharded_param = torch.randn(4)
        x = torch.randn(4)
        expected = _FakeCollectiveInterpreter(gm).run(sharded_param, x)
        parameter_values = _FakeCollectiveInterpreter(unshard.unshard_module).run(
            sharded_param
        )
        compute_outputs = _FakeCollectiveInterpreter(unshard.compute_module).run(
            *parameter_values, x
        )
        compute_grads_by_name = dict(
            zip(
                unshard.compute_output_names[1:],
                compute_outputs[1:],
                strict=True,
            )
        )
        reduce_grad_inputs = [
            compute_grads_by_name[name] for name in reduce_grad.reduce_grad_input_names
        ]
        reduced_grads = _FakeCollectiveInterpreter(reduce_grad.reduce_grad_module).run(
            *reduce_grad_inputs
        )
        actual = (compute_outputs[0], *reduced_grads)
        _assert_tensor_sequence_equal(self, actual, expected)

    def test_forward_split_keeps_quantization_in_unshard_chain(self) -> None:
        gm = _make_forward_graph_with_quantized_unshard()
        deduplicate_fsdp_unshard_chains_pass(gm)
        sharded_param = gm.graph.find_nodes(op="placeholder")[0]
        unshard_outputs = find_fsdp_unshard_save_nodes(sharded_param)

        self.assertEqual(len(unshard_outputs), 1)
        self.assertIs(unshard_outputs[0].target, _quantize_weight_for_graph_test)

        split = extract_fsdp_unshard_graph(
            gm,
            num_params=2,
            input_names=("sharded_param", "x"),
            flat_input_indices=(0, 2),
        )

        self.assertIsNotNone(split.unshard_module)
        if split.unshard_module is None:
            self.fail("Expected quantized FSDP unshard graph")
        self.assertIn(
            _quantize_weight_for_graph_test,
            _call_targets(split.unshard_module),
        )
        self.assertNotIn(
            _quantize_weight_for_graph_test,
            _call_targets(split.compute_module),
        )
        self.assertIn(operator.getitem, _call_targets(split.compute_module))
        self.assertEqual(split.compute_flat_input_indices, (2,))

    def test_forward_trace_annotation_stops_before_compute(self) -> None:
        gm = _make_forward_graph_with_annotated_unshard()
        sharded_param = gm.graph.find_nodes(op="placeholder")[0]

        (unshard_output,) = find_fsdp_unshard_save_nodes(sharded_param)

        self.assertIs(unshard_output.target, torch.ops.aten.clone.default)

    def test_forward_trace_annotation_includes_multi_input_reconstruction(self) -> None:
        gm = _make_forward_graph_with_annotated_unshard()
        graph = gm.graph
        sharded_param, reconstruction_input = graph.find_nodes(op="placeholder")
        (unsharded_param,) = graph.find_nodes(
            op="call_function", target=torch.ops.aten.clone.default
        )
        forward_consumer = next(
            user
            for user in unsharded_param.users
            if not user.meta.get("autograd_backward", False)
        )
        with graph.inserting_before(forward_consumer):
            reconstruction = graph.call_function(
                torch.ops.aten.mul.Tensor,
                args=(unsharded_param, reconstruction_input),
            )
        reconstruction.meta["custom"] = {FSDP_PARAM_FQNS_META: ("linear.weight",)}
        forward_consumer.replace_input_with(unsharded_param, reconstruction)
        graph.lint()
        gm.recompile()

        (unshard_output,) = find_fsdp_unshard_save_nodes(sharded_param)

        self.assertIs(unshard_output, reconstruction)

    def test_forward_trace_annotation_ignores_rematerialized_output(self) -> None:
        gm = _make_forward_graph_with_annotated_unshard()
        deduplicate_fsdp_unshard_chains_pass(gm)
        sharded_param = gm.graph.find_nodes(op="placeholder")[0]
        (unshard_output,) = find_fsdp_unshard_save_nodes(sharded_param)
        unshard_output.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE

        selective_activation_remat_pass(gm)

        self.assertEqual(find_fsdp_unshard_save_nodes(sharded_param), (unshard_output,))

    def test_forward_trace_annotation_ignores_unrelated_placeholder(self) -> None:
        gm = _make_forward_graph_with_annotated_unshard()
        deduplicate_fsdp_unshard_chains_pass(gm)
        sharded_param, unrelated = gm.graph.find_nodes(op="placeholder")
        (unshard_output,) = find_fsdp_unshard_save_nodes(sharded_param)
        unrelated.meta.update(unshard_output.meta)

        self.assertEqual(find_fsdp_unshard_save_nodes(sharded_param), (unshard_output,))

    def test_forward_trace_annotation_accepts_mutated_output_buffer(self) -> None:
        gm = _make_forward_graph_with_annotated_unshard()
        deduplicate_fsdp_unshard_chains_pass(gm)
        graph = gm.graph
        sharded_param = graph.find_nodes(op="placeholder")[0]
        (unshard_output,) = find_fsdp_unshard_save_nodes(sharded_param)
        source = unshard_output.args[0]
        with graph.inserting_before(unshard_output):
            buffer = graph.call_function(
                torch.ops.aten.empty.memory_format,
                args=([4],),
            )
            graph.call_function(torch.ops.aten.copy_.default, args=(buffer, source))
        unshard_output.replace_input_with(source, buffer)
        graph.lint()

        self.assertEqual(find_fsdp_unshard_save_nodes(sharded_param), (unshard_output,))

        traced = minimal_fx_tracer(lambda p, x: [p + x, -p])(
            torch.randn(4), torch.randn(4)
        )
        traced.gm = gm
        _, bw_module, meta = partition_joint_graph(traced, num_fwd_outputs=1)
        self.assertIn(sharded_param.name, meta.saved_for_backward_names)
        self.assertIn(unshard_output.name, meta.saved_for_backward_names)
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _call_targets(bw_module),
        )

    def test_forward_trace_annotation_rejects_mismatched_parameter(self) -> None:
        gm = _make_forward_graph_with_annotated_unshard()
        sharded_param = gm.graph.find_nodes(op="placeholder")[0]
        (wait,) = gm.graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.wait_tensor.default,
        )
        wait.meta["custom"] = {FSDP_PARAM_FQNS_META: ("other.weight",)}

        with self.assertRaisesRegex(ValueError, "trace metadata does not match"):
            find_fsdp_unshard_save_nodes(sharded_param)

    def test_forward_trace_ignores_unannotated_all_gather(self) -> None:
        gm = _make_forward_graph_with_direct_unshard()
        sharded_param = gm.graph.find_nodes(op="placeholder")[0]
        for node in gm.graph.nodes:
            node.meta.pop("custom", None)

        self.assertEqual(find_fsdp_unshard_save_nodes(sharded_param), ())

    def test_forward_trace_ignores_unannotated_downstream_all_gather(self) -> None:
        graph = fx.Graph()
        param = graph.placeholder("param")
        annotated_view = graph.call_function(
            torch.ops.aten.view.default,
            args=(param, [4]),
        )
        annotated_view.meta["custom"] = {FSDP_PARAM_FQNS_META: ("linear.weight",)}
        compute = graph.call_function(
            torch.ops.aten.relu.default, args=(annotated_view,)
        )
        all_gather = graph.call_function(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            args=(compute, 1, _FAKE_PG),
        )
        wait = graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            args=(all_gather,),
        )
        graph.output((wait,))
        gm = _make_graph_module(graph)

        self.assertEqual(find_fsdp_unshard_save_nodes(param), ())

    def test_forward_trace_annotation_rejects_lost_consumer_input(self) -> None:
        gm = _make_forward_graph_with_annotated_unshard()
        deduplicate_fsdp_unshard_chains_pass(gm)
        sharded_param = gm.graph.find_nodes(op="placeholder")[0]
        (unshard_output,) = find_fsdp_unshard_save_nodes(sharded_param)
        forward_consumer = next(
            user
            for user in unshard_output.users
            if not user.meta.get("autograd_backward", False)
        )
        unshard_output.meta.clear()
        forward_consumer.args = ()

        with self.assertRaisesRegex(ValueError, "lost input path"):
            find_fsdp_unshard_save_nodes(sharded_param)

    def test_forward_trace_annotation_rejects_lost_boundary(self) -> None:
        gm = _make_forward_graph_with_annotated_unshard()
        deduplicate_fsdp_unshard_chains_pass(gm)
        sharded_param = gm.graph.find_nodes(op="placeholder")[0]
        (unshard_output,) = find_fsdp_unshard_save_nodes(sharded_param)
        unshard_output.meta.clear()
        for consumer in unshard_output.users:
            consumer.meta.clear()

        with self.assertRaisesRegex(ValueError, "lost its annotated unshard output"):
            find_fsdp_unshard_save_nodes(sharded_param)

    def test_forward_trace_annotation_rejects_unrelated_replacement(self) -> None:
        gm = _make_forward_graph_with_annotated_unshard()
        deduplicate_fsdp_unshard_chains_pass(gm)
        sharded_param, unrelated = gm.graph.find_nodes(op="placeholder")
        (unshard_output,) = find_fsdp_unshard_save_nodes(sharded_param)
        unshard_output.meta.clear()
        unshard_output.replace_all_uses_with(unrelated)

        with self.assertRaisesRegex(ValueError, "no longer depends on parameter"):
            find_fsdp_unshard_save_nodes(sharded_param)

    def test_forward_split_finds_bucketed_unshard_from_annotation(self) -> None:
        gm = _make_forward_graph_with_direct_unshard()
        deduplicate_fsdp_unshard_chains_pass(gm)
        graph = gm.graph
        sharded_param = graph.find_nodes(op="placeholder")[0]
        old_all_gather = graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        )[0]
        old_wait = graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.wait_tensor.default,
        )[0]

        with graph.inserting_before(old_all_gather):
            bucket = graph.call_function(
                torch.ops.bucketing._pre_bucket_all_gather.default,
                args=([sharded_param], 1, torch.float32, [0], 0),
            )
            shard = graph.call_function(
                torch.ops.aten.slice.Tensor,
                args=(bucket, 0, 0, 4),
            )
            bucket_all_gather = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor_out.default,
                args=(shard, 1, _FAKE_PG),
                kwargs={"out": bucket},
            )
            bucket_wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                args=(bucket_all_gather,),
            )
            replacement = graph.call_function(
                torch.ops.aten.view.default,
                args=(bucket_wait, [4]),
            )
        old_wait.replace_all_uses_with(replacement)
        graph.erase_node(old_wait)
        graph.erase_node(old_all_gather)
        output = graph.find_nodes(op="output")[0]
        output.args = ((*output.args[0], sharded_param, bucket_all_gather),)
        graph.lint()
        gm.recompile()

        self.assertEqual(find_fsdp_unshard_save_nodes(sharded_param), (replacement,))
        split = extract_fsdp_unshard_graph(
            gm,
            num_params=1,
            input_names=("sharded_param", "x"),
            flat_input_indices=(0, 1),
            side_effect_output_names=(bucket_all_gather.name,),
        )

        self.assertIsNotNone(split.unshard_module)
        if split.unshard_module is None:
            self.fail("Expected bucketed FSDP unshard graph")
        self.assertIn(
            torch.ops._c10d_functional.all_gather_into_tensor_out.default,
            _call_targets(split.unshard_module),
        )
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor_out.default,
            _call_targets(split.compute_module),
        )
        self.assertNotIn(
            bucket_all_gather.name,
            split.compute_output_names,
        )

    def test_forward_split_no_fsdp_is_noop(self) -> None:
        gm = _make_forward_graph_without_fsdp()
        split = extract_fsdp_unshard_graph(
            gm,
            num_params=1,
            input_names=("param", "x"),
            flat_input_indices=(0, 1),
        )

        self.assertIsNone(split.unshard_module)
        self.assertIs(split.compute_module, gm)
        self.assertEqual(split.compute_input_names, ("param", "x"))
        self.assertEqual(split.compute_flat_input_indices, (0, 1))

    def test_forward_split_requires_wait_after_all_gather(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected wait_tensor"):
            extract_fsdp_unshard_graph(
                _make_forward_graph_without_wait(),
                num_params=1,
                input_names=("param", "x"),
                flat_input_indices=(0, 1),
            )

    def test_backward_split_extracts_reduce_grad_epilogues(self) -> None:
        split = extract_fsdp_reduce_grad_graph(
            _make_backward_graph_with_reduce_grad_epilogues(),
            num_param_grads=3,
        )

        self.assertIsNotNone(split.reduce_grad_module)
        if split.reduce_grad_module is None:
            self.fail("Expected backward FSDP split to extract reduce-grad graph")
        compute_targets = _call_targets(split.compute_module)
        reduce_grad_targets = _call_targets(split.reduce_grad_module)
        self.assertIn(torch.ops.aten._to_copy.default, compute_targets)
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            compute_targets,
        )
        self.assertNotIn(
            torch.ops._c10d_functional.all_reduce.default,
            compute_targets,
        )
        self.assertNotIn(torch.ops.aten._to_copy.default, reduce_grad_targets)
        self.assertIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            reduce_grad_targets,
        )
        self.assertIn(
            torch.ops._c10d_functional.all_reduce.default,
            reduce_grad_targets,
        )
        self.assertEqual(
            split.reduce_grad_input_names,
            split.compute_output_names[:2],
        )
        self.assertEqual(len(split.compute_output_names), 4)
        self.assertEqual(split.compute_output_names[-1], "input_grad")

    def test_backward_split_keeps_expert_fsdp_reduction_in_compute(self) -> None:
        gm = _make_backward_graph_with_reduce_grad_epilogues()
        reduce_scatter = next(
            node
            for node in gm.graph.nodes
            if node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
        )
        cast = reduce_scatter.all_input_nodes[0]
        (wait,) = reduce_scatter.users
        for node in (cast, reduce_scatter, wait):
            node.meta["custom"] = {FSDP_MESH_AXIS_NAMES_META: ("efsdp",)}

        split = extract_fsdp_reduce_grad_graph(gm, num_param_grads=3)

        self.assertIsNotNone(split.reduce_grad_module)
        if split.reduce_grad_module is None:
            self.fail("Expected the dense reduction to be extracted")
        self.assertIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            _call_targets(split.compute_module),
        )
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            _call_targets(split.reduce_grad_module),
        )
        self.assertIn(
            torch.ops._c10d_functional.all_reduce.default,
            _call_targets(split.reduce_grad_module),
        )

    def test_reduce_grad_split_preserves_leading_joint_loss(self) -> None:
        gm = _make_backward_graph_with_reduce_grad_epilogues()
        first_compute = next(
            node for node in gm.graph.nodes if node.op != "placeholder"
        )
        with gm.graph.inserting_before(first_compute):
            loss = gm.graph.placeholder("loss")
        output = gm.graph.find_nodes(op="output")[0]
        output.args = ((loss, *output.args[0]),)
        gm.graph.lint()
        gm.recompile()

        split = extract_fsdp_reduce_grad_graph(
            gm,
            num_param_grads=3,
            param_grad_output_start=1,
        )

        self.assertIsNotNone(split.reduce_grad_module)
        self.assertEqual(split.compute_output_names[0], "loss")
        self.assertEqual(
            split.reduce_grad_input_names,
            split.compute_output_names[1:3],
        )
        self.assertEqual(split.compute_output_names[-1], "input_grad")

    def test_unbucketed_split_accumulates_before_one_final_bucket(self) -> None:
        gm = _make_unbucketed_action_graph("reduce_scatter")
        split = extract_fsdp_reduce_grad_graph(gm, num_param_grads=2)
        self.assertIsNotNone(split.reduce_grad_module)
        if split.reduce_grad_module is None:
            self.fail("Expected an extracted reduce-grad graph")

        accumulators = insert_graph_gradient_accumulation(
            split.compute_module,
            num_param_grads=2,
            device=torch.device("cpu"),
        )
        torch._foreach_zero_(list(accumulators))
        microbatches = (
            (torch.tensor([1.0, 3.0]), torch.tensor([-1.0, 2.0, 4.0])),
            (torch.tensor([-2.0, 1.0]), torch.tensor([3.0, 0.5, -1.0])),
            (torch.tensor([4.0, -3.0]), torch.tensor([2.0, -2.0, 0.25])),
        )
        for microbatch in microbatches:
            _FakeCollectiveInterpreter(split.compute_module).run(
                *microbatch,
                *accumulators,
            )

        input_names = _placeholder_names(split.reduce_grad_module)
        num_outputs = len(
            split.reduce_grad_module.graph.find_nodes(op="output")[0].args[0]
        )
        merge_all_reduce_scatters(
            split.reduce_grad_module,
        )
        actual = _FakeCollectiveInterpreter(split.reduce_grad_module).run(*accumulators)
        expected = tuple(sum(values) for values in zip(*microbatches, strict=True))
        _assert_tensor_sequence_equal(self, actual, expected)
        self.assertEqual(_placeholder_names(split.reduce_grad_module), input_names)
        self.assertEqual(
            len(split.reduce_grad_module.graph.find_nodes(op="output")[0].args[0]),
            num_outputs,
        )
        self.assertEqual(
            sum(
                node.target == torch.ops.bucketing._pre_bucket_reduce_scatter.default
                for node in split.reduce_grad_module.graph.nodes
            ),
            1,
        )

    def test_unbucketed_hsdp_chain_is_extracted(self) -> None:
        for order in ("reduce_scatter_first", "all_reduce_first"):
            with self.subTest(order=order):
                graph = fx.Graph()
                grad = graph.placeholder("grad")
                value = grad
                targets = (
                    (
                        torch.ops._c10d_functional.reduce_scatter_tensor.default,
                        torch.ops._c10d_functional.all_reduce.default,
                    )
                    if order == "reduce_scatter_first"
                    else (
                        torch.ops._c10d_functional.all_reduce.default,
                        torch.ops._c10d_functional.reduce_scatter_tensor.default,
                    )
                )
                for target in targets:
                    args = (
                        (value, "sum", 1, _FAKE_PG)
                        if target
                        is torch.ops._c10d_functional.reduce_scatter_tensor.default
                        else (value, "sum", _FAKE_PG)
                    )
                    value = graph.call_function(target, args=args)
                    value = graph.call_function(
                        torch.ops._c10d_functional.wait_tensor.default,
                        args=(value,),
                    )
                graph.output((value,))
                split = extract_fsdp_reduce_grad_graph(
                    _make_graph_module(graph),
                    num_param_grads=1,
                )

                self.assertIsNotNone(split.reduce_grad_module)
                if split.reduce_grad_module is None:
                    self.fail("Expected HSDP reduce-grad graph")
                self.assertEqual(split.reduce_grad_input_names, ("grad",))
                self.assertIn(
                    torch.ops._c10d_functional.reduce_scatter_tensor.default,
                    _call_targets(split.reduce_grad_module),
                )
                self.assertIn(
                    torch.ops._c10d_functional.all_reduce.default,
                    _call_targets(split.reduce_grad_module),
                )

    def test_non_fsdp_split_is_not_a_reduce_grad_candidate(self) -> None:
        gm = _make_non_fsdp_split_backward_graph()
        output = gm.graph.find_nodes(op="output")[0].args[0][0]
        self.assertIsNone(find_fsdp_reduce_grad_input(output))
        split = extract_fsdp_reduce_grad_graph(gm, num_param_grads=1)
        self.assertIs(split.compute_module, gm)
        self.assertIsNone(split.reduce_grad_module)

    def test_backward_split_no_fsdp_is_noop_and_validates_grad_count(self) -> None:
        gm = _make_backward_graph_without_fsdp()
        split = extract_fsdp_reduce_grad_graph(gm, num_param_grads=1)

        self.assertIsNone(split.reduce_grad_module)
        self.assertIs(split.compute_module, gm)

        with self.assertRaisesRegex(ValueError, "num_param_grads cannot exceed"):
            extract_fsdp_reduce_grad_graph(gm, num_param_grads=2)


class GraphPPFSDPCollectiveSplitDsv3Test(_GraphPPDsv3FSDPTest):
    def test_real_dsv3_moe_block_fsdp_split_reconstructs_graphs(self) -> None:
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("real FSDP collective trace requires 2 GPUs")

        self._setup()
        fsdp_mesh = get_simple_fsdp_mesh(self.parallel_dims)
        traced_block = _trace_dsv3_moe_block_stage(fsdp_mesh=fsdp_mesh)
        deduplicate_fsdp_unshard_chains_pass(
            traced_block.traced.gm,
            traced_block.traced.example_inputs,
        )

        fw_module, bw_module, meta = partition_joint_graph(
            traced_block.traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(len(traced_block.traced.example_inputs) - 1,),
        )
        unshard_extraction = extract_fsdp_unshard_graph(
            fw_module,
            num_params=traced_block.num_flat_param_values,
            input_names=meta.fwd_input_names,
            flat_input_indices=meta.fwd_flat_input_indices,
        )
        reduce_grad_extraction = extract_fsdp_reduce_grad_graph(
            bw_module,
            num_param_grads=traced_block.num_param_grad_values,
        )

        self.assertIsNotNone(unshard_extraction.unshard_module)
        self.assertIsNotNone(reduce_grad_extraction.reduce_grad_module)
        if (
            unshard_extraction.unshard_module is None
            or reduce_grad_extraction.reduce_grad_module is None
        ):
            self.fail("Expected real DSV3 FSDP trace to contain split collectives")
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _call_targets(unshard_extraction.compute_module),
        )
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            _call_targets(reduce_grad_extraction.compute_module),
        )

        fw_args = [
            traced_block.flat_inputs[index] for index in meta.fwd_flat_input_indices
        ]
        fw_outputs = _boxed_run(fw_module, list(fw_args))
        unshard_args = [
            traced_block.flat_inputs[index]
            for index in unshard_extraction.unshard_flat_param_indices
        ]
        unsharded_params = _boxed_run(unshard_extraction.unshard_module, unshard_args)
        compute_args = [
            *unsharded_params,
            *(
                traced_block.flat_inputs[index]
                for index in unshard_extraction.compute_flat_input_indices
            ),
        ]
        split_fw_outputs = _boxed_run(
            unshard_extraction.compute_module,
            list(compute_args),
        )
        _assert_tensor_sequence_equal(self, split_fw_outputs, fw_outputs)

        bw_args = _backward_args_from_partition(
            meta,
            fw_outputs,
            (traced_block.output_grad,),
        )
        bw_outputs = _boxed_run(bw_module, list(bw_args))
        compute_outputs = _boxed_run(
            reduce_grad_extraction.compute_module, list(bw_args)
        )
        grad_values_by_name = dict(
            zip(
                reduce_grad_extraction.compute_output_names[
                    : traced_block.num_param_grad_values
                ],
                compute_outputs[: traced_block.num_param_grad_values],
                strict=True,
            )
        )
        reduce_grad_args = [
            grad_values_by_name[name]
            for name in reduce_grad_extraction.reduce_grad_input_names
        ]
        reduced_grads = _boxed_run(
            reduce_grad_extraction.reduce_grad_module, reduce_grad_args
        )
        split_bw_outputs = (
            *reduced_grads,
            *compute_outputs[traced_block.num_param_grad_values :],
        )
        _assert_tensor_sequence_equal(self, split_bw_outputs, bw_outputs)


if __name__ == "__main__":
    unittest.main()
