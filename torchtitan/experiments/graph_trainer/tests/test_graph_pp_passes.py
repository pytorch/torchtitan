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
import torch.utils._pytree as pytree
from graph_trainer.fsdp_passes import (
    deduplicate_fsdp_unshard_chains_pass,
    joint_transformer_block_bucketing_reordering_pass,
    merge_all_all_gathers,
    merge_all_all_reduces,
    merge_all_reduce_scatters,
)
from graph_trainer.graph_pp import (
    extract_fsdp_reduce_grad_graph,
    extract_fsdp_unshard_graph,
    partition_joint_graph,
    split_di_dw_graph,
)
from graph_trainer.graph_pp.partition import GraphMeta
from graph_trainer.graph_pp.utils import flatten_graph_values
from graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
    TracedResult,
)
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.nn.attention.flex_attention import flex_attention
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    get_simple_fsdp_mesh,
    maybe_register_blockmask_pytree_node,
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)
from torchtitan.experiments.graph_trainer.graph_builder_utils import (
    _configure_fsdp_bucketing_pass,
    _find_fsdp_bucketing_pass,
)
from torchtitan.experiments.graph_trainer.passes import apply_graph_passes
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    FSDP_MESH_AXIS_NAMES_META,
    FSDP_PARAM_FQNS_META,
)
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.moe import MoE


@dataclass(frozen=True, slots=True)
class _Dsv3MoeBlockTrace:
    traced: TracedResult
    flat_inputs: list[Any]
    output_grad: torch.Tensor
    num_param_grad_values: int
    num_sharded_param_values: int


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
        parallelism = ParallelismConfig(expert_parallel_degree=2)
        model_config = build_deepseek_v3_model_config(
            "debugmodel", attn_backend="flex", seq_len=seq_len, parallelism=parallelism
        )
        moe_layer_config = model_config.layers[1]
        if not isinstance(moe_layer_config.feed_forward, MoE.Config):
            raise AssertionError("DeepSeek V3 MoE layer must contain an MoE block")

        with (
            patch(
                "torchtitan.distributed.spmd_types.spmd_mesh_size",
                side_effect=lambda axis: 2 if axis == "ep" else 1,
            ),
            torch.device("meta"),
        ):
            model = model_config.build()
        model.to_empty(device="cuda")
        with torch.no_grad():
            model.init_states(buffer_device=None)
        model._apply(
            lambda tensor: (
                tensor.to(dtype=torch.bfloat16)
                if tensor.is_floating_point()
                else tensor
            )
        )
        model.train()

        block = model.layers["1"]
        if not isinstance(block.feed_forward, MoE):
            raise AssertionError("DeepSeek V3 debug layer 1 must be a MoE layer")
        inner_attention_type = type(block.attention.inner_attention)
        if fsdp_mesh is not None:
            block = data_parallel(
                block,
                device_mesh=fsdp_mesh,
                mode="fully_shard",
            )

        num_tokens = batch_size * seq_len
        aux_loss_denominator = torch.tensor(num_tokens, device="cuda")
        x = torch.randn(
            num_tokens,
            model_config.dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=include_input_grad,
        )
        positions = torch.arange(seq_len, device="cuda").repeat(batch_size)
        attention_metadata = model._get_attention_metadata(positions)
        output_grad = torch.randn_like(x)

        def stage_step(
            x: torch.Tensor,
            positions: torch.Tensor,
            attention_metadata: Any,
            output_grad: torch.Tensor,
        ):
            out = block(
                x,
                attention_metadata.get(inner_attention_type),
                positions,
                aux_loss_denominator=aux_loss_denominator,
            )
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
                attention_metadata,
                output_grad,
            )

        user_flat_inputs, _ = pytree.tree_flatten(
            ((x, positions, attention_metadata, output_grad), {})
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
            num_sharded_param_values=len(flatten_graph_values(state_params)),
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

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0),
        "MXFP8 requires SM100 or later",
    )
    def test_partition_saves_mxfp8_wgrad_inputs_across_mutation(self) -> None:
        try:
            from torchtitan.quantization.mxfp8.tensor import _quantize_mxfp8_weight
        except ImportError as error:
            raise unittest.SkipTest("TorchAO MXFP8 is unavailable") from error

        torch.manual_seed(0)
        activation = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn_like(activation)
        with torch.no_grad():
            weight_operands = _quantize_mxfp8_weight(weight)
        weight_qdata = weight_operands.weight_qdata_fprop_KN
        weight_scale = weight_operands.weight_scale_fprop_swizzled
        base_traced = minimal_fx_tracer(lambda a, b, c, d: [a, b, c, d])(
            activation,
            weight_qdata,
            weight_scale,
            grad_output,
        )

        graph = fx.Graph()
        graph_activation = graph.placeholder("activation")
        graph_weight_qdata = graph.placeholder("weight_qdata")
        graph_weight_scale = graph.placeholder("weight_scale")
        graph_grad_output = graph.placeholder("grad_output")

        quantized_activation = graph.call_function(
            torch.ops.torchao.mxfp8_quantize.default,
            args=(graph_activation, True, True, 32, 32, "e4m3", "rceil"),
        )
        quantized_activation.meta["val"] = (None, None, None, None)
        activation_qdata_row = graph.call_function(
            operator.getitem, args=(quantized_activation, 0)
        )
        activation_qdata_col = graph.call_function(
            operator.getitem, args=(quantized_activation, 1)
        )
        activation_scale_row = graph.call_function(
            operator.getitem, args=(quantized_activation, 2)
        )
        activation_scale_col = graph.call_function(
            operator.getitem, args=(quantized_activation, 3)
        )
        activation_scale_row_swizzled = graph.call_function(
            torch.ops.torchao.triton_mx_block_rearrange.default,
            args=(activation_scale_row,),
        )
        activation_scale_col_swizzled = graph.call_function(
            torch.ops.torchao.triton_mx_block_rearrange.default,
            args=(activation_scale_col,),
        )
        output_buffer = graph.call_function(
            torch.ops.aten.empty_like.default,
            args=(graph_activation,),
        )
        graph.call_function(
            torch.ops.aten._scaled_mm_v2.out,
            args=(
                activation_qdata_row,
                graph_weight_qdata,
                [activation_scale_row_swizzled],
                [3],
                [1],
                [graph_weight_scale],
                [3],
                [1],
                None,
                torch.bfloat16,
                [],
                False,
            ),
            kwargs={"out": output_buffer},
        )
        graph.call_function(
            torch.ops.aten.add_.Tensor,
            args=(graph_activation, 256.0),
        )
        fwd_output = graph.call_function(
            torch.ops.aten.cos.default,
            args=(output_buffer,),
        )

        quantized_grad_output = graph.call_function(
            torch.ops.torchao.mxfp8_quantize.default,
            args=(graph_grad_output, False, True, 32, 32, "e4m3", "rceil"),
        )
        grad_output_qdata_col = graph.call_function(
            operator.getitem, args=(quantized_grad_output, 1)
        )
        grad_output_scale_col = graph.call_function(
            operator.getitem, args=(quantized_grad_output, 3)
        )
        grad_output_scale_col_swizzled = graph.call_function(
            torch.ops.torchao.triton_mx_block_rearrange.default,
            args=(grad_output_scale_col,),
        )
        grad_output_qdata_col_t = graph.call_function(
            torch.ops.aten.t.default,
            args=(grad_output_qdata_col,),
        )
        wgrad = graph.call_function(
            torch.ops.aten._scaled_mm_v2.default,
            args=(
                grad_output_qdata_col_t,
                activation_qdata_col,
                [grad_output_scale_col_swizzled],
                [3],
                [1],
                [activation_scale_col_swizzled],
                [3],
                [1],
                None,
                torch.bfloat16,
            ),
        )
        for node in (
            quantized_grad_output,
            grad_output_qdata_col,
            grad_output_scale_col,
            grad_output_scale_col_swizzled,
            grad_output_qdata_col_t,
            wgrad,
        ):
            node.meta["autograd_backward"] = True
        graph.output((fwd_output, wgrad))
        joint = _make_graph_module(graph)
        traced = replace(base_traced, gm=joint)

        fw_module, bw_module, meta = partition_joint_graph(
            traced,
            num_fwd_outputs=1,
            backward_only_input_indices=(3,),
        )

        inputs = [activation, weight_qdata, weight_scale, grad_output]
        joint_inputs = [value.clone() for value in inputs]
        joint_outputs = _boxed_run(joint, joint_inputs)
        split_inputs = [value.clone() for value in inputs]
        fw_args = [split_inputs[index] for index in meta.fwd_flat_input_indices]
        fw_outputs = _boxed_run(fw_module, fw_args)
        bw_args = _backward_args_from_partition(
            meta,
            fw_outputs,
            (split_inputs[3],),
        )
        bw_outputs = _boxed_run(bw_module, bw_args)

        _assert_tensor_sequence_equal(self, fw_outputs[:1], joint_outputs[:1])
        torch.testing.assert_close(
            bw_outputs[0],
            joint_outputs[1],
            rtol=0,
            atol=0,
        )
        self.assertIn(activation_qdata_col.name, meta.saved_for_backward_names)
        self.assertIn(activation_scale_col.name, meta.saved_for_backward_names)
        (backward_quantizer,) = bw_module.graph.find_nodes(
            op="call_function",
            target=torch.ops.torchao.mxfp8_quantize.default,
        )
        self.assertEqual(backward_quantizer.args[0].name, graph_grad_output.name)


class _GraphPPDsv3FSDPTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return max(1, min(torch.cuda.device_count(), 2))

    def _setup(self) -> None:
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


class GraphPPPartitionFSDPTest(_GraphPPDsv3FSDPTest):
    def test_real_dsv3_moe_block_fsdp_partition_matches_joint_graph(self) -> None:
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("real FSDP collective trace requires 2 GPUs")

        self._setup()
        traced_block = _trace_dsv3_moe_block_stage(
            fsdp_mesh=get_simple_fsdp_mesh(self.parallelism_context)
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


def _backward_with_routed_token_live_ins() -> fx.GraphModule:
    """Build a backward graph whose dW live-ins are shaped ``(u0 + u1, D)``.

    This mirrors routed MoE tokens received from an EP all-to-all: ``u0`` and
    ``u1`` are SymInt inputs that bind those shapes, but no dW op reads them.
    """
    aten = torch.ops.aten
    shape_env = ShapeEnv()
    graph = fx.Graph()
    with FakeTensorMode(shape_env=shape_env):
        with shape_env.ignore_fresh_unbacked_symbols():
            u0 = shape_env.create_unbacked_symint()
            u1 = shape_env.create_unbacked_symint()
        placeholder_vals = {
            "u0": u0,
            "u1": u1,
            "x": torch.empty(u0 + u1, 4),
            "w": torch.empty(3, 4),
            "tangents_1": torch.empty(u0 + u1, 3),
        }
        nodes = {}
        for name, val in placeholder_vals.items():
            nodes[name] = graph.placeholder(name)
            nodes[name].meta["val"] = val

        def call(target, *args: fx.Node) -> fx.Node:
            node = graph.call_function(target, args)
            node.meta["val"] = target(*(arg.meta["val"] for arg in args))
            return node

        grad_x = call(aten.mm.default, nodes["tangents_1"], nodes["w"])
        grad_out_t = call(aten.t.default, nodes["tangents_1"])
        grad_w = call(aten.mm.default, grad_out_t, nodes["x"])
        graph.output((grad_w, grad_x))
    return _make_graph_module(graph)


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


def _make_unbucketed_action_graph(
    collective: str,
    *,
    separate_process_groups: bool = False,
) -> fx.GraphModule:
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
        group_name = _FAKE_PG_2 if separate_process_groups and index == 1 else _FAKE_PG
        if collective == "all_gather":
            start = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor.default,
                args=(value, 1, group_name),
            )
        elif collective == "reduce_scatter":
            start = graph.call_function(
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                args=(value, "sum", 1, group_name),
            )
        elif collective == "all_reduce":
            start = graph.call_function(
                torch.ops._c10d_functional.all_reduce.default,
                args=(value, "sum", group_name),
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


def _make_multilayer_unbucketed_action_graph(collective: str) -> fx.GraphModule:
    graph = fx.Graph()
    outputs = []
    with FakeTensorMode() as fake_mode:
        values = []
        for layer_id in range(2):
            for part_id in range(2):
                name = f"layer_{layer_id}_part_{part_id}"
                value = graph.placeholder(name)
                fake_value = fake_mode.from_tensor(torch.empty(2 + part_id))
                value.meta["val"] = fake_value
                values.append((layer_id, part_id, value, fake_value))
        for layer_id, part_id, value, fake_value in values:
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
            module_fqn = f"layers.{layer_id}.part{part_id}"
            for node in (start, wait):
                node.meta["val"] = fake_value
                node.meta["custom"] = {
                    _MODULE_FQN: module_fqn,
                    FSDP_PARAM_FQNS_META: (f"{module_fqn}.weight",),
                }
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
            ("expert_param", 4, 2, _FAKE_PG_2, "edp_shard"),
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


def _make_backward_graph_with_reduce_grad_layout(
    *,
    reduce_dtype_cast: bool = False,
) -> fx.GraphModule:
    graph = fx.Graph()
    local_grad = graph.placeholder("local_grad")
    local_grad.meta["val"] = torch.empty(3, 2, dtype=torch.bfloat16)

    layout_input = local_grad
    if reduce_dtype_cast:
        layout_input = graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(local_grad,),
            kwargs={"dtype": torch.float32},
        )
        layout_input.meta["val"] = torch.empty(3, 2, dtype=torch.float32)
    layout_dtype = layout_input.meta["val"].dtype

    split = graph.call_function(
        torch.ops.aten.split_with_sizes.default,
        args=(layout_input, [2, 1], 0),
    )
    split.meta["val"] = (
        torch.empty(2, 2, dtype=layout_dtype),
        torch.empty(1, 2, dtype=layout_dtype),
    )
    first = graph.call_function(operator.getitem, args=(split, 0))
    first.meta["val"] = split.meta["val"][0]
    second = graph.call_function(operator.getitem, args=(split, 1))
    second.meta["val"] = split.meta["val"][1]
    padded = graph.call_function(
        torch.ops.aten.constant_pad_nd.default,
        args=(second, (0, 0, 0, 1), 0.0),
    )
    padded.meta["val"] = torch.empty(2, 2, dtype=layout_dtype)
    packed = graph.call_function(
        torch.ops.aten.cat.default,
        args=([first, padded], 0),
    )
    packed.meta["val"] = torch.empty(4, 2, dtype=layout_dtype)
    reduce_scatter = graph.call_function(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        args=(packed, "sum", 4, _FAKE_PG),
    )
    reduce_scatter.meta["val"] = torch.empty(1, 2, dtype=layout_dtype)
    reduce_scatter_wait = graph.call_function(
        torch.ops._c10d_functional.wait_tensor.default,
        args=(reduce_scatter,),
    )
    reduce_scatter_wait.meta["val"] = reduce_scatter.meta["val"]

    fsdp_meta = {FSDP_PARAM_FQNS_META: ("weight",)}
    layout_nodes = (
        *((layout_input,) if reduce_dtype_cast else ()),
        split,
        first,
        second,
        padded,
        packed,
        reduce_scatter,
        reduce_scatter_wait,
    )
    for node in layout_nodes:
        node.meta.setdefault("custom", {}).update(fsdp_meta)
    reduce_scatter_wait.meta["custom"][PARAMETER_GRADIENT_FQNS_META] = ("weight",)
    graph.output((reduce_scatter_wait,))
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

    def test_action_bucketing_keeps_process_groups_separate(self) -> None:
        cases = (
            (
                "all_gather",
                merge_all_all_gathers,
                torch.ops._c10d_functional.all_gather_into_tensor.default,
            ),
            (
                "reduce_scatter",
                merge_all_reduce_scatters,
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
            ),
            (
                "all_reduce",
                merge_all_all_reduces,
                torch.ops._c10d_functional.all_reduce.default,
            ),
        )
        for collective, pass_fn, collective_target in cases:
            with self.subTest(collective=collective):
                gm = _make_unbucketed_action_graph(
                    collective,
                    separate_process_groups=True,
                )
                pass_fn(gm)
                self.assertEqual(
                    sum(node.target == collective_target for node in gm.graph.nodes),
                    2,
                )

    def test_extracted_actions_preserve_per_layer_bucket_scopes(self) -> None:
        scoped_pass = functools.partial(
            joint_transformer_block_bucketing_reordering_pass,
            module_bucket_plans=[
                ["layers.0.part0", "layers.0.part1"],
                ["layers.1.part0", "layers.1.part1"],
            ],
            bucket_mode="custom_ops",
        )
        compile_config = GraphTrainerCompileConfig()

        cases = (
            (
                "all_gather",
                (True, False, False),
                torch.ops.bucketing._pre_bucket_all_gather.default,
            ),
            (
                "reduce_scatter",
                (False, True, False),
                torch.ops.bucketing._pre_bucket_reduce_scatter.default,
            ),
            (
                "all_reduce",
                (False, False, True),
                torch.ops.aten.cat.default,
            ),
        )
        for collective, enabled_types, bucket_target in cases:
            with (
                self.subTest(collective=collective),
                patch(
                    "torch.distributed.distributed_c10d._resolve_process_group",
                    return_value=object(),
                ),
                patch("torch.distributed.get_rank", return_value=0),
            ):
                configured_pass = _configure_fsdp_bucketing_pass(
                    scoped_pass,
                    bucket_all_gathers=enabled_types[0],
                    bucket_reduce_scatters=enabled_types[1],
                    bucket_all_reduces=enabled_types[2],
                )
                if configured_pass is None:
                    self.fail("Expected a configured FSDP bucketing pass")
                scoped = apply_graph_passes(
                    _make_multilayer_unbucketed_action_graph(collective),
                    (),
                    [configured_pass],
                    compile_config=compile_config,
                )

            self.assertEqual(
                sum(node.target == bucket_target for node in scoped.graph.nodes),
                2,
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
                    bucketing_pass = _find_fsdp_bucketing_pass(passes)
                    self.assertIs(bucketing_pass, original_pass)
                    configured_pass = _configure_fsdp_bucketing_pass(
                        bucketing_pass,
                        bucket_all_gathers=not extract_unshard,
                        bucket_reduce_scatters=not extract_reduce_grad,
                        bucket_all_reduces=not extract_reduce_grad,
                    )
                    if extract_unshard and extract_reduce_grad:
                        self.assertIsNone(configured_pass)
                    else:
                        assert isinstance(configured_pass, functools.partial)
                        self.assertEqual(
                            configured_pass.keywords["bucket_all_gathers"],
                            not extract_unshard,
                        )
                        self.assertEqual(
                            configured_pass.keywords["bucket_reduce_scatters"],
                            not extract_reduce_grad,
                        )
                        self.assertEqual(
                            configured_pass.keywords["bucket_all_reduces"],
                            not extract_reduce_grad,
                        )
                else:
                    self.assertEqual(passes, [original_pass])


class GraphPPFSDPCollectiveSplitDsv3Test(_GraphPPDsv3FSDPTest):
    def test_real_dsv3_moe_block_fsdp_split_reconstructs_graphs(self) -> None:
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("real FSDP collective trace requires 2 GPUs")

        self._setup()
        fsdp_mesh = get_simple_fsdp_mesh(self.parallelism_context)
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
            num_params=traced_block.num_sharded_param_values,
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
