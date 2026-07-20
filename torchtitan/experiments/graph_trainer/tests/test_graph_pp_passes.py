# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import operator
import unittest
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch.nn.attention.flex_attention import flex_attention
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import BatchConfig, DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    model_registry as dsv3_model_registry,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    deduplicate_fsdp_unshard_chains_pass,
)
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_unshard_output,
    find_fsdp_unshard_save_node,
    find_fsdp_unshard_save_nodes,
)
from torchtitan.experiments.graph_trainer.graph_pp import (
    partition_joint_graph,
    split_backward_fsdp_collectives,
    split_di_dw_graph,
    split_forward_fsdp_collectives,
)
from torchtitan.experiments.graph_trainer.graph_pp.partition import GraphMeta
from torchtitan.experiments.graph_trainer.graph_pp.utils import flatten_graph_values
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel
from torchtitan.models.common.attention import FlexAttention
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
    original_configs = FlexAttention.inductor_configs
    original_compiled_flex_attn = FlexAttention._compiled_flex_attn
    FlexAttention.inductor_configs = {
        **original_configs,
        "max_autotune": False,
        "coordinate_descent_tuning": False,
    }
    FlexAttention._compiled_flex_attn = torch.compile(
        flex_attention,
        options=FlexAttention.inductor_configs,
    )
    try:
        yield
    finally:
        FlexAttention.inductor_configs = original_configs
        FlexAttention._compiled_flex_attn = original_compiled_flex_attn


def _trace_dsv3_moe_block_stage(
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    include_input_grad: bool = True,
    fsdp_mesh: Any | None = None,
) -> _Dsv3MoeBlockTrace:
    """Trace a real DeepSeek V3 MoE decoder block as a GraphPP stage.

    This is intentionally CUDA-only: FlexAttention backward is not supported on
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
        model_spec = dsv3_model_registry("debugmodel", attn_backend="flex")
        model_config = model_spec.model
        runtime_config = Trainer.Config(
            model_spec=model_spec,
            training=TrainingConfig(
                batch=BatchConfig(local_batch_size=batch_size, seq_len=seq_len),
                steps=1,
            ),
            parallelism=ParallelismConfig(expert_parallel_degree=2),
            checkpoint=CheckpointManager.Config(initial_load_model_only=False),
            debug=DebugConfig(seed=0, deterministic=True),
        )
        model_config.update_from_config(config=runtime_config)
        moe_layer_config = model_config.layers[1]
        if moe_layer_config.moe is None or not moe_layer_config.moe.seq_dim_tp_sharded:
            raise AssertionError("DeepSeek V3 MoE layer must be configured with EP")

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

        x = torch.randn(
            batch_size,
            seq_len,
            model_config.dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=include_input_grad,
        )
        positions = torch.arange(seq_len, device="cuda").repeat(batch_size, 1)
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
        )


class GraphPPPartitionFSDPTest(_GraphPPDsv3FSDPTest):
    def test_real_dsv3_moe_block_fsdp_partition_matches_joint_graph(self) -> None:
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("real FSDP collective trace requires 2 GPUs")

        self._setup()
        traced_block = _trace_dsv3_moe_block_stage(
            fsdp_mesh=self.parallel_dims.get_mesh("fsdp")
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


def _call_targets(gm: fx.GraphModule) -> set[object]:
    return {node.target for node in gm.graph.nodes if node.op == "call_function"}


def _placeholder_names(gm: fx.GraphModule) -> tuple[str, ...]:
    return tuple(node.name for node in gm.graph.find_nodes(op="placeholder"))


def _make_graph_module(graph: fx.Graph) -> fx.GraphModule:
    gm = fx.GraphModule({}, graph)
    gm.graph.lint()
    gm.recompile()
    return gm


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


def _make_backward_graph_without_fsdp() -> fx.GraphModule:
    graph = fx.Graph()
    grad = graph.placeholder("grad")
    out = graph.call_function(torch.ops.aten.neg.default, args=(grad,))
    graph.output((out,))
    return _make_graph_module(graph)


class GraphPPFSDPCollectiveSplitTest(unittest.TestCase):
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

        split = split_forward_fsdp_collectives(
            gm,
            num_params=2,
            fwd_input_names=("sharded_param", "replicated_param", "x"),
            fwd_flat_input_indices=(0, 1, 2),
        )

        self.assertIsNotNone(split.unshard_module)
        if split.unshard_module is None:
            self.fail("Expected forward FSDP split to extract an unshard graph")
        self.assertEqual(
            _placeholder_names(split.unshard_module),
            ("sharded_param", "replicated_param"),
        )
        self.assertEqual(split.unshard_flat_param_indices, (0, 1))
        self.assertEqual(split.num_fw_unsharded_param_inputs, 2)
        self.assertEqual(split.fw_no_fsdp_flat_input_indices, (2,))
        self.assertIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _call_targets(split.unshard_module),
        )
        self.assertIn(torch.ops.aten.cat.default, _call_targets(split.unshard_module))
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _call_targets(split.fw_no_fsdp_module),
        )

    def test_forward_split_no_fsdp_is_noop(self) -> None:
        gm = _make_forward_graph_without_fsdp()
        split = split_forward_fsdp_collectives(
            gm,
            num_params=1,
            fwd_input_names=("param", "x"),
            fwd_flat_input_indices=(0, 1),
        )

        self.assertIsNone(split.unshard_module)
        self.assertIs(split.fw_no_fsdp_module, gm)
        self.assertEqual(split.fw_no_fsdp_input_names, ("param", "x"))
        self.assertEqual(split.fw_no_fsdp_flat_input_indices, (0, 1))

    def test_forward_split_requires_wait_after_all_gather(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected wait_tensor"):
            split_forward_fsdp_collectives(
                _make_forward_graph_without_wait(),
                num_params=1,
                fwd_input_names=("param", "x"),
                fwd_flat_input_indices=(0, 1),
            )

    def test_backward_split_extracts_reduce_grad_epilogues(self) -> None:
        split = split_backward_fsdp_collectives(
            _make_backward_graph_with_reduce_grad_epilogues(),
            num_param_grads=3,
        )

        self.assertIsNotNone(split.reduce_grad_module)
        if split.reduce_grad_module is None:
            self.fail("Expected backward FSDP split to extract reduce-grad graph")
        bw_no_fsdp_targets = _call_targets(split.bw_no_fsdp_module)
        reduce_grad_targets = _call_targets(split.reduce_grad_module)
        self.assertIn(torch.ops.aten._to_copy.default, bw_no_fsdp_targets)
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            bw_no_fsdp_targets,
        )
        self.assertNotIn(
            torch.ops._c10d_functional.all_reduce.default,
            bw_no_fsdp_targets,
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
            split.bw_no_fsdp_output_names[:2],
        )
        self.assertEqual(len(split.bw_no_fsdp_output_names), 4)
        self.assertEqual(split.bw_no_fsdp_output_names[-1], "input_grad")

    def test_backward_split_no_fsdp_is_noop_and_validates_grad_count(self) -> None:
        gm = _make_backward_graph_without_fsdp()
        split = split_backward_fsdp_collectives(gm, num_param_grads=1)

        self.assertIsNone(split.reduce_grad_module)
        self.assertIs(split.bw_no_fsdp_module, gm)

        with self.assertRaisesRegex(ValueError, "num_param_grads cannot exceed"):
            split_backward_fsdp_collectives(gm, num_param_grads=2)


class GraphPPFSDPCollectiveSplitDsv3Test(_GraphPPDsv3FSDPTest):
    def test_real_dsv3_moe_block_fsdp_split_reconstructs_graphs(self) -> None:
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("real FSDP collective trace requires 2 GPUs")

        self._setup()
        fsdp_mesh = self.parallel_dims.get_mesh("fsdp")
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
        fw_split = split_forward_fsdp_collectives(
            fw_module,
            num_params=traced_block.num_flat_param_values,
            fwd_input_names=meta.fwd_input_names,
            fwd_flat_input_indices=meta.fwd_flat_input_indices,
        )
        bw_split = split_backward_fsdp_collectives(
            bw_module,
            num_param_grads=traced_block.num_param_grad_values,
        )

        self.assertIsNotNone(fw_split.unshard_module)
        self.assertIsNotNone(bw_split.reduce_grad_module)
        if fw_split.unshard_module is None or bw_split.reduce_grad_module is None:
            self.fail("Expected real DSV3 FSDP trace to contain split collectives")
        self.assertNotIn(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            _call_targets(fw_split.fw_no_fsdp_module),
        )
        self.assertNotIn(
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            _call_targets(bw_split.bw_no_fsdp_module),
        )

        fw_args = [
            traced_block.flat_inputs[index] for index in meta.fwd_flat_input_indices
        ]
        fw_outputs = _boxed_run(fw_module, list(fw_args))
        unshard_args = [
            traced_block.flat_inputs[index]
            for index in fw_split.unshard_flat_param_indices
        ]
        unsharded_params = _boxed_run(fw_split.unshard_module, unshard_args)
        fw_no_fsdp_args = [
            *unsharded_params,
            *(
                traced_block.flat_inputs[index]
                for index in fw_split.fw_no_fsdp_flat_input_indices
            ),
        ]
        split_fw_outputs = _boxed_run(
            fw_split.fw_no_fsdp_module,
            list(fw_no_fsdp_args),
        )
        _assert_tensor_sequence_equal(self, split_fw_outputs, fw_outputs)

        bw_args = _backward_args_from_partition(
            meta,
            fw_outputs,
            (traced_block.output_grad,),
        )
        bw_outputs = _boxed_run(bw_module, list(bw_args))
        bw_no_fsdp_outputs = _boxed_run(bw_split.bw_no_fsdp_module, list(bw_args))
        grad_values_by_name = dict(
            zip(
                bw_split.bw_no_fsdp_output_names[: traced_block.num_param_grad_values],
                bw_no_fsdp_outputs[: traced_block.num_param_grad_values],
                strict=True,
            )
        )
        reduce_grad_args = [
            grad_values_by_name[name] for name in bw_split.reduce_grad_input_names
        ]
        reduced_grads = _boxed_run(bw_split.reduce_grad_module, reduce_grad_args)
        split_bw_outputs = (
            *reduced_grads,
            *bw_no_fsdp_outputs[traced_block.num_param_grad_values :],
        )
        _assert_tensor_sequence_equal(self, split_bw_outputs, bw_outputs)


if __name__ == "__main__":
    unittest.main()
