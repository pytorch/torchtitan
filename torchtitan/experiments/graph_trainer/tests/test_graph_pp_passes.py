# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import unittest
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch.nn.attention.flex_attention import flex_attention

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.graph_trainer.common_utils import (
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.deepseek_v3 import (
    model_registry as dsv3_model_registry,
)
from torchtitan.experiments.graph_trainer.graph_pp import partition_joint_graph
from torchtitan.experiments.graph_trainer.graph_pp.partition import GraphMeta
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    extract_module_state,
    minimal_fx_tracer,
    TracedResult,
)
from torchtitan.models.common.attention import FlexAttention
from torchtitan.trainer import Trainer


@dataclass(frozen=True, slots=True)
class _Dsv3MoeBlockTrace:
    traced: TracedResult
    flat_inputs: list[Any]
    output_grad: torch.Tensor
    num_param_grads: int


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
) -> _Dsv3MoeBlockTrace:
    """Trace a real DeepSeek V3 MoE decoder block as a GraphPP stage.

    This is intentionally CUDA-only: FlexAttention backward is not supported on
    CPU, and these pass tests need to exercise the real BlockMask tracing path
    rather than a maskless SDPA shortcut. The unit test enables EP sharding
    metadata and traces the first MoE layer, but does not run true multi-rank
    all-to-all. True EP numerics are covered by the distributed GraphPP DSV3
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
                local_batch_size=batch_size,
                seq_len=seq_len,
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
        flat_inputs = [*extract_module_state(block).values(), *user_flat_inputs]
        if len(flat_inputs) != len(traced.example_inputs):
            raise AssertionError(
                "Real flat inputs must match traced flat input count: "
                f"{len(flat_inputs)} != {len(traced.example_inputs)}"
            )

        num_param_grads = sum(
            1
            for _, param in block.named_parameters(remove_duplicate=False)
            if param.requires_grad
        )
        return _Dsv3MoeBlockTrace(
            traced=traced,
            flat_inputs=flat_inputs,
            output_grad=output_grad,
            num_param_grads=num_param_grads,
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
        self.assertEqual(meta.num_bwd_outputs, traced_block.num_param_grads + 1)
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


if __name__ == "__main__":
    unittest.main()
