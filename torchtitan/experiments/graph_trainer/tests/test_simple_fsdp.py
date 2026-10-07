# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    apply_simple_fsdp,
    compute_parameter_gradients,
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_reduce_grad_input,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    remove_parameter_gradient_markers_pass,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    FSDP_MESH_AXIS_NAMES_META,
    FSDP_PARAM_FQNS_META,
    FSDP_REDUCE_DTYPE_META,
)
from torchtitan.models.common.attention import ScaledDotProductInnerAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock


class _RoutedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w13 = nn.Module()
        self.w13.group_size = 64
        self.w2 = nn.Module()


def _transformer_block(*, with_moe: bool) -> TransformerBlock:
    block = TransformerBlock.__new__(TransformerBlock)
    nn.Module.__init__(block)
    if with_moe:
        moe = nn.Module()
        moe.routed_experts = _RoutedExperts()
        block.moe = moe
    else:
        block.moe = None
    return block


def _decoder(*, with_auxiliary_moe: bool) -> Decoder:
    model = Decoder.__new__(Decoder)
    nn.Module.__init__(model)
    model.layers = nn.ModuleDict(
        {
            "0": _transformer_block(with_moe=True),
            "1": _transformer_block(with_moe=False),
        }
    )
    if with_auxiliary_moe:
        model.auxiliary_blocks = nn.ModuleList([_transformer_block(with_moe=True)])
    return model


class TestApplySimpleFSDPSingleRank(unittest.TestCase):
    """Verify simple_fsdp's MixedPrecisionPolicy actually casts params at NGPU=1."""

    def setUp(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12358",
                world_size=1,
                rank=0,
            )

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    @patch("torchtitan.distributed.parallelism_context.device_type", "cpu")
    def test_uses_dtensor_storage_and_local_compute(self):
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
            enable_sequence_parallel=False,
        )
        training = TrainingConfig(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        )

        model = apply_simple_fsdp(
            nn.Linear(8, 8),
            parallelism_context=parallelism_context,
            training=training,
        )

        self.assertIsInstance(
            model._parameters["weight"], torch.distributed.tensor.DTensor
        )
        self.assertEqual(model._parameters["weight"].dtype, torch.float32)
        self.assertNotIsInstance(model.weight, torch.distributed.tensor.DTensor)
        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertEqual(
            model(torch.randn(2, 8, dtype=torch.bfloat16)).dtype, torch.bfloat16
        )

    @patch("torchtitan.distributed.parallelism_context.device_type", "cpu")
    def test_preserves_inner_attention_metadata_key(self):
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
            enable_sequence_parallel=False,
        )
        training = TrainingConfig(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        )
        inner_attention = ScaledDotProductInnerAttention(
            ScaledDotProductInnerAttention.Config()
        )

        model = apply_simple_fsdp(
            inner_attention,
            parallelism_context=parallelism_context,
            training=training,
        )

        self.assertIs(model.attention_metadata_key, ScaledDotProductInnerAttention)

    def _assert_reduce_dtype_annotation_reaches_backward_cast(
        self,
        model: nn.Module,
        *,
        expected_param_fqn: str,
        expected_module_fqn: str | None,
    ) -> None:
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
            enable_sequence_parallel=False,
        )
        training = TrainingConfig(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="bfloat16",
        )
        annotate_module_fqns(model)
        model = apply_simple_fsdp(
            model,
            parallelism_context=parallelism_context,
            training=training,
        )

        def train_step(x):
            return compute_parameter_gradients(model(x).sum(), model.named_parameters())

        traced = minimal_fx_tracer(train_step, module=model)(
            torch.randn(2, 8, dtype=torch.bfloat16)
        )
        remove_parameter_gradient_markers_pass(traced.gm, ())
        persistent_casts = [
            node
            for node in traced.gm.graph.nodes
            if node.target is torch.ops.aten._to_copy.default
            and isinstance(node.meta.get("val"), torch.Tensor)
            and node.meta["val"].dtype == torch.float32
        ]
        self.assertEqual(len(persistent_casts), 1)
        persistent_cast = persistent_casts[0]
        custom = persistent_cast.meta["custom"]
        self.assertEqual(custom[FSDP_REDUCE_DTYPE_META], torch.bfloat16)
        self.assertEqual(custom[FSDP_PARAM_FQNS_META], (expected_param_fqn,))
        self.assertEqual(custom[FSDP_MESH_AXIS_NAMES_META], ("fsdp",))
        self.assertEqual(custom[PARAMETER_GRADIENT_FQNS_META], (expected_param_fqn,))
        self.assertEqual(custom.get("module_fqn"), expected_module_fqn)
        cast_input = persistent_cast.all_input_nodes[0]
        self.assertEqual(cast_input.meta["val"].dtype, torch.bfloat16)
        self.assertEqual(
            cast_input.meta["custom"].get("module_fqn"), expected_module_fqn
        )
        self.assertIs(find_fsdp_reduce_grad_input(persistent_cast), cast_input)

    @patch("torchtitan.distributed.parallelism_context.device_type", "cpu")
    def test_reduce_dtype_annotation_reaches_backward_cast(self):
        self._assert_reduce_dtype_annotation_reaches_backward_cast(
            nn.Sequential(nn.Linear(8, 8, bias=False)),
            expected_param_fqn="0.weight",
            expected_module_fqn="0",
        )

    @patch("torchtitan.distributed.parallelism_context.device_type", "cpu")
    def test_root_module_reduce_dtype_annotation_reaches_backward_cast(self):
        self._assert_reduce_dtype_annotation_reaches_backward_cast(
            nn.Linear(8, 8, bias=False),
            expected_param_fqn="weight",
            expected_module_fqn=None,
        )


class TestApplySimpleFSDPExpertTraversal(unittest.TestCase):
    def test_wraps_all_nested_moe_transformer_blocks(self):
        fsdp_mesh = object()
        edp_mesh = MagicMock()
        edp_mesh["edp_shard"].size.return_value = 1
        ep_mesh = object()
        tp_mesh = object()
        parallelism_context = SimpleNamespace(
            dp_replicate_enabled=False,
            ep_enabled=True,
            ep=2,
        )

        def get_optional_mesh(mesh_axis_names, **kwargs):
            del kwargs
            if mesh_axis_names == ["edp_shard"]:
                return edp_mesh
            if mesh_axis_names == "ep":
                return ep_mesh
            if mesh_axis_names == "tp":
                return tp_mesh
            raise AssertionError(f"Unexpected mesh axes: {mesh_axis_names}")

        parallelism_context.get_optional_mesh = get_optional_mesh
        training = TrainingConfig(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        )

        for with_auxiliary_moe, num_expert_calls in ((False, 1), (True, 2)):
            with self.subTest(with_auxiliary_moe=with_auxiliary_moe):
                model = _decoder(with_auxiliary_moe=with_auxiliary_moe)
                main_routed_experts = model.layers["0"].moe.routed_experts
                auxiliary_routed_experts = (
                    model.auxiliary_blocks[0].moe.routed_experts
                    if with_auxiliary_moe
                    else None
                )

                with (
                    patch(
                        "torchtitan.experiments.graph_trainer.common_utils."
                        "get_simple_fsdp_mesh",
                        return_value=fsdp_mesh,
                    ),
                    patch(
                        "torchtitan.experiments.graph_trainer.common_utils."
                        "data_parallel",
                        side_effect=lambda module, *args, **kwargs: module,
                    ) as data_parallel,
                ):
                    apply_simple_fsdp(
                        model,
                        parallelism_context=parallelism_context,
                        training=training,
                    )

                expert_calls = data_parallel.call_args_list[:num_expert_calls]
                self.assertIs(expert_calls[0].args[0], main_routed_experts)
                if auxiliary_routed_experts is not None:
                    self.assertIs(expert_calls[1].args[0], auxiliary_routed_experts)
                for expert_call in expert_calls:
                    self.assertIs(expert_call.args[1], edp_mesh)
                    self.assertEqual(expert_call.kwargs["shard_dim"], 0)
                    self.assertEqual(expert_call.kwargs["param_shard_placements"], {})
                    self.assertIs(expert_call.kwargs["non_dp_mesh"], ep_mesh)

                self.assertEqual(data_parallel.call_count, num_expert_calls + 1)
                model_call = data_parallel.call_args_list[-1]
                self.assertIs(model_call.args[0], model)
                self.assertIs(model_call.args[1], fsdp_mesh)
                self.assertIs(model_call.kwargs["non_dp_mesh"], tp_mesh)


if __name__ == "__main__":
    unittest.main()
