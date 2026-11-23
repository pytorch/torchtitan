# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import spmd_types as spmd
import torch
import torch.distributed as dist

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v4 import build_model_config
from torchtitan.models.deepseek_v4.model import DeepSeekV4Model
from torchtitan_recipes.tests.models.deepseek_v4 import (
    deepseek_v4_debugmodel,
    deepseek_v4_mtp_debugmodel,
)


class TestDeepSeekV4MTPConfig(unittest.TestCase):
    def test_dataloader_mtp_depth_matches_model(self):
        for config in (deepseek_v4_debugmodel(), deepseek_v4_mtp_debugmodel()):
            self.assertEqual(
                config.dataloader.num_mtp_layers, config.model.n_mtp_layers
            )

    def test_mtp_debugmodel_builds_mtp_layers(self):
        config = deepseek_v4_mtp_debugmodel()
        model_config = config.model
        self.assertEqual(model_config.n_mtp_layers, 1)
        self.assertIsNotNone(model_config.mtp_layers)
        self.assertEqual(len(model_config.mtp_layers), 1)

    def test_mtp_metadata_remains_replicated_at_block_boundary(self):
        config = build_model_config(
            "debugmodel",
            n_mtp_layers=1,
            parallelism=ParallelismConfig(
                enable_sequence_parallel=True, expert_parallel_degree=2
            ),
        )

        assert config.mtp_layers is not None
        mtp_config = config.mtp_layers[0].sharding_config
        assert mtp_config is not None
        self.assertIsNotNone(mtp_config.input_shardings)

    @patch("torchtitan.distributed.fsdp.resolve_sparse_fsdp_mesh")
    @patch("torchtitan.distributed.fsdp.resolve_fsdp_mesh")
    @patch("torchtitan.models.common.mtp.apply_fsdp_to_mtp_decoder")
    def test_uses_mtp_fsdp_path(
        self,
        apply_fsdp_to_mtp_decoder,
        resolve_fsdp_mesh,
        resolve_sparse_fsdp_mesh,
    ):
        model = MagicMock(spec=DeepSeekV4Model)
        model.mtp_layers = [MagicMock()]
        dp_mesh = object()
        dp_mesh_dims = object()
        edp_mesh = object()
        edp_mesh_dims = object()
        resolve_fsdp_mesh.return_value = (dp_mesh, dp_mesh_dims)
        resolve_sparse_fsdp_mesh.return_value = (edp_mesh, edp_mesh_dims)
        parallelism_context = SimpleNamespace(pp_enabled=False, ep=2)
        precision = SimpleNamespace(param_dtype="bfloat16", reduce_dtype="float32")
        parallelism = SimpleNamespace(
            fsdp_enable_cpu_offloading=False,
            fsdp_reshard_after_forward="default",
            fsdp_symm_mem_scope=None,
            fsdp_unshard_in_current_stream=False,
        )

        DeepSeekV4Model._apply_fsdp(
            model,
            parallelism_context=parallelism_context,
            precision=precision,
            parallelism=parallelism,
        )

        apply_fsdp_to_mtp_decoder.assert_called_once_with(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            cpu_offload=False,
            reshard_after_forward_policy="default",
            ep_degree=2,
            edp_mesh=edp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh_dims=edp_mesh_dims,
            symm_mem_scope=None,
            unshard_in_current_stream=False,
        )


def _preprocess(model, input_dict, **kwargs):
    # SPMD annotation needs a device mesh; these tests check the plumbing only.
    with patch(
        "torchtitan.models.common.mtp.annotate_input_spmd_types",
        side_effect=lambda _parallelism_context, batch, _input_sharding: batch,
    ):
        return model.preprocess_inputs(
            input_dict,
            parallelism_context=SimpleNamespace(
                cp_enabled=False, activate_spmd=contextlib.nullcontext
            ),
            parallelism=SimpleNamespace(),
            **kwargs,
        )


class TestDeepSeekV4MTPInputs(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(AuxLoss._group_counts, clear=True))
        self.enterContext(patch.dict(AuxLoss.group_acc, clear=True))

    def test_preprocess_inputs_prepares_each_depth(self):
        num_mtp_layers = 4
        with torch.device("meta"):
            model = build_model_config(
                "debugmodel",
                seq_len=128,
                n_mtp_layers=num_mtp_layers,
                parallelism=ParallelismConfig(expert_parallel_degree=1),
            ).build()
        # Two packed documents (70 and 50 tokens) and 8 padding tokens.
        document_1 = torch.arange(71, dtype=torch.long) + 100
        document_2 = torch.arange(51, dtype=torch.long) + 1000
        positions = torch.cat(
            [
                torch.arange(70, dtype=torch.long),
                torch.arange(50, dtype=torch.long),
                torch.arange(8, dtype=torch.long),
            ]
        )
        padding_mask = torch.cat(
            [
                torch.zeros(70, dtype=torch.bool),
                torch.zeros(50, dtype=torch.bool),
                torch.ones(8, dtype=torch.bool),
            ]
        )
        tokens = torch.cat(
            [
                document_1[:-1],
                document_2[:-1],
                torch.zeros(8, dtype=torch.long),
            ]
        )
        labels = torch.cat(
            [
                document_1[1:],
                document_2[1:],
                torch.full((8,), IGNORE_INDEX, dtype=torch.long),
            ]
        )

        inputs, loss_labels, model_kwargs, _ = _preprocess(
            model,
            {
                "input": tokens,
                "labels": labels,
                "positions": positions,
                "padding_mask": padding_mask,
            },
        )

        self.assertEqual(len(inputs), num_mtp_layers + 1)
        self.assertTrue(torch.equal(inputs[0], tokens))
        self.assertTrue(torch.equal(model_kwargs["input_ids_T"], tokens))
        self.assertEqual(len(model_kwargs["mtp_input_valid_masks"]), num_mtp_layers)
        self.assertEqual(len(loss_labels), num_mtp_layers + 1)
        self.assertTrue(torch.equal(loss_labels[0], labels))
        # Depth d is shifted by d inside each document: nothing crosses a
        # document boundary, enters the padding, or runs past the end.
        for depth in range(1, num_mtp_layers + 1):
            reference_valid = torch.cat(
                [
                    torch.ones(70 - depth, dtype=torch.bool),
                    torch.zeros(depth, dtype=torch.bool),
                    torch.ones(50 - depth, dtype=torch.bool),
                    torch.zeros(depth, dtype=torch.bool),
                    torch.zeros(8, dtype=torch.bool),
                ]
            )
            reference_labels = torch.cat(
                [
                    document_1[depth + 1 :],
                    torch.full((depth,), IGNORE_INDEX, dtype=torch.long),
                    document_2[depth + 1 :],
                    torch.full((depth,), IGNORE_INDEX, dtype=torch.long),
                    torch.full((8,), IGNORE_INDEX, dtype=torch.long),
                ]
            )
            reference_inputs = torch.cat(
                [
                    document_1[depth:-1],
                    torch.zeros(depth, dtype=torch.long),
                    document_2[depth:-1],
                    torch.zeros(depth, dtype=torch.long),
                    torch.zeros(8, dtype=torch.long),
                ]
            )
            self.assertTrue(
                torch.equal(
                    model_kwargs["mtp_input_valid_masks"][depth - 1], reference_valid
                )
            )
            self.assertTrue(torch.equal(inputs[depth], reference_inputs))
            self.assertTrue(torch.equal(loss_labels[depth], reference_labels))


class TestDeepSeekV4MTPInputLayout(unittest.TestCase):
    def setUp(self):
        dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
        self.addCleanup(dist.destroy_process_group)

    @patch("torchtitan.distributed.parallelism_context.device_type", "cpu")
    def test_mtp_inputs_have_the_main_layouts(self):
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
        parallelism_context.build_mesh()
        with torch.device("meta"):
            model = deepseek_v4_mtp_debugmodel(seq_len=16).model.build()
        inputs, loss_labels, model_kwargs, _ = model.preprocess_inputs(
            {
                "input": torch.zeros(16, dtype=torch.long),
                "labels": torch.arange(16),
                "positions": torch.arange(16),
                "padding_mask": torch.zeros(16, dtype=torch.bool),
            },
            parallelism_context=parallelism_context,
            parallelism=SimpleNamespace(),
        )

        # On one rank an unannotated tensor also reads as the empty local type,
        # so check that each per-depth tensor was annotated at all.
        for depth_tensor, main_tensor in (
            (inputs[1], inputs[0]),
            (model_kwargs["mtp_input_valid_masks"][0], inputs[0]),
            (loss_labels[1], loss_labels[0]),
        ):
            self.assertTrue(spmd.has_local_type(depth_tensor))
            self.assertEqual(
                spmd.get_local_type(depth_tensor), spmd.get_local_type(main_tensor)
            )


if __name__ == "__main__":
    unittest.main()
