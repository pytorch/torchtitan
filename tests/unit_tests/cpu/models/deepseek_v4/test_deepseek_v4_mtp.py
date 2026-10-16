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
        self.assertIsNotNone(mtp_config.in_src_shardings)
        self.assertIsNone(mtp_config.in_dst_shardings)

    @patch("torchtitan.distributed.fsdp.resolve_sparse_fsdp_mesh")
    @patch("torchtitan.distributed.fsdp.resolve_fsdp_mesh")
    @patch("torchtitan.models.deepseek_v4.model.apply_fsdp_to_mtp_decoder")
    def test_uses_mtp_fsdp_path(
        self,
        apply_fsdp_to_mtp_decoder,
        resolve_fsdp_mesh,
        resolve_sparse_fsdp_mesh,
    ):
        model = MagicMock(spec=DeepSeekV4Model)
        dp_mesh = object()
        dp_mesh_dims = object()
        edp_mesh = object()
        edp_mesh_dims = object()
        resolve_fsdp_mesh.return_value = (dp_mesh, dp_mesh_dims)
        resolve_sparse_fsdp_mesh.return_value = (edp_mesh, edp_mesh_dims)
        parallelism_context = SimpleNamespace(pp_enabled=False, ep=2)
        training = SimpleNamespace(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            enable_cpu_offload=False,
        )
        parallelism = SimpleNamespace(
            fsdp_reshard_after_forward="default",
            fsdp_symm_mem_scope=None,
        )

        DeepSeekV4Model._apply_fsdp(
            model,
            parallelism_context=parallelism_context,
            training=training,
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
        )


def _preprocess(model, input_dict, **kwargs):
    # SPMD annotation needs a device mesh; these tests check the plumbing only.
    no_annotation = dict(
        side_effect=lambda _parallelism_context, batch, _input_sharding: batch
    )
    with (
        patch(
            "torchtitan.models.common.decoder.annotate_input_spmd_types",
            **no_annotation,
        ),
        patch(
            "torchtitan.models.deepseek_v3.mtp.annotate_input_spmd_types",
            **no_annotation,
        ),
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
        with torch.device("meta"):
            model = deepseek_v4_mtp_debugmodel(seq_len=128).model.build()
        # Two packed documents (70 and 50 tokens) and 8 padding tokens.
        positions = torch.cat([torch.arange(70), torch.arange(50), torch.arange(8)])
        padding_mask = torch.arange(128) >= 120
        tokens = torch.arange(128) + 1000
        labels = torch.arange(128)
        inputs, loss_labels, kwargs = _preprocess(
            model,
            {
                "input": tokens,
                "labels": labels,
                "positions": positions,
                "padding_mask": padding_mask,
            },
        )

        # Depth 1 is shifted by one inside each document: nothing crosses a
        # document boundary, enters the padding, or runs past the end.
        valid = torch.zeros(128, dtype=torch.bool)
        valid[:69] = True
        valid[70:119] = True
        self.assertEqual(len(inputs), 2)
        self.assertTrue(torch.equal(inputs[0], tokens))
        self.assertTrue(torch.equal(inputs[1], torch.where(valid, tokens.roll(-1), 0)))
        self.assertEqual(len(kwargs["mtp_input_valid_masks"]), 1)
        self.assertTrue(torch.equal(kwargs["mtp_input_valid_masks"][0], valid))
        self.assertEqual(len(loss_labels), 2)
        self.assertTrue(torch.equal(loss_labels[0], labels))
        self.assertTrue(
            torch.equal(
                loss_labels[1], torch.where(valid, labels.roll(-1), IGNORE_INDEX)
            )
        )

    def test_mtp_debugmodel_loss_runs(self):
        torch.manual_seed(0)
        config = deepseek_v4_mtp_debugmodel(seq_len=128)
        model = config.model.build()
        model.init_weights(buffer_device=torch.device("cpu"))
        tokens = torch.randint(0, config.model.vocab_size, (128,))
        inputs, labels, kwargs = _preprocess(
            model,
            {
                "input": tokens,
                "labels": torch.roll(tokens, -1),
                "positions": torch.arange(128),
                "padding_mask": torch.zeros(128, dtype=torch.bool),
            },
            global_routing_token_counts=torch.tensor([128, 127]),
        )
        with torch.no_grad():
            predictions = model(inputs, **kwargs)

        loss, _ = config.loss.build()(predictions, labels, torch.tensor([128, 127]))
        self.assertTrue(torch.isfinite(loss))


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
        inputs, loss_labels, kwargs = model.preprocess_inputs(
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
            (kwargs["mtp_input_valid_masks"][0], inputs[0]),
            (loss_labels[1], loss_labels[0]),
        ):
            self.assertTrue(spmd.has_local_type(depth_tensor))
            self.assertEqual(
                spmd.get_local_type(depth_tensor), spmd.get_local_type(main_tensor)
            )


if __name__ == "__main__":
    unittest.main()
