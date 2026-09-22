# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torchtitan.models.deepseek_v4.config_registry import deepseek_v4_mtp_debugmodel

from torchtitan.models.deepseek_v4.model import DeepSeekV4Model


class TestDeepSeekV4MTPConfig(unittest.TestCase):
    def test_mtp_debugmodel_builds_mtp_layers(self):
        config = deepseek_v4_mtp_debugmodel()
        model_config = config.model
        self.assertEqual(model_config.n_mtp_layers, 1)
        self.assertIsNotNone(model_config.mtp_layers)
        self.assertEqual(len(model_config.mtp_layers), 1)

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
        parallel_dims = SimpleNamespace(pp_enabled=False, ep=2)
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
            parallel_dims=parallel_dims,
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


if __name__ == "__main__":
    unittest.main()
