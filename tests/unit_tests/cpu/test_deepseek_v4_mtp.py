# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.token_dispatcher import MinimalAsyncEPTokenDispatcher
from torchtitan.models.deepseek_v4 import model_registry
from torchtitan.models.deepseek_v4.config_registry import deepseek_v4_mtp_debugmodel
from torchtitan.protocols.module import ModuleList


class TestDeepSeekV4MTPConfig(unittest.TestCase):
    def test_mtp_debugmodel_builds_mtp_layers(self):
        config = deepseek_v4_mtp_debugmodel()
        model_config = config.model_spec.model
        self.assertEqual(model_config.n_mtp_layers, 1)
        self.assertIsNotNone(model_config.mtp_layers)
        self.assertEqual(len(model_config.mtp_layers), 1)

    def test_mtp_model_satisfies_module_protocol(self):
        config = deepseek_v4_mtp_debugmodel(seq_len=32)
        model = config.model_spec.model.build()

        self.assertIsInstance(model.mtp_layers, ModuleList)
        model.verify_module_protocol()

    def test_mtp_moe_minimal_async_ep_gets_runtime_config(self):
        config = deepseek_v4_mtp_debugmodel(seq_len=32)
        config.model_spec = model_registry(
            "debugmodel",
            seq_len=32,
            moe_comm_backend="minimal_async_ep",
            n_mtp_layers=1,
        )
        config.parallelism.expert_parallel_degree = 2
        config.activation_checkpoint = FullAC.Config()

        model_config = config.model_spec.model
        model_config.update_from_config(config=config)

        dispatcher_cfgs = {
            fqn: dispatcher
            for fqn, dispatcher, _, _ in model_config.traverse(
                MinimalAsyncEPTokenDispatcher.Config
            )
        }
        mtp_dispatcher = dispatcher_cfgs[
            "mtp_layers.0.moe.routed_experts.token_dispatcher"
        ]
        self.assertEqual(mtp_dispatcher.hidden_dim, model_config.dim)
        self.assertEqual(
            mtp_dispatcher.num_max_tokens_per_rank,
            config.training.num_tokens_per_microbatch_per_dp_rank,
        )
        self.assertEqual(mtp_dispatcher.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
