# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.token_dispatcher import MinimalAsyncEPTokenDispatcher
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel_minimal_async_ep,
)


def _first_minimal_async_ep_config(config):
    for layer_config in config.model_spec.model.layers:
        if layer_config.moe is None:
            continue
        dispatcher_config = layer_config.moe.routed_experts.token_dispatcher
        if isinstance(dispatcher_config, MinimalAsyncEPTokenDispatcher.Config):
            return dispatcher_config
    raise AssertionError("MinimalAsyncEP dispatcher config not found")


class TestMinimalAsyncEPTensorParallel(unittest.TestCase):
    def test_tp_capacity_is_rank_local(self):
        config = deepseek_v3_debugmodel_minimal_async_ep()
        self.assertFalse(config.parallelism.enable_sequence_parallel)
        config.parallelism.data_parallel_shard_degree = 2
        config.parallelism.tensor_parallel_degree = 2
        config.parallelism.expert_parallel_degree = 4
        config.activation_checkpoint = FullAC.Config()

        config.model_spec.model.update_from_config(config=config)

        dispatcher_config = _first_minimal_async_ep_config(config)
        expected_capacity = config.training.local_batch_size * (
            (config.training.seq_len + 1) // 2
        )
        self.assertEqual(
            dispatcher_config.num_max_tokens_per_rank,
            expected_capacity,
        )
        self.assertEqual(dispatcher_config.hidden_dim, config.model_spec.model.dim)

    def test_tp_capacity_with_dense_sequence_parallel(self):
        config = deepseek_v3_debugmodel_minimal_async_ep()
        config.parallelism.data_parallel_shard_degree = 2
        config.parallelism.tensor_parallel_degree = 2
        config.parallelism.expert_parallel_degree = 4
        config.parallelism.enable_sequence_parallel = True
        config.activation_checkpoint = FullAC.Config()

        config.model_spec.model.update_from_config(config=config)

        dispatcher_config = _first_minimal_async_ep_config(config)
        expected_capacity = config.training.local_batch_size * (
            (config.training.seq_len + 1) // 2
        )
        self.assertEqual(
            dispatcher_config.num_max_tokens_per_rank,
            expected_capacity,
        )

    def test_restores_uneven_local_token_shard(self):
        dispatcher = MinimalAsyncEPTokenDispatcher.Config(
            num_experts=4,
            top_k=1,
            hidden_dim=1,
            num_max_tokens_per_rank=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).build()
        dispatcher.sp_size = 2
        dispatcher.sp_rank = 1

        combined_TD = torch.tensor([[1.0], [2.0], [3.0]])  # noqa: N806
        restored_TD = dispatcher._restore_sp_output(  # noqa: N806
            combined_TD,
            num_local_tokens_after_padding=4,
            local_seq_len_after_padding=2,
        )

        expected_TD = torch.zeros(8, 1)  # noqa: N806
        expected_TD[[2, 3, 6]] = combined_TD  # noqa: N806
        self.assertTrue(torch.equal(restored_TD, expected_TD))


if __name__ == "__main__":
    unittest.main()
