# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-token loss arguments follow the labels through ``preprocess_inputs``."""

from unittest.mock import patch

import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.llama3 import build_model_config


class TestDecoderLossKwargsContextParallel(DTensorTestBase):
    """CP=2 on CPU + gloo, with and without a load-balancing permutation."""

    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_loss_kwargs_are_sharded_like_labels(self):
        with patch("torchtitan.distributed.parallelism_context.device_type", "cpu"):
            parallelism_context = ParallelismContext(
                dp_replicate=1,
                dp_shard=1,
                cp=2,
                tp=1,
                pp=1,
                ep=1,
                world_size=2,
                enable_sequence_parallel=False,
            )
            parallelism_context.build_mesh()
        with torch.device("meta"):
            model = build_model_config(
                "debugmodel", parallelism=ParallelismConfig()
            ).build()

        num_tokens = 16
        for load_balancer in (None, HeadTailCPLoadBalancer.Config()):
            with self.subTest(load_balancer=type(load_balancer).__name__):
                labels_T = torch.arange(num_tokens)
                with parallelism_context.activate_spmd():
                    (
                        _,
                        local_labels_T,
                        model_kwargs,
                        loss_kwargs,
                    ) = model.preprocess_inputs(
                        {
                            "input": labels_T.clone(),
                            "labels": labels_T,
                            "loss_kwargs": {
                                "advantages": labels_T.float() / 10,
                                "loss_mask": labels_T % 3 != 0,
                                "reduction": "sum",
                            },
                        },
                        parallelism_context=parallelism_context,
                        parallelism=ParallelismConfig(
                            context_parallel_load_balancer=load_balancer
                        ),
                    )

                cp_rank = parallelism_context.get_mesh("cp").get_local_rank()
                local_num_tokens = num_tokens // 2
                contiguous_shard_T = torch.arange(
                    cp_rank * local_num_tokens, (cp_rank + 1) * local_num_tokens
                )
                self.assertEqual(
                    torch.equal(local_labels_T, contiguous_shard_T),
                    load_balancer is None,
                )
                torch.testing.assert_close(
                    loss_kwargs["advantages"], local_labels_T.float() / 10
                )
                torch.testing.assert_close(
                    loss_kwargs["loss_mask"], local_labels_T % 3 != 0
                )
                self.assertEqual(loss_kwargs["reduction"], "sum")
                self.assertFalse(model_kwargs.keys() & loss_kwargs.keys())
