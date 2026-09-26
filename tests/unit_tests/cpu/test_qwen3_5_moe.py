# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import ANY, call, patch

import spmd_types as spmd
import torch

from torchtitan.models.common.linear import Linear, RowParallelLinear
from torchtitan.models.qwen3_5.moe import SigmoidGatedFeedForward


class TestSigmoidGatedFeedForward(unittest.TestCase):
    def test_shared_input_gather_and_projections_use_one_remat_region(self):
        x_TD = torch.randn(4, 4)
        tp_group = object()

        for sp_enabled, expected_names, expected_redistributions in (
            (
                False,
                ["input_projections", "w2"],
                [
                    call(
                        x_TD,
                        tp_group,
                        src=spmd.I,
                        dst=spmd.R,
                        backward_options={"op_dtype": x_TD.dtype},
                    )
                ],
            ),
            (
                True,
                ["input_projections", "w2"],
                [
                    call(
                        x_TD,
                        tp_group,
                        src=spmd.S(0),
                        dst=spmd.R,
                        backward_options={"op_dtype": x_TD.dtype},
                    ),
                    call(
                        ANY,
                        tp_group,
                        src=spmd.R,
                        dst=spmd.S(0),
                        backward_options={"op_dtype": x_TD.dtype},
                    ),
                    call(
                        ANY,
                        tp_group,
                        src=spmd.P,
                        dst=spmd.S(0),
                        backward_options={"op_dtype": x_TD.dtype},
                    ),
                ],
            ),
        ):
            w2_config = (
                RowParallelLinear.Config(in_features=8, out_features=4)
                if sp_enabled
                else Linear.Config(in_features=8, out_features=4)
            )
            shared_expert = SigmoidGatedFeedForward.Config(
                w13=Linear.Config(in_features=4, out_features=8, num_linears=2),
                w2=w2_config,
                gate=Linear.Config(in_features=4, out_features=1),
            ).build()
            with (
                self.subTest(sp_enabled=sp_enabled),
                patch(
                    "torchtitan.models.qwen3_5.moe.spmd_sparse_mesh",
                    return_value=object(),
                ),
                patch(
                    "torchtitan.models.qwen3_5.moe.spmd_dense_sp_enabled",
                    return_value=sp_enabled,
                ),
                patch(
                    "torchtitan.models.qwen3_5.moe.spmd_mesh_group",
                    return_value=tp_group,
                ),
                patch(
                    "torchtitan.models.common.linear.spmd_dense_sp_enabled",
                    return_value=sp_enabled,
                ),
                patch(
                    "torchtitan.models.common.linear.spmd_mesh_group",
                    return_value=tp_group,
                ),
                patch(
                    "torchtitan.models.qwen3_5.moe.spmd.redistribute",
                    side_effect=lambda tensor, *_args, **_kwargs: tensor,
                ) as redistribute,
                patch(
                    "torchtitan.models.qwen3_5.moe.remat.region",
                    side_effect=lambda function, *_args, **_kwargs: function,
                ) as region,
            ):
                output_TD = shared_expert(x_TD)

            self.assertEqual(output_TD.shape, x_TD.shape)
            self.assertEqual(
                [region_call.args[1] for region_call in region.call_args_list],
                expected_names,
            )
            self.assertEqual(redistribute.call_args_list, expected_redistributions)


if __name__ == "__main__":
    unittest.main()
