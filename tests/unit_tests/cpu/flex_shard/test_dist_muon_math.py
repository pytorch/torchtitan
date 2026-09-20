# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.flex_shard.dist_muon import (
    _apply_muon_update,
    _compute_muon_direction,
    _MatrixBatchView,
)


class TestMuonMath(unittest.TestCase):
    def test_tensor_views_match_independent_muon_matrices(self):
        parameter = torch.arange(24).reshape(12, 2).float().div_(7).sin_()
        gradient = torch.arange(24).reshape(12, 2).float().mul_(0.137).cos_()
        reference = [
            torch.nn.Parameter(matrix.clone()) for matrix in parameter.split(4)
        ]
        reference_optimizer = torch.optim.Muon(
            reference,
            lr=0.03,
            weight_decay=0.2,
            momentum=0,
            nesterov=False,
            ns_steps=2,
            adjust_lr_fn="match_rms_adamw",
        )
        for matrix, matrix_gradient in zip(reference, gradient.split(4), strict=True):
            matrix.grad = matrix_gradient.clone()

        compute = gradient.clone()
        _compute_muon_direction(
            compute,
            matrix_views=(
                _MatrixBatchView(
                    shape=torch.Size((3, 4, 2)), strides=(8, 2, 1), offset=0
                ),
            ),
            ns_coefficients=(3.4445, -4.775, 2.0315),
            ns_steps=2,
            eps=1e-7,
        )
        _apply_muon_update(
            parameter,
            compute,
            lr=0.03,
            weight_decay=0.2,
            adjust_lr_fn="match_rms_adamw",
            compute_matrix_shape=torch.Size((4, 2)),
        )
        reference_optimizer.step()
        torch.testing.assert_close(
            parameter,
            torch.cat([matrix.detach() for matrix in reference]),
            rtol=0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
