# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch
from torch.distributed.tensor import DTensor

from torchtitan.distributed.flex_shard.dist_muon import (
    _matrix_batch_views_from_shape,
    DistMuon,
)


class TestDistMuonStorageValidation(unittest.TestCase):
    def _create_mock_dtensor(self, device: torch.device | MagicMock | str) -> MagicMock:
        mock_dtensor = MagicMock(spec=DTensor)
        local_tensor = MagicMock(spec=torch.Tensor)
        if isinstance(device, str):
            local_tensor.device = torch.device(device)
        else:
            local_tensor.device = device
        mock_dtensor.to_local.return_value = local_tensor
        return mock_dtensor

    def test_single_cpu_device_accepted(self):
        optimizer = object.__new__(DistMuon)
        optimizer.param_groups = [
            {
                "params": [
                    self._create_mock_dtensor("cpu"),
                    self._create_mock_dtensor("cpu"),
                ],
                "param_names": ["layer1.weight", "layer2.weight"],
            }
        ]
        validated_device = optimizer._validate_parameter_storage()
        self.assertEqual(validated_device, torch.device("cpu"))

    def test_uniform_matrix_batches_preserve_storage_aliases(self):
        backing = torch.arange(72).view(24, 3)
        compute = backing[4:10]
        expected_matrices = torch.stack((compute[:2], compute[2:4], compute[4:6]))
        (view,) = _matrix_batch_views_from_shape(compute.shape, matrix_rows=2)
        matrices = view.view_as_matrix_batch(compute)
        torch.testing.assert_close(matrices, expected_matrices)

        expected_backing = backing.clone()
        expected_backing[6:8].fill_(-1)
        matrices[1].fill_(-1)
        torch.testing.assert_close(backing, expected_backing)

    def test_matrix_batch_view_cannot_escape_supplied_tensor(self):
        backing = torch.arange(24).view(8, 3)
        (view,) = _matrix_batch_views_from_shape(torch.Size((6, 3)), matrix_rows=2)
        with self.assertRaises(RuntimeError):
            view.view_as_matrix_batch(backing[2:6])

    def test_matrix_batch_view_rejects_noncontiguous_input(self):
        compute = torch.arange(24).view(4, 6)[:, ::2]
        (view,) = _matrix_batch_views_from_shape(compute.shape, matrix_rows=2)
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            view.view_as_matrix_batch(compute)


if __name__ == "__main__":
    unittest.main()
