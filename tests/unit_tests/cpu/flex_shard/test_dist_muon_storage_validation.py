# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import torch
from torch.distributed.tensor import DTensor

from torchtitan.distributed.flex_shard.dist_muon import DistMuon


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


if __name__ == "__main__":
    unittest.main()
