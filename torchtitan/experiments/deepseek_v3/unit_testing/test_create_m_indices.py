# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan.experiments.deepseek_v3.dsgemm_utils import (
    create_indices_from_offsets_nosync,
)


def test_create_indices_from_offsets_nosync():
    # Test case 1: Regular offsets with increasing values
    m_offsets = torch.tensor([128, 256, 384, 512], device="cuda", dtype=torch.int32)
    indices = create_indices_from_offsets_nosync(m_offsets)

    # Expected: 128 zeros, 128 ones, 128 twos, 128 threes
    expected = torch.cat(
        [
            torch.zeros(128, dtype=torch.int32, device="cuda"),
            torch.ones(128, dtype=torch.int32, device="cuda"),
            2 * torch.ones(128, dtype=torch.int32, device="cuda"),
            3 * torch.ones(128, dtype=torch.int32, device="cuda"),
        ]
    )

    assert torch.all(indices == expected), "Test case 1 failed"

    # Test case 2: Offsets with empty groups
    m_offsets = torch.tensor([128, 128, 256, 384], device="cuda", dtype=torch.int32)
    indices = create_indices_from_offsets_nosync(m_offsets)

    # Expected: 128 zeros, 0 ones (empty group), 128 twos, 128 threes
    expected = torch.cat(
        [
            torch.zeros(128, dtype=torch.int32, device="cuda"),
            2 * torch.ones(128, dtype=torch.int32, device="cuda"),
            3 * torch.ones(128, dtype=torch.int32, device="cuda"),
        ]
    )

    assert torch.all(indices == expected), "Test case 2 failed"

    print("All tests passed!")


if __name__ == "__main__":
    test_create_indices_from_offsets_nosync()
