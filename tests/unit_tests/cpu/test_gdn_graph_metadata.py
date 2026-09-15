# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan.experiments.rl.models.gdn_metadata import GDNGraphMetadata


def test_metadata_refresh_keeps_addresses_and_clears_padding() -> None:
    metadata = GDNGraphMetadata.allocate(2, device=torch.device("cpu"))
    for tensor in vars(metadata).values():
        tensor.fill_(1)
    addresses = {name: tensor.data_ptr() for name, tensor in vars(metadata).items()}
    metadata.update(
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([False]),
        num_reqs=1,
        num_tokens=3,
        token_capacity=8,
    )
    assert metadata.query_start_loc.tolist() == [0, 3, 8, 8]
    assert metadata.state_indices.tolist() == [2, 0, 0]
    assert metadata.has_initial_state.tolist() == [False, False, False]
    assert {
        name: tensor.data_ptr() for name, tensor in vars(metadata).items()
    } == addresses
