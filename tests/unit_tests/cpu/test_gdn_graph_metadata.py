# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan.experiments.rl.models.gdn_metadata import GDNGraphMetadata


def test_metadata_updates_preserve_addresses_and_clear_stale_requests() -> None:
    storage = GDNGraphMetadata.allocate(4, device=torch.device("cpu"))
    metadata = storage.for_capacity(192)
    tensors = (
        metadata.query_start_loc,
        metadata.state_indices,
        metadata.has_initial_state,
    )
    addresses = [tensor.data_ptr() for tensor in tensors]
    metadata.update(
        torch.tensor([0, 1, 31, 127], dtype=torch.int32),
        torch.tensor([5, 1, 3], dtype=torch.int32),
        torch.tensor([True, False, True]),
        num_reqs=3,
        num_tokens=127,
        token_capacity=192,
    )
    assert metadata.query_start_loc.tolist() == [0, 1, 31, 127, 192, 192]
    assert metadata.state_indices.tolist() == [5, 1, 3, 0, 0]
    assert metadata.has_initial_state.tolist() == [True, False, True, False, False]

    metadata.update(
        torch.tensor([0, 65], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        None,
        num_reqs=1,
        num_tokens=65,
        token_capacity=192,
    )
    assert [tensor.data_ptr() for tensor in tensors] == addresses
    assert metadata.query_start_loc.tolist() == [0, 65, 192, 192, 192, 192]
    assert metadata.state_indices.tolist() == [2, 0, 0, 0, 0]
    assert metadata.has_initial_state.tolist() == [True, False, False, False, False]

    metadata.clear(token_capacity=192)
    assert metadata.query_start_loc.tolist() == [0, 192, 192, 192, 192, 192]
    assert not metadata.state_indices.any()
    assert not metadata.has_initial_state.any()


@pytest.mark.parametrize("capacity", [1, 2, 4, 192])
def test_bucket_views_share_storage_and_reserve_a_null_interval(capacity: int) -> None:
    storage = GDNGraphMetadata.allocate(4, device=torch.device("cpu"))
    first = storage.for_capacity(capacity)
    second = storage.for_capacity(capacity)
    assert first.state_indices.numel() == min(capacity, 4) + 1
    assert first.query_start_loc.numel() == first.state_indices.numel() + 1
    assert first.query_start_loc.data_ptr() == second.query_start_loc.data_ptr()
    assert first.state_indices.data_ptr() == storage.state_indices.data_ptr()
    assert first.has_initial_state.data_ptr() == storage.has_initial_state.data_ptr()
    first.clear(token_capacity=capacity)
    assert first.query_start_loc[0] == 0
    assert (first.query_start_loc[1:] == capacity).all()


@pytest.mark.parametrize(("num_reqs", "num_tokens"), [(5, 5), (1, 193)])
def test_rejects_batches_exceeding_capacity(num_reqs: int, num_tokens: int) -> None:
    metadata = GDNGraphMetadata.allocate(4, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="exceeds captured"):
        metadata.update(
            torch.zeros(num_reqs + 1, dtype=torch.int32),
            torch.ones(num_reqs, dtype=torch.int32),
            None,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            token_capacity=192,
        )
