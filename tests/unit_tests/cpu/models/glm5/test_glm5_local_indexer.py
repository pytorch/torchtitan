# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch

from torchtitan.models.glm5.indexer import select_local_topk


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("topk", [0, 8, 2056])
def test_local_selector_streaming_matches_dense_scores(dtype, topk):
    torch.manual_seed(42)
    q = torch.randn(19, 2, 16, dtype=dtype, requires_grad=True)
    k = torch.randn(2053, 16, dtype=dtype, requires_grad=True)
    weights = torch.randn(19, 2, dtype=dtype, requires_grad=True)
    bounds = torch.tensor(
        [[0, 0], [0, 3], [1020, 1040], [1024, 2053]] * 4
        + [[2050, 2053], [5, 5], [7, 2049]],
        dtype=torch.int32,
    )
    actual = select_local_topk(q, k, weights, bounds, topk)
    assert actual.shape == (19, topk)
    assert actual.dtype == torch.int32 and actual.is_contiguous()
    assert not actual.requires_grad
    keys = torch.arange(k.shape[0])
    valid = (keys >= bounds[:, :1]) & (keys < bounds[:, 1:])
    scores = (
        (torch.einsum("qnh,kh->qnk", q, k).relu() * weights[:, :, None]).sum(1)
        / math.sqrt(32)
    ).masked_fill(~valid, -torch.inf)
    expected = scores.topk(min(topk, k.shape[0]), dim=-1).values
    expected = torch.nn.functional.pad(
        expected, (0, max(0, topk - k.shape[0])), value=-torch.inf
    )
    assert torch.all((actual == -1) | ((actual >= 0) & (actual < k.shape[0])))
    selected = scores.gather(1, actual.clamp_min(0).long()).masked_fill(
        actual < 0, -torch.inf
    )
    torch.testing.assert_close(selected.sort(-1).values, expected.sort(-1).values)
    for row in actual:
        ids = row[row >= 0]
        assert ids.unique().numel() == ids.numel()


def test_local_selector_rejects_invalid_metadata_layout():
    q, k, weights = torch.ones(3, 2, 16), torch.ones(8, 16), torch.ones(3, 2)
    with pytest.raises(ValueError, match="contiguous INT32"):
        select_local_topk(q, k, weights, torch.zeros(3, 2, dtype=torch.int64), 2)


def test_local_selector_empty_key_pool():
    output = select_local_topk(
        torch.ones(3, 2, 16), torch.empty(0, 16), torch.ones(3, 2),
        torch.zeros(3, 2, dtype=torch.int32), 4,
    )
    torch.testing.assert_close(output, torch.full((3, 4), -1, dtype=torch.int32))
