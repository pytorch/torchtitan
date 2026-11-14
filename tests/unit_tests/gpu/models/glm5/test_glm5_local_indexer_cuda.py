# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch

from torchtitan.models.glm5.indexer import _local_topk_op, select_local_topk


@pytest.fixture(autouse=True)
def require_cuda_selector():
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("NVIDIA CUDA required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("SM80 or newer required")


def inputs(dtype, heads=4, width=16):
    torch.manual_seed(42)
    q = torch.randn(7, heads, width, device="cuda", dtype=dtype)
    k = torch.randn(2053, width, device="cuda", dtype=dtype)
    weights = torch.randn(7, heads, device="cuda", dtype=dtype)
    bounds = torch.tensor(
        [[0, 0], [0, 1], [0, 2053], [1000, 2053], [5, 8], [2052, 2053], [4, 4]],
        device="cuda", dtype=torch.int32,
    )
    return q, k, weights, bounds


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads,width,topk", [(4, 16, 0), (4, 16, 8), (32, 128, 2048)])
def test_fused_selector_matches_dense_oracle(dtype, heads, width, topk):
    q, k, weights, bounds = inputs(dtype, heads, width)
    actual = select_local_topk(q, k, weights, bounds, topk)
    keys = torch.arange(k.shape[0], device=k.device)
    valid = (keys >= bounds[:, :1]) & (keys < bounds[:, 1:])
    scores = (
        (torch.einsum("qnh,kh->qnk", q.double(), k.double()).relu()
         * weights.double()[:, :, None]).sum(1) / math.sqrt(heads * width)
    ).masked_fill(~valid, -torch.inf)
    assert torch.all((actual == -1) | ((actual >= 0) & (actual < k.shape[0])))
    selected = scores.gather(1, actual.clamp_min(0).long()).masked_fill(
        actual < 0, -torch.inf
    )
    expected = scores.topk(topk, dim=-1).values
    torch.testing.assert_close(
        selected.sort(-1).values, expected.sort(-1).values, rtol=1e-5, atol=1e-6
    )
    for row in actual:
        ids = row[row >= 0]
        assert ids.unique().numel() == ids.numel()


@pytest.mark.parametrize("permuted", [False, True])
def test_selector_opcheck(permuted):
    q, k, weights, bounds = inputs(torch.bfloat16)
    if permuted:
        q = q.transpose(0, 1).contiguous().transpose(0, 1)
        k = k.T.contiguous().T
        weights = weights.T.contiguous().T
    # Unique single-key support makes integer ordering stable for AOT opcheck.
    bounds[:, 0] = torch.arange(7, device="cuda")
    bounds[:, 1] = bounds[:, 0] + 1
    torch.library.opcheck(_local_topk_op, (q, k, weights, bounds, 2))


def test_public_selector_fullgraph_and_cuda_graph():
    args = inputs(torch.bfloat16)
    compiled = torch.compile(select_local_topk, fullgraph=True)
    expected = select_local_topk(*args, 8)
    actual = compiled(*args, 8)
    torch.testing.assert_close(actual, expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            select_local_topk(*args, 8)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = select_local_topk(*args, 8)
    args[3][2, 1] = 512
    graph.replay()
    torch.testing.assert_close(output, select_local_topk(*args, 8))
