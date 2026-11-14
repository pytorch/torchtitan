# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest import mock

import pytest
import torch
from torch.testing._internal.common_utils import DeterministicGuard
from torch.utils._pytree import tree_leaves, tree_map

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import (
    QSAIndexerLoss,
    QSAIndexMetadata,
)
from torchtitan.tools.utils import has_cuda_capability

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and has_cuda_capability(9, 0)),
    reason="the fused lightning_indexer requires SM90 or newer",
)

_NUM_TOKENS = 256
_MAX_NUM_DOCUMENTS = 4


def _layout(lengths, padded):
    positions = torch.cat([torch.arange(n) for n in lengths]).cuda()
    padding = torch.zeros(_NUM_TOKENS, dtype=torch.bool, device="cuda")
    padding[padded] = True
    metadata = QSAIndexMetadata.from_inputs(
        positions,
        padding_mask_T=padding,
        compress_ratio=4,
        max_num_documents=_MAX_NUM_DOCUMENTS,
    )
    return positions, padding, metadata


def test_indexer_graph_replay_refreshes_document_layout():
    """A captured indexer step replays a new document layout exactly.

    Guards the static-shape QSA metadata: capture fails on a host sync, and
    replay is wrong if any metadata shape depends on the captured batch.
    """
    torch.manual_seed(0)
    config = build_model_config(
        "debugmodel",
        seq_len=_NUM_TOKENS,
        parallelism=ParallelismConfig(),
        indexer_loss=QSAIndexerLoss.Config(coeff=1.0),
    )
    indexer = config.layers[3].attention.indexer.build().cuda()
    indexer.init_states()
    for parameter in indexer.parameters():
        parameter.data = parameter.data.bfloat16()
    x = torch.randn(
        _NUM_TOKENS, indexer.index_qk_proj.in_features, device="cuda"
    ).bfloat16()
    # Two documents with trailing padding, then four with padding inside one.
    captured = _layout((100, 156), slice(236, None))
    replayed = _layout((40, 60, 70, 86), slice(50, 58))
    assert captured[2].block_indices_GR.shape == replayed[2].block_indices_GR.shape

    def step(positions, padding, metadata):
        selected, scores = indexer(
            x,
            positions=positions,
            document_positions_T=positions,
            padding_mask_T=padding,
            metadata=metadata,
        )
        loss = scores.masked_fill(~scores.isfinite(), 0).float().square().sum()
        return selected, loss, torch.autograd.grad(loss, list(indexer.parameters()))

    # cuBLAS needs a fixed workspace for deterministic algorithms.
    with (
        mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}),
        DeterministicGuard(True),
    ):
        expected = tree_map(torch.Tensor.detach, step(*replayed))
        static = tree_map(torch.Tensor.clone, captured)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step(*static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = step(*static)
        for target, source in zip(
            tree_leaves(static), tree_leaves(replayed), strict=True
        ):
            target.copy_(source)
        graph.replay()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
