# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint interop tests for FusedQKVLinear.

FusedQKVLinear stores a single fused ``wqkv`` parameter but checkpoints in the
stock ``QKVLinear`` layout (``wq.weight`` / ``wk.weight`` / ``wv.weight``) via
state_dict hooks, so its checkpoints round-trip with the non-fused module.

All tests run on CPU.
"""

import unittest

import torch
from torchtitan.models.common.attention import FusedQKVLinear, QKVLinear
from torchtitan.models.common.nn_modules import Linear

_DIM = 16
_N_HEADS = 4
_N_KV_HEADS = 2
_HEAD_DIM = 8
_HPK = _N_HEADS // _N_KV_HEADS  # heads_per_kv = 2
_R_DIM = _HPK + 2  # 4
_WQKV_OUT = (_N_HEADS + 2 * _N_KV_HEADS) * _HEAD_DIM  # 64
_WQ_OUT = _N_HEADS * _HEAD_DIM  # 32
_WK_OUT = _N_KV_HEADS * _HEAD_DIM  # 16


def _build_fused(with_bias: bool = False) -> FusedQKVLinear:
    fused = FusedQKVLinear.Config(
        head_dim=_HEAD_DIM,
        n_heads=_N_HEADS,
        n_kv_heads=_N_KV_HEADS,
        wqkv=Linear.Config(in_features=_DIM, out_features=_WQKV_OUT, bias=with_bias),
    ).build()
    with torch.no_grad():
        fused.wqkv.weight.copy_(torch.randn(_WQKV_OUT, _DIM))
        if with_bias:
            fused.wqkv.bias.copy_(torch.randn(_WQKV_OUT))
    return fused


def _build_stock(with_bias: bool = False) -> QKVLinear:
    stock = QKVLinear.Config(
        head_dim=_HEAD_DIM,
        wq=Linear.Config(in_features=_DIM, out_features=_WQ_OUT, bias=with_bias),
        wkv=Linear.Config(in_features=_DIM, out_features=_WK_OUT, bias=with_bias),
    ).build()
    with torch.no_grad():
        for p in stock.parameters():
            p.copy_(torch.randn_like(p))
    return stock


class TestFusedQKVCheckpointInterop(unittest.TestCase):
    def test_fused_checkpoint_loads_into_stock(self):
        """Fused state_dict loads into stock QKVLinear with correct weights."""
        fused = _build_fused(with_bias=True)
        stock = _build_stock(with_bias=True)
        stock.load_state_dict(fused.state_dict())

        n_kv = _WQKV_OUT // (_R_DIM * _HEAD_DIM)
        wqkv = fused.wqkv.weight.reshape(n_kv, _R_DIM, _HEAD_DIM, _DIM)
        self.assertTrue(torch.equal(stock.wq.weight, wqkv[:, :_HPK].reshape(-1, _DIM)))
        self.assertTrue(torch.equal(stock.wk.weight, wqkv[:, _HPK].reshape(-1, _DIM)))
        self.assertTrue(
            torch.equal(stock.wv.weight, wqkv[:, _HPK + 1].reshape(-1, _DIM))
        )

        b_3d = fused.wqkv.bias.reshape(n_kv, _R_DIM, _HEAD_DIM)
        self.assertTrue(torch.equal(stock.wq.bias, b_3d[:, :_HPK].reshape(-1)))
        self.assertTrue(torch.equal(stock.wk.bias, b_3d[:, _HPK].reshape(-1)))
        self.assertTrue(torch.equal(stock.wv.bias, b_3d[:, _HPK + 1].reshape(-1)))

    def test_stock_checkpoint_loads_into_fused(self):
        """A stock checkpoint loads into FusedQKVLinear."""
        stock = _build_stock(with_bias=True)
        fused = _build_fused(with_bias=True)
        fused.load_state_dict(stock.state_dict())

        n_kv = _WQKV_OUT // (_R_DIM * _HEAD_DIM)
        wqkv = fused.wqkv.weight.reshape(n_kv, _R_DIM, _HEAD_DIM, _DIM)
        self.assertTrue(torch.equal(wqkv[:, :_HPK].reshape(-1, _DIM), stock.wq.weight))
        self.assertTrue(torch.equal(wqkv[:, _HPK].reshape(-1, _DIM), stock.wk.weight))
        self.assertTrue(
            torch.equal(wqkv[:, _HPK + 1].reshape(-1, _DIM), stock.wv.weight)
        )

        wqkv_b = fused.wqkv.bias.reshape(n_kv, _R_DIM, _HEAD_DIM)
        self.assertTrue(torch.equal(wqkv_b[:, :_HPK].reshape(-1), stock.wq.bias))
        self.assertTrue(torch.equal(wqkv_b[:, _HPK].reshape(-1), stock.wk.bias))
        self.assertTrue(torch.equal(wqkv_b[:, _HPK + 1].reshape(-1), stock.wv.bias))


if __name__ == "__main__":
    unittest.main()
