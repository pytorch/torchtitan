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

    def test_hf_adapter_roundtrip(self):
        """HF adapter works with FusedQKVLinear's hook-produced wq/wk/wv keys."""
        from torchtitan.models.llama3 import llama3_configs
        from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter

        for config_name, configs, adapter_cls in (
            ("llama3", llama3_configs, Llama3StateDictAdapter),
            ("qwen3", qwen3_configs, Qwen3StateDictAdapter),
        ):
            with self.subTest(model=config_name):
                model_config = configs["debugmodel_fused_qkv"](attn_backend="flex")
                model = model_config.build()
                model.eval()

                sd_original = model.state_dict()
                adapter = adapter_cls(model_config, hf_assets_path=None)

                hf_sd = adapter.to_hf(sd_original)
                sd_restored = adapter.from_hf(hf_sd)

                model2 = model_config.build()
                model2.load_state_dict(sd_restored)

                sd_after = model2.state_dict()
                self.assertEqual(set(sd_original.keys()), set(sd_after.keys()))
                for k in sd_original:
                    self.assertTrue(torch.equal(sd_original[k], sd_after[k]), k)


class TestFusedQKVForwardContiguity(unittest.TestCase):
    """The forward must emit contiguous, head-major q/k/v.

    Splitting the fused ``wqkv`` output along the R dim leaves xk/xv as strided
    views into the fused buffer. PyTorch ops respect strides, but vLLM's
    attention/KV-cache CUDA kernels index q/k/v by raw ``data_ptr()`` assuming a
    contiguous head-major layout, so the forward must materialize them
    contiguously. These run on CPU.
    """

    def test_forward_outputs_are_contiguous_and_correct(self):
        """q/k/v are contiguous and match an independent per-projection matmul."""
        fused = _build_fused()
        x = torch.randn(2, 3, _DIM)
        xq, xk, xv = fused(x)

        # The fix: all three are contiguous.
        self.assertTrue(xq.is_contiguous())
        self.assertTrue(xk.is_contiguous())
        self.assertTrue(xv.is_contiguous())

        # Correctness: compare against an independent reference computed from the
        # stock-layout weights the state_dict hook produces (wq/wk/wv) via plain
        # matmul -- this does not reuse the fused forward's split logic.
        sd = fused.state_dict()
        ref_q = (x @ sd["wq.weight"].T).view(2, 3, _N_HEADS, _HEAD_DIM)
        ref_k = (x @ sd["wk.weight"].T).view(2, 3, _N_KV_HEADS, _HEAD_DIM)
        ref_v = (x @ sd["wv.weight"].T).view(2, 3, _N_KV_HEADS, _HEAD_DIM)
        torch.testing.assert_close(xq, ref_q)
        torch.testing.assert_close(xk, ref_k)
        torch.testing.assert_close(xv, ref_v)

    def test_raw_pointer_read_needs_contiguous(self):
        """A consumer reading the base pointer with contiguous head-major strides
        (what vLLM's kernels do) gets the wrong bytes from the strided split, and
        the correct bytes only after the forward's ``.contiguous()``.
        """
        fused = _build_fused()
        x = torch.randn(2, 3, _DIM)
        bs, seqlen, _ = x.shape

        # Reconstruct the pre-fix strided split (no .contiguous()).
        qkv = fused.wqkv(x).view(bs, seqlen, _N_KV_HEADS, _R_DIM, _HEAD_DIM)
        _, xk_strided, _ = torch.split(qkv, [_HPK, 1, 1], dim=-2)
        xk_strided = xk_strided.reshape(bs, seqlen, _N_KV_HEADS, _HEAD_DIM)
        self.assertFalse(xk_strided.is_contiguous())  # the bug precondition

        # Strides a contiguous tensor of this shape would have.
        contig_strides = torch.empty(xk_strided.shape).stride()

        # Simulate a raw-pointer kernel: read xk's storage with contiguous
        # strides. Because consecutive KV groups are R*head_dim apart in the
        # fused buffer, this lands on interleaved Q bytes -> wrong values.
        raw = xk_strided.as_strided(
            xk_strided.shape, contig_strides, xk_strided.storage_offset()
        )
        self.assertFalse(torch.equal(raw, xk_strided))

        # The fix: the real forward returns contiguous xk, so the same raw read
        # now lands on the correct values.
        xk_fixed = fused(x)[1]
        self.assertTrue(xk_fixed.is_contiguous())
        raw_fixed = xk_fixed.as_strided(
            xk_fixed.shape, contig_strides, xk_fixed.storage_offset()
        )
        self.assertTrue(torch.equal(raw_fixed, xk_fixed))


if __name__ == "__main__":
    unittest.main()
