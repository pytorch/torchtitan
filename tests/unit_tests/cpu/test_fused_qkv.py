# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint and forward tests for FusedQKVLinear.

FusedQKVLinear stores a single fused ``wqkv`` parameter but checkpoints in the
logical ``wq.weight`` / ``wk.weight`` / ``wv.weight`` layout via state_dict
hooks.

All tests run on CPU.
"""

import unittest

import torch
from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.models.common.linear import Linear

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


def _logical_state_dict(with_bias: bool = False) -> dict[str, torch.Tensor]:
    state_dict = {
        "wq.weight": torch.randn(_WQ_OUT, _DIM),
        "wk.weight": torch.randn(_WK_OUT, _DIM),
        "wv.weight": torch.randn(_WK_OUT, _DIM),
    }
    if with_bias:
        state_dict.update(
            {
                "wq.bias": torch.randn(_WQ_OUT),
                "wk.bias": torch.randn(_WK_OUT),
                "wv.bias": torch.randn(_WK_OUT),
            }
        )
    return state_dict


class TestFusedQKVCheckpointInterop(unittest.TestCase):
    def test_state_dict_exposes_logical_qkv(self):
        """The fused parameter is exposed as logical Q/K/V tensors."""
        fused = _build_fused(with_bias=True)
        state_dict = fused.state_dict()

        n_kv = _WQKV_OUT // (_R_DIM * _HEAD_DIM)
        wqkv = fused.wqkv.weight.reshape(n_kv, _R_DIM, _HEAD_DIM, _DIM)
        self.assertTrue(
            torch.equal(state_dict["wq.weight"], wqkv[:, :_HPK].reshape(-1, _DIM))
        )
        self.assertTrue(
            torch.equal(state_dict["wk.weight"], wqkv[:, _HPK].reshape(-1, _DIM))
        )
        self.assertTrue(
            torch.equal(state_dict["wv.weight"], wqkv[:, _HPK + 1].reshape(-1, _DIM))
        )

        b_3d = fused.wqkv.bias.reshape(n_kv, _R_DIM, _HEAD_DIM)
        self.assertTrue(torch.equal(state_dict["wq.bias"], b_3d[:, :_HPK].reshape(-1)))
        self.assertTrue(torch.equal(state_dict["wk.bias"], b_3d[:, _HPK].reshape(-1)))
        self.assertTrue(
            torch.equal(state_dict["wv.bias"], b_3d[:, _HPK + 1].reshape(-1))
        )

    def test_logical_checkpoint_loads_into_fused(self):
        """Logical Q/K/V checkpoint tensors are packed into wqkv."""
        state_dict = _logical_state_dict(with_bias=True)
        fused = _build_fused(with_bias=True)
        fused.load_state_dict(state_dict)

        n_kv = _WQKV_OUT // (_R_DIM * _HEAD_DIM)
        wqkv = fused.wqkv.weight.reshape(n_kv, _R_DIM, _HEAD_DIM, _DIM)
        self.assertTrue(
            torch.equal(wqkv[:, :_HPK].reshape(-1, _DIM), state_dict["wq.weight"])
        )
        self.assertTrue(
            torch.equal(wqkv[:, _HPK].reshape(-1, _DIM), state_dict["wk.weight"])
        )
        self.assertTrue(
            torch.equal(wqkv[:, _HPK + 1].reshape(-1, _DIM), state_dict["wv.weight"])
        )

        wqkv_b = fused.wqkv.bias.reshape(n_kv, _R_DIM, _HEAD_DIM)
        self.assertTrue(
            torch.equal(wqkv_b[:, :_HPK].reshape(-1), state_dict["wq.bias"])
        )
        self.assertTrue(torch.equal(wqkv_b[:, _HPK].reshape(-1), state_dict["wk.bias"]))
        self.assertTrue(
            torch.equal(wqkv_b[:, _HPK + 1].reshape(-1), state_dict["wv.bias"])
        )

    def test_hf_adapter_roundtrip(self):
        """HF adapter works with FusedQKVLinear's hook-produced wq/wk/wv keys."""
        from torchtitan.models.llama3 import llama3_configs
        from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
        from torchtitan.models.muse_glimmer import muse_glimmer_configs
        from torchtitan.models.muse_glimmer.state_dict_adapter import (
            MuseGlimmerStateDictAdapter,
        )
        from torchtitan.models.qwen3 import qwen3_configs
        from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter

        for config_name, configs, adapter_cls in (
            ("llama3", llama3_configs, Llama3StateDictAdapter),
            ("qwen3", qwen3_configs, Qwen3StateDictAdapter),
            ("muse_glimmer", muse_glimmer_configs, MuseGlimmerStateDictAdapter),
        ):
            with self.subTest(model=config_name):
                build_config, max_context_length = configs["debugmodel"]
                model_config = build_config(
                    attn_backend="flex", seq_len=max_context_length
                )
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
        num_tokens = 6
        x_TD = torch.randn(num_tokens, _DIM)
        xq_THK, xk_THK, xv_THV = fused(x_TD)

        self.assertEqual(xq_THK.shape, (num_tokens, _N_HEADS, _HEAD_DIM))
        self.assertEqual(xk_THK.shape, (num_tokens, _N_KV_HEADS, _HEAD_DIM))
        self.assertEqual(xv_THV.shape, (num_tokens, _N_KV_HEADS, _HEAD_DIM))
        self.assertTrue(xq_THK.is_contiguous())
        self.assertTrue(xk_THK.is_contiguous())
        self.assertTrue(xv_THV.is_contiguous())

        sd = fused.state_dict()
        ref_q_THK = (x_TD @ sd["wq.weight"].T).view(num_tokens, _N_HEADS, _HEAD_DIM)
        ref_k_THK = (x_TD @ sd["wk.weight"].T).view(num_tokens, _N_KV_HEADS, _HEAD_DIM)
        ref_v_THV = (x_TD @ sd["wv.weight"].T).view(num_tokens, _N_KV_HEADS, _HEAD_DIM)
        torch.testing.assert_close(xq_THK, ref_q_THK)
        torch.testing.assert_close(xk_THK, ref_k_THK)
        torch.testing.assert_close(xv_THV, ref_v_THV)

    def test_raw_pointer_read_needs_contiguous(self):
        """A consumer reading the base pointer with contiguous head-major strides
        (what vLLM's kernels do) gets the wrong bytes from the strided split, and
        the correct bytes only after the forward's ``.contiguous()``.
        """
        fused = _build_fused()
        num_tokens = 6
        x_TD = torch.randn(num_tokens, _DIM)

        # Reconstruct the pre-fix strided split (no .contiguous()).
        qkv = fused.wqkv(x_TD).view(num_tokens, _N_KV_HEADS, _R_DIM, _HEAD_DIM)
        _, xk_strided, _ = torch.split(qkv, [_HPK, 1, 1], dim=-2)
        xk_strided = xk_strided.reshape(num_tokens, _N_KV_HEADS, _HEAD_DIM)
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
        xk_fixed = fused(x_TD)[1]
        self.assertTrue(xk_fixed.is_contiguous())
        raw_fixed = xk_fixed.as_strided(
            xk_fixed.shape, contig_strides, xk_fixed.storage_offset()
        )
        self.assertTrue(torch.equal(raw_fixed, xk_fixed))


if __name__ == "__main__":
    unittest.main()
