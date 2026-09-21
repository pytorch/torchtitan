# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-dict adapter and forward tests for QKVLinear.

QKVLinear keeps its packed ``wqkv`` parameter in native state dicts. Hugging
Face adapters expose the logical ``wq`` / ``wk`` / ``wv`` projections.
"""

import unittest

import torch
from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.common.linear import Linear

_DIM = 16
_N_HEADS = 4
_N_KV_HEADS = 2
_HEAD_DIM = 8
_HPK = _N_HEADS // _N_KV_HEADS  # heads_per_kv = 2
_R_DIM = _HPK + 2  # 4
_WQKV_OUT = (_N_HEADS + 2 * _N_KV_HEADS) * _HEAD_DIM  # 64


def _build_qkv_linear(with_bias: bool = False) -> QKVLinear:
    fused = QKVLinear.Config(
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


class TestQKVLinearCheckpointInterop(unittest.TestCase):
    def test_state_dict_retains_native_qkv(self):
        """Native state dicts expose the physical packed parameter."""
        fused = _build_qkv_linear(with_bias=True)
        state_dict = fused.state_dict()

        self.assertEqual(set(state_dict), {"wqkv.weight", "wqkv.bias"})
        self.assertEqual(
            state_dict["wqkv.weight"].data_ptr(), fused.wqkv.weight.data_ptr()
        )
        self.assertEqual(state_dict["wqkv.bias"].data_ptr(), fused.wqkv.bias.data_ptr())

    def test_native_checkpoint_loads_into_qkv_linear(self):
        """A native packed checkpoint loads without layout conversion."""
        source = _build_qkv_linear(with_bias=True)
        target = _build_qkv_linear(with_bias=True)

        target.load_state_dict(source.state_dict())

        torch.testing.assert_close(target.wqkv.weight, source.wqkv.weight)
        torch.testing.assert_close(target.wqkv.bias, source.wqkv.bias)

    def test_adapter_split_and_merge_preserve_qkv_layout(self):
        """HF-boundary conversion preserves the packed QKV ordering."""
        fused = _build_qkv_linear(with_bias=True)
        native_state_dict = dict(fused.state_dict())
        state_dict = dict(native_state_dict)
        from torchtitan.models.llama3 import llama3_configs
        from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

        build_config, max_context_length = llama3_configs["debugmodel"]
        model_config = build_config(attn_backend="flex", seq_len=max_context_length)
        adapter = Llama3StateDictAdapter(model_config, hf_assets_path=None)

        adapter._split_qkv_linear(
            state_dict,
            prefix="",
            head_dim=_HEAD_DIM,
            heads_per_kv=_HPK,
        )

        n_kv_heads = _WQKV_OUT // (_R_DIM * _HEAD_DIM)
        weight = fused.wqkv.weight.reshape(n_kv_heads, _R_DIM, _HEAD_DIM, _DIM)
        bias = fused.wqkv.bias.reshape(n_kv_heads, _R_DIM, _HEAD_DIM)
        torch.testing.assert_close(
            state_dict["wq.weight"], weight[:, :_HPK].reshape(-1, _DIM)
        )
        torch.testing.assert_close(
            state_dict["wk.weight"], weight[:, _HPK].reshape(-1, _DIM)
        )
        torch.testing.assert_close(
            state_dict["wv.weight"], weight[:, _HPK + 1].reshape(-1, _DIM)
        )
        torch.testing.assert_close(state_dict["wq.bias"], bias[:, :_HPK].reshape(-1))
        torch.testing.assert_close(state_dict["wk.bias"], bias[:, _HPK].reshape(-1))
        torch.testing.assert_close(state_dict["wv.bias"], bias[:, _HPK + 1].reshape(-1))

        adapter._merge_qkv_linear(
            state_dict,
            prefix="",
            head_dim=_HEAD_DIM,
            heads_per_kv=_HPK,
        )

        self.assertEqual(state_dict.keys(), native_state_dict.keys())
        for key, value in native_state_dict.items():
            torch.testing.assert_close(state_dict[key], value)

    def test_hf_adapter_roundtrip(self):
        """HF adapters split and restore QKVLinear's native packed parameter."""
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
        fused = _build_qkv_linear()
        num_tokens = 6
        x_TD = torch.randn(num_tokens, _DIM)
        xq_THK, xk_THK, xv_THV = fused(x_TD)

        self.assertEqual(xq_THK.shape, (num_tokens, _N_HEADS, _HEAD_DIM))
        self.assertEqual(xk_THK.shape, (num_tokens, _N_KV_HEADS, _HEAD_DIM))
        self.assertEqual(xv_THV.shape, (num_tokens, _N_KV_HEADS, _HEAD_DIM))
        self.assertTrue(xq_THK.is_contiguous())
        self.assertTrue(xk_THK.is_contiguous())
        self.assertTrue(xv_THV.is_contiguous())

        wqkv = fused.wqkv.weight.reshape(_N_KV_HEADS, _R_DIM, _HEAD_DIM, _DIM)
        wq = wqkv[:, :_HPK].reshape(-1, _DIM)
        wk = wqkv[:, _HPK].reshape(-1, _DIM)
        wv = wqkv[:, _HPK + 1].reshape(-1, _DIM)
        ref_q_THK = (x_TD @ wq.T).view(num_tokens, _N_HEADS, _HEAD_DIM)
        ref_k_THK = (x_TD @ wk.T).view(num_tokens, _N_KV_HEADS, _HEAD_DIM)
        ref_v_THV = (x_TD @ wv.T).view(num_tokens, _N_KV_HEADS, _HEAD_DIM)
        torch.testing.assert_close(xq_THK, ref_q_THK)
        torch.testing.assert_close(xk_THK, ref_k_THK)
        torch.testing.assert_close(xv_THV, ref_v_THV)

    def test_raw_pointer_read_needs_contiguous(self):
        """A consumer reading the base pointer with contiguous head-major strides
        (what vLLM's kernels do) gets the wrong bytes from the strided split, and
        the correct bytes only after the forward's ``.contiguous()``.
        """
        fused = _build_qkv_linear()
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
