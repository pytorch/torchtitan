# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.deepseek_v4 import build_model_config

# Documents longer than 128 tokens so the ratio-128 compressor emits entries.
DOC_LENGTHS = (300, 260, 72)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCompressorLocalCompile(unittest.TestCase):
    def tearDown(self):
        apply_local_compile([])
        torch._dynamo.reset()

    def _compressor(self, ratio):
        torch.manual_seed(0)
        for layer in build_model_config("debugmodel").layers:
            attention = layer.attention
            if attention.compress_ratio == ratio:
                config = (
                    attention.compressor if ratio == 4 else attention.compressor_128
                )
                with torch.device("cuda"):
                    compressor = config.build()
                    compressor.init_states(buffer_device=torch.device("cuda"))
                return compressor
        raise AssertionError(f"debug model has no compress_ratio={ratio} layer")

    def _run(self, compressor, x, positions, cu_seqlens):
        x = x.detach().clone().requires_grad_()
        out = compressor(x, positions, cu_seqlens)
        generator = torch.Generator(device="cuda").manual_seed(1)
        grad_out = torch.randn(
            out.shape, dtype=out.dtype, device="cuda", generator=generator
        )
        params = list(compressor.parameters())
        grads = torch.autograd.grad(out, [x, *params], grad_out)
        return (out, *grads)

    def _packed(self, doc_lengths):
        positions = torch.cat([torch.arange(n, device="cuda") for n in doc_lengths])
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(doc_lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        return positions, cu_seqlens

    def test_compiled_matches_eager(self):
        # Packed documents, and one unpacked sequence of 5 groups of 128 tokens.
        inputs = (self._packed(DOC_LENGTHS), (torch.arange(640, device="cuda"), None))
        for ratio in (4, 128):
            compressor = self._compressor(ratio)
            for positions, cu_seqlens in inputs:
                x = torch.randn(
                    len(positions), compressor.wkv.in_features, device="cuda"
                ).bfloat16()
                apply_local_compile([])
                eager = self._run(compressor, x, positions, cu_seqlens)
                apply_local_compile(["compressor"])
                compiled = self._run(compressor, x, positions, cu_seqlens)
                # Parameter grads sum over every group, and the compiled region
                # keeps fp32 where eager rounds to bf16, so compare whole tensors.
                for actual, expected in zip(compiled, eager, strict=True):
                    error = (actual.float() - expected.float()).norm()
                    self.assertLess(error / expected.float().norm(), 1e-2)

    def test_one_graph_per_compressor_and_grad_mode(self):
        apply_local_compile(["compressor"])
        compressors = [self._compressor(ratio) for ratio in (4, 128)]
        torch._dynamo.utils.counters.clear()
        for doc_lengths, grad in (
            (DOC_LENGTHS, True),
            ((520, 300, 204), True),
            (DOC_LENGTHS, False),
        ):
            positions, cu_seqlens = self._packed(doc_lengths)
            for compressor in compressors:
                x = torch.randn(
                    len(positions), compressor.wkv.in_features, device="cuda"
                ).bfloat16()
                with torch.set_grad_enabled(grad):
                    compressor(x, positions, cu_seqlens)
        self.assertEqual(torch._dynamo.utils.counters["stats"]["unique_graphs"], 4)


if __name__ == "__main__":
    unittest.main()
