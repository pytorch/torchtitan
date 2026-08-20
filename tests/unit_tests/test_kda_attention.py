# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the KDA linear-attention layer."""

import importlib.util
import unittest

import torch

from torchtitan.models.common.attention import create_varlen_metadata_for_document

_HAS_BLACKWELL = (
    importlib.util.find_spec("attn_gym") is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (10, 0)
)


@unittest.skipUnless(
    _HAS_BLACKWELL, "KDA requires attention-gym and CUDA capability 10.0 or newer"
)
class TestKDA(unittest.TestCase):
    def _make_kda(self, *, backend: str = "chunked"):
        """Build a KDA layer with deterministic weights."""
        from torchtitan.models.common import Conv1d, Linear, RMSNorm
        from torchtitan.models.common.attention import (
            KDA,
            KDAAttention,
            KDAInnerAttention,
        )

        def linear(in_features: int, out_features: int) -> Linear.Config:
            return Linear.Config(
                in_features=in_features, out_features=out_features, bias=False
            )

        model = KDA.Config(
            num_heads=2,
            head_dim=128,
            in_proj_qkv=linear(32, 768),
            conv_qkv=Conv1d.Config(
                in_channels=768,
                out_channels=768,
                kernel_size=4,
                groups=768,
                bias=False,
            ),
            gate_proj_a=linear(32, 128),
            gate_proj_b=linear(128, 256),
            beta_proj=linear(32, 2),
            out_gate_proj_a=linear(32, 128),
            out_gate_proj_b=linear(128, 256),
            out_norm=RMSNorm.Config(normalized_shape=128),
            out_proj=linear(256, 32),
            attention=KDAAttention.Config(
                head_dim=128,
                inner_attention=KDAInnerAttention.Config(backend=backend),
            ),
        ).build()
        model = model.to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for param in model.parameters():
                values = torch.linspace(
                    -0.2, 0.2, param.numel(), dtype=param.dtype, device=param.device
                )
                param.copy_(values.reshape_as(param))
            model.A_log.fill_(0.0)
            model.dt_bias.zero_()
            model.out_norm.weight.fill_(1.0)
        return model

    def _inputs(self, seed: int, tokens: int = 128) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(1, tokens, 32, device="cuda", dtype=torch.bfloat16)

    def test_chunked_and_recurrent_backends_agree(self):
        chunked = self._make_kda(backend="chunked")
        recurrent = self._make_kda(backend="recurrent")
        recurrent.load_state_dict(chunked.state_dict())

        x_BLD = self._inputs(seed=0)
        torch.testing.assert_close(
            chunked(x_BLD).float(), recurrent(x_BLD).float(), rtol=5e-2, atol=5e-2
        )

    def test_varlen_matches_independent_document_forwards(self):
        lengths = (37, 64, 91)
        x_BLD = self._inputs(seed=2, tokens=sum(lengths))
        positions = torch.tensor(
            [[index for length in lengths for index in range(length)]],
            device="cuda",
            dtype=torch.int32,
        )
        masks = create_varlen_metadata_for_document(
            positions, include_host_offsets=True
        )

        for backend in ("chunked", "recurrent"):
            model = self._make_kda(backend=backend)
            packed = model(x_BLD, masks)
            start = 0
            for document, length in enumerate(lengths):
                with self.subTest(backend=backend, document=document):
                    document_slice = slice(start, start + length)
                    torch.testing.assert_close(
                        packed[:, document_slice].float(),
                        model(x_BLD[:, document_slice], None).float(),
                        rtol=2e-2,
                        atol=2e-2,
                    )
                start += length


if __name__ == "__main__":
    unittest.main()
