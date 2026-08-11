# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the KDA linear-attention layer.

Every test needs a Blackwell (SM 10.0+) GPU and attention-gym: the layer's L2 norm
and gate are Triton kernels and the chunked core is CuTe, so there is no CPU path.
Kernel numerics are attention-gym's own test suite; these tests cover the layer's
composition -- that the pieces are wired together correctly.

The layer is built with 2 heads of head_dim 128 (the chunked core is specialized
to 128) over a model dim of 32, so the QKV projection is 3 * 2 * 128 = 768 wide.

Tensor shape suffixes: B batch, L seq len, D model dim, N heads, K head dim.
"""

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
        """The two algorithms must produce the same result on the same input.

        Parallel-within-chunks against token-at-a-time is the strongest numerical
        check available at this level: two different decompositions, including the
        two different gate conventions ``KDAAttention._gate`` produces for them.
        """
        chunked = self._make_kda(backend="chunked")
        recurrent = self._make_kda(backend="recurrent")
        recurrent.load_state_dict(chunked.state_dict())

        x_BLD = self._inputs(seed=0)
        torch.testing.assert_close(
            chunked(x_BLD).float(), recurrent(x_BLD).float(), rtol=5e-2, atol=5e-2
        )

    def test_backward_reaches_every_parameter(self):
        """Every parameter must get a finite, nonzero gradient.

        ``A_log`` and the ``[num_heads, head_dim]`` ``dt_bias`` reach the kernels
        through the gate map rather than a matmul, so they are the ones most
        likely to end up silently detached.
        """
        model = self._make_kda()
        model(self._inputs(seed=1)).float().square().mean().backward()
        for name, param in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(param.grad, f"{name} received no gradient")
                self.assertTrue(torch.isfinite(param.grad).all())
                self.assertGreater(param.grad.abs().sum().item(), 0.0)

    def test_varlen_matches_independent_document_forwards(self):
        """Every packed document must match forwarding that document alone.

        Both stages that carry information across tokens reset at the boundaries:
        the recurrence, and the short convolution -- the latter because
        ``cute_causal_conv1d_silu`` takes ``cu_seqlens``. So a packed document
        cannot see any token of its predecessors.

        Both backends are checked: they take different varlen paths, and the
        recurrence is the stage most likely to leak state across a boundary.

        The lengths are deliberately ragged. A document ending mid-chunk is the
        case the chunked core's partial-chunk predication exists for, so it is
        where a boundary bug would show up; 64 alone would never exercise it.
        """
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
