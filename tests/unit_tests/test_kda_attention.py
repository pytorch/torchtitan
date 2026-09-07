# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the KDA linear-attention layer."""

import importlib.util
import unittest
from typing import cast
from unittest.mock import patch, PropertyMock

import torch
from attn_gym.linear.context_parallel import ContextParallelRouting

from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import create_varlen_metadata_for_document
from torchtitan.models.kimi_k3.cp_kda import (
    ContextParallelInnerKDA,
    partition_fragments,
)
from torchtitan.models.kimi_k3.kda import InnerKDA, KDA, KDAKernel, KimiRMSNormGated

_HAS_KDA_GPU = (
    importlib.util.find_spec("attn_gym") is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() in {(9, 0), (10, 0), (10, 3)}
)


class _TestRouting:
    __slots__ = ("cu_seqlens", "tail_sources")

    def __init__(
        self,
        cu_seqlens: torch.Tensor,
        tail_sources: torch.Tensor,
    ) -> None:
        self.cu_seqlens = cu_seqlens
        self.tail_sources = tail_sources


def _kda_config() -> KDA.Config:
    def linear(in_features: int, out_features: int) -> Linear.Config:
        return Linear.Config(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )

    projection_dim = 256

    def conv() -> Conv1d.Config:
        return Conv1d.Config(
            in_channels=projection_dim,
            out_channels=projection_dim,
            kernel_size=4,
            groups=projection_dim,
            bias=False,
        )

    return KDA.Config(
        num_heads=2,
        head_dim=128,
        conv_kernel_size=4,
        q_proj=linear(32, projection_dim),
        k_proj=linear(32, projection_dim),
        v_proj=linear(32, projection_dim),
        q_conv=conv(),
        k_conv=conv(),
        v_conv=conv(),
        forget_a=linear(32, 128),
        forget_b=linear(128, projection_dim),
        beta=linear(32, 2),
        output_gate=linear(32, projection_dim),
        inner_kda=InnerKDA.Config(
            head_dim=128,
            kernel=KDAKernel.Config(),
        ),
        output_norm=KimiRMSNormGated.Config(dim=128),
        output_proj=linear(projection_dim, 32),
    )


class TestKDAContextParallelPartition(unittest.TestCase):
    def test_contiguous_fragments(self):
        self.assertEqual(
            partition_fragments(16, 2, None),
            [[(0, 8)], [(8, 16)]],
        )

    def test_headtail_fragments(self):
        self.assertEqual(
            partition_fragments(16, 2, "headtail"),
            [[(0, 4), (12, 16)], [(4, 8), (8, 12)]],
        )

    def test_rejects_an_uneven_partition(self):
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            partition_fragments(15, 2, "headtail")

    def test_rejects_an_unsupported_partition(self):
        with self.assertRaisesRegex(ValueError, "contiguous or headtail"):
            partition_fragments(16, 2, "ptrr")

    def test_inner_kda_uses_cp_convolution_history_and_offsets(self):
        num_tokens = 4
        head_dim = 128
        num_heads = 2
        channels = num_heads * head_dim
        local_offsets = torch.tensor([0, 2, 4], dtype=torch.int32)
        routing = cast(
            ContextParallelRouting,
            _TestRouting(
                local_offsets,
                torch.zeros(1, 2, dtype=torch.int64),
            ),
        )
        initial_state = torch.randn(2, 2, 3 * channels)
        q_1THK = torch.randn(1, num_tokens, num_heads, head_dim)
        k_1THK = torch.randn_like(q_1THK)
        gate_1THK = torch.randn_like(q_1THK)
        beta_1TH = torch.randn(1, num_tokens, num_heads)
        output_1THV = torch.randn(1, num_tokens, num_heads, head_dim)
        cp_group = object()
        inner_kda = ContextParallelInnerKDA.Config(
            head_dim=head_dim,
            kernel=KDAKernel.Config(),
        ).build()

        def fake_conv(x_1TC, _weight_CW, **kwargs):
            self.assertIs(kwargs["cu_seqlens"], local_offsets)
            self.assertIs(kwargs["initial_state"], initial_state)
            return x_1TC

        with (
            patch(
                "torchtitan.models.kimi_k3.cp_kda.context_parallel_conv_history",
                return_value=initial_state,
            ) as conv_history,
            patch("torchtitan.models.kimi_k3.kda.causal_conv1d", fake_conv),
            patch.object(
                inner_kda.kernel,
                "prepare_inputs",
                return_value=(q_1THK, k_1THK, gate_1THK, beta_1TH),
            ),
            patch(
                "torchtitan.models.kimi_k3.cp_kda.context_parallel_kda",
                return_value=(output_1THV, None),
            ) as cp_kda,
            patch.object(
                ContextParallelInnerKDA,
                "cp_group",
                new_callable=PropertyMock,
                return_value=cp_group,
            ),
        ):
            output_THV = inner_kda(
                torch.randn(num_tokens, channels),
                torch.randn(num_tokens, channels),
                torch.randn(num_tokens, channels),
                torch.randn(num_tokens, num_heads, head_dim),
                torch.randn(num_tokens, num_heads),
                torch.randn(channels, 1, 3),
                torch.randn(channels, 1, 3),
                torch.randn(channels, 1, 3),
                torch.randn(num_heads),
                torch.randn(num_heads, head_dim),
                cu_seqlens=None,
                routing=routing,
            )

        self.assertEqual(output_THV.shape, (num_tokens, num_heads, head_dim))
        conv_history.assert_called_once()
        self.assertIs(cp_kda.call_args.kwargs["routing"], routing)
        self.assertIs(cp_kda.call_args.kwargs["group"], cp_group)


@unittest.skipUnless(_HAS_KDA_GPU, "KDA requires Attention Gym on Hopper or Blackwell")
class TestKDA(unittest.TestCase):
    def _make_kda(self):
        model = _kda_config().build()
        model = model.to(device="cuda", dtype=torch.bfloat16)
        torch.manual_seed(1)
        with torch.no_grad():
            for param in model.parameters():
                param.normal_(mean=0.0, std=0.02)
            model.A_log.uniform_(1.0, 16.0).log_()
            model.dt_bias.zero_()
            model.output_norm.weight.fill_(1.0)
        return model

    def _inputs(self, seed: int, tokens: int = 128) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(tokens, 32, device="cuda", dtype=torch.bfloat16)

    def test_varlen_matches_independent_documents(self):
        lengths = (37, 64, 91)
        x_TD = self._inputs(seed=2, tokens=sum(lengths)).requires_grad_()
        positions_T = torch.tensor(
            [index for length in lengths for index in range(length)],
            device="cuda",
            dtype=torch.int32,
        )
        masks = create_varlen_metadata_for_document(
            positions_T,
            include_host_offsets=True,
        )
        self.assertEqual(masks.cu_seq_q_host, (0, 37, 101, 192))

        model = self._make_kda()
        packed_TD = model(x_TD, masks)
        independent_TD = torch.cat(
            [model(document_TD, None) for document_TD in x_TD.split(lengths)]
        )
        torch.testing.assert_close(
            packed_TD.float(),
            independent_TD.float(),
            rtol=2e-2,
            atol=2e-2,
        )
        output_grad_TD = torch.randn_like(packed_TD)
        parameters = tuple(model.parameters())
        packed_grads = torch.autograd.grad(
            packed_TD,
            (x_TD, *parameters),
            output_grad_TD,
        )
        independent_grads = torch.autograd.grad(
            independent_TD,
            (x_TD, *parameters),
            output_grad_TD,
        )
        torch.testing.assert_close(
            packed_grads,
            independent_grads,
            rtol=2e-2,
            atol=2e-2,
        )


if __name__ == "__main__":
    unittest.main()
