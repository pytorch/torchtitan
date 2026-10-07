# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the KDA linear-attention layer."""

import importlib.util
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch
from attn_gym.linear.context_parallel import ContextParallelRouting

from torchtitan.models.common import Conv1d, GatedRMSNorm, Linear, Sigmoid
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    KDAAttentionMetadata,
)
from torchtitan.models.common.attention.cp_kda import (
    ContextParallelInnerKDA,
    ContextParallelKDAAttentionMetadata,
)
from torchtitan.models.common.attention.kda import InnerKDA, KDA, KDAKernel

_HAS_ATTENTION_GYM_KDA = (
    importlib.util.find_spec("attn_gym") is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (9, 0)
)


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
            conv_kernel_size=4,
            kernel=KDAKernel.Config(),
        ),
        output_norm=GatedRMSNorm.Config(
            dim=128,
            eps=1e-5,
            activation_fn=Sigmoid.Config(),
        ),
        output_proj=linear(projection_dim, 32),
    )


class TestKDAContextParallelMetadata(unittest.TestCase):
    def test_fragments_follow_the_explicit_permutation(self):
        from torchtitan.distributed.context_parallel import get_token_fragments

        permutation = torch.tensor([[0, 1, 6, 7, 2, 3, 4, 5]])

        self.assertEqual(
            get_token_fragments(8, cp_size=2, permutation=permutation),
            [[(0, 2), (6, 8)], [(2, 6)]],
        )

    def test_backend_builds_routing_from_global_varlen_metadata(self):
        varlen = create_varlen_metadata_for_document(torch.tensor([0, 1, 0, 1]))
        attention_metadata = KDAAttentionMetadata(
            varlen=varlen,
            num_conv_history_tokens=3,
        )
        group = SimpleNamespace(size=lambda: 2)
        permutation = torch.tensor([[0, 3, 1, 2]])
        routing = cast(ContextParallelRouting, object())

        with patch(
            "torchtitan.models.common.attention.cp_kda.ContextParallelRouting.from_fragments",
            return_value=routing,
        ) as build_routing, patch(
            "torchtitan.models.common.attention.cp_kda.spmd_mesh_group",
            return_value=group,
        ), patch(
            "torchtitan.models.common.attention.cp_kda.dist.get_rank", return_value=0
        ):
            kda_metadata = ContextParallelInnerKDA.prepare_cp_metadata(
                attention_metadata,
                permutation=permutation,
            )

        self.assertIs(kda_metadata.cp_routing, routing)
        build_routing.assert_called_once_with(
            cu_seqlens_global=[0, 2, 4],
            fragments=[[(0, 1), (3, 4)], [(1, 3)]],
            cp_rank=0,
            device=varlen.cu_seq_q.device,
            conv_history=3,
        )

    def test_metadata_annotates_cp_routing_tensors(self):
        varlen = create_varlen_metadata_for_document(torch.tensor([0, 1, 0, 1]))
        routing = ContextParallelRouting.from_fragments(
            cu_seqlens_global=[0, 2, 4],
            fragments=[[(0, 2)], [(2, 4)]],
            cp_rank=0,
            device="cpu",
            conv_history=3,
        )
        metadata = ContextParallelKDAAttentionMetadata(
            varlen=varlen,
            num_conv_history_tokens=3,
            cp_routing=routing,
        )

        with patch(
            "torchtitan.models.common.attention.cp_kda.spmd.assert_type"
        ) as annotate:
            metadata.annotate_spmd_types()

        annotated_tensors = [call.args[0] for call in annotate.call_args_list]
        for value in vars(routing).values():
            if isinstance(value, torch.Tensor):
                self.assertTrue(
                    any(value is annotated for annotated in annotated_tensors)
                )


@unittest.skipUnless(
    _HAS_ATTENTION_GYM_KDA, "KDA requires Attention Gym on CUDA capability 9.0+"
)
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
        attention_metadata = KDAAttentionMetadata(
            varlen=create_varlen_metadata_for_document(positions_T),
            num_conv_history_tokens=3,
        )

        model = self._make_kda()
        packed_TD = model(x_TD, attention_metadata)
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
