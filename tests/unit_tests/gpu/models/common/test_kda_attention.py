# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the KDA linear-attention layer."""

import importlib.util
import unittest
from copy import deepcopy

import torch

from torchtitan.distributed.accelerator_graph import (
    accelerator_graph_teardown,
    wrap_fwd_bwd_with_accelerator_graph,
)

from torchtitan.models.common import Conv1d, GatedRMSNorm, Linear, Sigmoid
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    LinearAttentionMetadata,
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
        n_heads=2,
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
        inner_attention=InnerKDA.Config(
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


@unittest.skipUnless(
    _HAS_ATTENTION_GYM_KDA, "KDA requires Attention Gym on CUDA capability 9.0+"
)
class TestKDA(unittest.TestCase):
    def _make_kda(self, seed: int = 1):
        model = _kda_config().build()
        model = model.to(device="cuda", dtype=torch.bfloat16)
        torch.manual_seed(seed)
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
        attention_metadata = LinearAttentionMetadata(
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

    def test_graph_replay_refreshes_packed_documents(self):
        """Changing padded offsets must preserve exact eager loss and updates."""
        deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        accelerator_graph_teardown()
        try:
            model = self._make_kda(seed=42)
            reference = deepcopy(model)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)

            def forward_backward(inputs, metadata):
                loss = model(inputs, metadata).float().square().sum()
                loss.backward()
                return loss.detach()

            run = wrap_fwd_bwd_with_accelerator_graph(
                forward_backward,
                parameters=model.parameters(),
                num_warmup_iterations=2,
            )
            gradient_addresses = None
            for step in range(5):
                inputs = self._inputs(seed=42 + step, tokens=96)
                lengths = (96,) if step % 2 == 0 else (17, 31, 48)
                positions = torch.tensor(
                    [index for length in lengths for index in range(length)],
                    device="cuda",
                    dtype=torch.int32,
                )
                metadata = LinearAttentionMetadata(
                    varlen=create_varlen_metadata_for_document(
                        positions, max_num_documents=4, max_context_length=96
                    ),
                    num_conv_history_tokens=3,
                )
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                expected = reference(inputs, metadata).float().square().sum()
                expected.backward()
                actual = run(inputs, metadata).clone()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for parameter, expected_parameter in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        parameter.grad, expected_parameter.grad, rtol=0, atol=0
                    )
                if step >= 2:
                    addresses = [
                        parameter.grad.data_ptr() for parameter in model.parameters()
                    ]
                    gradient_addresses = gradient_addresses or addresses
                    self.assertEqual(addresses, gradient_addresses)
                optimizer.step()
                reference_optimizer.step()
                for parameter, expected_parameter in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        parameter, expected_parameter, rtol=0, atol=0
                    )
        finally:
            accelerator_graph_teardown()
            torch.use_deterministic_algorithms(deterministic)


if __name__ == "__main__":
    unittest.main()
