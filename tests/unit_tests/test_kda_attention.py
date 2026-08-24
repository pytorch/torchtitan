# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the KDA linear-attention layer."""

import importlib.util
import unittest

import spmd_types as spmd
import torch

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common import Conv1d, Linear, RMSNorm
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    KDA,
    KDAAttention,
    KDABackend,
    KDAInnerAttention,
)
from torchtitan.models.common.decoder_sharding import (
    dense_sequence_parallel_placement,
    set_kda_sharding,
)


_HAS_BLACKWELL = (
    importlib.util.find_spec("attn_gym") is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (10, 0)
)
_HAS_FLA = importlib.util.find_spec("fla") is not None


def _kda_config(*, backend: KDABackend) -> KDA.Config:
    def linear(in_features: int, out_features: int) -> Linear.Config:
        return Linear.Config(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )

    projection_dim = 2 * 128

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
        attention=KDAAttention.Config(
            head_dim=128,
            inner_attention=KDAInnerAttention.Config(backend=backend),
        ),
        output_norm=RMSNorm.Config(normalized_shape=128),
        output_proj=linear(projection_dim, 32),
    )


class TestKDASharding(unittest.TestCase):
    def test_folded_token_tp_contracts(self):
        config = _kda_config(backend="reference")
        input_layout = dense_sequence_parallel_placement()
        set_kda_sharding(
            config,
            attention_input_layout=input_layout,
            enable_sp=True,
            cp_enabled=False,
        )

        tp_axis = MeshAxisName.TP
        q_proj_sharding = config.q_proj.sharding_config
        output_proj_sharding = config.output_proj.sharding_config
        attention_sharding = config.attention.sharding_config
        kda_sharding = config.sharding_config
        assert q_proj_sharding is not None
        assert output_proj_sharding is not None
        assert attention_sharding is not None
        assert attention_sharding.in_src_shardings is not None
        assert attention_sharding.local_map is not None
        assert attention_sharding.local_map.in_grad_placements is not None
        assert kda_sharding is not None
        self.assertEqual(
            q_proj_sharding.state_shardings["weight"].axis_types[tp_axis],
            spmd.S(0),
        )
        self.assertEqual(
            output_proj_sharding.state_shardings["weight"].axis_types[tp_axis],
            spmd.S(1),
        )
        self.assertEqual(
            set(attention_sharding.in_src_shardings),
            {
                "query_TC",
                "key_TC",
                "value_TC",
                "raw_gate_TNK",
                "raw_beta_TN",
                "conv_q_weight_C1W",
                "conv_k_weight_C1W",
                "conv_v_weight_C1W",
                "A_log_N",
                "dt_bias_NK",
                "cu_seqlens",
            },
        )
        head_layout = attention_sharding.in_src_shardings["raw_gate_TNK"]
        self.assertEqual(head_layout.per_axis_spmd_types()[tp_axis], spmd.S(1))
        self.assertEqual(
            kda_sharding.in_src_shardings,
            {"x_TD": input_layout},
        )
        self.assertEqual(
            len(attention_sharding.local_map.in_grad_placements),
            11,
        )

    def test_context_parallel_is_rejected(self):
        config = _kda_config(backend="reference")
        with self.assertRaisesRegex(NotImplementedError, "Context parallel"):
            set_kda_sharding(
                config,
                attention_input_layout=dense_sequence_parallel_placement(),
                enable_sp=True,
                cp_enabled=True,
            )


@unittest.skipUnless(
    _HAS_BLACKWELL, "KDA requires Attention Gym and CUDA capability 10.0 or newer"
)
class TestKDA(unittest.TestCase):
    def _make_kda(self, *, backend: KDABackend):
        model = _kda_config(backend=backend).build()
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

    def test_fused_and_reference_backends_agree(self):
        fused = self._make_kda(backend="fused")
        reference = self._make_kda(backend="reference")
        reference.load_state_dict(fused.state_dict())

        x_fused_TD = self._inputs(seed=0).requires_grad_()
        x_reference_TD = x_fused_TD.detach().clone().requires_grad_()
        actual_TD = fused(x_fused_TD)
        expected_TD = reference(x_reference_TD)
        torch.testing.assert_close(
            actual_TD.float(), expected_TD.float(), rtol=2e-2, atol=2e-2
        )

        output_grad_TD = torch.randn_like(actual_TD)
        actual_grads = torch.autograd.grad(
            actual_TD,
            (x_fused_TD, *fused.parameters()),
            grad_outputs=output_grad_TD,
        )
        expected_grads = torch.autograd.grad(
            expected_TD,
            (x_reference_TD, *reference.parameters()),
            grad_outputs=output_grad_TD,
        )
        for actual_grad, expected_grad in zip(
            actual_grads,
            expected_grads,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_grad.float(),
                expected_grad.float(),
                rtol=5e-2,
                atol=5e-2,
            )

    @unittest.skipUnless(_HAS_FLA, "FLA is required for fallback parity")
    def test_fused_and_fla_backends_agree(self):
        fused = self._make_kda(backend="fused")
        fla = self._make_kda(backend="fla")
        fla.load_state_dict(fused.state_dict())

        x_fused_TD = self._inputs(seed=4).requires_grad_()
        x_fla_TD = x_fused_TD.detach().clone().requires_grad_()
        actual_TD = fused(x_fused_TD)
        expected_TD = fla(x_fla_TD)
        torch.testing.assert_close(
            actual_TD.float(), expected_TD.float(), rtol=2e-2, atol=2e-2
        )

        output_grad_TD = torch.randn_like(actual_TD)
        actual_grads = torch.autograd.grad(
            actual_TD,
            (x_fused_TD, *fused.parameters()),
            grad_outputs=output_grad_TD,
        )
        expected_grads = torch.autograd.grad(
            expected_TD,
            (x_fla_TD, *fla.parameters()),
            grad_outputs=output_grad_TD,
        )
        for actual_grad, expected_grad in zip(
            actual_grads,
            expected_grads,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_grad.float(),
                expected_grad.float(),
                rtol=5e-2,
                atol=5e-2,
            )

    def test_rejects_unfolded_batch_dimension(self):
        model = self._make_kda(backend="reference")
        x_TD = self._inputs(seed=3)
        with self.assertRaisesRegex(ValueError, r"\[T, D\]"):
            model(x_TD.unsqueeze(0))

    def test_varlen_matches_independent_document_forwards(self):
        lengths = (37, 64, 91)
        x_TD = self._inputs(seed=2, tokens=sum(lengths))
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

        backends: list[KDABackend] = ["fused", "reference"]
        if _HAS_FLA:
            backends.append("fla")
        for backend in backends:
            model = self._make_kda(backend=backend)
            packed_TD = model(x_TD, masks)
            start = 0
            for document, length in enumerate(lengths):
                with self.subTest(backend=backend, document=document):
                    document_slice = slice(start, start + length)
                    torch.testing.assert_close(
                        packed_TD[document_slice].float(),
                        model(x_TD[document_slice], None).float(),
                        rtol=2e-2,
                        atol=2e-2,
                    )
                start += length


if __name__ == "__main__":
    unittest.main()
