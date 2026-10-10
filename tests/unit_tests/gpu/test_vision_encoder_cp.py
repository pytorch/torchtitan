# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import pytest
import spmd_types as spmd
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.fsdp import (
    apply_fsdp_to_multimodal_encoder,
    disable_fsdp_gradient_division,
    resolve_fsdp_mesh,
)
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.distributed.utils import clip_grad_norm_
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.multimodal import (
    gather_vision_embeds,
    replicate_cp_vision_output,
)
from torchtitan.models.common.vision_encoder_sharding import (
    set_vision_encoder_cp_invariant,
    vision_invariant_linear_config,
)
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig


pytestmark = [
    pytest.mark.multi_gpu,
    pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four GPUs"),
]


class _Encoder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        first: Linear.Config
        second: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.first = config.first.build()
        self.second = config.second.build()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.second(torch.tanh(self.first(x_TD)))


def _config(invariant: bool) -> _Encoder.Config:
    layout = spmd.SpmdType({"dp": spmd.V, "cp": spmd.R, "tp": spmd.I})
    config = _Encoder.Config(
        first=Linear.Config(
            in_features=4,
            out_features=8,
            sharding_config=vision_invariant_linear_config(),
        ),
        second=Linear.Config(
            in_features=8,
            out_features=4,
            sharding_config=vision_invariant_linear_config(),
        ),
        sharding_config=ShardingConfig(
            out_src_shardings=layout, out_dst_shardings=layout
        ),
    )
    if invariant:
        set_vision_encoder_cp_invariant(config)
    return config


class TestVisionEncoderCP(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_dp_cp_gradients_match_unsharded_reference(self) -> None:
        context = ParallelismContext(
            dp_replicate=1,
            dp_shard=2,
            cp=2,
            tp=1,
            pp=1,
            ep=1,
            world_size=4,
            enable_sequence_parallel=False,
        )
        context.build_mesh()
        dp_rank, cp_rank = self.rank // 2, self.rank % 2

        def build(invariant: bool):
            torch.manual_seed(42)
            return (
                _config(invariant)
                .build()
                .to(device=self.device_type, dtype=torch.float64)
            )

        def inputs(dp: int, cp: int, text_only: bool):
            pixels = (
                torch.arange(16, device=self.device_type, dtype=torch.float64).reshape(
                    4, 4
                )
                / 16
                + dp
            )
            indices = torch.tensor(
                [-1, 0, 2, 2] if cp == 0 else [-1, 1, 2, 3],
                device=self.device_type,
            )
            if text_only and cp == 1:
                indices.fill_(-1)
            return pixels, indices

        def loss(model, dp: int, cp: int, text_only: bool, invariant: bool = False):
            pixels, indices = inputs(dp, cp, text_only)
            bank = model(pixels)
            if invariant:
                bank = replicate_cp_vision_output(bank)
            text = torch.zeros(4, 4, device=self.device_type, dtype=torch.float64)
            fused = gather_vision_embeds(
                text, vision_bank_VD=bank, vision_bank_indices_T=indices
            )
            return fused.square().sum() / 64

        for text_only in (False, True):
            reference = build(False)
            expected_loss = sum(
                loss(reference, dp, cp, text_only) for dp in range(2) for cp in range(2)
            )
            expected_loss.backward()
            for invariant in (False, True):
                model = build(invariant)
                with context.activate_spmd():
                    annotate_replicated_parameters(model, context)
                    model._parallelize(context)
                    mesh, axes = resolve_fsdp_mesh(context, shard_cp=not invariant)
                    apply_fsdp_to_multimodal_encoder(
                        model,
                        mesh,
                        param_dtype=torch.float64,
                        reduce_dtype=torch.float64,
                        dp_mesh_dims=axes,
                    )
                    disable_fsdp_gradient_division(model)
                    actual_loss = loss(model, dp_rank, cp_rank, text_only, invariant)
                    actual_loss.backward()
                for actual, expected in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        actual.grad.full_tensor(), expected.grad, atol=1e-12, rtol=1e-12
                    )
                    assert actual.to_local().numel() == expected.numel() // (
                        2 if invariant else 4
                    )
                    if invariant:
                        cp_axis = actual.device_mesh.mesh_dim_names.index("cp")
                        assert actual.placements[cp_axis].is_replicate()

                # Encoder and decoder use different FSDP meshes in the new mode.
                # Their gradients must still participate in one global norm.
                dense = torch.nn.Linear(4, 4, bias=False).to(
                    device=self.device_type, dtype=torch.float64
                )
                with context.activate_spmd():
                    annotate_replicated_parameters(dense, context)
                    mesh, axes = resolve_fsdp_mesh(context)
                    apply_fsdp_to_multimodal_encoder(
                        dense,
                        mesh,
                        param_dtype=torch.float64,
                        reduce_dtype=torch.float64,
                        dp_mesh_dims=axes,
                    )
                    disable_fsdp_gradient_division(dense)
                    dense(
                        torch.ones(1, 4, device=self.device_type, dtype=torch.float64)
                    ).sum().backward()
                norm = clip_grad_norm_(
                    [*model.parameters(), *dense.parameters()],
                    max_norm=1000,
                    foreach=True,
                )
                expected_norm = (
                    sum(p.grad.square().sum() for p in reference.parameters()) + 256
                ).sqrt()
                torch.testing.assert_close(norm, expected_norm, atol=1e-12, rtol=1e-12)
                for norm_type in (1.0, float("inf")):
                    norm = clip_grad_norm_(
                        [*model.parameters(), *dense.parameters()],
                        max_norm=float("inf"),
                        norm_type=norm_type,
                        foreach=True,
                    )
                    expected = torch.nn.utils.get_total_norm(
                        [
                            *[p.grad for p in reference.parameters()],
                            torch.full(
                                (4, 4),
                                4.0,
                                device=self.device_type,
                                dtype=torch.float64,
                            ),
                        ],
                        norm_type=norm_type,
                    )
                    torch.testing.assert_close(norm, expected, atol=1e-12, rtol=1e-12)
                clipped_norm = clip_grad_norm_(
                    [*model.parameters(), *dense.parameters()],
                    max_norm=1.0,
                    foreach=True,
                    ep_enabled=True,
                )
                torch.testing.assert_close(
                    clipped_norm, expected_norm, atol=1e-12, rtol=1e-12
                )
                scale = 1.0 / (expected_norm + 1e-6)
                for actual, expected in zip(
                    model.parameters(), reference.parameters(), strict=True
                ):
                    torch.testing.assert_close(
                        actual.grad.full_tensor(),
                        expected.grad * scale,
                        atol=1e-12,
                        rtol=1e-12,
                    )
