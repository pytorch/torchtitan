# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from dataclasses import replace
from typing import cast

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.optim import AdamW, DistMuon
from torchtitan.distributed.flex_shard.dist_muon import _adjust_muon_learning_rate
from torchtitan.models.kimi_k3.flavors import _vision_encoder_config
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan_recipes.tests.models.kimi_k3 import (
    _dist_muon_optimizer,
    kimi_k3_debugmodel,
)


pytestmark = pytest.mark.multi_gpu


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestKimiK3VisionDistMuon(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cuda"

    @with_comms
    def test_fsdp_matches_independent_matrix_updates_and_resume(self) -> None:
        torch.manual_seed(42)
        device = torch.device(self.device_type, self.rank)
        recipe = kimi_k3_debugmodel(seq_len=16)
        model_config = cast(KimiK3Model.Config, recipe.model)
        # Three heads force the two-rank FSDP storage boundary to split a head.
        vision_config = _vision_encoder_config(
            text_dim=model_config.dim,
            dim=32,
            qkv_dim=48,
            hidden_dim=64,
            num_layers=1,
            num_heads=3,
            patch_size=2,
            init_pos_emb_height=2,
            init_pos_emb_width=2,
            max_num_frames=1,
        )
        model_config = replace(model_config, vision_encoder=vision_config)
        optimizer_config = _dist_muon_optimizer(
            model_config, muon_lr=1e-3, adamw_lr=1e-3, parallelism=recipe.parallelism
        )
        muon_config, adamw_config = optimizer_config.optimizers
        self.assertIsInstance(muon_config, DistMuon.Config)
        self.assertIsInstance(adamw_config, AdamW.Config)

        with torch.device(device):
            model = nn.Module()
            model.vision_encoder = vision_config.build()
            model.vision_encoder.init_states(buffer_device=device)
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp_shard",)
        )
        fully_shard(
            model.vision_encoder,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
        optimizer = optimizer_config.build(model_parts=[model])
        muon, adamw = optimizer.optimizers
        parameters = dict(model.named_parameters())
        muon_names = muon.param_groups[0]["param_names"]
        adamw_names = adamw.param_groups[0]["param_names"]
        head_dim = vision_config.block.attn.dim // vision_config.block.attn.num_heads
        reference_parameters = {}
        for fqn, parameter in parameters.items():
            value = parameter.full_tensor().detach()
            blocks = (
                value.split(head_dim, dim=0)
                if fqn.endswith(
                    (".attn.wq.weight", ".attn.wk.weight", ".attn.wv.weight")
                )
                else (value,)
            )
            reference_parameters[fqn] = tuple(
                nn.Parameter(block.clone()) for block in blocks
            )

        reference_muon = torch.optim.Muon(
            [block for fqn in muon_names for block in reference_parameters[fqn]],
            **{
                name: getattr(muon_config, name)
                for name in (
                    "lr",
                    "weight_decay",
                    "momentum",
                    "nesterov",
                    "ns_coefficients",
                    "eps",
                    "ns_steps",
                    "adjust_lr_fn",
                )
            },
        )
        reference_adamw = adamw_config.build(
            params=[reference_parameters[fqn][0] for fqn in adamw_names]
        )
        generator = torch.Generator(device=device).manual_seed(42 + self.rank)
        decay = 1 - muon_config.lr * muon_config.weight_decay

        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            # Text-only batches run dummy vision and contribute zero gradients.
            has_image = step == 0 or (step == 1 and self.rank == 1)
            grid_size = 4 if has_image else 2
            grid_thw_N3 = torch.tensor([[1, grid_size, grid_size]], device=device)
            pixel_values_TP = (
                torch.randn(16, 12, device=device, generator=generator)
                if has_image
                else torch.zeros(4, 12, device=device)
            )
            target_TO = torch.randn(
                grid_size**2 // 4,
                model_config.dim,
                device=device,
                generator=generator,
            )
            output_TO = model.vision_encoder(pixel_values_TP, grid_thw=grid_thw_N3)
            loss = (
                (output_TO.float() - target_TO).square().mean()
                if has_image
                else (output_TO.float() * 0.0).sum()
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()

            actual_before = {}
            reference_before = {}
            for fqn, parameter in parameters.items():
                self.assertIsNotNone(parameter.grad, fqn)
                gradient = parameter.grad.full_tensor().detach()
                blocks = reference_parameters[fqn]
                gradients = (
                    gradient.split(head_dim, dim=0) if len(blocks) > 1 else (gradient,)
                )
                for block, grad in zip(blocks, gradients, strict=True):
                    block.grad = grad.clone()
                if fqn in muon_names:
                    actual_before[fqn] = parameter.full_tensor().detach().clone()
                    reference_before[fqn] = (
                        torch.cat([block.detach() for block in blocks], dim=0)
                        if len(blocks) > 1
                        else blocks[0].detach().clone()
                    )

            optimizer.step()
            reference_muon.step()
            reference_adamw.step()
            for fqn, parameter in parameters.items():
                blocks = reference_parameters[fqn]
                expected = (
                    torch.cat([block.detach() for block in blocks], dim=0)
                    if len(blocks) > 1
                    else blocks[0].detach()
                )
                actual = parameter.full_tensor().detach()
                if fqn in adamw_names:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    continue
                adjusted_lr = _adjust_muon_learning_rate(
                    muon_config.lr, muon_config.adjust_lr_fn, blocks[0].shape
                )
                actual_update = (actual_before[fqn] * decay - actual) / adjusted_lr
                expected_update = (
                    reference_before[fqn] * decay - expected
                ) / adjusted_lr
                # Batched BF16 Newton-Schulz can differ slightly across GEMM schedules.
                torch.testing.assert_close(
                    actual_update, expected_update, rtol=0, atol=2e-2, msg=fqn
                )

            if step == 0:
                state = copy.deepcopy(optimizer.state_dict())
                optimizer = optimizer_config.build(model_parts=[model])
                optimizer.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
