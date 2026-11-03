# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""BF16 optimization and frozen-backbone contracts for vision and DSpark.

T: context tokens, B: draft blocks, S: draft positions, V: vocabulary,
D: hidden width, P: image patches, F: flattened patch features.
"""

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4_1 import build_model_config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


@pytest.fixture
def model():
    previous_dtype = torch.get_default_dtype()
    previous_device = torch.get_default_device()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    config = build_model_config(
        "debugmodel",
        parallelism=ParallelismConfig(data_parallel_shard_degree=1),
        seq_len=32,
        indexer_loss_coeff=0.0,
        vision_enabled=True,
        num_draft_layers=3,
    )
    config.local_compile.regions = []
    try:
        model = config.build()
        model.init_states(buffer_device=torch.device("cuda"))
        yield model
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_default_device(previous_device)
        torch.use_deterministic_algorithms(previous_deterministic)


def test_vision_optimization_reaches_encoder_and_markers(model):
    model.train()
    tokens_T = torch.arange(18)
    types_T = torch.tensor([-1, -1, 0, 1, 1, 2, 1, 1, 2, 3] + [-1] * 8)
    pixels_PF = torch.randn(36, 12, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("drafter.")
        ],
        lr=0.002,
    )
    losses = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        pixels_PF.grad = None
        logits_TV = model(
            tokens_T,
            token_types=types_T,
            pixel_values=pixels_PF,
            grid_hw=((6, 6),),
        )
        loss = F.cross_entropy(logits_TV, tokens_T + 1)
        loss.backward()
        assert pixels_PF.grad is not None and pixels_PF.grad.float().norm() > 0
        assert model.image_embeddings.weight.grad is not None
        assert model.vision.patch_embed.weight.grad.float().norm() > 0
        assert model.aligner.w1.weight.grad.float().norm() > 0
        assert torch.isfinite(loss)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_dspark_optimization_preserves_frozen_backbone(model):
    model.eval()
    tokens_T = torch.arange(16)
    anchors_B = torch.tensor([5, 10])
    with torch.no_grad():
        _, context_TD = model(tokens_T, return_draft_context=True)
        draft_ids_BS = torch.full((2, 5), model.drafter.config.noise_token_id)
        draft_ids_BS[:, 0] = tokens_T[anchors_B]
        embeddings_TD = model.tok_embeddings(draft_ids_BS.flatten())
        output_weight_VD = model.lm_head.weight.detach()
    previous_BS = torch.arange(10).view(2, 5)
    model.drafter.train()
    optimizer = torch.optim.AdamW(model.drafter.parameters(), lr=0.002)
    losses = []
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        logits_BSV, confidence_BS = model.forward_draft(
            context_TD,
            embeddings_TD,
            previous_BS,
            anchors_B,
            main_positions_T=torch.arange(16),
            output_weight_VD=output_weight_VD,
        )
        loss = F.cross_entropy(logits_BSV.flatten(0, 1), (previous_BS + 1).flatten())
        loss = loss + F.binary_cross_entropy_with_logits(
            confidence_BS, torch.full_like(confidence_BS, 0.5)
        )
        loss.backward()
        for parameter in (
            model.drafter.layers["0"].main_proj.weight,
            model.drafter.layers["2"].markov_head.embed.weight,
            model.drafter.layers["2"].confidence_head.proj.weight,
        ):
            assert parameter.grad is not None and parameter.grad.float().norm() > 0
        assert all(
            parameter.grad is None
            for name, parameter in model.named_parameters()
            if not name.startswith("drafter.")
        )
        assert torch.isfinite(loss)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
