# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.glm5 import build_model_config
from torchtitan.tools.utils import set_default_dtype


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                    reason="requires NVIDIA fused gather attention")
@pytest.mark.parametrize("stage", ["warmup", "sparse"])
def test_bf16_indexer_training_with_default_remat(stage):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("stock fused indexer requires SM90 or newer")
    config = build_model_config(
        "glm5_2_debugmodel", parallelism=ParallelismConfig(), seq_len=128,
        indexer_stage=stage,
    )
    config.layers = config.layers[:4]
    config.index_sources = (0, 1, 2, 2)
    torch.manual_seed(61)
    with patch.dict(AuxLoss.group_acc, clear=True), patch.dict(AuxLoss._group_counts, clear=True):
        # Match low-precision parameter construction, preserving explicit FP32
        # indexer mixing weights instead of casting the entire model afterward.
        with set_default_dtype(torch.bfloat16):
            model = config.build().to(device="cuda")
        model.init_states()
        DefaultActivationRematConfig().apply(model)
        inputs = torch.arange(128, device="cuda")
        positions = inputs % 64
        padding = torch.zeros_like(inputs, dtype=torch.bool)
        padding[[63, 127]] = True
        logits = model(
            inputs, positions=positions, padding_mask=padding,
            aux_loss_denominators=torch.tensor([126.0], device="cuda"),
        )
        loss = logits if stage == "warmup" else logits.float().square().mean()
        loss.backward()
        gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        assert torch.isfinite(loss)
        assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
        assert any(".indexer." in name and g.abs().sum() > 0 for name, g in gradients.items())
        assert any(".indexer." not in name for name in gradients) == (stage == "sparse")
        print(f"GLM BF16 {stage}: fused gather/LSE/KL, shared-index consumer and default remat backward completed")
