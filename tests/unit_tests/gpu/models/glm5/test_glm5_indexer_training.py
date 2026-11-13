# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
import torch.distributed.checkpoint as dcp

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.components.loss import CrossEntropyLoss
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


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                    reason="requires NVIDIA fused gather attention")
def test_bf16_warmup_dcp_handoff_to_sparse_with_default_remat(tmp_path):
    """Model-only DCP handoff must not restore an indexer-only optimizer."""
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("stock fused indexer requires SM90 or newer")

    def build(stage, seed):
        config = build_model_config(
            "glm5_2_debugmodel", parallelism=ParallelismConfig(),
            seq_len=128, indexer_stage=stage,
        )
        config.layers = config.layers[:4]
        config.index_sources = (0, 1, 2, 2)
        torch.manual_seed(seed)
        with set_default_dtype(torch.bfloat16):
            model = config.build().to(device="cuda")
        model.init_states()
        DefaultActivationRematConfig().apply(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=8e-4,
        )
        return model, optimizer

    inputs = torch.arange(128, device="cuda")
    positions = inputs % 64
    padding = torch.zeros_like(inputs, dtype=torch.bool)
    padding[[63, 127]] = True
    labels = (inputs + 1).masked_fill(padding, -100)
    denominator = torch.tensor([126.0], device="cuda")
    language_loss = CrossEntropyLoss.Config(global_vocab_size=2048).build()

    def snapshot(model):
        return {name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()}

    def step(model, optimizer, stage, number):
        optimizer.zero_grad(set_to_none=True)
        for module in model.modules():
            if isinstance(module, AuxLoss):
                module.instance_acc.zero_()
        output = model(inputs, positions=positions, padding_mask=padding,
                       aux_loss_denominators=denominator)
        loss = (
            output if stage == "warmup"
            else language_loss(output, labels, global_loss_token_counts=denominator[0])[0]
        )
        assert torch.isfinite(loss)
        loss.backward()
        gradients = {name: parameter.grad for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
        assert any(".indexer." in name and value.abs().sum() > 0
                   for name, value in gradients.items())
        assert any(".indexer." not in name for name in gradients) == (stage == "sparse")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        assert torch.isfinite(norm)
        optimizer.step()
        print(f"{stage} step={number} loss={loss.item():.9g} grad_norm={norm.item():.9g}")

    with patch.dict(AuxLoss.group_acc, clear=True), patch.dict(AuxLoss._group_counts, clear=True):
        warmup, optimizer = build("warmup", 61)
        initial = snapshot(warmup)
        for number in (1, 2):
            step(warmup, optimizer, "warmup", number)
        assert optimizer.state
        saved = snapshot(warmup)
        assert any(not torch.equal(saved[name], initial[name])
                   for name in saved if ".indexer." in name)
        for name in saved:
            if ".indexer." not in name:
                torch.testing.assert_close(saved[name], initial[name], rtol=0, atol=0)
        checkpoint = tmp_path / "warmup-model"
        dcp.save({"model": warmup.state_dict()}, checkpoint_id=checkpoint)
        del optimizer, warmup, initial

        sparse, optimizer = build("sparse", 62)
        assert not optimizer.state
        assert any(not torch.equal(value, saved[name])
                   for name, value in snapshot(sparse).items())
        restored = {"model": sparse.state_dict()}
        dcp.load(restored, checkpoint_id=checkpoint)
        sparse.load_state_dict(restored["model"], strict=True)
        for name, value in snapshot(sparse).items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
        assert not optimizer.state
        for number in (1, 2):
            step(sparse, optimizer, "sparse", number)
        assert optimizer.state
        updated = snapshot(sparse)
        for indexer in (False, True):
            assert any(not torch.equal(updated[name], saved[name])
                       for name, _ in sparse.named_parameters()
                       if (".indexer." in name) == indexer)
        print("GLM BF16 DCP handoff: warmup 2 steps -> strict model load -> sparse 2 steps; "
              "fresh optimizer, gradient ownership and default remat passed")
        print(f"checkpoint={checkpoint}")
