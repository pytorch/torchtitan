# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan.models.common.linear import Linear
from torchtitan.models.muse_spark import build_model_config
from torchtitan.models.muse_spark.state_dict_adapter import MuseSparkStateDictAdapter


def _build_model():
    config = build_model_config("debugmodel_mm", seq_len=8)
    for _, linear, _, _ in config.traverse(Linear.Config, recurse=True):
        linear.inplace_wgrad_accum = False
    model = config.build().to(device="cuda")
    model.init_states()
    return model


def _train_step(model, optimizer, tokens):
    optimizer.zero_grad(set_to_none=True)
    positions = torch.arange(tokens.numel(), device=tokens.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            tokens,
            positions=positions,
            attention_metadata=model._get_attention_metadata(positions),
        )
    loss = logits.float().square().mean()
    loss.backward()
    optimizer.step()
    return logits.detach(), loss.detach()


def test_dummy_media_preserves_text_and_runs_backward():
    model = _build_model()
    hidden = torch.randn(8, model.config.dim, device="cuda", requires_grad=True)
    result = model._prepare_multimodal_embeds(
        hidden,
        pixel_values=None,
        pixel_values_videos=None,
        grid_thw=None,
        grid_thw_videos=None,
        waveforms=None,
        waveform_padded_lengths=None,
        image_bank_indices_T=None,
        video_bank_indices_T=None,
        audio_bank_indices_T=None,
    )
    torch.testing.assert_close(result, hidden, rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden), rtol=0, atol=0)
    for encoder in (model.vision_encoder, model.audio_encoder):
        for name, parameter in encoder.named_parameters():
            assert parameter.grad is not None, name
            assert torch.count_nonzero(parameter.grad) == 0, name


def test_checkpoint_restores_next_optimizer_step(tmp_path):
    model = _build_model()
    adapter = MuseSparkStateDictAdapter(model.config, hf_assets_path=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    tokens = torch.arange(8, device="cuda")
    _train_step(model, optimizer, tokens)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": adapter.to_hf(model.state_dict()),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    expected_logits, expected_loss = _train_step(model, optimizer, tokens)

    restored_model = _build_model()
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=0.01, foreach=False
    )
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    restored_model.load_state_dict(adapter.from_hf(checkpoint["model"]), strict=True)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    actual_logits, actual_loss = _train_step(restored_model, restored_optimizer, tokens)

    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(
        restored_model.state_dict(), model.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        restored_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0
    )
