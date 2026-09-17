# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib
from types import SimpleNamespace

import torch


flux_infer = importlib.import_module("torchtitan.models.flux.inference.infer")


def test_inference_runs_model_in_engine_context(monkeypatch, tmp_path):
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("a prompt\n")

    context_active = False

    @contextlib.contextmanager
    def train_context():
        nonlocal context_active
        assert not context_active
        context_active = True
        try:
            yield
        finally:
            context_active = False

    engine = SimpleNamespace(
        device=torch.device("cpu"),
        model_parts=[object()],
        load_checkpoint=lambda: None,
        train_context=train_context,
    )
    trainer = SimpleNamespace(
        engine=engine,
        _dtype=torch.float32,
        autoencoder=object(),
        t5_encoder=object(),
        clip_encoder=object(),
    )
    config = SimpleNamespace(
        inference=SimpleNamespace(
            prompts_path=str(prompts_path),
            local_batch_size=1,
            img_size=16,
            save_img_folder="images",
            sampling=SimpleNamespace(
                enable_classifier_free_guidance=False,
                denoising_steps=1,
                classifier_free_guidance_scale=1.0,
            ),
        ),
        checkpointer=SimpleNamespace(load_step=-1),
        tokenizer=SimpleNamespace(build=lambda: object()),
        dump_folder=str(tmp_path),
    )

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(flux_infer, "FluxTrainer", lambda config: trainer)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: None)
    monkeypatch.setattr(flux_infer, "save_image", lambda **kwargs: None)

    def generate_image(**kwargs):
        assert context_active
        return torch.zeros(1, 3, 16, 16)

    monkeypatch.setattr(flux_infer, "generate_image", generate_image)

    flux_infer.inference(config)
    assert not context_active
