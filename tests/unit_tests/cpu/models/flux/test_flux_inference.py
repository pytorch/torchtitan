# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib
from types import SimpleNamespace

import pytest
import torch

from torchtitan.config import ConfigLoader


flux_infer = importlib.import_module("torchtitan.models.flux.inference.infer")
flux_sampling = importlib.import_module("torchtitan.models.flux.inference.sampling")


def test_documented_inference_config_loads():
    """The config named in run_train.sh's Flux ENTRYPOINT example loads."""
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_recipes.tests.models.flux",
            "--config",
            "flux_debugmodel_inference",
        ]
    )

    assert config.inference is not None


def test_inference_runs_model_in_engine_context(monkeypatch, tmp_path):
    prompts_path = tmp_path / "prompts.txt"
    prompts_path.write_text("a prompt\n")

    context_active = False

    @contextlib.contextmanager
    def spmd_context():
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
        parallelism_context=SimpleNamespace(
            activate_spmd=lambda **kwargs: spmd_context()
        ),
        config=SimpleNamespace(debug=SimpleNamespace(spmd_typechecking=False)),
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


# 256px -> 32x32 latents -> 256 packed tokens; 512px -> 1024 packed tokens. The
# expected values are the reference Flux schedule (mu = 0.5 at 256 tokens) for 4 steps.
@pytest.mark.parametrize(
    "img_size, expected",
    [(256, [1.0, 0.8318, 0.6225, 0.3547]), (512, [1.0, 0.8492, 0.6525, 0.3849])],
)
def test_denoise_shifts_schedule_by_packed_sequence_length(img_size, expected):
    """The schedule shift uses the packed 2x2-patch sequence length, as in the
    reference Flux sampler, not the number of unpacked latent positions."""
    denoising_steps = 4
    seen_timesteps = []

    def model(*, img, timesteps, **kwargs):
        seen_timesteps.append(timesteps[0].item())
        return torch.zeros_like(img)

    flux_sampling.denoise(
        device=torch.device("cpu"),
        dtype=torch.float32,
        model=model,
        img_width=img_size,
        img_height=img_size,
        denoising_steps=denoising_steps,
        clip_encodings=torch.zeros(1, 768),
        t5_encodings=torch.zeros(1, 1, 4096),
    )

    torch.testing.assert_close(
        torch.tensor(seen_timesteps), torch.tensor(expected), atol=1e-4, rtol=0
    )
