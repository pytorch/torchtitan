# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from torchtitan.models.flux.configs import FluxEncoderConfig, Inference


def test_flux_dataclass_defaults_use_shipped_asset_paths():
    encoder = FluxEncoderConfig()
    inference = Inference()

    assert "experiments/flux" not in encoder.autoencoder_path
    assert "experiments/flux" not in inference.prompts_path
    assert encoder.autoencoder_path == "assets/hf/FLUX.1-dev/ae.safetensors"

    repo_root = Path(__file__).parents[3]
    prompts_file = (repo_root / inference.prompts_path).resolve()
    shipped_prompts = (
        repo_root / "torchtitan/models/flux/inference/prompts.txt"
    ).resolve()
    assert prompts_file.is_file()
    assert prompts_file == shipped_prompts

    launcher = (repo_root / "torchtitan/models/flux/run_infer.sh").read_text()
    assert "models/flux/run_train.sh" not in launcher
