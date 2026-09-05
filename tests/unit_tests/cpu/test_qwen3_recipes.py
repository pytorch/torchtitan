# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.qwen3.config_registry import qwen3_8b


def test_qwen3_8b_pretrain_recipe_uses_8b_flavor() -> None:
    config = qwen3_8b()

    assert config.model_spec is not None
    assert config.model_spec.name == "qwen3"
    assert config.model_spec.flavor == "8B"
    assert config.hf_assets_path.endswith("Qwen3-8B")
    assert config.optimizer.param_groups[0].optimizer_kwargs["lr"] == 8e-4
