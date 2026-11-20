# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan.components.optim import OptimizersContainer
from torchtitan.config.transform import transform_model_config_

from torchtitan_recipes.models.muse_spark import MUSE_SPARK_LORA_DECODER_TARGETS
from torchtitan_recipes.sft import slake
from torchtitan_recipes.tests.models.muse_glimmer import muse_glimmer_debugmodel_mm
from torchtitan_recipes.tests.models.muse_spark import muse_spark_debugmodel_mm


_VISION_PREFIXES = ("vision_encoder.", "vision_adapter.", "vision_projection.")


@pytest.mark.parametrize(
    "debugmodel_mm, decoder_targets",
    [
        (muse_spark_debugmodel_mm, MUSE_SPARK_LORA_DECODER_TARGETS),
        (muse_glimmer_debugmodel_mm, slake._MUSE_GLIMMER_LORA_DECODER_TARGETS),
    ],
)
@pytest.mark.parametrize("variant", ["lora_decoder", "lora_vision_decoder"])
def test_slake_lora_trainable_split(debugmodel_mm, decoder_targets, variant):
    # LoRA only warns on unmatched targets; this catches a renamed module.
    transform = slake._slake_lora_transform(variant, decoder_targets=decoder_targets)
    model_config = transform_model_config_(debugmodel_mm().model, [transform])
    with torch.device("meta"):
        model = model_config.build()
    trainable = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    vision = {name for name in trainable if name.startswith(_VISION_PREFIXES)}
    lora = {name for name in trainable if ".lora_" in name}
    assert trainable == lora
    expected_vision = {
        "vision_encoder.layers.0.attn.wq",
        "vision_encoder.layers.0.attn.wk",
        "vision_encoder.layers.0.attn.wv",
        "vision_encoder.layers.0.attn.proj",
        "vision_encoder.layers.0.mlp.linear_fc1",
        "vision_encoder.layers.0.mlp.linear_fc2",
        "vision_adapter.c_fc",
        "vision_adapter.c_proj",
        "vision_projection",
    }
    adapted_vision = {name.rsplit(".lora_", 1)[0] for name in vision}
    decoder = {
        name.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1]
        for name in lora
        if name.startswith("layers.")
    }
    # Every decoder target must match a decoder module.
    assert decoder == set(decoder_targets)
    if variant == "lora_decoder":
        assert not vision
    else:
        assert expected_vision <= adapted_vision
        assert not any("conv1" in name for name in adapted_vision)


def test_slake_full_vision_optimizer_groups():
    # Groups are first-match-wins and the LoRA group matches ".*", so a vision
    # param that the vision pattern misses would silently train at the LoRA lr.
    config = slake.muse_spark_816b_a42b_slake_full_vision_lora_decoder_sft()
    transform = slake._slake_lora_transform(
        "full_vision_lora_decoder", decoder_targets=MUSE_SPARK_LORA_DECODER_TARGETS
    )
    model_config = transform_model_config_(
        muse_spark_debugmodel_mm().model, [transform]
    )
    with torch.device("meta"):
        model = model_config.build()
        # The container raises unless every trainable param is in exactly one group.
        container = OptimizersContainer(config.optim.optimizer, model_parts=[model])
    vision_group, lora_group = (
        optimizer.param_groups[0] for optimizer in container.optimizers
    )
    vision = {
        name
        for name, _ in model.named_parameters()
        if name.startswith(_VISION_PREFIXES)
    }
    assert vision_group["lr"] == 1e-5
    assert set(vision_group["param_names"]) == vision
    assert lora_group["param_names"]
    assert all(".lora_" in name for name in lora_group["param_names"])
