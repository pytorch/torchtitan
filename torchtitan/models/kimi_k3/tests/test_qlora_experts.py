# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for MXFP4-packed grouped experts under the LoRA converter."""

import pytest
import torch

pytest.importorskip("torchao.prototype.mx_formats.mx_tensor")

from torchtitan.components.lora import (
    LoRAConverter,
    merge_lora_state_dict,
    MXFP4ExpertsBase,
)
from torchtitan.models.kimi_k3 import model_registry


def _build():
    torch.manual_seed(0)
    spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=4,
                alpha=8.0,
                target_modules=["wo"],
                quantize_experts="mxfp4",
            ),
        ],
    )
    model = spec.model.build()
    model.init_states()
    return model


def test_experts_pack_at_build_and_dequant_property():
    model = _build()
    packed = [m for _, m in model.named_modules() if isinstance(m, MXFP4ExpertsBase)]
    assert packed
    m = packed[0]
    for name, shape in m._mxfp4_shapes.items():
        assert name not in m._parameters
        assert m._parameters[name + "_qdata"].dtype == torch.uint8
        # The property restores the logical [E, A, B] view in bf16.
        w = getattr(m, name)
        assert tuple(w.shape) == shape
        assert w.abs().sum() > 0, "packed experts left zero-initialized"


def test_experts_merge_restores_original_keys():
    model = _build()
    merged = merge_lora_state_dict(model)
    packed = [
        (n, m) for n, m in model.named_modules() if isinstance(m, MXFP4ExpertsBase)
    ]
    name, m = packed[0]
    for wname, shape in m._mxfp4_shapes.items():
        assert f"{name}.{wname}" in merged
        assert tuple(merged[f"{name}.{wname}"].shape) == shape
        assert f"{name}.{wname}_qdata" not in merged
        # The packed layout is back in place after the export.
        assert wname not in m._parameters
