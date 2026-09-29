# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from torchtitan.config import CompileConfig
from torchtitan.models.common.multimodal import MultimodalModel


class _DummyVisionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_heads = 4


class _DummyVisionAdapter(nn.Module):
    """Two-layer MLP stand-in: no ``layers`` attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Identity()


class _DummyMuseGlimmer(nn.Module):
    multimodal_encoder_fqns = ("vision_encoder",)

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.vision_encoder = _DummyVisionEncoder()
        self.vision_adapter = _DummyVisionAdapter()

    def _parallelize(self, parallelism_context) -> None:
        del parallelism_context


def test_compile_skips_vision_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled: list[nn.Module] = []

    def fake_apply_compile(model, *, compile_config, parallelism_context):
        del compile_config, parallelism_context
        compiled.append(model)

    monkeypatch.setattr(
        "torchtitan.distributed.compile.apply_compile",
        fake_apply_compile,
    )

    model = _DummyMuseGlimmer()
    parallelism_context = MagicMock()
    parallelism_context.tp_enabled = False

    MultimodalModel.parallelize(
        model,
        parallelism_context=parallelism_context,
        training=MagicMock(),
        parallelism=MagicMock(),
        compile_config=CompileConfig(components=["model"]),
        ac_config=None,
        dump_folder="",
        skip_dp=True,
    )

    assert compiled == [model, model.vision_encoder]
    assert model.vision_adapter not in compiled
    assert not hasattr(model.vision_adapter, "layers")
