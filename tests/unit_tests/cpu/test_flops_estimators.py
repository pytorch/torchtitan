# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections.abc import Callable

import pytest
import torch

from torchtitan.experiments.transformers_modeling_backend.config_registry import (
    transformers_modeling_backend_debugmodel,
    transformers_modeling_backend_debugmodel_moe,
)
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v3 import model_registry as deepseek_v3_registry
from torchtitan.models.flops import get_parameter_counts
from torchtitan.models.flux.config_registry import flux_debugmodel
from torchtitan.models.flux.flux_datasets import FluxTrainingMicrobatch
from torchtitan.models.gpt_oss import model_registry as gpt_oss_registry
from torchtitan.models.kimi_k2_7 import model_registry as kimi_k2_7_registry
from torchtitan.models.kimi_k3 import model_registry as kimi_k3_registry
from torchtitan.models.llama3 import model_registry as llama3_registry
from torchtitan.models.muse_glimmer import model_registry as muse_glimmer_registry
from torchtitan.models.qwen3 import model_registry as qwen3_registry
from torchtitan.models.qwen3_5 import model_registry as qwen3_5_registry
from torchtitan.protocols import BaseModel
from torchtitan.trainer import Trainer


@pytest.fixture(scope="module", autouse=True)
def _isolate_aux_loss_registry():
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None
    yield
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None


def _build_estimator(
    model_registry: Callable[..., BaseModel.Config],
    flavor: str,
    *,
    seq_len: int = 16,
    **config_overrides: int,
):
    model_config = model_registry(
        flavor,
        seq_len=seq_len,
        **config_overrides,
    )
    with torch.device("meta"):
        model = model_config.build()
    return model_config.build_flops_estimator(model, seq_len=seq_len)


@pytest.mark.parametrize(
    ("model_registry", "flavor", "seq_len", "config_overrides", "expected_flops"),
    [
        pytest.param(llama3_registry, "debugmodel", 16, {}, 546_103_296, id="llama3"),
        pytest.param(
            qwen3_registry, "debugmodel_moe", 16, {}, 4_945_698_816, id="qwen3"
        ),
        pytest.param(
            gpt_oss_registry, "debugmodel", 16, {}, 4_416_359_424, id="gpt_oss"
        ),
        pytest.param(
            deepseek_v3_registry,
            "debugmodel",
            8,
            {"num_mtp_layers": 1},
            1_586_823_168,
            id="deepseek_v3_mtp",
        ),
    ],
)
def test_decoder_estimator_preserves_flops(
    model_registry: Callable[..., BaseModel.Config],
    flavor: str,
    seq_len: int,
    config_overrides: dict[str, int],
    expected_flops: int,
) -> None:
    estimator = _build_estimator(
        model_registry,
        flavor,
        seq_len=seq_len,
        **config_overrides,
    )

    result = estimator({"input": torch.empty(seq_len)})

    assert result == expected_flops
    assert type(result) is int


def _text_batch() -> dict[str, torch.Tensor]:
    return {"input": torch.zeros((1, 16), dtype=torch.long)}


def _image_fields(grids: list[tuple[int, int, int]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.zeros((sum(t * h * w for t, h, w in grids), 1)),
        "grid_thw": torch.tensor(grids),
    }


def _video_fields(grids: list[tuple[int, int, int]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values_videos": torch.zeros((sum(t * h * w for t, h, w in grids), 1)),
        "grid_thw_videos": torch.tensor(grids),
    }


def test_vision_attention_exposes_flops_per_query_key_pair() -> None:
    model_config = qwen3_5_registry("debugmodel")
    vision_config = model_config.vision_encoder
    assert vision_config is not None
    block_config = vision_config.block

    assert block_config.attn.flops_per_query_key_pair() == (
        6
        * block_config.attn.num_heads
        * 2
        * (block_config.attn.dim // block_config.attn.num_heads)
    )


def test_qwen_estimator_counts_mixed_image_and_video_work() -> None:
    batch = {
        **_text_batch(),
        **_image_fields([(1, 4, 4)]),
        **_video_fields([(2, 4, 4)]),
    }

    assert _build_estimator(qwen3_5_registry, "debugmodel_moe")(batch) == 7_081_963_776


@pytest.mark.parametrize(
    ("modality_fields", "expected_flops"),
    [
        (_image_fields([(1, 4, 4)]), 2_945_955_840),
        (_video_fields([(2, 4, 4)]), 3_172_325_376),
    ],
)
def test_kimi_k2_7_estimator_counts_vision_work(
    modality_fields: dict[str, torch.Tensor],
    expected_flops: int,
) -> None:
    batch = {**_text_batch(), **modality_fields}

    assert _build_estimator(kimi_k2_7_registry, "debugmodel")(batch) == expected_flops


def test_kimi_k3_estimator_counts_image_work() -> None:
    batch = {**_text_batch(), **_image_fields([(1, 4, 4)])}

    assert _build_estimator(kimi_k3_registry, "debugmodel")(batch) == 55_845_153_792


@pytest.mark.parametrize(
    ("grids", "expected_flops"),
    [
        ([(1, 4, 4)], 731_043_840),
        ([(1, 4, 4), (1, 6, 8)], 791_298_048),
    ],
)
def test_muse_glimmer_estimator_counts_grid_dependent_vision_flops(
    grids: list[tuple[int, int, int]],
    expected_flops: int,
) -> None:
    batch = {**_text_batch(), **_image_fields(grids)}

    assert (
        _build_estimator(muse_glimmer_registry, "debugmodel_mm")(batch)
        == expected_flops
    )


def test_flux_estimator_uses_image_batch_size() -> None:
    trainer_config = flux_debugmodel()
    model_config = trainer_config.model
    with torch.device("meta"):
        model = model_config.build()
    estimator = model_config.build_flops_estimator(
        model,
        seq_len=trainer_config.training.max_context_length,
    )
    batch = FluxTrainingMicrobatch(
        labels=torch.zeros(2, 3, 256, 256),
        t5=torch.zeros(2, 256, dtype=torch.int64),
        clip=torch.zeros(2, 77, dtype=torch.int64),
        prompt=["a test image"] * 2,
        num_valid_tokens=2 * 16 * 32 * 32,
    ).as_input_dict()

    assert estimator(batch) == 814_721_925_120


@pytest.mark.parametrize(
    ("config_factory", "expected_flops"),
    [
        (transformers_modeling_backend_debugmodel, 5_583_839_232),
        (transformers_modeling_backend_debugmodel_moe, 35_340_976_128),
    ],
)
def test_transformers_backend_estimator_preserves_flops(
    config_factory: Callable[..., Trainer.Config],
    expected_flops: int,
) -> None:
    trainer_config = config_factory(seq_len=16)
    model_config = copy.deepcopy(trainer_config.model)
    model_config.update_from_config(config=trainer_config)
    with torch.device("meta"):
        model = model_config.build()
    estimator = model_config.build_flops_estimator(model, seq_len=16)

    assert estimator({"input": torch.zeros(1, 16)}) == expected_flops


def test_transformers_backend_parameter_counts_weight_preparallel_moe() -> None:
    trainer_config = transformers_modeling_backend_debugmodel_moe(seq_len=16)
    model_config = copy.deepcopy(trainer_config.model)
    model_config.update_from_config(config=trainer_config)
    with torch.device("meta"):
        model = model_config.build()

    total_params = sum(param.numel() for param in model.parameters())
    expected_active_params = total_params
    for layer in model.layers.values():
        moe_config = getattr(layer, "_native_moe_config", None)
        if moe_config is None:
            continue
        expert_params = sum(param.numel() for param in layer.mlp.experts.parameters())
        expected_active_params -= expert_params
        expected_active_params += (
            expert_params * moe_config.router.top_k // moe_config.num_experts
        )

    assert model_config.get_parameter_counts(model) == (
        total_params,
        expected_active_params,
    )


def test_model_config_parameter_counts_default_to_model_structure() -> None:
    model_config = llama3_registry("debugmodel", seq_len=16)
    with torch.device("meta"):
        model = model_config.build()

    assert model_config.get_parameter_counts(model) == get_parameter_counts(model)
