# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from inspect import signature

import pytest

from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
    graph_trainer_deepseek_v3_debugmodel,
)
from torchtitan.experiments.graph_trainer.llama3.config_registry import (
    graph_trainer_llama3_debugmodel,
)
from torchtitan.experiments.graph_trainer.muse_glimmer.config_registry import (
    graph_trainer_muse_glimmer_debugmodel,
)
from torchtitan.experiments.graph_trainer.qwen3.config_registry import (
    graph_trainer_qwen3_debugmodel,
    graph_trainer_qwen3_debugmodel_moe,
)
from torchtitan.experiments.torchft.llama3.config_registry import (
    llama3_torchft_debugmodel,
)
from torchtitan.experiments.transformers_modeling_backend.config_registry import (
    transformers_modeling_backend_debugmodel,
    transformers_modeling_backend_debugmodel_moe,
)
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_hybridep,
    deepseek_v3_debugmodel_mtp,
    deepseek_v3_debugmodel_mxfp8,
)
from torchtitan.models.deepseek_v4.config_registry import (
    deepseek_v4_debugmodel,
    deepseek_v4_mtp_debugmodel,
)
from torchtitan.models.gpt_oss.config_registry import (
    gpt_oss_debugmodel,
    gpt_oss_debugmodel_flex,
)
from torchtitan.models.kimi_k2_7.config_registry import kimi_k2_5_debugmodel
from torchtitan.models.llama3.config_registry import (
    llama3_debugmodel,
    llama3_debugmodel_ce_loss,
    llama3_debugmodel_dist_gemm,
    llama3_debugmodel_first_85_pct_layers_nvfp4,
    llama3_debugmodel_float8,
    llama3_debugmodel_float8_emulate_lora,
    llama3_debugmodel_mxfp8,
    llama3_debugmodel_nvfp4,
    llama3_debugmodel_varlen_attn,
    sft_debugmodel,
)
from torchtitan.models.muse_glimmer.config_registry import (
    muse_glimmer_debugmodel,
    muse_glimmer_debugmodel_mm,
)
from torchtitan.models.qwen3.config_registry import (
    qwen3_debugmodel,
    qwen3_debugmodel_first_85_pct_layers_nvfp4,
    qwen3_debugmodel_flex_flash,
    qwen3_debugmodel_moe_param_groups,
    qwen3_debugmodel_non_fused_qkv,
    qwen3_debugmodel_nvfp4,
    qwen3_moe_debug,
    qwen3_moe_deepep,
)
from torchtitan.models.qwen3_5.config_registry import (
    qwen35_debugmodel,
    qwen35_debugmodel_moe,
    qwen35_debugmodel_varlen_attn,
)
from torchtitan.models.qwen3_6.config_registry import (
    qwen36_debugmodel,
    qwen36_debugmodel_moe,
    qwen36_debugmodel_varlen_attn,
)
from torchtitan.models.qwen3_8.config_registry import (
    qwen38_debugmodel,
    qwen38_debugmodel_moe,
    qwen38_debugmodel_varlen_attn,
)
from torchtitan.trainer import Trainer


DebugConfigFactory = Callable[..., Trainer.Config]

_DEBUG_CONFIG_FACTORIES: tuple[DebugConfigFactory, ...] = (
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_hybridep,
    deepseek_v3_debugmodel_mtp,
    deepseek_v3_debugmodel_mxfp8,
    deepseek_v4_debugmodel,
    deepseek_v4_mtp_debugmodel,
    gpt_oss_debugmodel,
    gpt_oss_debugmodel_flex,
    kimi_k2_5_debugmodel,
    llama3_debugmodel,
    llama3_debugmodel_ce_loss,
    llama3_debugmodel_dist_gemm,
    llama3_debugmodel_first_85_pct_layers_nvfp4,
    llama3_debugmodel_float8,
    llama3_debugmodel_float8_emulate_lora,
    llama3_debugmodel_mxfp8,
    llama3_debugmodel_nvfp4,
    llama3_debugmodel_varlen_attn,
    sft_debugmodel,
    muse_glimmer_debugmodel,
    muse_glimmer_debugmodel_mm,
    qwen3_debugmodel,
    qwen3_debugmodel_first_85_pct_layers_nvfp4,
    qwen3_debugmodel_flex_flash,
    qwen3_debugmodel_moe_param_groups,
    qwen3_debugmodel_non_fused_qkv,
    qwen3_debugmodel_nvfp4,
    qwen3_moe_debug,
    qwen3_moe_deepep,
    qwen35_debugmodel,
    qwen35_debugmodel_moe,
    qwen35_debugmodel_varlen_attn,
    qwen36_debugmodel,
    qwen36_debugmodel_moe,
    qwen36_debugmodel_varlen_attn,
    qwen38_debugmodel,
    qwen38_debugmodel_moe,
    qwen38_debugmodel_varlen_attn,
    llama3_torchft_debugmodel,
    transformers_modeling_backend_debugmodel,
    transformers_modeling_backend_debugmodel_moe,
)


@pytest.mark.parametrize(
    "config_factory",
    _DEBUG_CONFIG_FACTORIES,
    ids=[factory.__name__ for factory in _DEBUG_CONFIG_FACTORIES],
)
def test_debug_config_default_seq_len(config_factory: DebugConfigFactory) -> None:
    assert (
        signature(config_factory).parameters["seq_len"].default
        == DEFAULT_DEBUG_MODEL_SEQ_LEN
    )


@pytest.mark.parametrize(
    "config_factory",
    (
        graph_trainer_deepseek_v3_debugmodel,
        graph_trainer_llama3_debugmodel,
        graph_trainer_muse_glimmer_debugmodel,
        graph_trainer_qwen3_debugmodel,
        graph_trainer_qwen3_debugmodel_moe,
    ),
)
def test_graph_trainer_debug_config_default_seq_len(
    config_factory: DebugConfigFactory,
) -> None:
    config = config_factory()
    assert config.training.max_context_length == DEFAULT_DEBUG_MODEL_SEQ_LEN
    assert config.model_spec is not None
    assert config.model_spec.max_context_length == DEFAULT_DEBUG_MODEL_SEQ_LEN
