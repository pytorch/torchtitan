# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check that RL recipes select the intended loss and compile configurations."""

import dataclasses
import importlib
from unittest.mock import patch

import pytest
import torch

_ALPHABET = "torchtitan_recipes.rl.alphabet_sort"
_ALPHABET_TEST = "torchtitan_recipes.tests.rl.alphabet_sort"
_SEARCH = "torchtitan_recipes.rl.search_r1"

_BATCH_INVARIANT_CONFIGS = [
    (_ALPHABET_TEST, "rl_grpo_gpt_oss_debug_varlen_batch_invariant"),
    (_ALPHABET_TEST, "rl_grpo_qwen3_moe_debug_varlen_batch_invariant"),
    (_ALPHABET, "rl_grpo_qwen3_0_6b_varlen_batch_invariant"),
    (_ALPHABET, "rl_grpo_qwen3_0_6b_flex_batch_invariant"),
    (_ALPHABET, "rl_grpo_qwen3_5_9b_varlen_batch_invariant"),
    (_ALPHABET_TEST, "rl_grpo_qwen3_5_debug_varlen_batch_invariant"),
]


@pytest.mark.parametrize(
    "module_name, factory_name",
    [
        (_ALPHABET, "rl_grpo_qwen3_0_6b_varlen"),
        (_ALPHABET, "rl_grpo_qwen3_0_6b_flex"),
        (_ALPHABET_TEST, "rl_grpo_gpt_oss_debug_varlen"),
        (_ALPHABET_TEST, "rl_grpo_qwen3_5_debug_varlen"),
        (_ALPHABET, "rl_grpo_qwen3_6_27b_varlen_perf"),
        (_SEARCH, "rl_grpo_qwen3_1_7b_search_r1"),
        (_SEARCH, "rl_grpo_qwen3_8b_search_r1"),
        (_ALPHABET_TEST, "rl_grpo_gpt_oss_debug_varlen_batch_invariant"),
        (_ALPHABET_TEST, "rl_grpo_qwen3_moe_debug_varlen_batch_invariant"),
        (_ALPHABET, "rl_grpo_qwen3_0_6b_varlen_batch_invariant"),
        (_ALPHABET, "rl_grpo_qwen3_0_6b_flex_batch_invariant"),
        (_ALPHABET, "rl_grpo_qwen3_5_9b_varlen_batch_invariant"),
        (_ALPHABET_TEST, "rl_grpo_qwen3_5_debug_varlen_batch_invariant"),
    ],
    ids=lambda value: value.rsplit(".", 1)[-1],
)
def test_policy_loss_vocab_size_matches_model_and_mode(module_name, factory_name):
    pytest.importorskip("vllm")
    pytest.importorskip("renderers")
    if module_name == _SEARCH:
        pytest.importorskip("datasets")

    from torchtitan.components.loss import ChunkedLossWrapper
    from torchtitan.models.common.config_utils import decoder_vocab_size
    from torchtitan.rl.trainer.losses import DAPOLoss, GRPOLoss

    factory = getattr(importlib.import_module(module_name), factory_name)
    config = factory()
    loss_config = config.trainer.loss
    if isinstance(loss_config, ChunkedLossWrapper.Config):
        loss_config = loss_config.loss_fn
    assert isinstance(loss_config, (DAPOLoss.Config, GRPOLoss.Config))

    assert loss_config.global_vocab_size == decoder_vocab_size(config.trainer.model)


@pytest.mark.parametrize(
    "module_name, factory_name",
    _BATCH_INVARIANT_CONFIGS,
    ids=lambda value: value.rsplit(".", 1)[-1],
)
def test_batch_invariant_configs_disable_local_compile(module_name, factory_name):
    pytest.importorskip("vllm")
    pytest.importorskip("renderers")

    factory = getattr(importlib.import_module(module_name), factory_name)
    config = factory()

    assert config.trainer.model.local_compile.regions == []
    assert config.generator.model.local_compile.regions == []


def test_rocm_batch_invariance_rejects_varlen_attention():
    pytest.importorskip("vllm")
    pytest.importorskip("renderers")
    alphabet_sort = importlib.import_module(_ALPHABET)

    with patch.object(torch.version, "hip", "10.0"):
        alphabet_sort.rl_grpo_qwen3_0_6b_flex_batch_invariant()
        with pytest.raises(ValueError, match="requires FlexAttention"):
            alphabet_sort.rl_grpo_qwen3_0_6b_varlen_batch_invariant()


def test_batch_invariance_requires_matching_generator_debug_config():
    pytest.importorskip("vllm")
    pytest.importorskip("renderers")
    from torchtitan.rl.controller.controller import Controller

    alphabet_sort = importlib.import_module(_ALPHABET)
    config = alphabet_sort.rl_grpo_qwen3_0_6b_flex_batch_invariant()

    config.generator.debug = dataclasses.replace(
        config.generator.debug, batch_invariant=False
    )
    with pytest.raises(ValueError, match="same batch_invariant setting"):
        Controller(config)

    config = alphabet_sort.rl_grpo_qwen3_0_6b_flex_batch_invariant()
    config.generator.debug = dataclasses.replace(
        config.generator.debug, deterministic=False
    )
    with pytest.raises(ValueError, match="deterministic=True for both"):
        Controller(config)
