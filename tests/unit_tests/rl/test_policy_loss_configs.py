# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check that RL recipes select the intended policy-statistics path."""

import importlib

import pytest

_ALPHABET = "torchtitan.rl.examples.alphabet_sort.config_registry"
_SEARCH = "torchtitan.rl.examples.search_r1.config_registry"


@pytest.mark.parametrize(
    "module_name, factory_name",
    [
        (_ALPHABET, "rl_grpo_qwen3_0_6b_varlen"),
        (_ALPHABET, "rl_grpo_qwen3_0_6b_flex"),
        (_ALPHABET, "rl_grpo_gpt_oss_debug_varlen"),
        (_ALPHABET, "rl_grpo_qwen3_5_debug_varlen"),
        (_ALPHABET, "rl_grpo_qwen3_6_27b_varlen_perf"),
        (_SEARCH, "rl_grpo_qwen3_1_7b_search_r1"),
        (_SEARCH, "rl_grpo_qwen3_8b_search_r1"),
        (_ALPHABET, "rl_grpo_gpt_oss_debug_varlen_batch_invariant"),
        (_ALPHABET, "rl_grpo_qwen3_moe_debug_varlen_batch_invariant"),
        (_ALPHABET, "rl_grpo_qwen3_0_6b_varlen_batch_invariant"),
        (_ALPHABET, "rl_grpo_qwen3_0_6b_flex_batch_invariant"),
        (_ALPHABET, "rl_grpo_qwen3_5_9b_varlen_batch_invariant"),
        (_ALPHABET, "rl_grpo_qwen3_5_debug_varlen_batch_invariant"),
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
    from torchtitan.rl.losses import DAPOLoss, GRPOLoss

    factory = getattr(importlib.import_module(module_name), factory_name)
    config = factory()
    loss_config = config.trainer.loss
    if isinstance(loss_config, ChunkedLossWrapper.Config):
        loss_config = loss_config.loss_fn
    assert isinstance(loss_config, (DAPOLoss.Config, GRPOLoss.Config))

    assert loss_config.global_vocab_size == decoder_vocab_size(config.model_spec)
