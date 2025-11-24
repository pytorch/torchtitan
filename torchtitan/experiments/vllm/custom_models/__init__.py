# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for integrating custom model implementations with vLLM."""

from .attention_replacement import replace_with_trainable_attention
from .custom_model_wrapper import VLLMModelForCausalLM
from .trainable_attention import TrainableFlashAttention
from .trainable_mla_attention import MLAConfig, TrainableMLA
from .utils import (
    convert_freqs_cis_to_real,
    create_mla_kv_cache_spec,
    load_external_weights,
    store_positions_in_context,
)

__all__ = [
    # Attention modules
    "TrainableFlashAttention",
    "TrainableMLA",
    "MLAConfig",
    "replace_with_trainable_attention",
    # Base wrapper
    "VLLMModelForCausalLM",
    # Utilities
    "convert_freqs_cis_to_real",
    "create_mla_kv_cache_spec",
    "load_external_weights",
    "store_positions_in_context",
]
