# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DeepSeek-V3 model wrapper for DeepEP experiments.

This module provides a DeepSeekV3 model class that is compatible with
DeepEP's MoE parallelization strategy.
"""

from torchtitan.models.deepseek_v3 import DeepSeekV3Model, DeepSeekV3ModelArgs


class DeepEPDeepSeekV3Model(DeepSeekV3Model):
    """
    DeepSeek-V3 model with DeepEP-compatible initialization.
    
    This class extends the base DeepSeekV3Model to ensure proper
    initialization for DeepEP experiments. The main difference is
    that MoE layers will be replaced with DeepEP versions during
    the parallelization step.
    """
    
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__(model_args)
        self.init_weights()
    
    def init_weights(self, *args, **kwargs):
        """Initialize model weights."""
        super().init_weights(*args, **kwargs)

