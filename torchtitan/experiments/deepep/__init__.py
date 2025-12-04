# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from .moe_deepep import MoEWithDeepEP, get_deepep_buffer, get_hidden_bytes
from .expert_parallel import DeepEPExpertParallel

__all__ = [
    "MoEWithDeepEP",
    "get_deepep_buffer",
    "get_hidden_bytes",
    "DeepEPExpertParallel",
]

__version__ = "1.0.0"
