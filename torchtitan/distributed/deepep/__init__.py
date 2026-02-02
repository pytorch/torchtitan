# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Expert Parallel Communication Backends for MoE Training.

- DeepEP: Support for NVLink8 (H100 / B200)
- HybridEP: Optimized for GB200/NVLink72

Backend is selected via job_config.parallelism.expert_parallel_comm_backend.
HybridEP config is in job_config.parallelism.hybridep.

Usage:
    from torchtitan.distributed.deepep import deepep, hybridep
    
    # For H100/NVLink8:
    hidden, tpe, state = deepep.dispatch_tokens(...)
    output = deepep.combine_tokens(hidden, state)
    
    # For GB200/NVLink72:
    hidden, tpe, state = hybridep.dispatch_tokens(...)
    output = hybridep.combine_tokens(hidden, state)
"""

from . import deepep, hybridep

__all__ = ["deepep", "hybridep"]
