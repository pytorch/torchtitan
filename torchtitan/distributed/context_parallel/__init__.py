# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Context Parallel APIs

``apply_cp_to_forward`` is for the legacy spmd backend. Full DTensor and spmd_types
don't need this API.

``cp_shard`` is only used by Flux, which has a different input pattern from LLMs.

``prepare_context_parallel_input`` is the main API.
TODO: we should generalize this API to cover even Flux's use case.
"""

from .api import apply_cp_to_forward, cp_shard, prepare_context_parallel_input

__all__ = [
    "apply_cp_to_forward",
    "cp_shard",
    "prepare_context_parallel_input",
]
