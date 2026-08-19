# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Context Parallel APIs

``cp_shard`` is only used by Flux, which has a different input pattern from LLMs.

``prepare_context_parallel_input`` is the main API.
TODO: we should generalize this API to cover even Flux's use case.

``validate_cp_backend`` is called from a model config's
``update_from_config`` by the models that declare CP in ``ShardingConfig``.
"""

from .api import cp_shard, prepare_context_parallel_input, validate_cp_backend

__all__ = [
    "cp_shard",
    "prepare_context_parallel_input",
    "validate_cp_backend",
]
