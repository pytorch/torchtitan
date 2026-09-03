# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Hunks in this file are copied from upstream open PR 4322/4449/4450 (fegin's CP stack) to unblock running;
# pending rebase and reconcile.

"""
Context Parallel APIs

``cp_shard`` is only used by Flux, which has a different input pattern from LLMs.

``prepare_context_parallel_input`` is the main API.
TODO: we should generalize this API to cover even Flux's use case.

``validate_context_parallel`` holds every config-time CP check and is called
from ``Trainer.Config.__post_init__``.
"""

from .api import cp_shard, prepare_context_parallel_input
from .validation import validate_context_parallel

__all__ = [
    "cp_shard",
    "prepare_context_parallel_input",
    "validate_context_parallel",
]
