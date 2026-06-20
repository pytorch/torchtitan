# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OLMo-style scaling ladder for TorchTitan's Llama3 decoder.

``LADDER`` is the default showcase ladder and the single source of truth for the
config registry, CLI, and agent-facing API. Building it is cheap: rungs are
audited on meta lazily (on first plan/trainer_config), not at import time.
"""

from .ladder import ComputeSpec, debug_ladder, default_ladder, Llama3Ladder
from .model import model_registry, RUNGS
from .policy import WSDSChinchillaPolicy

LADDER = default_ladder()

__all__ = [
    "ComputeSpec",
    "LADDER",
    "Llama3Ladder",
    "RUNGS",
    "WSDSChinchillaPolicy",
    "debug_ladder",
    "default_ladder",
    "model_registry",
]
