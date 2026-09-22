# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Allow CPU-safe RL unit tests to collect without vLLM.

``torchtitan.rl.__init__`` re-exports the vLLM wrapper, so importing any RL
submodule fails when vLLM is not installed. CPU unit CI does not install
vLLM; GPU/RL jobs do. When vLLM is missing, register ``torchtitan.rl`` as a
namespace package so submodule imports skip that initializer.

This is a no-op when vLLM is installed, so GPU RL workflows keep the real
package initializer (including ``apply_env_defaults``).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

if importlib.util.find_spec("vllm") is None and "torchtitan.rl" not in sys.modules:
    _rl = types.ModuleType("torchtitan.rl")
    _rl.__path__ = [str(Path(__file__).resolve().parents[3] / "torchtitan" / "rl")]
    _rl.__package__ = "torchtitan.rl"
    sys.modules["torchtitan.rl"] = _rl
