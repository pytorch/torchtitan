# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Weight conversion utilities for vLLM and TorchTitan."""

from .converter import torchtitan_to_vllm, vllm_to_torchtitan

__all__ = ["vllm_to_torchtitan", "torchtitan_to_vllm"]
