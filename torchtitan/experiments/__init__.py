# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_supported_experiments = frozenset(
    ["flux", "simple_fsdp.llama3", "simple_fsdp.deepseek_v3", "vlm", "transformers_backend"]
)
