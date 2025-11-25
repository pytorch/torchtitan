# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_supported_experiments = frozenset(
    [
        "gpt_oss",
        "simple_fsdp.llama3",
        "simple_fsdp.deepseek_v3",
        "vlm",
        "compiler_toolkit.deepseek_v3",
        "compiler_toolkit.llama3",
        "transformers_modeling_backend",
        "auto_parallel.llama3",
        "auto_parallel.deepseek_v3",
    ]
)
