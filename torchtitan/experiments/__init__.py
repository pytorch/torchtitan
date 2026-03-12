# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_supported_experiments = frozenset(
    [
        "graph_trainer.llama3",
        "graph_trainer.deepseek_v3",
        "vlm",
        "transformers_modeling_backend",
        "autoparallel.llama3",
        "autoparallel.deepseek_v3",
        "autoparallel.local_map_deepseek_v3",
        "ft.llama3",
        "rl",
    ]
)
