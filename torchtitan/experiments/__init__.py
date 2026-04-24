# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_supported_experiments = frozenset(
    [
        "graph_trainer.llama3",
        "graph_trainer.deepseek_v3",
        "graph_trainer.qwen3",
        "transformers_modeling_backend",
        "graph_trainer.autoparallel_llama3",
        "graph_trainer.autoparallel_deepseek_v3",
        "autoparallel.llama3",
        "autoparallel.local_map_deepseek_v3",
        "ft.llama3",
        "rl",
    ]
)
