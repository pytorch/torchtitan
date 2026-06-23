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
        "autoparallel.llama3",
        "autoparallel.local_map_deepseek_v3",
        "path",
        "plan_vit",
        "torchft.llama3",
        "rl",
        # RL examples own a per-example config_registry under rl/examples/<name>;
        # listed here so `--module <name>` resolves (see ConfigManager).
        "alphabet_sort",
        "search_r1",
    ]
)
