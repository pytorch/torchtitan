# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests that load_hf_config never runs config code shipped in a model repo."""

import json
import os
import tempfile
import unittest

from transformers import LlamaConfig

from torchtitan.experiments.transformers_modeling_backend import TitanModelConfig
from torchtitan.experiments.transformers_modeling_backend.model import (
    HFTransformerModel,
)

_REMOTE_CONFIG = """
import os
from transformers import LlamaConfig

open(os.environ["TT_REMOTE_CODE_MARKER"], "w").close()


class RemoteConfig(LlamaConfig):
    pass
"""


def _write_repo(repo_dir: str) -> None:
    LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
        architectures=["LlamaForCausalLM"],
    ).save_pretrained(repo_dir)
    config_path = os.path.join(repo_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["auto_map"] = {"AutoConfig": "configuration_remote.RemoteConfig"}
    with open(config_path, "w") as f:
        json.dump(config, f)
    with open(os.path.join(repo_dir, "configuration_remote.py"), "w") as f:
        f.write(_REMOTE_CONFIG)


class TestRemoteCode(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo_dir = os.path.join(tmp.name, "repo")
        self.marker = os.path.join(tmp.name, "marker")
        os.environ["TT_REMOTE_CODE_MARKER"] = self.marker
        self.addCleanup(os.environ.pop, "TT_REMOTE_CODE_MARKER", None)
        _write_repo(self.repo_dir)

    def test_does_not_run_remote_code(self) -> None:
        config = HFTransformerModel.Config(
            model_config=TitanModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=4)
        )
        config.load_hf_config(
            hf_model_id=self.repo_dir, max_context_length=128, deterministic=False
        )
        self.assertFalse(os.path.exists(self.marker))


if __name__ == "__main__":
    unittest.main()
