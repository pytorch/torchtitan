# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
NUMERICAL_EXAMPLE = (
    REPO_ROOT / "scripts" / "checkpoint_conversion" / "numerical_tests_example.py"
)


def test_checkpoint_recipe_sets_scheduler_under_optim() -> None:
    from scripts._checkpoint_test_config import _MODULE_SOURCE

    assert "trainer_config.optim.lr_scheduler.total_steps" in _MODULE_SOURCE
    assert "trainer_config.lr_scheduler.total_steps" not in _MODULE_SOURCE


def test_rl_loss_compare_repo_root_contains_pretrain_script() -> None:
    from scripts.rl.loss_compare import _repo_root

    root = Path(_repo_root())
    assert (root / "scripts" / "loss_compare.py").is_file()


def test_numerical_example_loads_the_llama3_recipe() -> None:
    tree = ast.parse(NUMERICAL_EXAMPLE.read_text())
    module_names = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and node.value == "torchtitan_recipes.models.llama3"
    ]
    assert module_names

    tokenizer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "HuggingFaceTokenizer"
    ]
    assert tokenizer_calls
    for call in tokenizer_calls:
        assert call.args == []
        assert any(keyword.arg == "tokenizer_path" for keyword in call.keywords)
