# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM inference calls parallelize_fn(..., skip_dp=True).

Inspect source signatures rather than importing the functions: several
parallelize modules pull model packages (and GPU-only stacks such as
attn_gym/triton) through package ``__init__``.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PARALLELIZE_FNS = [
    ("torchtitan/models/llama3/parallelize.py", "parallelize_llama"),
    ("torchtitan/models/deepseek_v3/parallelize.py", "parallelize_deepseekv3"),
    ("torchtitan/models/kimi_k2_7/parallelize.py", "parallelize_kimi_k2_5"),
    ("torchtitan/models/kimi_k3/parallelize.py", "parallelize_kimi_k3"),
    ("torchtitan/models/qwen3/parallelize.py", "parallelize_qwen3"),
]


def _function_def(source: str, fn_name: str) -> ast.FunctionDef:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return node
    raise AssertionError(f"{fn_name} not found")


@pytest.mark.parametrize("relpath, fn_name", _PARALLELIZE_FNS)
def test_parallelize_fn_accepts_skip_dp(relpath: str, fn_name: str) -> None:
    fn = _function_def((_REPO_ROOT / relpath).read_text(), fn_name)
    for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        if arg.arg != "skip_dp":
            continue
        assert default is not None
        assert ast.literal_eval(default) is False
        assert isinstance(arg.annotation, ast.Name) and arg.annotation.id == "bool"
        return
    raise AssertionError(f"{fn_name} is missing keyword-only skip_dp: bool = False")
