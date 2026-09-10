# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``parallelism.enable_fsdp_symm_mem`` must reach apply_fsdp_to_decoder.

Inspect source rather than importing the functions: several parallelize
modules pull model packages (and GPU-only stacks) through package ``__init__``.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PARALLELIZE_FILES = [
    "torchtitan/models/llama3/parallelize.py",
    "torchtitan/models/qwen3_5/parallelize.py",
    "torchtitan/models/kimi_k2_7/parallelize.py",
]


def _apply_fsdp_to_decoder_call(source: str) -> ast.Call:
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "apply_fsdp_to_decoder":
            return node
    raise AssertionError("apply_fsdp_to_decoder call not found")


def _is_parallelism_enable_fsdp_symm_mem(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "enable_fsdp_symm_mem"
        and isinstance(node.value, ast.Name)
        and node.value.id == "parallelism"
    )


@pytest.mark.parametrize("relpath", _PARALLELIZE_FILES)
def test_apply_fsdp_to_decoder_honors_enable_fsdp_symm_mem(relpath: str) -> None:
    call = _apply_fsdp_to_decoder_call((_REPO_ROOT / relpath).read_text())
    for kw in call.keywords:
        if kw.arg != "enable_symm_mem":
            continue
        assert _is_parallelism_enable_fsdp_symm_mem(kw.value), (
            f"{relpath}: enable_symm_mem must be parallelism.enable_fsdp_symm_mem"
        )
        return
    raise AssertionError(
        f"{relpath}: apply_fsdp_to_decoder is missing keyword enable_symm_mem"
    )
