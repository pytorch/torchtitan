# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
# This file names the removed command so it can reject it. Once the test is
# tracked, git grep reports that name here; any other path is still a failure.
_THIS_TEST = "tests/unit_tests/cpu/test_train_entrypoint_docs.py"
_REMOVED_COMMAND = "torchtitan_train"
_TRAIN_MODULE = "python -m torchtitan.train"

_ENTRYPOINT_DOCS = (
    "torchtitan/config/OVERRIDE.md",
    "torchtitan_recipes/overrides/helion_rope.py",
)

_LLAMA3_8B_PREFIX = (
    f"{_TRAIN_MODULE} --module torchtitan_recipes.models.llama3 "
    "--config llama3_8b \\\n"
)
_JSON_OVERRIDE_COMMAND = (
    _LLAMA3_8B_PREFIX
    + "    --override 'my_pkg.triton_rope.triton_rope={\"block_size\": 256}'"
)
_FUSED_SWIGLU_COMMAND = (
    _LLAMA3_8B_PREFIX
    + "    --override torchtitan_recipes.overrides.fused_swiglu.fused_swiglu"
)
_DIST_GEMM_COMMAND = (
    f"{_TRAIN_MODULE} --module torchtitan_recipes.tests.models.llama3 "
    "--config llama3_debugmodel_dist_gemm \\\n"
    "    --override torchtitan_recipes.overrides.fused_swiglu.fused_swiglu"
)
_DEEPSEEK_COMMAND = (
    f"{_TRAIN_MODULE} --module torchtitan_recipes.models.deepseek_v3 "
    "--config deepseek_v3_671b \\\n"
    "    --override vendor_x.overrides.vendor_x_moe"
)
_HELION_COMMAND = (
    f"{_TRAIN_MODULE} ... --override "
    "torchtitan_recipes.overrides.helion_rope.helion_complex_rope"
)

_OVERRIDE_MD_SNIPPETS = (
    _JSON_OVERRIDE_COMMAND,
    _FUSED_SWIGLU_COMMAND,
    _DIST_GEMM_COMMAND,
    _DEEPSEEK_COMMAND,
)


def _tracked_files_containing(token: str) -> list[str]:
    completed = subprocess.run(
        ["git", "-c", "color.ui=false", "grep", "-l", "-F", "-I", "--", token],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 1:
        return []
    assert completed.returncode == 0, completed.stderr
    return sorted(line for line in completed.stdout.splitlines() if line)


def _assert_snippets(text: str, snippets: tuple[str, ...]) -> None:
    for snippet in snippets:
        assert "python -m torchtitan.train" in snippet
        assert "--override" in snippet
        assert snippet in text


def test_override_docs_invoke_train_module() -> None:
    hits = _tracked_files_containing(_REMOVED_COMMAND)
    allowed = set(_ENTRYPOINT_DOCS) | {_THIS_TEST}
    unexpected = [path for path in hits if path not in allowed]
    assert unexpected == []

    override_md = (_REPO_ROOT / _ENTRYPOINT_DOCS[0]).read_text(encoding="utf-8")
    helion_rope = (_REPO_ROOT / _ENTRYPOINT_DOCS[1]).read_text(encoding="utf-8")
    removed_lines = [
        line
        for text in (override_md, helion_rope)
        for line in text.splitlines()
        if _REMOVED_COMMAND in line
    ]
    assert removed_lines == []

    _assert_snippets(override_md, _OVERRIDE_MD_SNIPPETS)
    assert override_md.count(_JSON_OVERRIDE_COMMAND) == 2
    assert override_md.count(_TRAIN_MODULE) == 5
    _assert_snippets(helion_rope, (_HELION_COMMAND,))
    assert helion_rope.count(_TRAIN_MODULE) == 1
