# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path


def _deepseek_v3_readme() -> str:
    return (
        Path(__file__).parents[3] / "torchtitan/models/deepseek_v3/README.md"
    ).read_text()


def test_deepseek_v3_readme_training_commands():
    readme = _deepseek_v3_readme()
    lines = readme.splitlines()
    problems = []

    command_16b = (
        "MODULE=torchtitan_recipes.tests.models.deepseek_v3 CONFIG=deepseek_v3_16b"
    )
    if not any(command_16b in line for line in lines):
        problems.append(
            "16B command must contain "
            "MODULE=torchtitan_recipes.tests.models.deepseek_v3 CONFIG=deepseek_v3_16b"
        )
    for line in lines:
        if (
            "CONFIG=deepseek_v3_16b" in line
            and "torchtitan_recipes.models.deepseek_v3" in line
        ):
            problems.append(
                "CONFIG=deepseek_v3_16b is paired with "
                "torchtitan_recipes.models.deepseek_v3: " + line
            )

    command_671b = (
        "MODULE=torchtitan_recipes.models.deepseek_v3 CONFIG=deepseek_v3_671b"
    )
    if not any(command_671b in line for line in lines):
        problems.append(
            "671B command must stay "
            "MODULE=torchtitan_recipes.models.deepseek_v3 CONFIG=deepseek_v3_671b"
        )

    debug_lines = [line for line in lines if "CONFIG=deepseek_v3_debugmodel" in line]
    if not debug_lines:
        problems.append("missing deepseek_v3_debugmodel command")
    for line in debug_lines:
        if "torchtitan_recipes.tests.models.deepseek_v3" not in line:
            problems.append(
                "debug command must use torchtitan_recipes.tests.models.deepseek_v3: "
                + line
            )

    assert not problems
