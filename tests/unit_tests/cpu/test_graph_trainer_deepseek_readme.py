# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

_DEEPSEEK_COMMAND = (
    "MODULE=torchtitan_recipes.graph_trainer.deepseek_v3 "
    "CONFIG=graph_trainer_deepseek_v3_671b_dist_moe_bf16 ./run_train.sh"
)
_LLAMA_COMMAND = (
    "MODULE=torchtitan_recipes.graph_trainer.llama3 "
    "CONFIG=graph_trainer_llama3_8b ./run_train.sh"
)
_QWEN_COMMAND = (
    "MODULE=torchtitan_recipes.graph_trainer.qwen3 "
    "CONFIG=graph_trainer_qwen3_14b ./run_train.sh"
)


def _graph_trainer_readme() -> str:
    return (
        Path(__file__).parents[3] / "torchtitan/experiments/graph_trainer/README.md"
    ).read_text()


def _quick_start_section(readme: str) -> str:
    marker = "### Quick Start"
    assert marker in readme
    section = readme.split(marker, 1)[1]
    next_heading = section.find("\n### ")
    if next_heading != -1:
        section = section[:next_heading]
    return section


def _bash_commands(section: str) -> list[str]:
    commands: list[str] = []
    rest = section
    fence = "```bash\n"
    while True:
        start = rest.find(fence)
        if start == -1:
            break
        start += len(fence)
        end = rest.find("\n```", start)
        assert end != -1
        commands.append(rest[start:end].strip())
        rest = rest[end + len("\n```") :]
    return commands


def _deepseek_heading(section: str) -> str:
    headings = [
        line
        for line in section.splitlines()
        if line.startswith("####") and "DeepSeek" in line
    ]
    assert len(headings) == 1
    return headings[0]


def test_graph_trainer_deepseek_quick_start_names_production_recipe():
    section = _quick_start_section(_graph_trainer_readme())
    commands = _bash_commands(section)
    assert _DEEPSEEK_COMMAND in commands
    heading = _deepseek_heading(section)
    assert "671B" in heading
    assert "Dist-MoE" in heading
    assert "graph_trainer_deepseek_v3_16b" not in section
    assert _LLAMA_COMMAND in commands
    assert _QWEN_COMMAND in commands
