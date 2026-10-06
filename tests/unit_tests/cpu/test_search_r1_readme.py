# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path


def _search_r1_readme() -> str:
    return (
        Path(__file__).parents[3] / "torchtitan/rl/examples/search_r1/README.md"
    ).read_text()


def _run_section(readme: str) -> str:
    marker = "## Run"
    assert marker in readme
    section = readme.split(marker, 1)[1]
    next_heading = section.find("\n## ")
    if next_heading != -1:
        section = section[:next_heading]
    return section


def _bash_fence(section: str) -> str:
    fence = "```bash\n"
    start = section.find(fence)
    assert start != -1
    start += len(fence)
    end = section.find("\n```", start)
    assert end != -1
    return section[start:end]


def test_search_r1_readme_run_uses_python_module():
    readme = _search_r1_readme()
    command = _bash_fence(_run_section(readme))
    assert "python -m torchtitan.rl.train" in command
    assert "python torchtitan/rl/train.py" not in command
    assert "e28796e0df801a5b5b30c34043cdd8bba14a6ea5" in readme
    assert "4b183e3aa26cce5da4689593ffd0cd419a32a32e" in readme
