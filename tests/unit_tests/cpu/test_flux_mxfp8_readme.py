# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path


def _flux_readme() -> str:
    return (Path(__file__).parents[3] / "torchtitan/models/flux/README.md").read_text()


def _custom_configuration_example(readme: str) -> str:
    marker = "### Custom Configuration"
    assert marker in readme
    section = readme.split(marker, 1)[1]
    next_heading = section.find("\n## ")
    if next_heading != -1:
        section = section[:next_heading]
    fence = "```python\n"
    start = section.find(fence)
    assert start != -1
    start += len(fence)
    end = section.find("\n```", start)
    assert end != -1
    return section[start:end]


def test_flux_mxfp8_readme_custom_example_uses_converters():
    readme = _flux_readme()
    example = _custom_configuration_example(readme)
    problems = []
    if "ModelConvertersContainer" in readme:
        problems.append("mentions ModelConvertersContainer")
    if "model_converters" in readme:
        problems.append("assigns model_converters")
    if "build_model_config" not in example or "converters=" not in example:
        problems.append(
            "custom example does not show build_model_config(..., converters=)"
        )
    assert not problems
