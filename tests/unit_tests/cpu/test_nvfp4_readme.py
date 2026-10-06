# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path


def _nvfp4_readme() -> str:
    return (
        Path(__file__).parents[3] / "torchtitan/quantization/nvfp4/README.md"
    ).read_text()


def _llama3_8b_usage_example(readme: str) -> str:
    marker = "### Llama 3 8B Usage"
    assert marker in readme
    section = readme.split(marker, 1)[1]
    next_heading = section.find("\n### ")
    if next_heading != -1:
        section = section[:next_heading]
    fence = "```python\n"
    start = section.find(fence)
    assert start != -1
    start += len(fence)
    end = section.find("\n```", start)
    assert end != -1
    return section[start:end]


def test_nvfp4_readme_llama3_8b_usage_sets_lr_on_optim_optimizer():
    example = _llama3_8b_usage_example(_nvfp4_readme())
    assert "config.optim.optimizer.optimizers[0].lr" in example
    assert "config.optimizer" not in example
