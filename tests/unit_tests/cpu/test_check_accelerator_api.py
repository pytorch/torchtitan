# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "scripts"
    / "check_accelerator_api.py"
)


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_accelerator_api", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _check(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    return [v.found for v in checker.check_file(path, None)]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import torch\nx = torch.cuda.is_available()\n", ["torch.cuda.is_available"]),
        ("import torch\nx = torch.cuda.device_count()\n", ["torch.cuda.device_count"]),
        ("import torch\ntorch.cuda.empty_cache()\n", ["torch.cuda.empty_cache"]),
        ('import torch\nx = torch.randn(2, device="cuda")\n', ['"cuda"']),
        ('import torch\nx = torch.randn(2).to("cuda:0")\n', ['"cuda:0"']),
        ("x = model.cuda()\n", [".cuda()"]),
    ],
)
def test_flags_portable_cuda_apis(tmp_path, source, expected):
    assert _check(tmp_path, source) == expected


@pytest.mark.parametrize(
    "source",
    [
        # Already portable.
        "import torch\nx = torch.accelerator.is_available()\n",
        "import torch\nx = torch.randn(2, device=torch.accelerator.current_accelerator())\n",
        # Genuinely CUDA-specific: no portable equivalent.
        "import torch\ng = torch.cuda.CUDAGraph()\n",
        "import torch\nc = torch.cuda.get_device_capability()\n",
        "import torch\np = torch.backends.cuda.matmul.fp32_precision\n",
        "import torch\nr = torch.version.hip is not None\n",
        # Prose mentioning cuda must not trip the checker.
        'x = "the cuda backend"\n',
        "# torch.cuda.is_available() is the old spelling\nx = 1\n",
    ],
)
def test_ignores_portable_and_cuda_specific_code(tmp_path, source):
    assert _check(tmp_path, source) == []


def test_allow_cuda_annotation_exempts_line(tmp_path):
    source = (
        "import torch\n"
        'x = torch.randn(2, device="cuda")  # allow-cuda: B200-only kernel\n'
    )
    assert _check(tmp_path, source) == []


def test_allow_cuda_annotation_exempts_following_line(tmp_path):
    source = (
        "import torch\n"
        "# allow-cuda: B200-only kernel\n"
        'x = torch.randn(2, device="cuda")\n'
    )
    assert _check(tmp_path, source) == []


def test_allow_cuda_requires_a_reason(tmp_path):
    source = "import torch\nx = torch.cuda.is_available()  # allow-cuda\n"
    # A reasonless annotation grants no exemption, so the underlying violation
    # is still reported alongside the complaint about the annotation itself.
    assert sorted(_check(tmp_path, source)) == sorted(
        [
            "torch.cuda.is_available",
            "# allow-cuda without a reason",
        ]
    )


def test_limit_to_added_lines(tmp_path):
    source = (
        "import torch\n"
        "a = torch.cuda.is_available()\n"
        "b = torch.cuda.device_count()\n"
    )
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    # Only line 3 is "added", so the pre-existing line 2 stays unreported.
    found = [v.found for v in checker.check_file(path, {3})]
    assert found == ["torch.cuda.device_count"]


def test_syntax_error_is_not_reported(tmp_path):
    assert _check(tmp_path, "def broken(\n") == []
