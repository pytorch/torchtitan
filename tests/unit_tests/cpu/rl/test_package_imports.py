# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import subprocess
import sys


def test_rl_package_import_does_not_require_vllm() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import torchtitan.rl; assert 'vllm' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_rl_package_keeps_vllm_public_api_names() -> None:
    import torchtitan.rl as rl

    assert rl.__all__ == ["VLLMModelWrapper", "register_to_vllm"]
