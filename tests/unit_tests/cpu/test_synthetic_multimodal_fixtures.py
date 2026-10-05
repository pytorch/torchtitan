# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
from pathlib import Path


FIXTURE_ROOT = Path(__file__).parents[2] / "assets" / "synthetic_multimodal"


def test_committed_synthetic_fixtures_match_generator_from_any_cwd(tmp_path):
    subprocess.run(
        [sys.executable, FIXTURE_ROOT / "generate.py", "--check"],
        check=True,
        cwd=tmp_path,
    )
