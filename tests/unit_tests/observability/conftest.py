# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared fixtures for observability tests."""

import pytest


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Provides a temp directory for JSONL output, cleaned up after test."""
    output_dir = tmp_path / "observability_test_output"
    output_dir.mkdir()
    return str(output_dir)
