# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config import ParallelismConfig


def test_parallelism_config_default_load_balancer() -> None:
    assert ParallelismConfig().context_parallel_load_balancer == "headtail"


def test_parallelism_config_accepts_none_load_balancer() -> None:
    config = ParallelismConfig(context_parallel_load_balancer=None)
    assert config.context_parallel_load_balancer is None


def test_parallelism_config_accepts_ptrr_when_cp_disabled() -> None:
    config = ParallelismConfig(
        context_parallel_degree=1,
        context_parallel_load_balancer="ptrr",
    )
    assert config.context_parallel_load_balancer == "ptrr"


def test_parallelism_config_rejects_unknown_load_balancer() -> None:
    with pytest.raises(
        ValueError,
        match=r"must be one of: None, 'headtail', 'ptrr' \(got 'foo'\)",
    ):
        ParallelismConfig(context_parallel_load_balancer="foo")


def test_parallelism_config_rejects_empty_string_load_balancer() -> None:
    with pytest.raises(ValueError, match="cannot be an empty string"):
        ParallelismConfig(context_parallel_load_balancer="")


def test_parallelism_config_rejects_unknown_load_balancer_when_cp_disabled() -> None:
    with pytest.raises(
        ValueError,
        match=r"must be one of: None, 'headtail', 'ptrr' \(got 'foo'\)",
    ):
        ParallelismConfig(
            context_parallel_degree=1,
            context_parallel_load_balancer="foo",
        )
