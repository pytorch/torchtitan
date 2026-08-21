# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.models.llama3.config_registry import llama3_debugmodel

from tests.integration_tests import OverrideDefinitions, validate_fake_pg_compatibility
from tests.integration_tests.features import build_features_test_list
from tests.integration_tests.flux import build_flux_test_list
from tests.integration_tests.h100 import build_h100_tests_list
from tests.integration_tests.models import build_model_tests_list
from tests.integration_tests.run_tests import _parse_test_suites


def test_parse_multiple_integration_test_suites() -> None:
    assert _parse_test_suites("features,models,h100") == (
        "features",
        "models",
        "h100",
    )


def test_h100_tests_are_registered_in_separate_suite() -> None:
    assert {test.test_name for test in build_h100_tests_list()} == {
        "2d_asynctp_compile",
        "deepseek_v3_fsdp+cp+tp+minimal_async_ep",
        "deepseek_v3_fsdp+hybridep+compile",
        "dist_gemm",
        "float8",
        "fsdp+tp+cp+compile+float8",
        "fsdp_symm_mem",
        "hsdp+cp+compile+float8",
        "qwen3_fsdp+deepep",
    }
    assert all(not hasattr(test, "use_h100") for test in build_features_test_list())
    assert all(not hasattr(test, "use_h100") for test in build_model_tests_list())


def test_specialized_moe_backends_have_ep_coverage() -> None:
    specialized_names = {
        "deepseek_v3_fsdp+cp+tp+minimal_async_ep",
        "deepseek_v3_fsdp+hybridep+compile",
        "qwen3_fsdp+deepep",
    }
    h100_model_tests = [
        test for test in build_h100_tests_list() if test.test_name in specialized_names
    ]

    for test in h100_model_tests:
        config = test.configs[0]()
        assert config.parallelism.expert_parallel_degree > 1


def test_models_do_not_reserve_canonical_real_pg_cases() -> None:
    model_tests = build_model_tests_list()
    fake_pg_model_tests = {
        test.test_name for test in model_tests if not test.use_real_pg
    }

    assert {
        "qwen3_moe_fsdp+tp+cp+ep_param_groups",
        "kimi_k2_5_muon_fsdp+ep",
        "muse_glimmer_mm_fsdp+tp+sp",
    } <= fake_pg_model_tests


def test_flux_fake_pg_filters_real_collective_cases() -> None:
    flux_tests = build_flux_test_list()
    fake_pg_tests = {test.test_name for test in flux_tests if not test.use_real_pg}

    assert fake_pg_tests == {"flux_fsdp+compile"}


@pytest.mark.parametrize(
    ("test_name", "incompatibility"),
    [
        ("checkpoint", "checkpointing"),
        ("pipeline_parallel", "pipeline parallelism"),
        ("explicit_backend", "comm.mode=torchcomms"),
        ("fsdp+varlen_attn+per_op_sac", "selective AC"),
    ],
)
def test_fake_pg_incompatible_test_requires_explicit_marker(
    test_name: str, incompatibility: str
) -> None:
    config = llama3_debugmodel()
    if test_name == "checkpoint":
        config.checkpoint.enable = True
    elif test_name == "pipeline_parallel":
        config.parallelism.pipeline_parallel_degree = 2
    elif test_name == "explicit_backend":
        config.comm.mode = "torchcomms"

    test = OverrideDefinitions(configs=[llama3_debugmodel], test_name=test_name)

    with pytest.raises(ValueError, match=incompatibility):
        validate_fake_pg_compatibility(test, config)

    test.use_real_pg = True
    validate_fake_pg_compatibility(test, config)
