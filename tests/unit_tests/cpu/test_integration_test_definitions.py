# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from pathlib import Path

import pytest

from torchtitan.models.llama3.config_registry import llama3_debugmodel
from torchtitan_recipes.tests.features import llama3_debugmodel_hf_checkpoint_load
from torchtitan_recipes.tests.models import llama3_debugmodel_fsdp2_tp2_pp2

from tests.integration_tests import OverrideDefinitions, validate_fake_pg_compatibility
from tests.integration_tests.b200 import build_b200_tests_list
from tests.integration_tests.features import build_features_test_list
from tests.integration_tests.flux import build_flux_test_list
from tests.integration_tests.h100 import build_h100_tests_list
from tests.integration_tests.models import build_model_tests_list
from tests.integration_tests.run_tests import _parse_test_suites, run_single_test


def test_hf_checkpoint_load_path_comes_from_test_config(monkeypatch) -> None:
    test_output_dir = "/tmp/model_only_hf_checkpoint"
    monkeypatch.setenv("TORCHTITAN_TEST_OUTPUT_DIR", test_output_dir)

    config = llama3_debugmodel_hf_checkpoint_load()

    assert config.checkpoint.initial_load_path == (
        f"{test_output_dir}/hf_checkpoint/step-10/"
    )


def test_integration_run_exports_test_output_dir(monkeypatch, tmp_path: Path) -> None:
    captured_env = None

    def fake_run_cmd(cmd, timeout=None, env=None):
        nonlocal captured_env
        captured_env = env
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setattr("tests.integration_tests.run_tests._run_cmd", fake_run_cmd)
    test = OverrideDefinitions(
        configs=[llama3_debugmodel],
        test_name="output_dir_test",
        ngpu=1,
    )

    run_single_test(test, str(tmp_path))

    assert captured_env is not None
    assert captured_env["TORCHTITAN_TEST_OUTPUT_DIR"] == str(
        tmp_path / "output_dir_test"
    )


def test_llama3_pp_numerics_has_one_microbatch_per_stage() -> None:
    config = llama3_debugmodel_fsdp2_tp2_pp2()

    assert (
        config.parallelism.num_pp_microbatches
        >= config.parallelism.pipeline_parallel_degree
    )


def test_parse_multiple_integration_test_suites() -> None:
    assert _parse_test_suites("features,models,h100,b200") == (
        "features",
        "models",
        "h100",
        "b200",
    )


def test_h100_tests_are_registered_in_separate_suite() -> None:
    assert {test.test_name for test in build_h100_tests_list()} == {
        "2d_asynctp_compile",
        "deepseek_v3_fsdp+cp+tp+minimal_async_ep",
        "deepseek_v3_fsdp+cp+tp+minimal_async_ep+sdc_replay",
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


def test_b200_tests_are_registered_in_separate_suite() -> None:
    assert {test.test_name for test in build_b200_tests_list()} == {"kimi_k3_mm_fsdp"}
    assert "kimi_k3_mm_fsdp" not in {
        test.test_name for test in build_model_tests_list()
    }


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


def test_models_select_fake_and_real_pg_cases() -> None:
    model_tests = build_model_tests_list()
    fake_pg_model_tests = {
        test.test_name for test in model_tests if not test.use_real_pg
    }
    real_pg_model_tests = {test.test_name for test in model_tests if test.use_real_pg}

    assert {
        "deepseek_v3_fsdp+ep",
        "qwen3_moe_fsdp+tp+cp+ep_param_groups",
        "kimi_k2_5_muon_fsdp+ep",
        "muse_glimmer_text_fsdp",
        "muse_glimmer_mm_fsdp+tp+sp",
    } <= fake_pg_model_tests
    assert {"deepseek_v3_fsdp+cp+pp+ep"} <= real_pg_model_tests


def test_flux_fake_pg_filters_real_collective_cases() -> None:
    flux_tests = build_flux_test_list()
    fake_pg_tests = {test.test_name for test in flux_tests if not test.use_real_pg}

    assert fake_pg_tests == {"flux_fsdp+compile"}


@pytest.mark.parametrize(
    ("test_name", "incompatibility"),
    [
        ("checkpoint", "checkpointing"),
        ("pipeline_parallel", "pipeline parallelism"),
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

    test = OverrideDefinitions(configs=[llama3_debugmodel], test_name=test_name)

    with pytest.raises(ValueError, match=incompatibility):
        validate_fake_pg_compatibility(test, config)

    test.use_real_pg = True
    validate_fake_pg_compatibility(test, config)
