# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import shlex
import sys
from pathlib import Path

import pytest

from tests.integration_tests.h100 import (
    build_h100_tests_list,
    deepseek_v3_debugmodel_hybridep_fsdp4_ep2,
)
from tests.integration_tests.run_tests import (
    _build_test_list,
    _filter_tests,
    _validate_test_names,
    main,
)


@pytest.fixture
def select_tests(monkeypatch, tmp_path):
    selected = []

    def record_test(test, *args, **kwargs):
        selected.append(test)

    monkeypatch.setattr(
        "tests.integration_tests.run_tests.run_single_test", record_test
    )

    def select(*args):
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_tests", str(tmp_path / "output"), "--no-parallel", *args],
        )
        main()
        return selected

    return select


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--test_name", "deepseek_v3_fsdp+hybridep+compile"),
        ("--exclude", "qwen3_fsdp+deepep,deepseek_v3_fsdp+hybridep+compile"),
        ("--test_name", "default"),
    ],
)
def test_unknown_selector_fails_before_running_tests(
    select_tests, capsys, tmp_path, option, value
) -> None:
    with pytest.raises(SystemExit) as error:
        select_tests("--test_suite", "h100", option, value)

    assert error.value.code == 2
    assert not (tmp_path / "output").exists()
    stderr = capsys.readouterr().err
    assert f"Unknown test name(s) for {option} in suite(s) 'h100'" in stderr
    assert value.split(",")[-1] in stderr
    assert "Available test names:" in stderr


def test_hybridep_selector_runs_current_recipe(select_tests) -> None:
    selected = select_tests(
        "--test_suite", "h100", "--test_name", "deepseek_v3_fsdp+hybridep"
    )

    assert len(selected) == 1
    assert selected[0].configs == [deepseek_v3_debugmodel_hybridep_fsdp4_ep2]


def test_selector_can_belong_to_one_of_multiple_suites(select_tests) -> None:
    selected = select_tests(
        "--test_suite", "features,h100", "--test_name", "deepseek_v3_fsdp+hybridep"
    )

    assert [test.test_name for test in selected] == ["deepseek_v3_fsdp+hybridep"]


def test_exclusions_can_belong_to_different_suites(select_tests) -> None:
    selected = select_tests(
        "--test_suite",
        "features,h100",
        "--exclude",
        "default, deepseek_v3_fsdp+hybridep",
    )

    names = {test.test_name for test in selected}
    assert "default" not in names
    assert "deepseek_v3_fsdp+hybridep" not in names
    assert {"2d_eager", "qwen3_fsdp+deepep"} <= names


@pytest.mark.parametrize("args", [("--gpu_arch_type", "rocm"), ("--ngpu", "1")])
def test_known_but_unavailable_test_can_still_be_skipped(select_tests, args) -> None:
    assert not select_tests(
        "--test_suite", "h100", "--test_name", "deepseek_v3_fsdp+hybridep", *args
    )


@pytest.mark.parametrize(
    "script_name",
    ["run_8xgpu_integration_tests.sh", "validate_release_gpu.sh"],
)
def test_h100_scripts_select_backends_after_matching_install(script_name) -> None:
    script_path = Path(__file__).resolve().parents[3] / ".github/scripts" / script_name
    script = script_path.read_text()
    if script_name == "validate_release_gpu.sh":
        script = script.split("run_h100_tests() {", 1)[1].split("\n}", 1)[0]

    tests = build_h100_tests_list()
    phases = {"base": [], "deepep_v2": [], "hybrid_ep": []}
    phase = "base"
    for line in script.replace("\\\n", "").splitlines():
        command = shlex.split(line, comments=True)
        if not command:
            continue
        if command[0] == "bash" and command[1].endswith("install_deepep_v2.sh"):
            phase = "deepep_v2"
        elif command == ["bash", "/install_deepep.sh"] or command == [
            "install_hybrid_ep"
        ]:
            phase = "hybrid_ep"
        if "tests.integration_tests.run_tests" not in command:
            continue
        args = _parse_script_selectors(command)
        assert args.test_suite == "h100"
        args.gpu_arch_type = "cuda"
        _validate_test_names(args, tests)
        runnable, skipped = _filter_tests(args, tests)
        assert not skipped
        assert runnable
        phases[phase].extend(test.test_name for test in runnable)

    assert phases["deepep_v2"] == ["qwen3_fsdp+deepep"]
    assert phases["hybrid_ep"] == ["deepseek_v3_fsdp+hybridep"]
    assert sorted(name for names in phases.values() for name in names) == sorted(
        test.test_name for test in tests if not test.disabled
    )


def _parse_script_selectors(command: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_suite")
    parser.add_argument("--test_name", default="all")
    parser.add_argument("--exclude")
    parser.add_argument("--ngpu")
    parser.add_argument("--execution_mode", default="real_pg")
    parser.add_argument("--gpu_arch_type", default="cuda")
    args, _ = parser.parse_known_args(command)
    if args.ngpu and args.ngpu.isdecimal():
        args.ngpu = int(args.ngpu)
    return args


def test_pytorch_ci_script_selectors_exist() -> None:
    script_path = (
        Path(__file__).resolve().parents[3] / "scripts/ci/pytorch_ci_test_runner.sh"
    )
    suites = []
    for line in script_path.read_text().replace("\\\n", "").splitlines():
        command = shlex.split(line, comments=True)
        if "tests.integration_tests.run_tests" not in command:
            continue
        args = _parse_script_selectors(command)
        _validate_test_names(args, _build_test_list(args.test_suite))
        suites.append(args.test_suite)

    assert suites == ["features", "models"]
