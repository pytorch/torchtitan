# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the integration-test skip files."""

import argparse
import importlib.metadata

import pytest

from tests.integration_tests import IntegrationTestDefinition, skips
from tests.integration_tests.run_tests import _filter_tests
from tests.integration_tests.skips import SkipSection

CORE_SUITES = {"features", "models", "h100", "b200"}

DEFAULT_FILE = """[all]
features = ["sdc_replay_mismatch"]
models = ["muse_glimmer_text_fsdp"]
"""


@pytest.fixture(autouse=True)
def skip_environment(monkeypatch, tmp_path) -> None:
    """Give every test its own default file and a clean module state."""
    default = tmp_path / "skip_cases.toml"
    default.write_text(DEFAULT_FILE, encoding="utf-8")
    monkeypatch.setattr(skips, "SKIP_CASES_FILE", default)
    monkeypatch.setattr(skips, "_SECTIONS", [])
    monkeypatch.setattr(skips, "_OUTSIDE_SUITES", {})
    monkeypatch.setattr(skips, "_loaded", False)
    monkeypatch.setattr(skips, "_loading", False)
    monkeypatch.setenv("TORCHTITAN_SKIP_CASES_AUTOLOAD", "0")


def test_load_and_parse_errors(tmp_path) -> None:
    skips.validate_suite(None, set(), CORE_SUITES)

    bad = tmp_path / "bad.toml"
    bad.write_text('[all]\nfeatures = "not-a-list"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be an array of strings"):
        skips.register_skip_cases(bad)

    skips._SECTIONS.append(
        SkipSection(
            "all", {"graph_trainr": frozenset({"x"})}, str(skips.SKIP_CASES_FILE)
        )
    )
    with pytest.raises(ValueError, match="unknown suite"):
        skips.validate_suite(None, set(), CORE_SUITES)

    skips._SECTIONS[-1] = SkipSection(
        "all", {"b200": frozenset({"no_such_test"})}, str(skips.SKIP_CASES_FILE)
    )
    with pytest.raises(ValueError, match="unknown test"):
        skips.validate_suite("b200", set(), CORE_SUITES)


def test_suite_declaration(tmp_path) -> None:
    declared = tmp_path / "declared.toml"
    declared.write_text('[all]\nmy_suite = ["some_test"]\n', encoding="utf-8")
    skips.register_skip_cases(declared, suites={"my_suite"})
    skips.validate_suite(None, set(), CORE_SUITES)

    undeclared = tmp_path / "undeclared.toml"
    undeclared.write_text('[all]\nother_suite = ["some_test"]\n', encoding="utf-8")
    skips.register_skip_cases(undeclared)
    with pytest.raises(ValueError, match="unknown suite"):
        skips.validate_suite(None, set(), CORE_SUITES)


def test_section_matching(monkeypatch) -> None:
    for section, device_type, hip, expected in [
        ("all", "cuda", None, True),
        ("all", "xpu", None, True),
        ("cuda", "cuda", None, True),
        ("cuda", "cuda", "6.2", False),
        ("rocm", "cuda", "6.2", True),
        ("rocm", "cuda", None, False),
        ("rocm", "npu", None, False),
        ("xpu", "xpu", None, True),
        ("xpu", "cuda", None, False),
    ]:
        monkeypatch.setattr(skips.utils, "device_type", device_type)
        monkeypatch.setattr(skips.torch.version, "hip", hip)
        assert skips._section_applies(section) is expected

    monkeypatch.setattr(skips, "_loaded", True)
    skips._SECTIONS.extend(
        [
            SkipSection("all", {"features": frozenset({"a"})}, "default"),
            SkipSection("rocm", {"features": frozenset({"a", "b"})}, "default"),
            SkipSection("cuda", {"features": frozenset({"c"})}, "default"),
        ]
    )
    monkeypatch.setattr(skips.utils, "device_type", "cuda")
    monkeypatch.setattr(skips.torch.version, "hip", "6.2")
    assert skips.active_skips() == {
        ("features", "a"): "rocm",
        ("features", "b"): "rocm",
    }
    monkeypatch.setattr(skips.torch.version, "hip", None)
    assert skips.active_skips() == {
        ("features", "a"): "all",
        ("features", "c"): "cuda",
    }


def test_entry_point_registration(monkeypatch, tmp_path) -> None:
    entry_file = tmp_path / "entry.toml"
    entry_file.write_text('[all]\nfeatures = ["entry_case"]\n', encoding="utf-8")

    class FakeEntryPoint:
        name = "fake"

        def load(self):
            return lambda: skips.register_skip_cases(entry_file, suites={"features"})

    monkeypatch.setenv("TORCHTITAN_SKIP_CASES_AUTOLOAD", "1")
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: [FakeEntryPoint()],
    )
    skips._load()
    assert any(section.source == str(entry_file) for section in skips._SECTIONS)

    class BrokenEntryPoint:
        name = "broken"

        def load(self):
            raise RuntimeError("boom")

    skips._SECTIONS.clear()
    skips._loaded = False
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: [BrokenEntryPoint()],
    )
    with pytest.raises(RuntimeError, match="broken"):
        skips._load()


def test_runner_without_test_suite_does_not_load_skip_files(tmp_path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('[all]\nnot_a_suite = ["x"]\n', encoding="utf-8")
    skips.register_skip_cases(bad)

    definition = IntegrationTestDefinition(
        configs=[lambda: None], test_name="case", ngpu=1
    )
    args = argparse.Namespace(ngpu=8, test_name="all")
    runnable, skipped_ngpu, skipped_skips = _filter_tests(args, [definition])

    assert runnable == [definition]
    assert skipped_ngpu == []
    assert skipped_skips == []
