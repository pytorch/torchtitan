# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the integration-test hardware skips."""

import argparse
import importlib.metadata

import pytest

from tests.integration_tests import IntegrationTestDefinition, hardware_skips
from tests.integration_tests.hardware_skips import HardwareSkipTestCases
from tests.integration_tests.run_tests import _filter_tests

DEFAULT_FILE = """cuda = ["sdc_replay_mismatch", "muse_glimmer_text_fsdp"]
"""


@pytest.fixture(autouse=True)
def skip_environment(monkeypatch, tmp_path) -> None:
    """Give every test its own default file and a clean module state."""
    default_dir = tmp_path / "hardware_skips"
    default_dir.mkdir()
    (default_dir / "cuda.toml").write_text(DEFAULT_FILE, encoding="utf-8")
    monkeypatch.setattr(hardware_skips, "HARDWARE_SKIPS_DIR", default_dir)
    monkeypatch.setattr(hardware_skips, "_TESTCASES_TO_SKIP", [])
    monkeypatch.setattr(hardware_skips, "_loaded", False)
    monkeypatch.setattr(hardware_skips.utils, "device_type", "cuda")
    monkeypatch.setattr(hardware_skips.torch.version, "hip", None)
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: [],
    )


def test_parse_errors(tmp_path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('cuda = "not-a-list"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be an array of strings"):
        hardware_skips.register_hardware_skips(bad)

    bad_entry = tmp_path / "bad_entry.toml"
    bad_entry.write_text("cuda = [1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an array of strings"):
        hardware_skips.register_hardware_skips(bad_entry)


def test_get_testcases_should_skip_on_current_hardware(tmp_path, monkeypatch) -> None:
    declared_dir = tmp_path / "declared"
    declared_dir.mkdir()
    (declared_dir / "rocm.toml").write_text(
        'rocm = ["some_test"]\n', encoding="utf-8"
    )
    (declared_dir / "cuda.toml").write_text(
        'cuda = ["everywhere"]\n', encoding="utf-8"
    )
    hardware_skips.register_hardware_skips(declared_dir)

    monkeypatch.setattr(hardware_skips.utils, "device_type", "cuda")
    monkeypatch.setattr(hardware_skips.torch.version, "hip", "6.2")
    to_skip = hardware_skips.get_testcases_should_skip_on_current_hardware()
    assert "some_test" in to_skip
    assert "everywhere" not in to_skip
    assert "sdc_replay_mismatch" not in to_skip

    monkeypatch.setattr(hardware_skips.torch.version, "hip", None)
    to_skip = hardware_skips.get_testcases_should_skip_on_current_hardware()
    assert "everywhere" in to_skip
    assert "sdc_replay_mismatch" in to_skip
    assert "some_test" not in to_skip


def test_hardware_matching(monkeypatch) -> None:
    for skip_hardware, device_type, hip, expected in [
        ("cuda", "cuda", None, True),
        ("cuda", "cuda", "6.2", False),
        ("rocm", "cuda", "6.2", True),
        ("rocm", "cuda", None, False),
        ("rocm", "npu", None, False),
        ("xpu", "xpu", None, True),
        ("xpu", "cuda", None, False),
    ]:
        monkeypatch.setattr(hardware_skips.utils, "device_type", device_type)
        monkeypatch.setattr(hardware_skips.torch.version, "hip", hip)
        assert hardware_skips._should_skip_current_hardware(skip_hardware) is expected

    monkeypatch.setattr(hardware_skips, "_loaded", True)
    hardware_skips._TESTCASES_TO_SKIP.extend(
        [
            HardwareSkipTestCases("rocm", frozenset({"a"})),
            HardwareSkipTestCases("cuda", frozenset({"a", "b"})),
            HardwareSkipTestCases("xpu", frozenset({"c"})),
        ]
    )
    monkeypatch.setattr(hardware_skips.utils, "device_type", "cuda")
    monkeypatch.setattr(hardware_skips.torch.version, "hip", "6.2")
    assert hardware_skips.get_testcases_should_skip_on_current_hardware() == {"a"}
    monkeypatch.setattr(hardware_skips.torch.version, "hip", None)
    assert hardware_skips.get_testcases_should_skip_on_current_hardware() == {
        "a",
        "b",
    }


def test_entry_point_registration(monkeypatch, tmp_path) -> None:
    entry_file = tmp_path / "entry.toml"
    entry_file.write_text('cuda = ["entry_case"]\n', encoding="utf-8")

    class FakeEntryPoint:
        name = "fake"

        def load(self):
            return lambda: hardware_skips.register_hardware_skips(entry_file)

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: [FakeEntryPoint()],
    )
    hardware_skips._load()
    assert any(
        testcases.skip_hardware == "cuda"
        and "entry_case" in testcases.tests
        for testcases in hardware_skips._TESTCASES_TO_SKIP
    )

    class BrokenEntryPoint:
        name = "broken"

        def load(self):
            raise RuntimeError("boom")

    hardware_skips._TESTCASES_TO_SKIP.clear()
    hardware_skips._loaded = False
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group=None: [BrokenEntryPoint()],
    )
    with pytest.raises(RuntimeError, match="broken"):
        hardware_skips._load()


def test_runner_skips_by_test_name(tmp_path) -> None:
    declared = tmp_path / "declared.toml"
    declared.write_text('cuda = ["case"]\n', encoding="utf-8")
    hardware_skips.register_hardware_skips(declared)

    args = argparse.Namespace(ngpu=8, test_name="all")
    skipped_case = IntegrationTestDefinition(
        configs=[lambda: None], test_name="case", ngpu=1
    )
    runnable, skipped_ngpu, skipped_by_hardware = _filter_tests(args, [skipped_case])
    assert runnable == []
    assert skipped_ngpu == []
    assert skipped_by_hardware == [skipped_case]

    runnable_case = IntegrationTestDefinition(
        configs=[lambda: None], test_name="other", ngpu=1
    )
    runnable, skipped_ngpu, skipped_by_hardware = _filter_tests(args, [runnable_case])
    assert runnable == [runnable_case]
    assert skipped_ngpu == []
    assert skipped_by_hardware == []
