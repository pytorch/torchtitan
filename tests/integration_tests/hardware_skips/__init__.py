# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hardware-specific skip rules for integration tests.

The default rules live in the ``*.toml`` files next to this module, one file per
hardware backend. A hardware backend can register an additional file in two ways:

* an entry point in the ``torchtitan.hardware_skips`` group, loaded
  automatically;
* :func:`register_hardware_skips`, called from a test bootstrap.

All files use the same schema and are merged into one list of test-case groups.
Each top-level key names a hardware backend and its value is the list of test
names skipped on that hardware.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch

from torchtitan.tools import utils

HARDWARE_SKIPS_DIR = Path(__file__).parent


@dataclass(frozen=True)
class HardwareSkipTestCases:
    """The test cases skipped on one hardware backend from one source file."""

    skip_hardware: str
    tests: frozenset[str]


def _parse(text: str, source: str) -> list[HardwareSkipTestCases]:
    raw = tomllib.loads(text)
    testcases: list[HardwareSkipTestCases] = []
    for skip_hardware, tests in raw.items():
        if not isinstance(tests, list) or not all(
            isinstance(test, str) for test in tests
        ):
            raise ValueError(f"{source}: {skip_hardware} must be an array of strings")
        testcases.append(HardwareSkipTestCases(skip_hardware, frozenset(tests)))
    return testcases


_TESTCASES_TO_SKIP: list[HardwareSkipTestCases] = []


def register_hardware_skips(source: str | Path) -> None:
    """Register a hardware-skip file or directory.

    A directory registers every ``*.toml`` file in it. Registration must happen
    before the runner starts filtering tests.
    """
    path = Path(source)
    files = sorted(path.glob("*.toml")) if path.is_dir() else [path]
    for file in files:
        _TESTCASES_TO_SKIP.extend(
            _parse(file.read_text(encoding="utf-8"), str(file))
        )


def _import_hardware_skip_extensions() -> None:
    """Leverage the Python plugin mechanism to load out-of-tree hardware skips."""
    from importlib.metadata import entry_points

    group_name = "torchtitan.hardware_skips"
    hardware_skip_extensions = entry_points(group=group_name)

    for hardware_skip_extension in hardware_skip_extensions:
        try:
            entrypoint = hardware_skip_extension.load()
            entrypoint()
        except Exception as err:
            raise RuntimeError(
                f"Failed to load the hardware-skips extension: "
                f"{hardware_skip_extension.name}."
            ) from err


_loaded = False


def _load() -> None:
    global _loaded
    if _loaded:
        return
    register_hardware_skips(HARDWARE_SKIPS_DIR)
    _import_hardware_skip_extensions()
    _loaded = True


def _should_skip_current_hardware(skip_hardware: str) -> bool:
    if skip_hardware == "rocm":
        return utils.device_type == "cuda" and torch.version.hip is not None
    if skip_hardware == "cuda":
        return utils.device_type == "cuda" and torch.version.hip is None
    return utils.device_type == skip_hardware


def get_testcases_should_skip_on_current_hardware() -> set[str]:
    _load()
    testcases_to_skip: set[str] = set()
    for testcases in _TESTCASES_TO_SKIP:
        if _should_skip_current_hardware(testcases.skip_hardware):
            testcases_to_skip.update(testcases.tests)
    return testcases_to_skip
