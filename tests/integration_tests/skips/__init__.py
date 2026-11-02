# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Skip cases for integration tests.

The default rules live in ``skip_cases.toml`` next to this module. A hardware
backend can register an additional file in three ways:

* an entry point in the ``torchtitan.skip_cases`` group, loaded automatically
  on import and disabled with ``TORCHTITAN_SKIP_CASES_AUTOLOAD=0``;
* :func:`register_skip_cases`, called from a test bootstrap;
* the ``TORCHTITAN_SKIP_CASES`` environment variable, an ``os.pathsep``-separated
  list of paths.

All files use the same schema and are merged into one set of sections.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from torchtitan.tools import utils

logger = logging.getLogger(__name__)

SKIP_CASES_FILE = Path(__file__).with_name("skip_cases.toml")


@dataclass(frozen=True)
class SkipSection:
    name: str
    skips: Mapping[str, frozenset[str]]
    source: str


def _parse(text: str, source: str) -> list[SkipSection]:
    raw = tomllib.loads(text)
    sections: list[SkipSection] = []
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise TypeError(f"{source}: section {name!r} must be a table")
        skips: dict[str, frozenset[str]] = {}
        for suite, tests in cfg.items():
            if not isinstance(tests, list) or not all(
                isinstance(test, str) for test in tests
            ):
                raise TypeError(
                    f"{source}: {name}.{suite} must be an array of strings"
                )
            skips[suite] = frozenset(tests)
        sections.append(SkipSection(name, skips, source))
    return sections


_SECTIONS: list[SkipSection] = []


def register_skip_cases(source: str | Path) -> None:
    """Register an additional skip-cases file with the same schema.

    Registration must happen before the runner starts filtering tests.
    """
    path = Path(source)
    _SECTIONS.extend(_parse(path.read_text(), str(path)))


def _import_skip_case_extensions() -> None:
    """Leverage the Python plugin mechanism to load out-of-the-tree skip cases."""
    from importlib.metadata import entry_points

    group_name = "torchtitan.skip_cases"
    skip_case_extensions = entry_points(group=group_name)

    for skip_case_extension in skip_case_extensions:
        try:
            entrypoint = skip_case_extension.load()
            entrypoint()
        except Exception as err:
            raise RuntimeError(
                f"Failed to load the skip-cases extension: "
                f"{skip_case_extension.name}. You can disable extension "
                "auto-loading with TORCHTITAN_SKIP_CASES_AUTOLOAD=0."
            ) from err


def _is_skip_case_autoload_enabled() -> bool:
    return os.getenv("TORCHTITAN_SKIP_CASES_AUTOLOAD", "1") == "1"


_SECTIONS.extend(_parse(SKIP_CASES_FILE.read_text(), str(SKIP_CASES_FILE)))
for _path in os.environ.get("TORCHTITAN_SKIP_CASES", "").split(os.pathsep):
    if _path.strip():
        register_skip_cases(_path.strip())
if _is_skip_case_autoload_enabled():
    _import_skip_case_extensions()


def _section_applies(name: str) -> bool:
    if name == "all":
        return True
    if name == "rocm":
        return utils.device_type == "cuda" and torch.version.hip is not None
    if name == "cuda":
        return utils.device_type == "cuda" and torch.version.hip is None
    return utils.device_type == name


def validate_suite(suite: str, known: set[str]) -> None:
    """Reject entries that reference unknown suites or test names."""
    from tests.integration_tests.run_tests import _TEST_SUITES

    for section in _SECTIONS:
        for declared_suite, tests in section.skips.items():
            if declared_suite not in _TEST_SUITES:
                raise ValueError(
                    f"{section.source}: section {section.name!r} references "
                    f"unknown suite {declared_suite!r}"
                )
            if declared_suite != suite:
                continue
            unknown = tests - known
            if unknown:
                raise ValueError(
                    f"{section.source}: section {section.name!r}, suite "
                    f"{suite!r}: unknown test(s) {sorted(unknown)}"
                )


def active_skips() -> dict[tuple[str, str], str]:
    """Map (suite, test_name) to the section that applies on this machine."""
    active: dict[tuple[str, str], str] = {}
    for section in _SECTIONS:
        if not _section_applies(section.name):
            continue
        for suite, tests in section.skips.items():
            for test_name in tests:
                active[(suite, test_name)] = section.name
    return active
