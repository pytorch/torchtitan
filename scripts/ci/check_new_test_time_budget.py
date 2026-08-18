#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


class TestDefinition:
    def __init__(
        self,
        *,
        class_names: tuple[str, ...],
        function_name: str,
        line: int,
        module_name: str,
        path: str,
    ) -> None:
        self.class_names = class_names
        self.function_name = function_name
        self.line = line
        self.module_name = module_name
        self.path = path

    @property
    def key(self) -> tuple[tuple[str, ...], str]:
        return self.class_names, self.function_name

    @property
    def classname(self) -> str:
        return ".".join((self.module_name, *self.class_names))


class TestDefinitionVisitor(ast.NodeVisitor):
    def __init__(self, *, module_name: str, path: str) -> None:
        self.class_names: list[str] = []
        self.definitions: dict[tuple[tuple[str, ...], str], TestDefinition] = {}
        self.module_name = module_name
        self.path = path

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.class_names.append(node.name)
        self.generic_visit(node)
        self.class_names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._record_test(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._record_test(node)

    def _record_test(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if not node.name.startswith("test_"):
            return
        definition = TestDefinition(
            class_names=tuple(self.class_names),
            function_name=node.name,
            line=node.lineno,
            module_name=self.module_name,
            path=self.path,
        )
        self.definitions[definition.key] = definition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--junit-xml", type=Path, required=True)
    parser.add_argument("--threshold-seconds", type=float, required=True)
    return parser.parse_args()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout


def module_name(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def test_definitions(
    source: str, *, path: str
) -> dict[tuple[tuple[str, ...], str], TestDefinition]:
    visitor = TestDefinitionVisitor(module_name=module_name(path), path=path)
    visitor.visit(ast.parse(source, filename=path))
    return visitor.definitions


def source_at_revision(revision: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout if result.returncode == 0 else None


def find_new_test_definitions(base_sha: str, head_sha: str) -> list[TestDefinition]:
    changed_paths = git(
        "diff",
        "--name-only",
        "--diff-filter=AM",
        f"{base_sha}...{head_sha}",
        "--",
        "tests",
    ).splitlines()
    definitions: list[TestDefinition] = []
    for path in changed_paths:
        if not path.endswith(".py"):
            continue
        head_source = source_at_revision(head_sha, path)
        if head_source is None:
            continue
        current = test_definitions(head_source, path=path)
        base_source = source_at_revision(base_sha, path)
        previous = (
            test_definitions(base_source, path=path) if base_source is not None else {}
        )
        definitions.extend(current[key] for key in current.keys() - previous.keys())
    return definitions


def testcase_matches(definition: TestDefinition, testcase: ET.Element) -> bool:
    name = testcase.attrib.get("name", "")
    return testcase.attrib.get("classname") == definition.classname and (
        name == definition.function_name
        or name.startswith(f"{definition.function_name}[")
    )


def format_nodeid(definition: TestDefinition, testcase: ET.Element) -> str:
    parts = [definition.path, *definition.class_names, testcase.attrib["name"]]
    return "::".join(parts)


def write_summary(lines: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path is None:
        return
    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write("\n".join(lines))
        summary.write("\n")


def main() -> int:
    args = parse_args()
    definitions = find_new_test_definitions(args.base_sha, args.head_sha)
    testcases = list(ET.parse(args.junit_xml).iterfind(".//testcase"))

    measured: list[tuple[float, TestDefinition, ET.Element]] = []
    for definition in definitions:
        for testcase in testcases:
            if testcase_matches(definition, testcase):
                measured.append(
                    (float(testcase.attrib.get("time", "0")), definition, testcase)
                )

    measured.sort(reverse=True, key=lambda result: result[0])
    total_seconds = sum(duration for duration, _, _ in measured)
    individual_violations = [
        result for result in measured if result[0] > args.threshold_seconds
    ]
    aggregate_violation = total_seconds > args.threshold_seconds

    status = "FAILED" if individual_violations or aggregate_violation else "PASSED"
    output = [
        f"New test time budget: {status}",
        "",
        f"P90 threshold: {args.threshold_seconds:.2f}s",
        f"New test cases measured: {len(measured)}",
        f"Total added test time: {total_seconds:.2f}s",
        "",
        "New test durations:",
    ]
    if measured:
        output.extend(
            f"  {duration:7.2f}s  {format_nodeid(definition, testcase)}"
            for duration, definition, testcase in measured
        )
    else:
        output.append("  None")

    if individual_violations or aggregate_violation:
        output.extend(["", "Violations:"])
    for duration, definition, testcase in individual_violations:
        nodeid = format_nodeid(definition, testcase)
        message = (
            f"New test {nodeid} took {duration:.2f}s, above the "
            f"P90 threshold of {args.threshold_seconds:.2f}s"
        )
        output.append(f"  {message}")
        print(
            f"::error file={definition.path},line={definition.line},title=Slow new test::{message}"
        )
    if aggregate_violation:
        message = (
            f"New tests added {total_seconds:.2f}s, above the "
            f"P90 threshold of {args.threshold_seconds:.2f}s"
        )
        output.append(f"  {message}")
        print(f"::error title=New test time budget exceeded::{message}")

    print("\n".join(output))
    write_summary(["## New test time budget", "", "```text", *output, "```"])
    return 1 if individual_violations or aggregate_violation else 0


if __name__ == "__main__":
    sys.exit(main())
