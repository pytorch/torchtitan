#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flag newly added CUDA-only code that has a ``torch.accelerator`` equivalent.

torchtitan supports several accelerator backends, but it is easy to reach for
``torch.cuda`` or a hard-coded ``"cuda"`` device string out of habit. Such code
silently reduces the test matrix: a unit test pinned to CUDA is skipped
everywhere else, and a training path pinned to CUDA fails outright.

This check is a ratchet, not a cleanup. It only looks at lines *added* by the
change under review, so the CUDA references that already exist stay untouched
until someone ports them deliberately. It also only flags APIs with a genuine
portable equivalent; CUDA-specific functionality (CUDA graphs, SM capability
checks, ``torch.backends.cuda``) is never reported.

Usage::

    # Review lines added relative to a base revision (CI and pre-commit).
    python .github/scripts/check_accelerator_api.py --base-ref origin/main

    # Review whole files regardless of the diff (useful for one-off audits).
    python .github/scripts/check_accelerator_api.py --all torchtitan/

Escape hatch: when a CUDA-only reference is deliberate, annotate the line with
``# allow-cuda: <reason>``. The reason is mandatory so the exemption explains
itself to the next reader.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path


# ``torch.cuda`` members with a drop-in ``torch.accelerator`` counterpart. The
# value is the suggested replacement shown in the error message.
CUDA_API_REPLACEMENTS = {
    "is_available": "torch.accelerator.is_available()",
    "device_count": "torch.accelerator.device_count()",
    "current_device": "torch.accelerator.current_device_index()",
    "set_device": "torch.accelerator.set_device_index(...)",
    "synchronize": "torch.accelerator.synchronize()",
    "current_stream": "torch.accelerator.current_stream()",
    "set_stream": "torch.accelerator.set_stream(...)",
    "empty_cache": "torch.accelerator.empty_cache()",
    "memory_stats": "torch.accelerator.memory_stats()",
    "memory_allocated": "torch.accelerator.memory_allocated()",
    "memory_reserved": "torch.accelerator.memory_reserved()",
    "max_memory_allocated": "torch.accelerator.max_memory_allocated()",
    "max_memory_reserved": "torch.accelerator.max_memory_reserved()",
    "reset_peak_memory_stats": "torch.accelerator.reset_peak_memory_stats()",
    "reset_accumulated_memory_stats": (
        "torch.accelerator.reset_accumulated_memory_stats()"
    ),
    "mem_get_info": "torch.accelerator.get_memory_info()",
    "get_rng_state": "torch.accelerator.random.get_rng_state()",
    "get_rng_state_all": "torch.accelerator.random.get_rng_state_all()",
    "initial_seed": "torch.accelerator.random.initial_seed()",
}

# Kwargs whose value names a device, e.g. ``torch.randn(..., device="cuda")``.
DEVICE_KEYWORDS = frozenset({"device", "device_type", "map_location"})

# Calls whose first positional argument names a device, e.g. ``x.to("cuda")``.
DEVICE_POSITIONAL_CALLS = frozenset({"to", "device"})

ALLOW_PATTERN = re.compile(r"#\s*allow-cuda:\s*\S")
ALLOW_BARE_PATTERN = re.compile(r"#\s*allow-cuda\b")

ACCELERATOR_HINT = (
    "torch.accelerator.current_accelerator() selects the active backend; "
    "see torchtitan/tools/utils.py for device_type/device_module helpers."
)


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    col: int
    found: str
    suggestion: str

    def render(self) -> str:
        return (
            f"{self.path}:{self.line}:{self.col}: {self.found} is CUDA-only; "
            f"use {self.suggestion}"
        )


def _dotted_name(node: ast.expr) -> str | None:
    """Return the dotted source spelling of an attribute chain, if it is static.

    Args:
        node: Expression node to resolve, typically an ``ast.Attribute``.

    Returns:
        The dotted name (e.g. ``"torch.cuda.is_available"``), or ``None`` when
        the chain is not a plain sequence of names.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


class _CudaVisitor(ast.NodeVisitor):
    """Collect CUDA-only references that have a portable equivalent."""

    def __init__(self) -> None:
        self.violations: list[Violation] = []

    def _record(self, node: ast.expr, found: str, suggestion: str) -> None:
        self.violations.append(
            Violation(
                path="",
                line=node.lineno,
                col=node.col_offset + 1,
                found=found,
                suggestion=suggestion,
            )
        )

    # Method names are fixed by the ast.NodeVisitor dispatch protocol.
    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        dotted = _dotted_name(node)
        if dotted is not None and dotted.startswith("torch.cuda."):
            member = dotted[len("torch.cuda.") :]
            # Only the leaf member matters; nested access such as
            # torch.cuda.nvtx.range_push is left alone.
            suggestion = CUDA_API_REPLACEMENTS.get(member)
            if suggestion is not None:
                self._record(node, dotted, suggestion)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == "cuda":
                # Tensor/Module .cuda(), including the .cuda(index) form.
                self._record(
                    node,
                    ".cuda()",
                    ".to(torch.accelerator.current_accelerator())",
                )
            if func.attr in DEVICE_POSITIONAL_CALLS and node.args:
                self._check_device_value(node.args[0])

        if isinstance(func, ast.Attribute) or isinstance(func, ast.Name):
            for keyword in node.keywords:
                if keyword.arg in DEVICE_KEYWORDS:
                    self._check_device_value(keyword.value)

        self.generic_visit(node)

    def _check_device_value(self, node: ast.expr) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == "cuda" or node.value.startswith("cuda:"):
                self._record(
                    node,
                    f'"{node.value}"',
                    "torch.accelerator.current_accelerator()",
                )


def _comments(source: str) -> list[tuple[int, str, bool]]:
    """Return the real comment tokens in ``source``.

    Tokenizing rather than scanning raw text keeps prose inside string literals
    and docstrings (including this file's own documentation) from being read as
    an annotation.

    Args:
        source: Full file contents.

    Returns:
        Tuples of (1-based line, comment text, whether the comment is alone on
        its line).
    """
    found: list[tuple[int, str, bool]] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                own_line = not token.line[: token.start[1]].strip()
                found.append((token.start[0], token.string, own_line))
    except (tokenize.TokenError, SyntaxError):
        return []
    return found


def _allowed_lines(source: str) -> set[int]:
    """Return 1-based line numbers exempted by an ``# allow-cuda:`` comment.

    An annotation covers its own line and, when it sits on a line of its own,
    the following line. That lets multi-line calls be exempted from above.

    Args:
        source: Full file contents.

    Returns:
        The set of exempted line numbers.
    """
    allowed: set[int] = set()
    for line, text, own_line in _comments(source):
        if not ALLOW_PATTERN.search(text):
            continue
        allowed.add(line)
        if own_line:
            allowed.add(line + 1)
    return allowed


def _bare_allow_lines(source: str) -> list[int]:
    """Return line numbers using ``# allow-cuda`` without a reason.

    Args:
        source: Full file contents.

    Returns:
        Line numbers whose annotation is missing the mandatory reason.
    """
    return [
        line
        for line, text, _ in _comments(source)
        if ALLOW_BARE_PATTERN.search(text) and not ALLOW_PATTERN.search(text)
    ]


def added_lines(base_ref: str, paths: list[str]) -> dict[str, set[int]]:
    """Map each changed Python file to the line numbers the diff adds.

    Args:
        base_ref: Revision to diff against, e.g. ``origin/main``.
        paths: Optional pathspec limiting the diff.

    Returns:
        A mapping of file path to added 1-based line numbers.

    Raises:
        SystemExit: If git cannot resolve ``base_ref``.
    """
    merge_base = _merge_base(base_ref)
    command = ["git", "diff", "--no-color", "-U0", merge_base, "--", *paths]
    diff = subprocess.run(command, capture_output=True, text=True, check=False).stdout

    result: dict[str, set[int]] = {}
    current: str | None = None
    hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[len("+++ b/") :]
            if not current.endswith(".py"):
                current = None
            continue
        if current is None:
            continue
        match = hunk.match(line)
        if match:
            start = int(match.group(1))
            count = int(match.group(2) or 1)
            result.setdefault(current, set()).update(range(start, start + count))
    return {path: lines for path, lines in result.items() if lines}


def _merge_base(base_ref: str) -> str:
    """Resolve the merge base with ``base_ref``, falling back to the ref itself.

    Args:
        base_ref: Revision to diff against.

    Returns:
        A revision suitable for ``git diff``.

    Raises:
        SystemExit: If ``base_ref`` cannot be resolved at all.
    """
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", base_ref],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        sys.exit(
            f"check_accelerator_api: cannot resolve base ref {base_ref!r}. "
            "In CI, fetch the base branch first (actions/checkout fetch-depth)."
        )
    merge_base = subprocess.run(
        ["git", "merge-base", base_ref, "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return merge_base.stdout.strip() or base_ref


def check_file(path: Path, limit_to: set[int] | None) -> list[Violation]:
    """Report CUDA-only references in ``path``.

    Args:
        path: File to inspect.
        limit_to: When given, only violations on these 1-based lines are
            reported. ``None`` inspects the whole file.

    Returns:
        The violations found, sorted by position.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # check-ast already reports unparsable files; stay quiet here.
        return []

    visitor = _CudaVisitor()
    visitor.visit(tree)

    allowed = _allowed_lines(source)
    violations = [
        Violation(str(path), v.line, v.col, v.found, v.suggestion)
        for v in visitor.violations
        if v.line not in allowed and (limit_to is None or v.line in limit_to)
    ]

    for line in _bare_allow_lines(source):
        if limit_to is None or line in limit_to:
            violations.append(
                Violation(
                    str(path),
                    line,
                    1,
                    "# allow-cuda without a reason",
                    "# allow-cuda: <why this must stay CUDA-only>",
                )
            )

    return sorted(violations, key=lambda v: (v.line, v.col))


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` when no violations are found, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories to inspect (default: repository root).",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Revision to diff against when scoping to added lines.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Inspect whole files instead of only lines added by the diff.",
    )
    args = parser.parse_args(argv)

    violations: list[Violation] = []
    if args.all:
        for raw in args.paths:
            path = Path(raw)
            candidates = sorted(path.rglob("*.py")) if path.is_dir() else [path]
            for candidate in candidates:
                if candidate.suffix == ".py":
                    violations.extend(check_file(candidate, None))
    else:
        for path_str, lines in added_lines(args.base_ref, args.paths).items():
            violations.extend(check_file(Path(path_str), lines))

    if not violations:
        return 0

    print("Found CUDA-only code that has a torch.accelerator equivalent:\n")
    for violation in violations:
        print(f"  {violation.render()}")
    print(f"\n{ACCELERATOR_HINT}")
    print(
        "If the CUDA dependency is intentional, annotate the line with "
        "'# allow-cuda: <reason>'."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
