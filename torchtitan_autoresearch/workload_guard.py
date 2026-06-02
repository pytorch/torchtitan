# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Admissibility: reject candidates that violate the constitution before any run.

This is step 1 of the pipeline (ARCHITECTURE.md section 5.3) and the cheap
pre-flight that protects the locked invariants and editable scope. It is purely a
command/diff inspection — no GPU. `seq_len`, dataset, and model-flavor changes are
called out specifically as `WorkloadViolation`s because they redefine the
workload (raw throughput rewards shorter sequences); other fixed-field or
out-of-scope edits are rejected the same way with a generic reason.
"""

from __future__ import annotations

from torchtitan_autoresearch.constitution import Rules
from torchtitan_autoresearch.types import Candidate

# Banned workload fields -> the CLI tokens that would change them.
_WORKLOAD_CLI = {
    "seq_len": ["--training.seq_len"],
    "dataset": ["--dataloader.dataset", "--dataloader"],
    "model_flavor": ["--config", "--module"],
}

# Environment variables a candidate may NOT set. The agent's ``env`` is otherwise
# unrestricted (NCCL_*, CUDA_*, etc. are a real runtime lever the faithfulness gate
# adjudicates like any other change). These keys are rejected because they would
# game the harness rather than the workload: the operational keys define the run
# itself (model/config/world size/import path), and the determinism keys govern the
# verify run's reproducibility -- the one thing the faithfulness gate cannot defend
# against if the candidate subverts it. Determinism is also matched by name pattern
# so a candidate cannot smuggle it through an unlisted variable.
_PROTECTED_ENV_KEYS = {
    "NGPU",
    "MODULE",
    "CONFIG",
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_LAUNCH_BLOCKING",
}
_PROTECTED_ENV_PATTERNS = ("SEED", "DETERMINISTIC")


def _protected_env_key(key: str) -> bool:
    up = key.upper()
    if up in _PROTECTED_ENV_KEYS:
        return True
    return any(pat in up for pat in _PROTECTED_ENV_PATTERNS)


class WorkloadViolation(ValueError):
    """A candidate changes a banned workload field (seq_len / dataset / flavor)."""


def editable_paths(rules: Rules) -> set[str]:
    """The set of repo files a candidate may edit (editable scope)."""
    return set(rules.editable_files) | {"torchtitan/models/qwen3/config_registry.py"}


def in_scope(path: str, rules: Rules) -> bool:
    return path in editable_paths(rules)


def _sets_field(command: list[str], prefixes: list[str]) -> bool:
    for tok in command:
        for p in prefixes:
            if tok == p or tok.startswith(p + "=") or tok.startswith(p + "."):
                return True
    return False


def _fixed_prefix(field: str) -> str:
    # dotted fields match exactly; section names match any subkey.
    return f"--{field}" if "." in field else f"--{field}."


def admissible(c: Candidate, rules: Rules) -> tuple[bool, str]:
    """Return (ok, reason). Reason is empty when admissible."""
    # 1. Banned workload fields (the redefinition guard).
    for fieldname in rules.banned_workload_fields:
        prefixes = _WORKLOAD_CLI.get(fieldname, [f"--{fieldname}"])
        if _sets_field(c.command, prefixes):
            return (
                False,
                f"WorkloadViolation: candidate changes banned field '{fieldname}'",
            )

    # 2. Other fixed fields may not be overridden on the command line.
    for fieldname in rules.fixed_fields:
        if fieldname == "model_flavor":
            continue  # covered by workload check above
        if _sets_field(c.command, [_fixed_prefix(fieldname).rstrip(".")]):
            return False, f"candidate overrides fixed field '{fieldname}'"

    # 3. Changed files must lie within editable scope.
    for path in c.changed_files:
        if not in_scope(path, rules):
            if any(path.startswith(lp) for lp in rules.locked_paths):
                return False, f"candidate edits locked path: {path}"
            return False, f"candidate edits a file outside editable scope: {path}"

    # 4. Environment overrides must not subvert the harness itself: the verify
    #    run's determinism/seed (which the faithfulness gate relies on) or the
    #    operational keys that define the run. Everything else is allowed.
    for key in c.env:
        if _protected_env_key(key):
            return False, (
                f"candidate sets harness-controlled env '{key}' "
                "(determinism/seed or run-defining key); not permitted"
            )

    return True, ""


def enforce(c: Candidate, rules: Rules) -> None:
    """Raise on inadmissibility (for direct callers); WorkloadViolation for seq/etc."""
    ok, reason = admissible(c, rules)
    if not ok:
        if reason.startswith("WorkloadViolation"):
            raise WorkloadViolation(reason)
        raise ValueError(reason)
