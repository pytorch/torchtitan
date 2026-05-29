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
            return False, f"WorkloadViolation: candidate changes banned field '{fieldname}'"

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

    return True, ""


def enforce(c: Candidate, rules: Rules) -> None:
    """Raise on inadmissibility (for direct callers); WorkloadViolation for seq/etc."""
    ok, reason = admissible(c, rules)
    if not ok:
        if reason.startswith("WorkloadViolation"):
            raise WorkloadViolation(reason)
        raise ValueError(reason)
