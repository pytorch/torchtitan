#!/usr/bin/env python3
"""
check_submodule_updates.py

Checks whether any git submodule pointer in this repo is behind its remote.
Can be imported and called from Python, or run directly as a script.
"""

import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional


def _run_git(*args: str, cwd: Optional[Path] = None) -> Optional[str]:
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_git_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """Find the git repository root."""
    root = _run_git("rev-parse", "--show-toplevel", cwd=start_path)
    return Path(root) if root else None


def get_submodule_paths(git_root: Path) -> List[str]:
    """Parse .gitmodules for submodule paths."""
    gitmodules = git_root / ".gitmodules"
    if not gitmodules.exists():
        return []

    output = _run_git(
        "config", "--file", ".gitmodules",
        "--get-regexp", r"^submodule\..*\.path$",
        cwd=git_root,
    )
    if not output:
        return []

    paths = []
    for line in output.splitlines():
        # Each line: "submodule.<name>.path <value>"
        parts = line.split(None, 1)
        if len(parts) == 2:
            paths.append(parts[1])
    return paths


def check_submodule(git_root: Path, submodule_path: str) -> Optional[Dict]:
    """
    Check a single submodule for upstream updates.

    Returns a dict with update info if the submodule is behind, else None.
    """
    full_path = git_root / submodule_path

    # Skip if not initialised
    if not (full_path / ".git").exists():
        return None

    # Determine tracked branch
    branch = _run_git("symbolic-ref", "--short", "HEAD", cwd=full_path)
    if not branch:
        branch = "HEAD"

    # Fetch latest from remote
    if _run_git("fetch", "--quiet", "--no-tags", "origin", cwd=full_path) is None:
        return None  # offline or fetch failed

    local_commit = _run_git("rev-parse", "HEAD", cwd=full_path)
    remote_commit = _run_git("rev-parse", f"origin/{branch}", cwd=full_path)

    if not local_commit or not remote_commit:
        return None

    if local_commit == remote_commit:
        return None

    behind_count = _run_git(
        "rev-list", "--count", f"HEAD..origin/{branch}", cwd=full_path
    )
    if not behind_count or int(behind_count) <= 0:
        return None

    return {
        "path": submodule_path,
        "branch": branch,
        "behind": int(behind_count),
        "local": local_commit[:10],
        "remote": remote_commit[:10],
    }


def check_all_submodules(start_path: Optional[Path] = None, quiet: bool = False) -> List[Dict]:
    """
    Check all submodules for upstream updates.

    Args:
        start_path: Any path inside the git repo (defaults to cwd).
        quiet: If True, suppress printed output.

    Returns:
        List of dicts for submodules that are behind their remote.
    """
    git_root = get_git_root(start_path)
    if git_root is None:
        if not quiet:
            print("Not inside a git repository — skipping submodule check.")
        return []

    submodule_paths = get_submodule_paths(git_root)
    if not submodule_paths:
        return []

    behind = []
    for sm_path in submodule_paths:
        info = check_submodule(git_root, sm_path)
        if info:
            behind.append(info)

    if not quiet and behind:
        print()
        print("===== Submodule Update Check =====")
        for info in behind:
            print(f"  ⚠  {info['path']} is {info['behind']} commit(s) behind origin/{info['branch']}")
            print(f"     Pinned:  {info['local']}")
            print(f"     Remote:  {info['remote']}")
        print()
        print("  To update all submodule pointers to latest remote:")
        print(f"    cd {git_root}")
        print('    git submodule update --remote --merge')
        print('    git add -A && git commit -m "Update submodule pointers"')
        print("====================================")
        print()

    return behind


if __name__ == "__main__":
    results = check_all_submodules()
    if not results:
        print("All submodules are up to date.")
