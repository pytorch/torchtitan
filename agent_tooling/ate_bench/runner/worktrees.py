"""Git worktree helpers for isolating new-feature attempts.

Each new-feature attempt runs in its own detached worktree off ``main`` so that
attempts don't collide and the agent's full change can be diffed against ``main``
(paper B.3.2 judges the ``git diff`` against main).
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(repo: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout
    )


def create_worktree(repo: Path, path: Path, base: str = "main") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    res = _git(repo, "worktree", "add", "--detach", str(path), base)
    if res.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {res.stderr.strip()}")
    return path


def diff_vs_base(worktree: Path, base: str = "main") -> str:
    """Full diff of the agent's changes vs base, including new (untracked) files."""
    _git(worktree, "add", "-A")
    res = _git(worktree, "diff", "--cached", base)
    return res.stdout


def remove_worktree(repo: Path, path: Path) -> None:
    # Worktrees can end up "locked" (esp. on network mounts); a single --force won't
    # remove a locked worktree. Unlock, double-force, then fall back to rm + prune.
    _git(repo, "worktree", "unlock", str(path))
    res = _git(repo, "worktree", "remove", "--force", "--force", str(path))
    if res.returncode != 0:
        import shutil
        shutil.rmtree(path, ignore_errors=True)
        _git(repo, "worktree", "prune")
