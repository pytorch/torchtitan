"""Run bootstrap and git lifecycle — the Harness's branch owner.

Enforces, structurally, that every autoresearch loop runs on a fresh isolated
branch and that the agent never touches git (ARCHITECTURE.md control planes). The
human triggers `start_run`; the harness owns commits. Candidates submit file
content (not commits), the harness applies+commits them to the isolated branch,
and resets to the champion on any non-promotion. Because the agent has no git
access through the API, fresh-branch / no-resume / no-cross-branch-read are facts,
not rules the agent must obey.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from torchtitan_autoresearch import workload_guard as wg
from torchtitan_autoresearch.constitution import Rules
from torchtitan_autoresearch.types import Candidate


class ProvenanceViolation(RuntimeError):
    """The run cannot start cleanly (branch exists / dirty tree / resume)."""


def _git(repo: str, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


@dataclass
class Session:
    repo_root: str  # where experiment git ops + runs happen (an isolated worktree)
    branch: str
    base_commit: str
    main_repo: str = ""  # the primary checkout (untouched); set when using a worktree

    def head(self) -> str:
        return _git(self.repo_root, "rev-parse", "HEAD")

    def remove(self) -> None:
        """Remove the experiment worktree (the branch + its commits are kept)."""
        if self.main_repo and os.path.abspath(self.repo_root) != os.path.abspath(self.main_repo):
            try:
                _git(self.main_repo, "worktree", "remove", "--force", self.repo_root)
            except RuntimeError:
                pass

    def _assert_on_branch(self) -> None:
        cur = _git(self.repo_root, "rev-parse", "--abbrev-ref", "HEAD")
        if cur != self.branch:
            raise ProvenanceViolation(
                f"expected to be on {self.branch} but on {cur}; refusing git op"
            )

    def commit_candidate(self, c: Candidate, rules: Rules) -> str:
        """Apply the candidate's file content to the branch and commit; return sha.

        A command-only candidate (no file edits) makes no commit and returns the
        current HEAD, since only the launch command differs from the champion.
        Out-of-scope file edits are refused before anything is written.
        """
        self._assert_on_branch()
        if not c.file_edits:
            return self.head()
        for path in c.file_edits:
            if not wg.in_scope(path, rules):
                raise wg.WorkloadViolation(f"file edit outside editable scope: {path}")
        for path, content in c.file_edits.items():
            full = os.path.join(self.repo_root, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(content)
            _git(self.repo_root, "add", path)
        _git(self.repo_root, "commit", "-m", c.label or "autoresearch candidate")
        return self.head()

    def reset_to(self, commit: str) -> None:
        """Discard the working commit and return the branch tip to ``commit``."""
        self._assert_on_branch()
        _git(self.repo_root, "reset", "--hard", commit)


def _branch_exists(repo: str, name: str) -> bool:
    out = subprocess.run(
        ["git", "-C", repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}"],
        capture_output=True, text=True,
    )
    return out.returncode == 0


def _tree_clean(repo: str) -> bool:
    return _git(repo, "status", "--porcelain") == ""


def start_run(
    repo_root: str,
    tag: str,
    rules: Rules,
    base_commit: str | None = None,
    worktree_path: str | None = None,
) -> Session:
    """Create a fresh isolated experiment branch. Refuses to resume.

    This is the only place a branch is created, and it is harness/human owned --
    never the agent. With ``worktree_path`` (the default for real runs), the
    experiment gets its own git worktree at that path: all its commits, resets,
    and training runs happen there, so the primary checkout is never touched and a
    live experiment can never collide with editing/committing the tooling. A
    pre-existing branch (unless the constitution allows resume) is a hard
    `ProvenanceViolation`. Without a worktree it falls back to the in-place
    checkout (used by tests).
    """
    branch = rules.branch_pattern.format(tag=tag)
    if _branch_exists(repo_root, branch) and not rules.allow_resume:
        raise ProvenanceViolation(
            f"branch {branch} already exists and resume is disabled; pick a new tag"
        )
    base = _git(repo_root, "rev-parse", base_commit or rules.base_commit)

    if worktree_path:
        if os.path.exists(worktree_path):
            raise ProvenanceViolation(f"worktree path already exists: {worktree_path}")
        # An isolated worktree needs no clean primary tree -- the primary is untouched.
        _git(repo_root, "worktree", "add", worktree_path, "-b", branch, base)
        return Session(repo_root=worktree_path, branch=branch, base_commit=base,
                       main_repo=repo_root)

    if not _tree_clean(repo_root):
        raise ProvenanceViolation("working tree is dirty; commit or stash before starting a run")
    _git(repo_root, "checkout", "-b", branch, base)
    return Session(repo_root=repo_root, branch=branch, base_commit=base, main_repo=repo_root)
