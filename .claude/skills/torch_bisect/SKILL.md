---
name: torch_bisect
description: Bisect PyTorch commits to find the regression that breaks TorchTitan. Use when the user wants to bisect PyTorch or invokes /torch_bisect.
disable-model-invocation: true
---

# PyTorch Git Bisect for TorchTitan

Bisect the PyTorch repo to find the commit that broke TorchTitan.
Always use absolute paths — shell state does not persist between Bash calls.

## Phase 1: Gather Information

Record the TorchTitan directory (`pwd`). Then check if
`~/.claude/torchtitan_torch_bisect_cache.json` exists — if so, show cached
values and let the user reuse or override them.

Collect via AskUserQuestion (pre-fill from cache if available):

1. **PyTorch repo path** — absolute path to local checkout
2. **Build command** — shell command to recompile PyTorch after each checkout
3. **Test command** — run from TorchTitan dir; must exit 0 on good commits,
   non-zero on bad. Judge ONLY by exit code, never by log content.
4. **Good commit** — a commit hash or a date (e.g., `Feb 15`)
5. **Timeout policy** — if test exceeds 10 min, auto-mark bad or ask each time?

After collecting, offer to save answers to the cache file.

## Phase 2: Setup

1. **Check GitHub connectivity**: run `curl -s -o /dev/null --connect-timeout 20 -w "%{http_code}" https://github.com`.
   If it fails or returns non-200, tell the user GitHub is not reachable and
   suggest checking internet connectivity or proxy settings. On Meta internal
   servers, suggest setting `https_proxy=http://fwdproxy:8080`. Do not proceed
   until connectivity is confirmed.
2. `cd <pytorch> && git fetch origin` (timeout 300000ms). On failure, ask user
   to fix (remind about proxy/fwdproxy if it looks like a network error).
3. Check `git bisect log 2>&1` — if a bisect is active, ask user to reset or abort.
4. Resolve the good commit:
   - Hash: verify with `git log -1 <hash>`
   - Date: `git log origin/main --before="<DATE>T23:59:59" --format="%H %s" -1`
     — show result and confirm with user.
5. Bad commit is `origin/main`. Show it to user.
6. Show `git rev-list --count <good>..origin/main` and estimated steps (~log2).
7. Run each bisect command individually so failures can be reported cleanly:
   - `git bisect start`
   - `git bisect bad origin/main`
   - `git bisect good <good>`
   If any step fails, show the error and run `git bisect reset` before asking
   the user how to proceed.

## Phase 3: Bisect Loop

Repeat until bisect finds the first bad commit:

**A. Build** — run build command in PyTorch dir (timeout 600000ms).
On failure: show last 50 lines, ask user to **Skip** / **Retry** / **Abort**.

**B. Test** — run test command in TorchTitan dir (timeout 600000ms).
Exit 0 = good, non-zero = bad. On timeout, follow user's policy from Phase 1.
Never analyze logs — use only exit code.

**C. Record** — `git bisect good` or `git bisect bad` in PyTorch dir.
- Output contains "is the first bad commit" → done, go to Phase 4.
- Output contains "Bisecting:" → continue loop.
- Cannot narrow further (too many skips) → tell user, suggest abort.

**D. Progress** — print step number, commit hash, good/bad, remaining count.

## Phase 4: Report and Cleanup

1. Capture `git bisect log` and `git show --stat <bad_commit>`.
2. Extract PR number from commit subject (`(#NNNNN)` pattern). If found, run
   `gh pr view <N> --repo pytorch/pytorch --json title,body,url` to get PR
   details. Fallback URL: `https://github.com/pytorch/pytorch/pull/<N>`.
3. `git bisect reset`.
4. Present summary:

```
## Bisect Complete
**First bad commit:** <hash>
**Commit message:** <subject>
**Author:** <author> | **Date:** <date>

### Associated Pull Request
**PR:** #<N> — <title>
**Link:** https://github.com/pytorch/pytorch/pull/<N>
**Summary:** <first ~20 lines of PR body>

### Changed files
<git show --stat output>

### Full bisect log
<git bisect log output>
```

Omit the PR section if no PR number was found. Do NOT diagnose root cause.

## Rules

- Always use absolute paths.
- Judge test results by exit code only — never analyze output.
- On unexpected errors, show the error and ask the user before proceeding.
- Keep the user informed after every bisect step.
