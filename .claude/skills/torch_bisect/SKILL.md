---
name: torch_bisect
description: Bisect PyTorch commits to find the regression that breaks TorchTitan. Use when the user wants to bisect PyTorch or invokes /torch_bisect.
disable-model-invocation: true
---

# PyTorch Git Bisect for TorchTitan

You are performing a git bisect on the PyTorch repository to find the exact
commit that introduced a regression affecting TorchTitan. Follow each phase
below precisely.

## Phase 1: Gather Information

First, record the TorchTitan directory by running `pwd`. Store this absolute
path — you will need it throughout the bisect to switch back.

### Check for cached answers

Read the file `~/.claude/torchtitan_torch_bisect_cache.json`. If it exists and
contains valid JSON, show the cached values to the user and ask whether they
want to reuse them (with the option to override individual fields) or start
fresh.

### Collect inputs

Use AskUserQuestion to collect the following from the user. If cached values
are being reused, pre-fill them and only ask for overrides.

1. **PyTorch repo path** — Absolute path to their local PyTorch git checkout
   (e.g., `/data/users/chienchin/mywork/pytorch`).

2. **PyTorch build command** — The shell command to recompile PyTorch after
   each checkout (e.g., `python setup.py develop`).

3. **TorchTitan test command** — The command to run from the TorchTitan
   directory that reproduces the issue. Tell the user:
   > "This command must **exit 0 on a good PyTorch commit** and **exit non-zero
   > on a bad commit**. I will use only the exit code to judge good vs bad — I
   > will NOT analyze log output to determine the result."

4. **Known good commit** — Either:
   - A specific commit hash (e.g., `abc1234`), OR
   - A date when PyTorch was known to work (e.g., `Feb 15` or `2025-02-15`).

5. **Timeout policy** — If the test command exceeds the 10-minute timeout,
   should you automatically mark it as a **bad commit** (regressions can cause
   hangs), or **ask the user** each time a timeout occurs?

### Save answers

After collecting all answers, ask the user if they want to save them to
`~/.claude/torchtitan_torch_bisect_cache.json` for future runs. If yes, write
all 5 answers as JSON to that file.

## Phase 2: Setup

### Step 1: Fetch latest PyTorch

```bash
cd <pytorch_path> && git fetch origin
```

Use a timeout of 300000ms (5 minutes). If `git fetch` fails or times out, show
the error to the user and ask them to resolve it (e.g., network issues,
authentication). Do not retry automatically — wait for the user to confirm the
fix, then retry.

### Step 2: Check for existing bisect

```bash
cd <pytorch_path> && git bisect log 2>&1
```

If this succeeds (exit 0), a bisect is already in progress. Ask the user
whether to reset it with `git bisect reset` before continuing, or abort.

### Step 3: Resolve the good commit

**If the user provided a commit hash**, verify it exists:
```bash
cd <pytorch_path> && git log -1 --format="%H %s" <commit_hash>
```
If this fails, ask the user for a valid commit.

**If the user provided a date**, find the latest commit on or before that date:
```bash
cd <pytorch_path> && git log origin/main --before="<DATE>T23:59:59" --format="%H %s" -1
```
Show the resolved commit hash and message to the user and ask them to confirm
it looks correct before proceeding. If no commit is found, ask the user for a
different date or commit hash.

### Step 4: Confirm the bad commit

The bad commit is `origin/main`. Show the user its info:
```bash
cd <pytorch_path> && git log origin/main -1 --format="%H %s"
```

### Step 5: Preview the commit range

```bash
cd <pytorch_path> && git rev-list --count <good_commit>..origin/main
```

Report the total number of commits and the estimated bisect steps
(approximately ceil(log2(count))).

### Step 6: Start the bisect

```bash
cd <pytorch_path> && git bisect start && git bisect bad origin/main && git bisect good <good_commit>
```

This checks out the first commit to test and prints the bisect status.

## Phase 3: Bisect Loop

Initialize a step counter at 1. Repeat until git bisect reports the first bad
commit.

### Step A: Build PyTorch

Run the user's build command in the PyTorch repo:
```bash
cd <pytorch_path> && <build_command>
```
Use a timeout of 600000ms (10 minutes).

**If the build fails** (non-zero exit code):
1. Show the user the last 50 lines of output.
2. Ask the user to choose:
   - **Skip** — Run `git bisect skip` to skip this commit.
   - **Retry** — The user fixes something externally, then you retry the build.
   - **Abort** — Run `git bisect reset` and stop the entire bisect.
3. If "skip", check the bisect output for completion (same as Step C).

### Step B: Test in TorchTitan

Run the user's test command from the TorchTitan directory:
```bash
cd <torchtitan_path> && <test_command>
```
Use a timeout of 600000ms (10 minutes).

**If the command completes**: use its exit code:
- Exit 0 → good commit
- Non-zero → bad commit

**If the command times out**: follow the user's chosen timeout policy from
Phase 1 — either automatically mark as bad, or ask the user what to do.

Do NOT analyze the test output to determine pass/fail. Use ONLY the exit code.

### Step C: Record bisect result

Run the appropriate command in the PyTorch repo:
```bash
cd <pytorch_path> && git bisect good   # if exit code was 0
cd <pytorch_path> && git bisect bad    # if exit code was non-zero
```

Check the output:
- If it contains **"is the first bad commit"** → bisect is done, go to Phase 4.
- If it contains **"Bisecting:"** → bisect continues, go back to Step A.
- If it says it **cannot narrow further** (too many skips) → inform the user
  and suggest aborting with `git bisect reset`.

### Step D: Report progress

Print a status line:
```
Bisect step <N>: commit <short_hash> — <good|bad>. ~<remaining> revisions left (~<steps> steps).
```

Increment the step counter and loop back to Step A.

## Phase 4: Report and Cleanup

### Step 1: Capture the bisect log

```bash
cd <pytorch_path> && git bisect log
```

### Step 2: Show the offending commit

```bash
cd <pytorch_path> && git show --stat <first_bad_commit>
```

### Step 3: Extract the PR number and details

PyTorch merge commits typically reference a pull request in the commit message
with the pattern `(#NNNNN)`. Extract the PR number from the commit subject:

```bash
cd <pytorch_path> && git log -1 --format="%s" <first_bad_commit>
```

If a `(#NNNNN)` pattern is found, fetch the PR title, body (first 20 lines),
and URL using the GitHub CLI:

```bash
gh pr view <PR_NUMBER> --repo pytorch/pytorch --json title,body,url
```

If `gh` is not available or fails, construct the URL manually as
`https://github.com/pytorch/pytorch/pull/<PR_NUMBER>` and note that the title
and summary could not be fetched.

If no PR number is found in the commit message, skip this step and note that
no associated PR was identified.

### Step 4: Reset the bisect

```bash
cd <pytorch_path> && git bisect reset
```

### Step 5: Present the summary

Format your final report as:

```markdown
## Bisect Complete

**First bad commit:** `<full_hash>`
**Commit message:** <subject line>
**Author:** <author name>
**Date:** <commit date>

### Associated Pull Request
**PR:** #<number> — <PR title>
**Link:** https://github.com/pytorch/pytorch/pull/<number>
**Summary:** <first ~20 lines of PR body>

### Changed files
<output of git show --stat>

### Full bisect log
<output of git bisect log>
```

If no PR was found, omit the "Associated Pull Request" section.

Do NOT attempt to diagnose what the commit broke or suggest a fix. Your job is
solely to identify the commit.

## Important Rules

- **Always use absolute paths** when running Bash commands. The shell state does
  not persist between Bash tool calls.
- **Never analyze test output** to determine pass/fail. Use only the exit code.
- **If any unexpected error occurs** (disk full, permissions, segfault in git),
  show the full error to the user and ask how to proceed before taking any
  action.
- **Keep the user informed** of progress after every bisect step.
- **If the bisect gets stuck** (too many skips, git cannot narrow further),
  inform the user and suggest aborting with `git bisect reset`.
