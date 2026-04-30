---
name: fix_ci
description: Fix CI bugs against a PR
disable-model-invocation: true
---

# Prerequisite

Use the `AskUserQuestion` tool to gather the inputs below. Batch the questions
into a single `AskUserQuestion` call when possible so the user sees all
selections at once instead of being prompted serially.

1. **PR link** — ask the user to paste the PR URL. This is free-form input, so
   ask it as a plain text question (not multiple-choice).
2. **Confirm HEAD matches the PR** — ask whether the current `HEAD` commit is
   the one already pushed to the PR. Offer the options:
   - "Yes, HEAD is the PR commit"
   - "No, let me check / sync first" (if chosen, stop and let the user sync)
3. **Fix-commit workflow** — ask how to land the fix, with these options:
   - "New commit + `git push`" (creates a follow-up commit on this branch)
   - "Amend HEAD + `ghstack submit`" (rewrites the PR commit; use when the PR
     was submitted via ghstack)
   - "Let me decide later" (skip; ask again before committing)
4. **Automation mode** — Claude Code can make mistakes. Make sure the user
   sees that caveat in the question text, then offer:
   - "Full auto" — Claude pushes/submits fixes without asking. By selecting
     this, the user authorizes Claude Code to run the loop end-to-end
     unattended.
   - "Semi-auto" — Claude does triage and prepares the fix, but step 6
     requires explicit user confirmation before each push or `ghstack
     submit`.

Do NOT pop these as separate free-form prompts — surface them as selectable
options via `AskUserQuestion` so the user can click through.


# Steps
You need to finish the Prerequisite section first. The Prerequisite answers
are gathered **once** at the top of this invocation; the loop below reuses
them — do not re-ask the user on every iteration. The whole point of this
skill is autonomous CI babysitting, so push/submit without asking once the
fix-commit workflow has been chosen up front.

1. **Load commit context.** If you don't already have it from the conversation
   or `auto memory`, run `git show HEAD` to understand what this PR changes.
   Skip if context is already loaded.

2. **Fetch CI status.** Use `gh pr checks <PR>` (and `gh run view <run-id>
   --log-failed` for failing runs). Categorize each job as `passed`,
   `failed`, `pending`, `skipped`, or `flaky/infra`.

3. **Branch on the categorization:**
   - No `failed` and no `pending` -> jump to step 7.
   - Any `failed` -> go to step 4 immediately. Do **not** wait on `pending`
     jobs; fix what's already broken in parallel.
   - Only `pending` remain -> `sleep 600` (10 min) in a single Bash call,
     then go back to step 2. Tell the user once before the first sleep so
     they know the skill is babysitting; don't reprint the same status on
     every tick.

4. **Triage failures.** For each failed job, decide whether this PR is the
   cause: compare against a recent green run on `main` for the same job, and
   map the error in the failing log to files/lines this PR touched. Mark each
   as `caused-by-this-PR`, `pre-existing`, `flaky`, `infra`, or
   `needs-pytorch-fix`. Print the triage table to the user.

   - If **none** are `caused-by-this-PR` and there are still `pending` jobs,
     `sleep 600` and go back to step 2.
   - If **none** are `caused-by-this-PR` and nothing is `pending`, jump to
     step 7 — surface the unrelated failures in the summary.

5. **Fix the PR-caused failures.** Investigate root cause per the project
   `CLAUDE.md` — do not paper over symptoms. If the fix must land in
   PyTorch (or another upstream repo), mark it `needs-pytorch-fix` and skip;
   don't try to work around it locally. Before committing:
   - Run `pre-commit run --all-files` and fix every issue.
   - For numerics-sensitive changes, follow the loss-comparison guidance in
     `CLAUDE.md`.

6. **Commit and update the PR using the workflow chosen in the prerequisite.**
   - **Full-auto mode:** push/submit without asking — the user authorized
     this up front and confirming each push defeats the skill's purpose.
   - **Semi-auto mode:** show the diff and the planned commit message, then
     ask the user to confirm via `AskUserQuestion` ("Push", "Amend the fix
     first", "Abort"). Only proceed on explicit confirmation.

   Then run the chosen workflow:
   - "New commit + `git push`": stage specific files (no `git add -A`),
     create a new commit, then `git push`.
   - "Amend HEAD + `ghstack submit`": `git commit --amend --no-edit`, then
     `ghstack submit` for this PR only if it's part of a stack.

   Tell the user what you pushed (commit SHA + one-line summary), then
   `sleep 60` to give CI a moment to register the new commit, and go back
   to step 2.

7. **Write a summary.** Include: PR link, final triage table, what was
   fixed and why, what was deferred (with reasons — e.g. `needs-pytorch-fix`,
   flaky, infra, pre-existing on main), and the new commit / ghstack
   revision link(s). Stop the loop here.
