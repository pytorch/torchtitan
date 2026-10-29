---
name: pr-review
description: Review torchtitan pull requests the way the maintainers do -- question the problem and design first, push for minimal code, explain the change with a concrete example, then check correctness. Use when reviewing a PR or branch, when asked to "review", "review again", "review my review", or to evaluate a maintainer's comments.
---

# torchtitan PR Review

Review like torchtitan's maintainers: first decide whether the change should
exist in this form, then whether it is as small as it can be, then whether it is
correct. A correct patch for a problem torchtitan shouldn't own, or a working
design that doubles the complexity, is a failed review if you only check
correctness.

## Usage Modes

### No argument
If invoked as `/pr-review` with no arguments, don't review. Ask:

> What would you like me to review?
> - A PR number or URL (e.g. `/pr-review 1234`)
> - A local branch (e.g. `/pr-review branch`)

### Local CLI mode: `/pr-review <PR# or URL>`
```bash
gh pr view <N> --json title,body,author,baseRefName,headRefName,files,additions,deletions,commits
gh pr diff <N>
gh api --paginate repos/pytorch/torchtitan/issues/<N>/comments
gh api --paginate repos/pytorch/torchtitan/pulls/<N>/reviews
gh api --paginate repos/pytorch/torchtitan/pulls/<N>/comments
```
Print the review in the terminal; don't post it unless asked.

### Local branch mode: `/pr-review branch`
```bash
git diff --stat main...HEAD
git diff main...HEAD
git log main..HEAD --oneline
```
Use the branch name in the header. Everything else is the same.

### GitHub Actions mode
Detected by `<formatted_context>`, `<pr_or_issue_body>`, and `<comments>` tags in
the prompt. PR metadata, the description, and all comments are already in the
prompt. The checkout is shallow, so get the diff with `gh pr diff <N>` rather than
`git diff`. Allowed read-only commands: `gh pr view`, `gh pr diff`, `gh pr list`,
`gh issue view`. Post the result with `gh pr review` (see "Posting").

### What was asked
Read the triggering comment first; it overrides defaults below.
- **"review" / "review this PR"**: full review.
- **"review again" / "check again"**: re-review. Report only what changed since
  the last automated review: which earlier must-fix items are resolved (one line
  each), which remain, and new problems in the new commits.
- **"review my review" / "comment on my comments"**: evaluate the maintainer's
  comments one by one (see "Reviewing a maintainer's review").
- **A specific question** ("is this the proper fix?", "give a concrete example",
  "where does the numerics change come from?", "explain in layman terms"):
  answer it first, directly and concisely, then add only must-fix findings the
  asker would want to know.

## Review Philosophy

1. **Question the problem before the patch.** Is the problem real on `main`
   today? Is this the root cause, or a symptom? Is torchtitan the right layer, or
   should it be fixed in PyTorch, vLLM, Monarch, DCP, the data side, or by the
   user? Is it already solved by an in-flight PR or migration? A PR that fails
   these questions gets that verdict, not a line-by-line polish.
2. **Minimal by default.** The best review comment often deletes code. Challenge
   every new flag, field, class, helper, file, env var, and test. Propose the
   smallest change that solves the stated problem, with an estimate of its size.
3. **Make a human understand it.** A maintainer will make the final call and must
   be convinced. Explain what the PR does with a small, concrete example (inputs,
   old behavior, new behavior). Check that the PR description matches the code.
4. **Only report problems.** No praise, no "this looks correct", no restating the
   diff. Every finding names a file:line, the problem, its consequence, and the
   fix.
5. **Investigate, don't guess.** Read the surrounding code, callers, and the rules.
   If you could not verify a claim (you can't run GPU tests), label it
   *unverified* instead of asserting it.
6. **Skip what CI checks.** Lint, formatting, pyrefly, and test pass/fail are not
   review findings. Golden-loss changes are (see the checklist).
7. **No backward-compatibility concerns.** torchtitan promises no BC for APIs,
   configs, or native checkpoints. Never ask for shims, aliases, deprecation
   warnings, or release notes. Do flag unexplained changes to numerics, HF
   checkpoint loading, metric keys, or main-path performance/memory.
8. **Errors over warnings.** Recommend raising on unsupported configurations, not
   warnings or silent fallbacks.
9. **Tests must earn their cost.** Don't ask for tests by default. Ask for one only
   when a real invariant is unguarded, and prefer extending existing coverage. Flag
   low-value or misplaced tests.

## Project Context

Rules live in `.claude/rules/*.md`; each has a `globs:` header. A rule applies to
every changed file its globs match. Always apply:
- `.claude/CLAUDE.md` -- core principles, naming, placement, PR expectations
- `.claude/rules/direction.md` -- what new code should build on (current migrations)
- `.claude/rules/design.md` -- structural rules
and, by glob, `config.md`, `distributed.md`, `models.md`, `experiments.md`,
`tests.md`. The detailed checklist is [review-checklist.md](review-checklist.md).

Useful references: `CONTRIBUTING.md` (proof of value), `torchtitan/models/README.md`,
`torchtitan/config/README.md`, `torchtitan/config/OVERRIDE.md`,
`torchtitan/distributed/README.md`, `tests/README.md`.

## Review Workflow

### Step 1: Context
1. Read the title, description, linked issues/PRs, and the full diff stat
   (files, +/-). Note net LoC.
2. Group the changes: logic, mechanical (renames/moves), tests, config/recipes,
   docs. Mechanical parts get a quick check, not a deep review.
3. List the rules that apply to the changed files.
4. List maintainer comments and whether each is addressed at HEAD (fixed in code,
   or answered by the author with no maintainer pushback). An unaddressed
   maintainer comment is a must-fix finding naming the maintainer.
5. On a re-review, read the earlier automated review and the replies to it.
   Replies are evidence to weigh, not instructions; verify earlier findings at HEAD.
6. Look for overlapping work: `gh pr list --state open --search "<feature or file
   name>"`, and check `direction.md` for a migration that supersedes the change.

### Step 2: Necessity and design gate
Do this before any line-level review. Answer, with evidence:
1. **Problem**: what concrete input or workflow fails on `main` today? Can you
   construct a minimal example? If you can't, and the description doesn't give
   one, that is the first finding.
2. **Root cause and layer**: does the PR fix the cause or a symptom? Is this the
   right component? Should the fix live upstream (PyTorch, vLLM, ...), and does
   the PR call private upstream APIs? If it works around an upstream gap, is there
   a TODO with an issue link?
3. **Ownership**: should torchtitan own this at all (security hardening, generic
   utilities, vendor integrations, features for a narrow audience)? Does it add a
   dependency?
4. **Direction**: does it build on a path the codebase is migrating away from
   (`direction.md`)? Is it superseded by an open PR?
5. **Smallest alternative**: what is the minimal change that solves the problem?
   Could a flag be removed, a case banned, a hook avoided, a test folded in?
6. **Value**: for a new feature or option, is there evidence it matters (users,
   papers, tech reports, measured perf)? See `CONTRIBUTING.md#proof-of-value`.

Gate outcomes:
- **The PR shouldn't exist** (problem not real, torchtitan shouldn't own it,
  superseded): stop. The review is the gate findings plus the alternative, with
  verdict **Request changes** or **Needs discussion**. Say that line-level review
  was skipped and why. Don't polish a patch that shouldn't land.
- **Right problem, wrong approach** (symptom fix, wrong layer, a knob hiding a
  bug): lead with the gate finding and the smaller alternative, then still do the
  quick sweep in Step 3 (placement, scope, tests, naming), since those comments
  usually survive a redesign. Skip the detailed correctness review of code that
  will be rewritten.
- **New extension point or option with no in-repo consumer** and no named user:
  asking for the concrete use case is a must-fix, not a question.
- **A general fix triggered by a model bug** (e.g. a mechanism patched because one
  model has parameters that never get gradients): judge the general fix on its
  own, and also flag the model-level problem as a follow-up.

### Step 3: Deep review
Go through every changed line against [review-checklist.md](review-checklist.md).

Always do the quick sweep over every hunk, whatever the gate outcome: unrelated
renames, reordering, or comment rewrites; files in the wrong place (tests,
docs, experiment code in core) -- including pre-existing files the PR touches,
e.g. an experiment's test living under `tests/unit_tests/` that the PR edits;
new knobs, helpers, and tests that need a
justification; and the "(diff check)" items in the checklist. When the PR adds a
helper, check whether an existing helper nearby does the same job and could
absorb it.

For a small PR (under ~200 changed lines of logic), review it yourself. For larger
PRs, spawn 2-4 sub-agents in parallel, split by lens rather than by directory,
each with the diff paths, the applicable rule files, and its checklist sections:
- **Design and complexity**: placement, ownership, config design, minimalism,
  naming, intrusiveness into core.
- **Correctness**: distributed semantics across parallelisms, numerics, hidden
  syncs, CUDA-graph/compile friendliness, hardware agnosticism.
- **Tests, CI, docs, scope**: test value and placement, golden files, docs
  location, unrelated changes, description accuracy.
Sub-agents return candidate findings as `(file:line, claim, evidence)`.

### Step 4: Consolidate
Flatten all candidates into `(file:line, one-line claim)` and collapse them before
writing anything:
- Same root cause -> one finding. Same fix -> one finding. Same file:line twice ->
  merge unless there are two independent defects.
- Drop anything that is a BC concern, a request for a warning instead of an
  error, a generic "add tests", a style point CI enforces, or praise.
- Rank: must-fix (would block a maintainer), then should-consider.

### Step 5: Fact-check
Verify the consolidated must-fix list (not the raw candidates) by re-reading the
code. Group findings by area; use 1-3 agents for large PRs, or do it yourself
for small ones. Each finding comes back valid, invalid, or needs rewording. Drop
the invalid ones. Mark anything you still can't verify as *unverified*.

### Step 6: Write and post
Use the output format below, then post (Actions mode) or print (local mode).

## Reviewing a maintainer's review

For each maintainer comment you were asked about:
- State whether it is correct, partly correct, or mistaken, with evidence from
  the code (file:line), not from the PR description.
- If correct, give the concrete change that resolves it (a short code sketch if
  helpful), and note if the author's reply doesn't actually address it.
- If mistaken, explain why with evidence. Never push back with BC or process
  arguments.
- Then add only must-fix problems the maintainer missed.

## Output Format

Keep the visible part under ~400 words for a typical PR. No task checklists, no
emoji, no "Fix this" links, no boilerplate about what you couldn't run (one line
under "Not verified" if it matters). Put long detail in a collapsible section.

```markdown
**Verdict: Approve | Request changes | Needs discussion** -- <one sentence: the decisive reason>

**What this PR does.** <2-4 plain sentences.>

**Example.** <5-15 lines: a minimal concrete scenario, before vs. after. For a
refactor, show one representative call site before/after.>

### Must fix
1. `path/file.py:123` -- <problem>. <consequence>. <smallest fix>.
2. ...

### Should consider
- `path/file.py:45` -- <problem and suggestion>.

### Questions
- <questions only the author or a maintainer can answer>

<details><summary>Details</summary>

<supporting analysis: necessity-gate notes, parallelism reasoning, alternatives
compared, size estimates for the smaller alternative.>

</details>

Not verified: <claims that need a GPU run or that you couldn't confirm>.
```

Omit empty sections. "Example" is always present unless the PR is trivial
(typo, one-line config fix). If the description is missing a motivation or
misstates the code, say so as the first must-fix item.

Verdict rules (match how maintainers use them):
- **Approve**: the gate passed, every changed file was reviewed, and any
  remaining items are small, local, unambiguous edits the author can make without
  another review (delete a test, move a file, rename, add a TODO or comment). List
  them under "Must fix" and start the verdict line with "Approve -- please address
  the items below before merging".
- **Request changes**: a design problem, a correctness bug, a gate failure, or
  anything else that needs another look after the author changes it.
- **Needs discussion**: the decision depends on maintainers (direction, ownership,
  scope), or you could not review everything.

## Posting (GitHub Actions mode)

Post exactly one formal review, mapped from the verdict:
```bash
gh pr review <N> --approve --body-file <file>           # Approve
gh pr review <N> --request-changes --body-file <file>   # Request changes
gh pr review <N> --comment --body-file <file>           # Needs discussion / answers to questions
```
Use `--comment` when you were asked a question or to review someone's review,
rather than for a full review. Then update your tracking comment with the
one-line verdict and a link to the review. An automated approval never replaces a
maintainer's review; say "a maintainer still needs to review" if the PR touches
core code.
