# GraphTrainer Nightly Self-Improvement Scout

This prompt is designed to be run nightly by Claude Code. Its purpose is NOT to
check if things are broken (CI does that). Its purpose is to discover
**opportunities and risks we haven't acted on yet**.

Run with:
```bash
claude -p "$(cat torchtitan/experiments/graph_trainer/.claude/nightly.md)"
```

---

## 0. Prerequisites

Ensure you are on the torchtitan `main` branch with the latest upstream
changes:

```bash
git checkout main
git pull origin main
```

### 0a. Read Prior Reports

Reports are published as comments on the tracking issue
[pytorch/torchtitan#2856](https://github.com/pytorch/torchtitan/issues/2856).
Fetch them with:

```bash
# Fetch all comments, filter to nightly reports from the past 7 days
gh api repos/pytorch/torchtitan/issues/2856/comments --paginate \
  --jq '.[] | select(.body | startswith("# Nightly Scout Report")) | {date: .created_at, body: .body}'
```

Read each report found. Build a mental summary of:

1. **Open action items** — items flagged in prior reports that have NOT been
   resolved in the codebase yet (i.e., the underlying code issue still exists).
   These will be re-investigated in subsequent sections rather than discovered
   from scratch.
2. **Previously "all clear" areas** — areas that were stable in prior reports.
   Give these a lighter check (just verify nothing changed).
3. **Recurring FYI items** — context items that appeared in multiple reports.
   Don't re-report these unless the situation has materially changed.

Record the date of the most recent prior report. This becomes the baseline for
`git log` ranges in Section 1 (use `--since="<last_report_date>"` instead of
a hardcoded window).

If no prior reports exist, proceed as if this is the first run.

### 0b. Check Prior Self-Improvement Commits

Check what the nightly scout has already attempted to fix:

```bash
# List local self-improve branches
git branch --list "graph_trainer/self_improve/*" --sort=-committerdate

# Check commits on each branch from the past 7 days
for branch in $(git branch --list "graph_trainer/self_improve/*" --sort=-committerdate); do
    echo "=== $branch ==="
    git log --oneline --since="7 days ago" "$branch" --grep="self_improve"
done

# Also check main
git log --oneline --since="7 days ago" main --grep="self_improve"
```

Items that were already committed by a prior run should not be re-attempted
unless the commit was reverted or the fix was incomplete.

---

## 1. Core Torchtitan Delta Review

Check what changed in core torchtitan since the **last report date** (from
Step 0a) that graph_trainer should know about. Not "did it break us" but "did
we miss an opportunity or fall behind?" If no prior report exists, use
`--since="1 day ago"`.

```
git log --since="<last_report_date>" --oneline -- \
    torchtitan/trainer.py \
    torchtitan/config/ \
    torchtitan/distributed/ \
    torchtitan/models/common/ \
    torchtitan/models/llama3/ \
    torchtitan/models/deepseek_v3/ \
    torchtitan/protocols/ \
    torchtitan/components/ \
    torchtitan/experiments/__init__.py
```

For each commit found, answer:
- Does this add a new API that graph_trainer could use instead of its own
  implementation? (e.g., a new utility in `torchtitan/distributed/` that
  replaces something graph_trainer hand-rolls)
- Does this change a signature or field that graph_trainer depends on?
  Key fragile surfaces:
  - `Trainer.Config` fields (dict-spread copy in `configs.py:to_graph_trainer_config`)
  - `Trainer.post_dataloading_process()` return tuple
  - `CompileConfig` fields (extended by `GraphTrainerCompileConfig`)
  - `FlexAttention.forward`, `MoE.forward` signatures (monkey-patched)
  - `ParallelDims` properties and `build_mesh()`
- Does this add a new model variant that graph_trainer should consider supporting?
- Does this unify code across models in a way that makes graph_trainer's
  per-model wrappers redundant?

Output a brief summary with action items (or "nothing actionable").

## 2. TODO Unblock Detection

Graph_trainer has known TODOs blocked on upstream work. **Discover them
dynamically** — do NOT rely on a static list.

### 2a. Collect all TODOs

```bash
grep -rn "TODO\|FIXME\|HACK\|XXX" torchtitan/experiments/graph_trainer/ --include="*.py"
```

For each TODO found, read the surrounding context (a few lines above and below)
to understand what it's blocked on.

### 2b. Cross-reference with prior reports

If a prior report (from Step 0a) already flagged a TODO as "still blocked" and
no upstream activity has occurred since then, do a quick re-check but don't
deep-dive. Just note "still blocked, no upstream change since YYYY-MM-DD."

For TODOs NOT mentioned in prior reports (i.e., newly added), investigate fully.

### 2c. Check upstream progress

For each blocked TODO, check if the blocker has been resolved:
- If a TODO references a PyTorch PR (e.g., `pytorch/pytorch/pull/NNNNN`),
  check if that PR has been merged:
  ```bash
  gh pr view NNNNN --repo pytorch/pytorch --json state,mergedAt
  ```
- If a TODO mentions a specific API or feature, grep the installed torch
  package for it.
- If a TODO names an owner, note that — just flag them as "owned by X."

## 3. Test & CI Coverage Gap Analysis

Check the health of CI workflows. Look for things we should be testing but aren't,
things we test locally but CI doesn't pick up.

**Prior report context:** If a coverage gap was already flagged in a prior
report and the test file hasn't changed since, don't re-investigate — just
carry it forward with "still open since YYYY-MM-DD." Focus fresh analysis on
newly added code paths and passes since the last report.

This section requires the `gh` CLI with appropriate permissions to access the
GitHub Actions API. If you encounter an error like "api.github.com has not been
allowlisted", skip the CI monitoring parts and note "skipped — gh API not
allowlisted" in the report.

### CI failure monitoring

Graph_trainer has two GitHub Actions workflows:
- `.github/workflows/integration_test_8gpu_graph_trainer.yaml` (A10 GPUs)
- `.github/workflows/integration_test_8gpu_graph_trainer_h100.yaml` (H100 GPUs)

For each workflow, fetch the 5 most recent scheduled runs on `main`:

```bash
gh run list --workflow "<workflow_name>" --branch main --event schedule --limit 5
```

If the most recent run succeeded, note "CI green" and move on.
If it failed, find the most recent **passing** run among the 5 (or note
"no recent success in last 5 runs").

For each failing workflow, pull the failed job logs:

```bash
gh run view <run_id> --log 2>&1
```

From the logs, extract:
1. **Which test failed** — look for lines matching
   `RuntimeError: N integration test(s) failed:` and the test name that follows,
   or pytest `FAILED` summary lines.
2. **The error message** — capture the root-cause exception (e.g.,
   `torch._dynamo.exc.UserError`, `AssertionError`, `RuntimeError`) and its
   message. Include a few lines of traceback context pointing to the
   graph_trainer source file and line number.

#### Identify PyTorch nightly regression range

Extract the installed PyTorch version from both the failing and last-passing
runs by searching for `Successfully installed.*torch-` in their logs.

Then, for each nightly version, download the wheel and extract the git commit
hash from `torch/version.py`:

```bash
pip download torch==<version> \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --no-deps -d /tmp/torch_wheel --python-version 3.12 --only-binary :all:

unzip -p /tmp/torch_wheel/torch-<version>-cp312-cp312-manylinux_2_28_x86_64.whl \
    torch/version.py | grep git_version
```

Report a table:

| Nightly | Status | PyTorch Commit |
|---------|--------|----------------|
| `dev<good_date>` | Pass | `<commit>` |
| `dev<bad_date>` | Fail | `<commit>` |

#### Generate local repro command

From the failed test's `Command:` line in the logs, produce a minimal local
repro command. Strip the following from the original command:
- `TORCH_TRACE=...` environment variable
- `--dump_folder ...` flag and its value
- `LOG_RANK=...` environment variable

The result should look like:

```bash
NGPU=<n> ./run_train.sh \
  --module <module> \
  --config <config> \
  [remaining flags...]
```

Include this repro command in the report under a "CI Failures" section.

### Tests missing from CI

Compare every `test_*.py` file under
`torchtitan/experiments/graph_trainer/tests/` against the pytest/torchrun
invocations in both workflow YAML files. Any test file that exists locally
but is not invoked by either workflow is a CI gap. Report each missing test
with a suggested one-liner to add to the appropriate workflow.

### Test coverage gaps

- Are there parallelism combinations that core torchtitan tests but
  graph_trainer's integration_tests.py doesn't? Compare:
  - `tests/integration_tests/features.py` and `tests/integration_tests/h100.py`
    vs `torchtitan/experiments/graph_trainer/tests/integration_tests.py`
- Are there new graph passes in
  `torchtitan/experiments/graph_trainer/passes.py` without corresponding
  tests in `torchtitan/experiments/graph_trainer/tests/test_passes.py`?
- Are there new code paths added to graph_trainer files that have no test
  exercising them? Check recent commits for untested additions.
- Are there model configs registered in graph_trainer's `config_registry.py`
  files that no test or integration test uses?

Output: a list of specific coverage gaps with suggested test descriptions.

## 4. Code Freshness & Technical Debt

Detect places where graph_trainer has drifted from how core torchtitan
now does things.

**Prior report context:** For items flagged as "all clear" or "signature-stable"
in prior reports, only re-check if the relevant upstream files have changed
since the last report date. For debt items already reported, verify whether
they've been addressed; if not, carry forward.

- **Stale monkey-patches**: Graph_trainer patches `FlexAttention.forward`,
  `MoE.forward`, `ExpertParallel._token_dispatch/_token_combine`. Check if
  the upstream signatures have changed, making our patches do unnecessary
  work or miss new parameters.
- **Private API usage**: `simple_fsdp.py` uses `DTensorSpec`,
  `redistribute_local_tensor`, `_StridedShard`. Check if public alternatives
  have appeared in recent PyTorch releases.
- **Duplicated logic**: Check if graph_trainer reimplements anything that
  core torchtitan now provides as a shared utility. Common areas:
  - Activation checkpointing policy (`passes.py` vs `torchtitan/distributed/activation_checkpoint.py`)
  - FSDP wrapping logic (`simple_fsdp.py` vs `torchtitan/distributed/fsdp.py`)
  - Model parallelization patterns (graph_trainer's `parallelize.py` vs core's)
- **Config drift**: Run `to_graph_trainer_config()` mentally — are there new
  fields in `Trainer.Config` that aren't being copied over or that need
  graph_trainer-specific handling?

Output: specific debt items with severity (blocking / should-fix / nice-to-have).

## 5. Documentation Freshness

Check if graph_trainer's docs match reality.

- Does `.claude/CLAUDE.md` reference correct file paths, test commands, and
  CLI flags? Run a spot-check: do the test files mentioned still exist? Do
  the benchmark commands still parse correctly?
- Does `README.md` match the current feature set and supported models?
- Are the run commands stored in Claude's memory file still correct?
- Are there new features or passes that aren't documented anywhere?

Output: specific inaccuracies found, or "docs are current."

## 6. Open Work Tracking

Check the state of in-flight work. Use the prior report's PR/issue state as a
baseline — highlight what **changed** (new PRs, newly stale PRs, recently
merged) rather than re-listing everything.

```bash
# PRs touching graph_trainer
gh pr list --search "graph_trainer" --state open --repo pytorch/torchtitan

# Recent merged PRs (last 7 days)
gh pr list --search "graph_trainer" --state merged --limit 10 --repo pytorch/torchtitan

# Issues mentioning graph_trainer
gh issue list --search "graph_trainer" --state open --repo pytorch/torchtitan
```

Flag:
- Open PRs that have been waiting >5 days without review
- Merged PRs in the last 24h that might need follow-up (doc updates, new tests)
- Issues that have been open >14 days without activity

---

## 7. Publish Report

Publish the report as a comment on the tracking issue
[pytorch/torchtitan#2856](https://github.com/pytorch/torchtitan/issues/2856).

```bash
gh issue comment 2856 --repo pytorch/torchtitan --body "$(cat <<'EOF'
# Nightly Scout Report — YYYY-MM-DD

## CI Failures
- **Workflow**: name — status (green / N of last 5 failed)
- **Failing test**: test name
- **Error**: root-cause exception and message
- **Regression range**: `dev<good>` (`<commit>`) → `dev<bad>` (`<commit>`)
- **Repro**: `NGPU=... ./run_train.sh ...`
(or "CI green, nothing to report")

## Action Items (new findings)
- [ ] [P0/P1/P2] Description — why, what file/area

## Carried Forward (from prior reports, re-investigated)
- [ ] [Pn] Description — first reported YYYY-MM-DD, status update

## Opportunities (things to consider)
- Description — potential benefit

## FYI (awareness, no action needed)
- Description

## Docs
- Inaccuracies found, or "docs are current"

## All Clear
- Areas with nothing to report
EOF
)"
```

Keep it short. If a section has nothing, say "nothing to report" and move on.
Don't repeat what CI already tells us. Focus on what a human developer wouldn't
notice without actively looking.

**Deduplication rules for the report:**
- An item is "new" if it was NOT in any prior report from Step 0a.
- An item is "carried forward" if it appeared in a prior report and the
  underlying issue still exists. Include a brief status update (e.g.,
  "no upstream change", "partial fix landed", "situation worsened").
- An FYI item that appeared in 3+ consecutive reports with no change should
  be dropped from the report entirely — it's noise at that point.

---

## 8. Implement Action Items

After writing the report, implement every action item from the report.

### 8a. Create a feature branch

Before making any code changes, create a dedicated branch off `main`:

```bash
git checkout -b graph_trainer/self_improve/YYYY-MM-DD
```

All commits in this section go on this branch. Do NOT commit to `main` directly.

### 8b. Implementation rules

- **One commit per action item.** Do not bundle multiple items into one commit.
- **Priority order.** Work through P0 items first, then P1, then P2.
- **Commit messages.** Each commit must have a clear, reviewable message:
  - Prefix with `[graph_trainer][self_improve]`
  - Explain *why* the change is needed, not just *what* changed
  - Reference the nightly report priority (e.g., "P0 fix" or "P1 coverage gap")
- **Scope discipline.** Each commit should touch only what is necessary for that
  action item. Do not sneak in unrelated cleanups or refactors.
- **Loop until done.** When you finish one item, move on to the next. Do not
  stop until all action items are committed.
- **Don't re-attempt carried-forward items.** If a "Carried Forward" item was
  already committed by a prior nightly run (detected in Step 0b) and that
  commit hasn't been reverted, skip it. Only re-attempt if the prior fix was
  incomplete or reverted.
- **New items only.** Focus implementation effort on items in the "Action Items
  (new findings)" section. Carried-forward items should only be implemented if
  the situation has materially changed (e.g., an upstream blocker was removed).

### 8c. Push the branch

After all commits are done, push the branch to origin:

```bash
git push origin graph_trainer/self_improve/YYYY-MM-DD
```

Do NOT open a PR — leave that for the human reviewer to decide.

### 8d. Update the report comment

Edit the report comment (from Section 7) to link each action item to its
fix commit on the pushed branch. Use the GitHub commit URL format:

```
- [x] [P0] Description — [fix](https://github.com/pytorch/torchtitan/commit/<sha>)
```

To edit the comment, find its ID and use:

```bash
# Find the comment ID for today's report
COMMENT_ID=$(gh api repos/pytorch/torchtitan/issues/2856/comments --paginate \
  --jq '.[] | select(.body | startswith("# Nightly Scout Report — YYYY-MM-DD")) | .id')

# Update the comment body with commit links
gh api repos/pytorch/torchtitan/issues/comments/$COMMENT_ID \
  --method PATCH --field body="<updated report body>"
```
