---
name: graph_trainer_weekly_report
description: Generate an executive-summary weekly report for GraphTrainer PRs. Use when the user wants a weekly report or invokes /graph_trainer_weekly_report.
disable-model-invocation: true
---

# GraphTrainer Weekly Report Generator

Generate an executive-summary style weekly report of GraphTrainer activity
from the pytorch/torchtitan GitHub repo.

## Arguments

The user may optionally provide:
- **Number of days** to look back (default: 7). Example: `/graph_trainer_weekly_report 14`

## Phase 1: Preflight & Date Range

1. Run `gh auth status` first. If not authenticated, tell the user and stop.
2. Compute the start date as `today - N days` (default N=7).
3. Format as `YYYY-MM-DD` for GitHub search queries.
4. Tell the user: "Generating GraphTrainer weekly report for {start_date} to {today}..."

## Phase 2: Fetch PRs from GitHub

GraphTrainer work spans multiple naming conventions and related subsystems.
Run searches for **all** of the following terms in parallel, excluding drafts,
including both open and merged/closed PRs.

**Important:** Do NOT include `body` in the `--json` fields for list calls —
the list endpoint truncates long bodies, and we always re-fetch full
descriptions via `gh pr view` later. Including `body` here wastes bandwidth
and can mislead categorization with partial text.

### Required search terms (always run all of these)

1. `GraphTrainer` — primary naming convention
2. `graph_trainer` — underscore variant
3. `graph PP` — graph pipeline parallelism
4. `graph_pp` — underscore variant of graph PP
5. `autoparallel` — autoparallel integration with graph_trainer
6. `HybridEP` — hybrid expert parallelism (often graph_trainer-related)

```bash
# Run all 6 in parallel (as separate tool calls, not a for loop)
gh pr list --repo pytorch/torchtitan \
  --search "GraphTrainer in:title created:>={start_date} -is:draft" \
  --state all --limit 100 \
  --json number,title,state,mergedAt,createdAt,author,isDraft
# ...repeat for each term
```

### Catch recently merged PRs created before the date range

Some PRs are created weeks/months before they merge (e.g., long-running
stacked PR chains). To catch these, also search for PRs **merged** in the
date range regardless of creation date:

```bash
gh pr list --repo pytorch/torchtitan \
  --search "GraphTrainer in:title merged:>={start_date} -is:draft" \
  --state merged --limit 100 \
  --json number,title,state,mergedAt,createdAt,author,isDraft
# ...repeat for each term
```

### Discover PRs by file path (git log)

Title-based search misses PRs that touch graph_trainer code but don't mention
it in the title (e.g., MoE refactors, Full DTensor infra, ChunkedCELoss).
Use `git log` to find commits that modified files under the graph_trainer
directory, then extract PR numbers:

```bash
git log --since="{start_date}" --oneline \
  -- '**/graph_trainer/**' '**graph_trainer**' '**/graph_trainer*' \
  | grep -oP '#\K[0-9]+' | sort -un
```

For each PR number found that is NOT already in the deduplicated set, fetch
the PR details via `gh pr view` and add it to the set. Apply the same
draft/state filtering as the title-based results.

This step is critical — in a typical week it catches 5-10 additional PRs
that would otherwise be missed entirely.

Deduplicate by PR number across all results (title search + git log).
Some search terms may return zero results (e.g. "graph PP" when no graph
pipeline parallelism work happened that week) — this is expected, not an error.

### Fetch Full Descriptions

After deduplication, fetch the full description for every PR using
`gh pr view`. The list endpoint truncates bodies; the view endpoint does not.

Use `xargs -P` for concurrency — sequential `for` loops are the
bottleneck at 30+ PRs:

```bash
echo "$pr_numbers" | xargs -P 10 -I{} sh -c '
  echo "===PR {}==="
  gh pr view {} --repo pytorch/torchtitan \
    --json number,title,state,mergedAt,createdAt,author,body,isDraft
'
```

Do NOT fall back to sequential `for` loops inside a single Bash call.
If `xargs -P` is impractical for some reason, split the PR list into
groups and run each group as a **separate parallel tool call**, each
using `xargs -P` internally.

After deduplication, report the discovery breakdown to the user:
"Found {X} PRs via title search, {Y} additional via git log ({Z} total
after dedup)." This helps calibrate whether the search terms are
comprehensive enough.

Use the full body text for categorization and for writing the narrative
summaries in the report. This is critical for producing accurate, detailed
summaries — titles alone are not sufficient.

## Phase 3: Classify PRs

For each PR, classify as:
- **Merged** — `mergedAt` is non-empty. PRs that merged in the date range
  get full "What Landed" coverage **regardless of creation date**.
- **Closed** — state is CLOSED and not merged. If a closed PR's
  title/description matches another merged or open PR in the list, it was
  likely superseded — silently drop it. Only mention closed PRs that are
  genuinely noteworthy (e.g., a reland where the original is not in the list).
- **In Progress** — state is OPEN and `isDraft` is false. PRs created in
  the range but still open go here.

Drop any PRs where `isDraft` is true (safety net — the search should already
exclude them, but double-check).

### Scope: direct vs upstream

After classification, tag each PR as **direct** or **upstream**:

- **Direct** — title contains `GraphTrainer`, `graph_trainer`, `graph_pp`,
  or the PR's changed files are entirely/primarily under the graph_trainer
  experiment directory.
- **Upstream** — PR is primarily about core torchtitan infrastructure (MoE
  refactors, Full DTensor, ChunkedCELoss, model config changes, etc.) but
  was discovered via git log because it touched graph_trainer files.

Direct PRs get full narrative treatment in "What Landed." Upstream PRs
should be mentioned briefly — one sentence explaining what changed and why
it matters to graph_trainer — not given the same narrative depth. If there
are 3+ upstream PRs, group them in a dedicated "Upstream Changes Affecting
GraphTrainer" category rather than mixing them into direct categories.

## Phase 4: Categorize by Theme

Group PRs into thematic categories based on their titles and descriptions.
Use your judgment, but common categories include:

- **Activation Memory** — SAC, CPU offload, memory policy, recomputation
- **Compilation & Inductor** — inductor passes, compilation modes, make_fx
- **Parallelism & Communication** — TP, FSDP, EP, HybridEP, async TP, comm overlap
- **Quantization & Low-Precision** — float8, MXFP8, mixed precision
- **Testing & CI** — test fixes, CI workflow changes, deterministic tests
- **Debugging & Developer Experience** — logging, diagnostics, code annotations
- **Tracing** — FX tracing, tracer improvements, graph capture
- **Bug Fixes** — correctness fixes, gradient bugs
- **Cleanup & Maintenance** — removals, refactors, codeowners

Merge small categories (1-2 PRs) into the closest related category. Aim for
4-7 categories total. Don't force-fit — if a PR doesn't match any category,
put it in "Other."

**Paragraph size limit**: Each narrative paragraph should cover at most 4 PRs.
If a category has more than 4 PRs, either split it into sub-themes (e.g.
"Parallelism" → "HybridEP" + "Communication Overlap" + "FSDP Fixes") or
summarize the less important PRs in a trailing sentence ("Several smaller
fixes also landed: ...") rather than giving every PR equal narrative weight.

**Upstream PRs** (tagged in Phase 3) should be grouped in a dedicated
"Upstream Changes Affecting GraphTrainer" category if there are 3+, or
folded as brief mentions into related direct categories if fewer.

## Phase 5: Write the Report

Write the report to `graph_trainer_weekly_report_{YYYY_MM_DD}.md` in the repo
root directory, using today's date.

### Report Template

```markdown
# GraphTrainer Weekly Report — {Mon DD} – {Mon DD}, {YYYY}

**{N} PRs merged | {M} PRs in progress | {K} contributors**

---

## Executive Summary

{2-4 sentence high-level summary of the week's most important developments.
Focus on what changed architecturally or what's newly possible, not individual
PR details. Mention any bugs found/fixed that affect correctness.}

---

## What Landed

### {Category Name}
{1-3 sentence narrative paragraph explaining what happened in this area and why
it matters. Reference PRs inline as links.}

### {Category Name}
{...repeat for each category...}

---

## What's In Progress

| Area | PR | Description | Author | Created |
|------|----|-------------|--------|---------|
| {category} | [#{number}]({url}) | {short description} | {author} | {Mon DD} |
{...one row per open PR, sorted by creation date ascending (oldest first)...}

---

## Risks & Blockers

{List any notable blockers, upstream regressions, or temporarily disabled
features. If there are none, omit this section entirely.}

---

## Contributors

| Contributor | Merged | In Progress |
|-------------|--------|-------------|
| {author} | {count} | {count or —} |
{...sorted by total contribution count descending...}
```

### Writing Guidelines

- **Executive Summary**: Lead with the single most impactful change. Be specific
  about what's new ("async TP landed" not "parallelism improvements"). Mention
  correctness bugs if any were found. **Do not make comparative claims you
  cannot verify from the data** (e.g. "the largest improvement to date") —
  state the numbers and let the reader judge.
- **What Landed sections**: Write narrative paragraphs, not bullet lists. Each
  paragraph should explain the "so what" — why does this set of changes matter?
  Link PRs inline like `([#3118](url))`.
- **What's In Progress table**: Keep descriptions to ~10 words. This is a
  scannable reference, not a narrative. Sort by creation date ascending
  (oldest first) so long-running PRs are visible at the top. The Created
  column helps readers distinguish PRs that have been open for days from
  ones opened hours ago.
- **Risks & Blockers**: Only include genuine blockers or regressions. Don't
  manufacture risk. Omit the section if there's nothing notable.
- **Contributors table**: Use `—` (em dash) for zero counts. Sort by total
  (merged + in-progress) descending.

## Phase 6: Present to User

After writing the file, display:
1. The file path
2. The executive summary section (so the user sees the key points immediately)
3. The PR counts (merged / in-progress / contributors)

## Rules

- Always deduplicate PRs across the multiple search queries.
- Never include draft PRs.
- Closed-but-not-merged PRs should be silently dropped if superseded by another
  PR in the list. Only mention them if genuinely noteworthy (e.g., a reland
  where the original is not in the list).
- Always fetch full PR descriptions via `gh pr view` for each PR. Use the
  complete body text for categorization and narrative summaries.
- If `gh` CLI is not available or not authenticated, tell the user and stop.
  Check this at the very start (Phase 1), not after running searches.
- Write the report to the repo root, not inside `.claude/`.
