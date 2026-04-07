# GraphTrainer AutoDev Workflow

The GitHub Project board is the single source of truth for what needs doing,
what's in flight, and what's waiting on review. Both developers and the
AutoDev agent read and write to the same board.

**Board**: https://github.com/orgs/pytorch/projects/161
**CLI access**: `gh project` commands (requires `read:project` + `project` scopes)

**AutoDev loop**: The agent runs in a continuous loop — poll the board for
actionable items, do the work, then wait 10 minutes before checking again.
If there is nothing to do, the agent waits and re-checks. The loop continues
until the developer stops the session.

---

## Status Columns

| Status          | Meaning                                                           | Who moves items here          |
|-----------------|-------------------------------------------------------------------|-------------------------------|
| **Backlog**   | Ideas, rough observations, not yet actionable                  | Developer or nightly scout    |
| **Ready**     | Scoped and actionable — an agent can pick this up              | Developer                     |
| **In Progress** | An agent or developer is actively working on it              | Agent (when starting work)    |
| **Blocked**   | Work started but hit an external blocker                       | Agent or developer            |
| **Need Review** | Code is written, branch is pushed, waiting for developer review | Agent (when work is ready) |
| **Done**      | Merged or resolved                                             | Developer (after merge)       |

### Key rule: only developers move items to Ready, Done, and out of Blocked

Agents can propose items (as Backlog drafts) and advance them through
In Progress → Blocked → Need Review. But the developer decides what's worth
doing (Backlog → Ready), what's actually finished (Need Review → Done),
and when a blocker is resolved (Blocked → In Progress).

---

## Item Lifecycle

### 1. Item Creation

Items enter the board in one of three ways:

- **Developer creates** a draft issue or real issue directly on the board.
- **Nightly scout** discovers something and creates a Backlog draft
  (see [nightly.md](nightly.md) §7 for the scout's implementation rules).
- **Agent proposes** during a work session — creates a Backlog draft for
  things discovered while working on something else.

Draft issues are fine for small items. Create a real `pytorch/torchtitan`
issue when:
- The item needs discussion or will take multiple sessions.
- It should be visible outside the board (e.g., linked from a PR).
- It has sub-issues or dependencies.

### 2. Agent Picks Up Work

At the start of a session, the agent reads the board:

```bash
gh project item-list 161 --owner pytorch --format json
```

The agent should:
1. **Refresh option IDs** — run `gh project field-list 161 --owner pytorch --format json`
   and parse the current Status option name→ID mappings. Never reuse IDs from
   a previous session or from this document.
2. Check **In Progress** items first — these may have review feedback
   that needs addressing (see §6).
3. Look at **Ready** items next — these are pre-approved new work.
4. Pick the highest-priority item (or the one the developer points to).
5. Move it to **In Progress** before starting work (using the fresh ID from step 1).
   Then verify with `gh project item-list` that it landed in the right column.
6. If nothing actionable exists, wait 10 minutes and re-check the board.

### 3. Doing the Work

The agent should spin up a **subagent with empty context** to do the actual
implementation work. This keeps the main agent's context clean for board
management and review coordination.

While working, the agent (or its subagent):
- Follows the development instructions in CLAUDE.md (root and graph_trainer).
- Creates a feature branch: `graph_trainer/<topic>`
- Makes focused commits (one logical change per commit).
- If blocked by something external, moves the item to **Blocked** and
  adds a comment explaining what's blocking it.
- Before requesting review, the agent self-reviews its own changes:
  read the full diff, check for correctness, style, and adherence to
  CLAUDE.md rules. Ensure the change has good test coverage.
  Fix any issues found before proceeding.

### 4. Requesting Review

When the work is complete and self-reviewed:
1. Push the branch but **do not open a PR** — the developer will review the
   branch and decide whether the work is worth a PR.
2. Move the item to **Need Review**.
3. Leave a brief comment on the item summarizing what was done and
   which branch to look at.

### 5. Developer Review

The developer reviews the branch and either:
- **Opens a PR, approves and merges** → moves the item to **Done**.
- **Requests changes** → moves the item back to **In Progress** with
  review comments (on the branch or board item).

### 6. Addressing Review Feedback

The agent learns about review feedback in two ways:

- **Board poll (default)**: At session start, the agent checks for items
  that are **In Progress** with review comments on the board item.
  These are prioritized over new Ready items.
- **Developer-initiated (urgent)**: The developer starts a session and directly
  points the agent to the branch or pastes the review comments.

When addressing feedback, the agent:
1. Reads all review comments on the board item (or PR if one exists).
2. Addresses each comment — fix the code or reply explaining why not.
3. Self-reviews the updated diff.
4. Pushes and moves the item back to **Need Review**.

---

## CLI Reference

### Read the board
```bash
gh project item-list 161 --owner pytorch --format json
```

### Create a draft issue on the board
```bash
gh project item-create 161 --owner pytorch --title "Title here" --body "Description"
```

### Update item status

**MANDATORY**: Before every status update, run `gh project field-list` to get
the current option IDs. Do NOT use cached/hardcoded IDs — they drift when
columns are reordered on the web UI. After updating, verify with
`gh project item-list` that the item landed in the expected column.

```bash
# Step 1: ALWAYS get fresh status field and option IDs
gh project field-list 161 --owner pytorch --format json
# Look for the "Status" field → its "options" array has current name→ID mappings

# Step 2: Update status using the IDs from step 1
gh project item-edit --project-id PVT_kwDOAUB9vs4BT6Cu \
    --id <ITEM_ID> \
    --field-id PVTSSF_lADOAUB9vs4BT6CuzhBFuQs \
    --single-select-option-id <OPTION_ID_FROM_STEP_1>

# Step 3: Verify the item moved to the right column
gh project item-list 161 --owner pytorch --format json
```

---

## Conventions

- **Branch naming**: `graph_trainer/<topic>`
- **Commit prefix**: `[graph_trainer]` for all graph_trainer commits.
  Add `[self_improve]` if the commit came from the nightly scout.
- **One item = one branch** unless items are tightly coupled.
- **Link everything**: board item should reference the branch name.
  If the developer opens a PR, link that too.
