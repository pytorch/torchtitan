# GraphTrainer Board Workflow

The GitHub Project board is the single source of truth for what needs doing,
what's in flight, and what's waiting on review. Both humans and agents read
and write to the same board.

**Board**: https://github.com/orgs/pytorch/projects/161
**CLI access**: `gh project` commands (requires `read:project` + `project` scopes)

---

## Status Columns

| Status        | Meaning                                                        | Who moves items here       |
|---------------|----------------------------------------------------------------|----------------------------|
| **Backlog**   | Ideas, rough observations, not yet actionable                  | Human or nightly scout     |
| **Ready**     | Scoped and actionable — an agent can pick this up              | Human                      |
| **In Progress** | An agent or human is actively working on it                  | Agent (when starting work) |
| **Blocked**   | Work started but hit an external blocker                       | Agent or human             |
| **Need Review** | Code is written, PR is up, waiting for human review          | Agent (when PR is ready)   |
| **Done**      | Merged or resolved                                             | Human (after merge)        |

### Key rule: only humans move items to Ready and Done

Agents can propose items (as Backlog drafts) and advance them through
In Progress → Blocked → Need Review. But the human decides what's worth
doing (Backlog → Ready) and what's actually finished (Need Review → Done).

---

## Item Lifecycle

### 1. Item Creation

Items enter the board in one of three ways:

- **Human creates** a draft issue or real issue directly on the board.
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
1. Look at **Ready** items first — these are pre-approved work.
2. Pick the highest-priority Ready item (or the one the human points to).
3. Move it to **In Progress** before starting work.
4. If no Ready items exist, ask the human what to work on.

### 3. Doing the Work

While working, the agent:
- Creates a feature branch: `graph_trainer/<topic>/<YYYY-MM-DD>`
- Makes focused commits (one logical change per commit).
- If blocked by something external, moves the item to **Blocked** and
  adds a comment explaining what's blocking it.

### 4. Requesting Review

When the work is complete:
1. Push the branch and open a PR against `main`.
2. Link the PR to the board item.
3. Move the item to **Need Review**.
4. Leave a brief comment on the item summarizing what was done.

### 5. Completion

The human reviews the PR. After merge, the human moves the item to **Done**.
If changes are requested, the human moves it back to **In Progress** (or
the agent picks it up in the next session).

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
```bash
# Get the status field ID and option IDs
gh project field-list 161 --owner pytorch --format json

# Update status (use the actual field/option IDs)
gh project item-edit --project-id PVT_kwDOAUB9vs4BT6Cu \
    --id <ITEM_ID> \
    --field-id PVTSSF_lADOAUB9vs4BT6CuzhBFuQs \
    --single-select-option-id <OPTION_ID>
```

Status option IDs:
- Backlog: `f75ad846`
- Ready: `47fc9ee4`
- In Progress: `ea6e9bcb`
- Blocked: `98236657`
- Need Review: (check field-list — may have changed)
- Done: (check field-list — may have changed)

> **Note**: Option IDs can drift if columns are reordered on the web UI.
> Always verify with `gh project field-list` before scripting.

---

## Nightly Scout Integration

The [nightly scout](nightly.md) runs on a schedule and:
1. Reads the board to see what's already tracked (avoids duplicates).
2. Creates Backlog drafts for new findings from its report.
3. Picks up Ready items if any are tagged for automated fix.

The scout should **not** auto-promote its own items to Ready. That's a
human judgment call.

---

## Conventions

- **Branch naming**: `graph_trainer/<topic>/<YYYY-MM-DD>`
- **Commit prefix**: `[graph_trainer]` for all graph_trainer commits.
  Add `[self_improve]` if the commit came from the nightly scout.
- **One item = one PR** unless items are tightly coupled.
- **Link everything**: PR description should reference the board item,
  board item should link to the PR.
