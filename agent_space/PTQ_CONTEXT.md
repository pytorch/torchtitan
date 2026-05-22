# PTQ Job Context

This directory is a PTQ-managed job home for `orchestrator-3409`.

## Paths

- Job ID: `20260520-torchtitan-3409`
- Job directory: `~/.ptq_workspace/jobs/20260520-torchtitan-3409`
- Source worktree: `~/.ptq_workspace/jobs/20260520-torchtitan-3409/torchtitan`
- PyTorch support worktree: `~/.ptq_workspace/jobs/20260520-torchtitan-3409/pytorch`
- Python/venv: `~/.ptq_workspace/jobs/20260520-torchtitan-3409/.venv/bin/python`
- Artifacts: `~/.ptq_workspace/jobs/20260520-torchtitan-3409`

## Enter the PTQ job home

```bash
cd ~/.ptq_workspace/jobs/20260520-torchtitan-3409 && source .venv/bin/activate
```

The job source worktrees are under this directory.

## Source and environment rules

- Edit source in `~/.ptq_workspace/jobs/20260520-torchtitan-3409/torchtitan`.
- If the root cause is in PyTorch, edit `~/.ptq_workspace/jobs/20260520-torchtitan-3409/pytorch`. Do not edit `~/.ptq_workspace/pytorch` or another PyTorch checkout.
- Use `~/.ptq_workspace/jobs/20260520-torchtitan-3409/.venv/bin/python` for Python commands.
- Write scratch files and reports under `~/.ptq_workspace/jobs/20260520-torchtitan-3409` or `~/.ptq_workspace/jobs/20260520-torchtitan-3409/torchtitan/agent_space`.
- For PyTorch C++ changes, rebuild with:

```bash
bash ~/.ptq_workspace/scripts/rebuild.sh ~/.ptq_workspace/jobs/20260520-torchtitan-3409/pytorch
```

## PTQ commands

Run these from the PTQ repo, usually `~/meta/pt_job_queue`.

```bash
uv run ptq list
uv run ptq takeover 20260520-torchtitan-3409
uv run ptq run 20260520-torchtitan-3409 -m 'follow-up instructions here' --agent pi
uv run ptq peek 20260520-torchtitan-3409
uv run ptq results 20260520-torchtitan-3409
uv run ptq clean 20260520-torchtitan-3409
```

If this job has a name, you can also launch by name:

```bash
uv run ptq run orchestrator-3409 -m 'task instructions here' --agent pi
```

## Agent context

PTQ-launched agents receive a rendered system prompt at `~/.ptq_workspace/jobs/20260520-torchtitan-3409/system_prompt.md`.
Follow-up runs also receive prior context from `~/.ptq_workspace/jobs/20260520-torchtitan-3409/worklog.md` and `~/.ptq_workspace/jobs/20260520-torchtitan-3409/report.md`.

Manual agents launched from this job directory should read this file first, then follow repo-local instructions in `~/.ptq_workspace/jobs/20260520-torchtitan-3409/torchtitan/AGENTS.md` when editing source.

