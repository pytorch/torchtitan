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

Before running any checks, ensure you are on the torchtitan `main` branch
with the latest upstream changes:

```bash
git checkout main
git pull origin main
```

If not on `main`, the delta review and freshness checks will produce
misleading results based on stale or branch-specific state.

---

## 1. Core Torchtitan Delta Review

Check what changed in core torchtitan since yesterday that graph_trainer should
know about. Not "did it break us" but "did we miss an opportunity or fall behind?"

```
git log --since="1 day ago" --oneline -- \
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

Graph_trainer has known TODOs blocked on upstream work. Check if upstream has
made progress that unblocks any of them.

**Active TODOs to check:**

| TODO | File | Blocked on |
|------|------|------------|
| Avoid parametrization in checkpoint save/load | `simple_fsdp.py` | `get_model_state_dict` supporting parametrized modules |
| DTensor mesh collapsing | `simple_fsdp.py` | DTensor adding mesh collapse op |
| Make dump rank configurable | `graph_utils.py` | Design decision |
| MOE grouped_mm error | `graph_utils.py` | PyTorch fix for grouped_mm with use_grouped_mm=False |
| BlockMask DTensor conversion | `common_utils.py` | flex_attention DTensor support |
| EP support for transformer block bucketing | `common_utils.py` | Owner: ruisizhang123 |
| CPU shadow chain avoidance in tracer | `make_fx_tracer.py` | make_fx internals |
| TP uneven seq_len with use_local_output=True | `llama3/parallelize.py`, `deepseek_v3/parallelize.py` | TP fix upstream |
| float8 tensorwise TP for deepseekv3 | `deepseek_v3/parallelize.py` | Owner: jianiw, needs testing |
| Re-enable async TP test | `tests/integration_tests.py` | Async TP bug fix |
| Add inductor backend to numerics test | `tests/test_numerics.py` | Inductor numerical accuracy fix |

For each, search upstream PyTorch and torchtitan commits/issues for relevant
progress. Report any that look unblocked or close to unblocked.

## 3. Test Coverage Gap Analysis

Look for things we should be testing but aren't.

- Are there parallelism combinations that core torchtitan tests but
  graph_trainer's integration_tests.py doesn't? Compare:
  - `tests/integration_tests/test_llama3.py` vs `graph_trainer/tests/integration_tests.py`
  - Same for deepseek_v3
- Are there new graph passes in `passes.py` without corresponding tests
  in `test_passes.py`?
- Are there new code paths added to graph_trainer files that have no test
  exercising them? Check recent commits to graph_trainer for untested additions.
- Are there model configs registered in `config_registry.py` that no test
  or integration test uses?

Output: a list of specific coverage gaps with suggested test descriptions.

## 4. Performance Opportunity Discovery

Not "did perf regress" but "could perf be better?"

- Check recent PyTorch commits in `torch/_inductor/`, `torch/_dynamo/`,
  `torch/_functorch/`, `torch/distributed/_tensor/` for new optimization
  features that graph_trainer could leverage.
- Check if any new `torch.compile` modes, backend options, or config knobs
  have been added that graph_trainer's `compile.py` or `passes.py` should
  know about.
- Review the traced graph pass pipeline in `passes.py` — are there passes
  that could be combined or simplified given recent upstream changes?
- Check torchao for new quantization methods relevant to compiled training.

Output: specific opportunities with links to upstream commits/docs.

## 5. Code Freshness & Technical Debt

Detect places where graph_trainer has drifted from how core torchtitan
now does things.

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

## 6. Documentation Freshness

Check if graph_trainer's docs match reality.

- Does `.claude/CLAUDE.md` reference correct file paths, test commands, and
  CLI flags? Run a spot-check: do the test files mentioned still exist? Do
  the benchmark commands still parse correctly?
- Does `README.md` match the current feature set and supported models?
- Are the run commands stored in Claude's memory file still correct?
- Are there new features or passes that aren't documented anywhere?

Output: specific inaccuracies found, or "docs are current."

## 7. Open Work Tracking

Check the state of in-flight work.

```
# PRs touching graph_trainer
gh pr list --search "graph_trainer" --state open

# Recent merged PRs (last 7 days)
gh pr list --search "graph_trainer" --state merged --limit 10

# Issues mentioning graph_trainer
gh issue list --search "graph_trainer" --state open
```

Flag:
- Open PRs that have been waiting >5 days without review
- Merged PRs in the last 24h that might need follow-up (doc updates, new tests)
- Issues that have been open >14 days without activity

---

## Output Format

Write the report to `torchtitan/experiments/graph_trainer/reports/YYYY-MM-DD.md`
(create the `reports/` directory if it doesn't exist). Use the following template:

```markdown
# Nightly Scout Report — YYYY-MM-DD

## Action Items (things to do)
- [ ] [P0/P1/P2] Description — why, what file/area

## Opportunities (things to consider)
- Description — potential benefit

## FYI (awareness, no action needed)
- Description

## Docs
- Inaccuracies found, or "docs are current"

## All Clear
- Areas with nothing to report
```

Keep it short. If a section has nothing, say "nothing to report" and move on.
Don't repeat what CI already tells us. Focus on what a human developer wouldn't
notice without actively looking.
