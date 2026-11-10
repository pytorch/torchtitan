# DeepSeek-V3 671B: eager vs graph fusion-region finders

Two agents searched one DeepSeek-V3 671B training step for kernel-fusion regions, each limited to one
substrate:
- **eager:** the model code, the eager runtime, and its profiler and execution traces;
- **graph:** GraphTrainer's joint forward + backward FX graph and its per-node traces, without the model
  source.

Both ran the same program: plain BF16 `deepseek_v3_671b(seq_len=4096)`, fake PG with `NGPU=256` (FSDP 256,
EP 64), 98,304 tokens per microbatch, full recompute, `local_compile` regions off, deterministic, one GB300
each. Run on 2026-11-04 at torchtitan `c5c112d82`, one agent run per side (~50 and ~56 min).

## Files

| Path | Contents |
|---|---|
| `EAGER_REGION_FINDER_PROMPT.md`, `GRAPH_REGION_FINDER_PROMPT.md` | the task each agent followed: fixed setup, substrate rules, steps, output schema |
| `eager/regions.json`, `eager/report.md` | eager finder: 30 regions, 7 no-opportunity entries, substrate observations |
| `graph/regions.json`, `graph/report.md` | graph finder: 146 regions, 16 no-opportunity entries, substrate observations |
| `COMPARISON.md` | side-by-side comparison: same program?, family-by-family match, unique findings, tiering disagreements, recall check, what is inherent to each substrate |

Not committed: the traces, the op and node tables (`eager/analysis/op_table.parquet`,
`graph/analysis/node_table.csv.gz`), the calibrations and the scripts each agent wrote, about 1.3 GB in the
local directory `outputs/fusion_finder/{eager,graph}/`. Paths in the reports and in `COMPARISON.md` such as
`eager/scripts/` or `graph/analysis/` refer to that directory.

## How the agents were launched

Each agent ran as a Claude Code subagent and received only the message below. The prompt files it points to
lived in `torchtitan/experiments/kernel_fusion/` (not committed); they are copied here unchanged, and were
last edited before the launch.

Eager:

```
Your complete task is specified in /home/bahuang/local/torchtitan/torchtitan/experiments/kernel_fusion/EAGER_REGION_FINDER_PROMPT.md. Read that file first and follow it exactly. It defines the fixed setup, the substrate rules, the steps, the output files, and the done criteria. Don't spawn subagents.

Work from the repo root /home/bahuang/local/torchtitan. Use CUDA_VISIBLE_DEVICES=0 for the main capture and all timing; GPU 1 is spare for short probes. A sibling agent is using GPUs 2 and 3; don't touch them. Write everything under outputs/fusion_finder/eager/. Don't modify tracked files, commit, upload anything, or touch GitHub.

When you finish, reply with what the prompt's last paragraph asks for: the report path, the five largest gaps, the number of regions per scope tag, and the attribution coverage. Also list anything you couldn't complete and why.
```

Graph:

```
Your complete task is specified in /home/bahuang/local/torchtitan/torchtitan/experiments/kernel_fusion/GRAPH_REGION_FINDER_PROMPT.md. Read that file first and follow it exactly. It defines the fixed setup, the substrate rules, the steps, the output files, and the done criteria. Don't spawn subagents.

Work from the repo root /home/bahuang/local/torchtitan. Use CUDA_VISIBLE_DEVICES=2 for the main capture and all timing; GPU 3 is spare for short probes. A sibling agent is using GPUs 0 and 1; don't touch them. Write everything under outputs/fusion_finder/graph/. Don't modify tracked files (any temporary graph_trainer edit must be listed in the report and reverted), commit, upload anything, or touch GitHub.

When you finish, reply with what the prompt's last paragraph asks for: the report path, the five largest gaps, the number of regions per scope tag, and the attribution coverage. Also list anything you couldn't complete and why, and confirm that any temporary edits were reverted.
```

The eager agent also read `EAGER_FUSION_PLAN.md` from the same folder, which names the six `local_compile`
regions (`COMPARISON.md` sec. 1.1). This weakens only the eager side of the recall check.

## Repeating for another model

Change in both prompts:
- the goal sentence, the recipe (and the graph prompt's GraphTrainer conversion), `seq_len`, tokens per
  microbatch, the parallelism degrees, and the GEMM shape used to calibrate the roofline;
- the output directories, so a rerun doesn't overwrite these results;
- the graph prompt's pointer to `scripts/profile_dsv3_fake.sh` (DSv3-only, not in the repo): let the agent
  write its own launcher, as the eager prompt already does;
- stale notes: another process holding ~17.6 GB per GPU, and full recompute (`memory_policy="full"`), which
  ran out of memory for DSv3 at HEAD and needed an extra pass on the graph side.

Keep both agents away from earlier finder results (this folder, the loops' `regions.json`, the plan docs):
the comparison measures what each substrate shows. To score the finders against a kernel pack, run them on
the program the loop optimized; here 13% of the DSv3 pack's saving had no counterpart in the BF16 program.
