# New-Feature references

The agent is given, for each task, the architecture's **arXiv paper** and its
**reference implementation** — the same materials an engineer would consult. The
run_feature runner passes the links below into the prompt; for an offline/airgapped
run, clone the repos and download the PDFs first and point the agent at local
paths instead.

> Note: arXiv IDs / repos below are the best-known sources for each architecture
> (the PithTrain paper cites these by number, not URL). Verify against the paper's
> bibliography if exact-match matters.

| Task | Architecture | arXiv | Reference implementation |
|---|---|---|---|
| NF1 | Differential Transformer (Diff attention) | 2410.05258 | github.com/microsoft/unilm (Diff-Transformer) |
| NF2 | DynMoE | 2405.14297 | github.com/LINs-lab/DynMoE |
| NF3 | MoBA (Mixture of Block Attention) | 2502.13189 | github.com/MoonshotAI/MoBA |
| NF4 | MoE++ | 2410.07348 | github.com/SkyworkAI/MoE-plus-plus |
