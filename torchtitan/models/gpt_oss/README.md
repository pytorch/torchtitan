# gpt-oss Model in torchtitan

## Quick Start
```bash
MODULE=torchtitan_recipes.tests.models.gpt_oss CONFIG=gpt_oss_debugmodel ./run_train.sh
```

The released 20B and 120B flavors conventionally use
`assets/hf/gpt-oss-20b` and `assets/hf/gpt-oss-120b`. TorchTitan keeps those
asset-path hints without shipping unverified full training recipes.

## Supported Parallelisms

Coverage below matches the features listed for gpt-oss and the gpt-oss jobs in
`tests/integration_tests` (e.g. `gpt_oss_pp+fsdp+cp+ep+sacop`,
`gpt_oss_pp+fsdp+ep+sacop`).

| Feature | Notes |
|---------|-------|
| FSDP / HSDP | Default data-parallel path |
| Tensor Parallel (TP) | Supported |
| Expert Parallel (EP) | Composes with FSDP and PP in CI |
| Context Parallel (CP) | Composes with FSDP, EP, and PP in CI (`gpt_oss_debugmodel_flex_fsdp2_cp2_pp2_ep4_sac`) |
| Pipeline Parallel (PP) | Interleaved1F1B exercised in CI. FlexInnerAttention zero-bubble / split-backward PP tests are disabled in `tests/integration_tests/features.py` because FlexInnerAttention `BlockMask` is not a Tensor (`stage_backward_input` calls `requires_grad` on every stage input). Full-backward schedules (1F1B / GPipe / Interleaved1F1B) are unaffected. |
| Grouped matrix multiplication | For efficient MoE-style computation |
