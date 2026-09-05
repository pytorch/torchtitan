# gpt-oss Model in torchtitan

## Quick Start
```bash
MODULE=gpt_oss CONFIG=gpt_oss_debugmodel ./run_train.sh
```

## Supported Features
- FSDP/HSDP, TP, EP, CP, PP
- Grouped matrix multiplication for efficient computation

CI already runs CP and PP on the debug model:
- `gpt_oss_pp+fsdp+cp+ep+sacop` (`gpt_oss_debugmodel_flex_fsdp2_cp2_pp2_ep4_sac`)
- `gpt_oss_pp+fsdp+ep+sacop`

Those jobs use Interleaved1F1B. FlexAttention zero-bubble / split-backward PP
tests are disabled in `tests/integration_tests/features.py` because
FlexAttention `BlockMask` is not a Tensor (`stage_backward_input` calls
`requires_grad` on every stage input). Full-backward schedules
(1F1B / GPipe / Interleaved1F1B) are unaffected.
