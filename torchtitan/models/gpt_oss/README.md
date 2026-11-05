# gpt-oss Model in torchtitan

## Quick Start
```bash
MODULE=torchtitan_recipes.tests.models.gpt_oss CONFIG=gpt_oss_debugmodel ./run_train.sh
```

The released 20B and 120B flavors conventionally use
`assets/hf/gpt-oss-20b` and `assets/hf/gpt-oss-120b`. TorchTitan keeps those
asset-path hints without shipping unverified full training recipes.

## Supported Features
- FSDP/HSDP, TP, EP, CP, PP
- Grouped matrix multiplication for efficient computation

With DeepEP, router scores scale W2's input and its per-expert bias:

```text
s * (W2(h) + b2) = W2(s * h) + s * b2
```

W2 and bias addition follow the configured remat policy. Bias scaling adds
pointwise operations and reductions, with the same GEMM count as bias-free
absorption. Other dispatchers retain post-expert scoring. The equality holds in
exact arithmetic; moving the scores changes BF16 rounding and can change
outputs and gradients.

CI already runs CP and PP on the debug model:
- `gpt_oss_pp+fsdp+cp+ep+sacop` (`gpt_oss_debugmodel_flex_fsdp2_cp2_pp2_ep4_sac`)
- `gpt_oss_pp+fsdp+ep+sacop`

Those jobs use Interleaved1F1B. FlexInnerAttention zero-bubble / split-backward PP
tests are disabled in `tests/integration_tests/features.py` because
FlexInnerAttention `BlockMask` is not a Tensor (`stage_backward_input` calls
`requires_grad` on every stage input). Full-backward schedules
(1F1B / GPipe / Interleaved1F1B) are unaffected.
