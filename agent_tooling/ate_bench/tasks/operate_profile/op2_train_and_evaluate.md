# OP2: Train and Evaluate

Drive the full **setup → train → export → evaluate** pipeline for the base MoE
model under the fixed config.

1. Train from **random initialization** for **25 steps** using the provided
   launcher.
2. Export the resulting checkpoint to **HuggingFace format** (see
   `scripts/checkpoint_conversion/convert_to_hf.py`).
3. The read-only `{{EVALUATE_SH}}` will load your export into **vLLM** and run
   **lm-evaluation-harness HellaSwag (zero-shot)**.

This tests **pipeline correctness, not model quality**: the HellaSwag score is
expected to be near-random because 25 steps from random init barely trains the
model. Initialization **must** be random; everything outside the fixed mesh and
step count (LR, optimizer, scheduler, data preprocessing) is up to you.

Produce the checkpoint such that the grader can run:
```
CKPT_DIR=<your checkpoint dir> STEP=25 bash {{EVALUATE_SH}}
```
and have it complete and write a finite HellaSwag score.
