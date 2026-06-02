# OP1: Getting Started

Set up the Python environment for this training framework and run the provided
smoke training script for 5 steps. **Success means the script reaches step 5 with
a finite loss.**

Install all dependencies the script needs so that running
```
STEPS=5 bash {{TRAIN_SH}}
```
succeeds as-is. The script is read-only and documents best practices for training
MoE models under the fixed config. The tokenizer and a debug dataset are already
pre-staged in the repo. Do not modify the training script or the mesh — your job
is to make the environment able to run it.
