# Qwen3.5 Varlen Attention Notes

Local smoke testing on an RTX 4090 used:

```bash
source .venv/bin/activate
NGPU=1 MODULE=qwen3_5 CONFIG=qwen35_debugmodel_varlen_attn ./run_train.sh \
  --training.steps 2 \
  --training.mixed-precision-param float32
```

Notes from setup/debugging:

- `requirements.txt` includes the VLM requirements file with `-r` so `pip install -r torchtitan/models/qwen3_5/requirements.txt` works from the repo root.
- Use the virtualenv `torchrun`; a pyenv shim may not support `--local-ranks-filter`.
- CLI overrides need Tyro option syntax, e.g. `--training.seq-len 512` or `--training.seq_len 512`, not bare `training.seq_length 512`.
- On Ada/FA2, PyTorch varlen attention rejects `num_splits`; only pass `num_splits=1` when FA3 is active and batch-invariant mode needs deterministic split-K reductions.
- The local 2-step smoke test needed `--training.mixed-precision-param float32` to avoid FSDP's mixed gradient dtype assertion.
