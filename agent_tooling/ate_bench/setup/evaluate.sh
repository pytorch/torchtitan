#!/usr/bin/bash
# ATE-Bench OP2 (Train and Evaluate) — read-only evaluation script.
#
# Consumes the pipeline output the agent produced (a TorchTitan checkpoint), and:
#   1. exports it to HuggingFace format via TorchTitan's converter
#   2. runs lm-evaluation-harness HellaSwag (zero-shot) on it via vLLM
#   3. writes a finite score to $OUT/hellaswag.json
#
# The task tests PIPELINE correctness, not model quality: 25 steps from random
# init is expected to be near-random on HellaSwag. The harness accepts the
# attempt iff this script completes and writes a finite score.
#
# Prereqs (OP2 is GPU-gated): vllm + lm-eval installed, a GPU visible.
#
# Usage:
#   CKPT_DIR=<torchtitan checkpoint dir> STEP=25 \
#   HF_OUT=<dir> OUT=<dir> bash agent_tooling/ate_bench/setup/evaluate.sh
set -euo pipefail

: "${CKPT_DIR:?set CKPT_DIR to the TorchTitan checkpoint directory}"
STEP="${STEP:-25}"
HF_OUT="${HF_OUT:-$CKPT_DIR/hf_export}"
OUT="${OUT:-$(dirname "$CKPT_DIR")/eval}"
TOKENIZER="${TOKENIZER:-./tests/assets/tokenizer}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "$HF_OUT" "$OUT"

echo "[ate_bench] exporting TorchTitan checkpoint (step $STEP) -> HF format at $HF_OUT"
python scripts/checkpoint_conversion/convert_to_hf.py \
    "$CKPT_DIR" "$HF_OUT" --step "$STEP" || {
  echo "convert_to_hf.py failed; check its --help for the exact arg signature" >&2
  exit 2
}
# TorchTitan exports weights only; copy a tokenizer so vLLM can load the model.
cp -n "$TOKENIZER"/* "$HF_OUT"/ 2>/dev/null || true

echo "[ate_bench] running lm-eval HellaSwag (zero-shot) via vLLM"
lm_eval --model vllm \
    --model_args "pretrained=$HF_OUT,dtype=bfloat16,gpu_memory_utilization=0.8" \
    --tasks hellaswag --num_fewshot 0 \
    --output_path "$OUT" \
    --log_samples || {
  echo "lm_eval failed; ensure vllm + lm-eval-harness are installed" >&2
  exit 3
}

# Surface a single finite score for the harness check.
python - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
files = sorted(glob.glob(os.path.join(out, "**", "results*.json"), recursive=True))
if not files:
    print("no lm-eval results json found", file=sys.stderr); sys.exit(4)
res = json.load(open(files[-1]))
acc = res.get("results", {}).get("hellaswag", {})
score = acc.get("acc,none", acc.get("acc"))
json.dump({"hellaswag_acc": score}, open(os.path.join(out, "hellaswag.json"), "w"), indent=2)
print("hellaswag_acc =", score)
PY
