#!/bin/bash
#
# Bitwise numerics verification (+ perf comparison) for DeepSeek-V3:
#   eager Trainer   vs   graph_trainer (aot_fx_trace).
#
# Both sides use ChunkedLoss and FULL activation checkpointing:
#   - eager : ChunkedCELoss               + --activation_checkpoint.mode full
#   - graph : ChunkedCELossWithParamGrads + --compile.memory_policy full
#             (memory_policy=full is documented as the mirror of eager full AC)
#
# The eager baseline is run PURE eager: the 16B config enables
# `compile.components=["loss"]`, so we pass --compile.no-enable to keep the
# reference unfused (inductor loss fusion can shift bits). The graph_trainer
# side uses regional-inductor flex (required for bitwise flex match) and has
# cudagraph DISABLED so the comparison isolates the ChunkedLoss + full-AC
# numerics (and avoids the branch's force-cudagraph debug hack).
#
# loss_compare.py forces --debug.deterministic --debug.seed=42 and creates a
# shared seed checkpoint (single-GPU init), so both runs start from identical
# weights. --assert-equal makes it exit non-zero on any full-precision loss
# mismatch. After the runs we re-read TensorBoard for FULL-PRECISION loss and
# grad_norm (bitwise), plus memory / tps / tflops / mfu (perf, not expected to
# match).
#
# Usage:
#   ./run_loss_compare_dsv3.sh                 # 16b, 50 steps, dp8/tp1/ep4
#   MODEL_SIZE=debugmodel STEPS=20 ./run_loss_compare_dsv3.sh
#   STEPS=100 EP=8 ./run_loss_compare_dsv3.sh

set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

MODEL_SIZE="${MODEL_SIZE:-16b}"          # 16b | debugmodel | 671b
STEPS="${STEPS:-50}"
NGPU="${NGPU:-8}"
DP_SHARD="${DP_SHARD:-8}"
TP="${TP:-1}"
EP="${EP:-4}"
DATASET="${DATASET:-c4_test}"
# CONFIG_SUFFIX selects a config variant on BOTH sides, e.g. "_minimal_async_ep"
# to compare eager deepseek_v3_16b_minimal_async_ep vs graph_trainer_..._minimal_async_ep.
CONFIG_SUFFIX="${CONFIG_SUFFIX:-}"
# TEST_CUDAGRAPH=1 keeps cudagraph ENABLED on the graph side (tests the forced
# cudagraph numerics); default 0 disables it (isolates ChunkedLoss+AC+EP numerics).
TEST_CUDAGRAPH="${TEST_CUDAGRAPH:-0}"
JOB_DIR="${JOB_DIR:-$HOME/tmp/loss_compare_dsv3_${MODEL_SIZE}${CONFIG_SUFFIX}}"

# 16B's MoE kernels compile cold; cap inductor workers so the many-core host
# does not spike host RAM and OOM-kill regional_inductor (same reason as
# run_graph_trainer_dsv3.sh). loss_compare inherits this env into both runs.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"

PARALLELISM="--parallelism.data_parallel_shard_degree=${DP_SHARD}"
PARALLELISM="${PARALLELISM} --parallelism.tensor_parallel_degree=${TP}"
PARALLELISM="${PARALLELISM} --parallelism.expert_parallel_degree=${EP}"
COMMON="${PARALLELISM} --dataloader.dataset ${DATASET}"

# No seed checkpoint: with identical parallelism + --debug.seed=42 +
# --debug.deterministic, the eager and graph_trainer runs initialize to
# byte-identical weights on their own (verified: the debugmodel matches
# bitwise both with and without a seed checkpoint). This also avoids
# loss_compare's single-GPU seed creation, which for 16B/671B forces an
# unsharded CPU init of all params -- impractically slow.
BASE_OPTS="${COMMON} --compile.no-enable --activation_checkpoint.mode full"
TEST_OPTS="${COMMON} --compile.mode aot_fx_trace --compile.memory_policy full"
if [ "$TEST_CUDAGRAPH" != "1" ]; then
    TEST_OPTS="${TEST_OPTS} --compile.disable_passes cudagraph_pass"
fi

mkdir -p "${JOB_DIR}"
LOG="${JOB_DIR}/compare.log"
SUMMARY_MD="${JOB_DIR}/fullprec_summary.md"

echo "=============================================================="
echo " DSv3 numerics verification"
echo "   model        : deepseek_v3_${MODEL_SIZE}${CONFIG_SUFFIX}  vs  graph_trainer_deepseek_v3_${MODEL_SIZE}${CONFIG_SUFFIX}"
echo "   test cudagraph: ${TEST_CUDAGRAPH}"
echo "   parallelism  : dp_shard=${DP_SHARD} tp=${TP} ep=${EP} (NGPU=${NGPU})"
echo "   steps        : ${STEPS}    dataset: ${DATASET}"
echo "   job dir      : ${JOB_DIR}"
echo "   baseline opts: ${BASE_OPTS}"
echo "   test opts    : ${TEST_OPTS}"
echo "=============================================================="

python scripts/loss_compare.py . . \
  --baseline-module=deepseek_v3 --baseline-config="deepseek_v3_${MODEL_SIZE}${CONFIG_SUFFIX}" \
  --test-module=graph_trainer.deepseek_v3 --test-config="graph_trainer_deepseek_v3_${MODEL_SIZE}${CONFIG_SUFFIX}" \
  --baseline-options="${BASE_OPTS}" \
  --test-options="${TEST_OPTS}" \
  --assert-equal --steps="${STEPS}" --no-seed-checkpoint \
  --baseline-ngpus="${NGPU}" --test-ngpus="${NGPU}" \
  --job-dump-folder="${JOB_DIR}" 2>&1 | tee "${LOG}"
COMPARE_EXIT=${PIPESTATUS[0]}
echo "loss_compare exit code: ${COMPARE_EXIT}"

# --- Full-precision re-read from TensorBoard (loss/grad_norm bitwise; perf) ---
python - "${JOB_DIR}" "${MODEL_SIZE}" "${STEPS}" "${DP_SHARD}" "${TP}" "${EP}" "${SUMMARY_MD}" <<'PY'
import os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

job, model, steps, dp, tp, ep, out_md = sys.argv[1:8]

def load(folder, tag):
    base = os.path.join(job, folder)
    sub = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    ea = EventAccumulator(os.path.join(base, sub[0])); ea.Reload()
    return {s.step: s.value for s in ea.Scalars(tag)}

def both(tag):
    return load("tb_baseline", tag), load("tb_test", tag)

L = []
L.append(f"## {model} — eager vs graph_trainer ({steps} steps, dp_shard={dp} tp={tp} ep={ep})\n")

# ---- Bitwise metrics: loss + grad_norm ----
L.append("### Bitwise (loss + grad_norm, full precision)\n")
verdicts = []
for tag, name in (("loss_metrics/global_avg_loss", "loss"), ("grad_norm", "grad_norm")):
    b, t = both(tag)
    st = sorted(set(b) & set(t))
    maxd = max(abs(b[s] - t[s]) for s in st)
    ne = sum(1 for s in st if b[s] != t[s])
    verdicts.append((name, len(st), maxd, ne))
    print(f"\n=== {name}: steps={len(st)}  max|diff|={maxd:.3e}  bitwise-unequal-steps={ne}")
    for s in st:
        print(f"  step {s:3d}  baseline={b[s]!r}  test={t[s]!r}  diff={b[s]-t[s]:+.3e}")

L.append("| metric | steps | max \\|diff\\| | steps differing |")
L.append("|---|---|---|---|")
for name, n, maxd, ne in verdicts:
    L.append(f"| {name} | {n} | {maxd:.3e} | {ne} / {n} |")
allzero = all(v[2] == 0.0 and v[3] == 0 for v in verdicts)
if allzero:
    L.append("\n**Verdict: BITWISE IDENTICAL ✅** "
             "(loss and grad_norm match to full precision at every step).\n")
else:
    onset = []
    for tag, name in (("loss_metrics/global_avg_loss", "loss"), ("grad_norm", "grad_norm")):
        b, t = both(tag); st = sorted(set(b) & set(t))
        fd = next((s for s in st if b[s] != t[s]), None)
        onset.append(f"{name} from step {fd}")
    L.append("\n**Verdict: MISMATCH ❌** (first divergence: "
             + "; ".join(onset) + ").\n")

# Per-step full-precision table (sampled: first 3, every 10th, last)
bl, tl = both("loss_metrics/global_avg_loss")
bg, tg = both("grad_norm")
st = sorted(set(bl) & set(tl))
sample = sorted(set(st[:3]) | set(s for s in st if s % 10 == 0) | {st[-1]})
L.append("Per-step (sampled), full precision:\n")
L.append("| step | loss (both) | grad_norm (both) | loss Δ | grad Δ |")
L.append("|---:|---|---|---|---|")
for s in sample:
    L.append(f"| {s} | {bl[s]!r} | {bg[s]!r} | {bl[s]-tl[s]:+.1e} | {bg[s]-tg[s]:+.1e} |")
L.append("")

# ---- Perf metrics: memory / tps / tflops / mfu (not expected to match) ----
def peak(tag):
    b, t = both(tag); return max(b.values()), max(t.values())
def avg_after(tag, warmup):
    b, t = both(tag)
    bb = [v for s, v in b.items() if s > warmup]
    tt = [v for s, v in t.items() if s > warmup]
    return sum(bb)/len(bb), sum(tt)/len(tt)

warmup = max(10, int(steps) // 5)
mem_res_b, mem_res_t = peak("memory/max_reserved(GiB)")
mem_act_b, mem_act_t = peak("memory/max_active(GiB)")
tps_b, tps_t = avg_after("throughput(tps)", warmup)
tfl_b, tfl_t = avg_after("tflops", warmup)
mfu_b, mfu_t = avg_after("mfu(%)", warmup)

print(f"\n=== perf (memory=peak; tps/tflops/mfu=avg of steps>{warmup}) ===")
L.append(f"### Perf (memory = peak; tps/tflops/mfu = avg of steps > {warmup})\n")
L.append("| metric | eager | graph_trainer | graph/eager |")
L.append("|---|---:|---:|---:|")
def row(label, be, te, fmt="{:.2f}", ratio=True):
    r = f"{te/be:.3f}×" if (ratio and be) else ""
    line = f"| {label} | {fmt.format(be)} | {fmt.format(te)} | {r} |"
    L.append(line); print(line)
row("peak reserved mem (GiB)", mem_res_b, mem_res_t)
row("peak active mem (GiB)",   mem_act_b, mem_act_t)
row("throughput (tps)",        tps_b, tps_t, "{:,.0f}")
row("tflops",                  tfl_b, tfl_t)
row("mfu (%)",                 mfu_b, mfu_t)
L.append("")

with open(out_md, "w") as f:
    f.write("\n".join(L))
print(f"\nWrote markdown summary -> {out_md}")
PY

# --- Artifact: upload the full run log to pastry (internal; skipped if absent) ---
if command -v pastry >/dev/null 2>&1; then
    echo "Pastry (run log): $(pastry < "${LOG}" 2>/dev/null)"
fi
echo "Markdown summary: ${SUMMARY_MD}"
echo "COMPARE_EXIT=${COMPARE_EXIT}"
exit "${COMPARE_EXIT}"
