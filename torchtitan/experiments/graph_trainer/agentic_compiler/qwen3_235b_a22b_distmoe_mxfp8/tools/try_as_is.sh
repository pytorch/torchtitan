#!/bin/bash
# Run each 30B kernel unchanged on the matching 235B specialization (seed 0 + real input).
Q=/home/bahuang/local/torchtitan/torchtitan/experiments/graph_trainer/agentic_compiler/qwen3_30b_a3b_distmoe_mxfp8/kernel_cache
W=/home/bahuang/local/torchtitan/outputs/agentic_compiler/qwen3_235b_a22b_distmoe_mxfp8
GPU=$1; shift
for pair in "$@"; do
  g=${pair%%:*}; rest=${pair#*:}; p235=${rest%%,*}; p30=${rest#*,}
  for s in $W/kernel_cache/${p235}*/specialization_*.json; do
    s30=$(ls $Q/${p30}*/specialization_*.json | head -1)
    k=$(dirname $s30)/$(python -c "import json;print(json.load(open('$s30'))['kernel_name'])")
    [ -f "$k" ] || { echo "$g: no 30B kernel"; continue; }
    out=$W/autokernel/run1/as_is/$g.log
    (cd /tmp && CUDA_VISIBLE_DEVICES=$GPU timeout 300 python $W/autokernel/harness.py $s $k --seeds 0 --real $W/real_inputs/fusion_$g.pt > $out 2>&1)
    echo "$g: $(grep -E 'EXACT|MISMATCH|FAIL|PASS|Error' $out | grep -v Warn | tr '\n' ' ' | cut -c1-200)"
  done
done
