#!/bin/bash
# Usage: make_plots.sh <label> <caching_run> <trace_run> <planned_run> [planner]
set -e
LABEL=$1; CACHING=$2; TRACE=$3; PLANNED=$4; PLANNER=${5:-stalloc}
R=/data/users/bahuang/local/torchtitan_2/outputs/memplan
mkdir -p $R/plots
cd /data/users/bahuang/local/torchtitan_2
PYTHONPATH=/tmp/pdftools:mem_planning_study /home/bahuang/.conda/envs/pt_dev/bin/python \
  mem_planning_study/plot_alloc.py \
  --caching-dir $R/$CACHING --graph-info $R/$CACHING/graph_info.json \
  --window-trace $R/$TRACE/window_trace.txt \
  --planner-summary $R/$TRACE/summary_$PLANNER.json \
  --planned-memstat $R/$PLANNED/memstat.jsonl \
  --caching-label "$LABEL, caching allocator" \
  --out $R/plots/${LABEL}_${PLANNER}.png
