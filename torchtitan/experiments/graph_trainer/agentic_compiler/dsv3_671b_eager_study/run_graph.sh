#!/bin/bash
# run_graph.sh <gpu> <config> <run dir>: one GraphTrainer run of graph_recipe.py on the fake PG
# (NGPU=256, LOCAL_RANK=0), from the repo root, with the DSv3 loop's regions and kernel cache.
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=$(mkdir -p "$3" && cd "$3" && pwd)
cd "$HERE/../../../../.."
S=torchtitan/experiments/graph_trainer/.claude/skills/fusion_region_finder/scripts
CUDA_VISIBLE_DEVICES=$1 RUN="$RUN" PYTHONPATH="$HERE:$S:$HERE/../dsv3_671b_distmoe_mxfp8:." \
  NGPU=256 LOCAL_RANK=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m torchtitan.train --module graph_recipe --config "$2" --comm-backend fake > "$RUN/log.txt" 2>&1
echo "exit $?" >> "$RUN/log.txt"
