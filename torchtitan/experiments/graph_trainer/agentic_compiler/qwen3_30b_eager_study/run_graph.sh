#!/bin/bash
# run_graph.sh <gpu> <config> <run dir>: one GraphTrainer run of graph_recipe.py on the fake PG,
# from the repo root. PACKS: a checkout of github.com/SherlockNoMad/graph-trainer-kernel-packs.
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=$(mkdir -p "$3" && cd "$3" && pwd)
cd "$HERE/../../../../.."
CUDA_VISIBLE_DEVICES=$1 RUN="$RUN" PYTHONPATH="$HERE:${PACKS:?set PACKS to a graph-trainer-kernel-packs checkout}:." \
  NGPU=8 LOCAL_RANK=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m torchtitan.train --module graph_recipe --config "$2" --comm-backend fake > "$RUN/log.txt" 2>&1
echo "exit $?" >> "$RUN/log.txt"
