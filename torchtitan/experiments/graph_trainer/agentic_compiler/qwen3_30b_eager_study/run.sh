#!/bin/bash
# run.sh <gpu> <config> <run dir>: one eager run of eager_recipe.py on the fake PG (NGPU=8,
# LOCAL_RANK=0), from the repo root. GK, LOCAL_COMPILE and LC_REGIONS select kernels and
# local_compile regions (see eager_recipe.py). Example:
#   GK=attention_input,attention_output,router,loss,expert_weight_prep \
#     ./run.sh 0 eager_kernels_tps runs/gk_all5_nolc_tps
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=$(mkdir -p "$3" && cd "$3" && pwd)
cd "$HERE/../../../../.."
CUDA_VISIBLE_DEVICES=$1 RUN="$RUN" GK="${GK:-}" LOCAL_COMPILE="${LOCAL_COMPILE:-0}" \
  LC_REGIONS="${LC_REGIONS:-}" PYTHONPATH="$HERE:." NGPU=8 LOCAL_RANK=0 \
  python -m torchtitan.train --module eager_recipe --config "$2" --comm-backend fake > "$RUN/log.txt" 2>&1
echo "exit $?" >> "$RUN/log.txt"
