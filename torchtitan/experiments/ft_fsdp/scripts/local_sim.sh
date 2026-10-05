#!/bin/bash
# Simulate hosts on one machine: one GPU per "host", each in a restart loop.
# Usage: local_sim.sh <num_hosts> <out_dir> [kill_schedule]
# kill_schedule is a space separated list of "<seconds>:<host_index>" or
# "<seconds>:s<slot>" (kills the host currently holding that slot).
set -u
NUM_HOSTS=$1
OUT=$2
KILLS=${3:-}
MODULE=${MODULE:-torchtitan_recipes.ft_fsdp.llama3}
CONFIG=${CONFIG:-llama3_debugmodel_local}
MIN_REPLICAS=${MIN_REPLICAS:-$((NUM_HOSTS - 1))}
PY=${PY:-.venv/bin/python}
mkdir -p "$OUT"
rm -f "$OUT/finished"

export FTFSDP_STORE_ADDR="[::1]:29600" TORCHFT_LIGHTHOUSE="http://[::1]:29510"
export FTFSDP_RUN_ID="local-$(date +%s)"
FTFSDP_MIN_REPLICAS=$MIN_REPLICAS $PY -m torchtitan.experiments.ft_fsdp.lighthouse \
    > "$OUT/lighthouse.log" 2>&1 &
LH=$!
sleep 3
kill -0 $LH 2>/dev/null || { echo "lighthouse failed to start" >&2; exit 1; }

host_loop() {
    local i=$1 attempt=0
    while true; do
        CUDA_VISIBLE_DEVICES=$i LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 \
            MASTER_ADDR=localhost MASTER_PORT=$((29700 + i)) \
            FTFSDP_HOST_INDEX=$i FTFSDP_HOST_NAME=host$i \
            setsid $PY -m torchtitan.train --module "$MODULE" --config "$CONFIG" \
            --output-dir "$OUT/host$i" >> "$OUT/host$i.log" 2>&1 &
        echo $! > "$OUT/host$i.pid"
        wait $!
        rc=$?
        echo "host$i attempt $attempt exited rc=$rc at $(date +%s.%N)" >> "$OUT/events.log"
        # A clean exit means training finished; stop restarting everyone.
        [ $rc -eq 0 ] && touch "$OUT/finished"
        [ -e "$OUT/finished" ] && break
        attempt=$((attempt + 1))
        [ $attempt -gt 20 ] && break
    done
}

for ((i = 0; i < NUM_HOSTS; i++)); do
    host_loop $i &
done

START=$(date +%s)
for k in $KILLS; do
    t=${k%%:*}; h=${k##*:}
    while [ $(($(date +%s) - START)) -lt $t ]; do sleep 1; done
    [ -e "$OUT/finished" ] && break
    if [[ $h == s* ]]; then
        slot=${h#s}
        h=$(grep -H -o "gen [0-9]*: slot [0-9]*" "$OUT"/host*.log \
            | awk -v s="$slot" '{split($1, a, ":"); gen[a[1]] = $2 + 0; sl[a[1]] = $4}
                END {best = -1; for (f in gen) if (sl[f] == s && gen[f] > best) {best = gen[f]; host = f}
                     sub(/.*host/, "", host); sub(/\.log/, "", host); print host}')
    fi
    pid=$(cat "$OUT/host$h.pid")
    echo "kill host$h pid $pid at $(date +%s.%N)" >> "$OUT/events.log"
    kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid"
done

wait
kill $LH 2>/dev/null
echo done >> "$OUT/events.log"
