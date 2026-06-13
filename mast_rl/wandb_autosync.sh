#!/usr/bin/env bash
# Near-live mirror of a MAST OFFLINE wandb run to meta.wandb.io.
#
# The MAST job (offline, no egress) writes its wandb run to the Manifold bucket
# under ${DUMP_DIR}/wandb/offline-run-*. The devvm gvfs mount can't *enumerate*
# those dirs (Permission denied on listdir), so we use `manifold ls` to find the
# newest run, `manifold getr` to download it to local /tmp, then `wandb sync` it
# to meta.wandb.io (reachable only via fwdproxy; auth from ~/.netrc).
#
#   bash mast_rl/wandb_autosync.sh [<run-name>] [<interval_s>]
#   nohup bash mast_rl/wandb_autosync.sh torchtitan-rl-search-r1-qwen3-0.6b 120 \
#       > /home/yichuan/wandb_autosync.log 2>&1 &
set -uo pipefail

RUN_NAME="${1:-torchtitan-rl-search-r1-qwen3-0.6b}"
INTERVAL="${2:-120}"
WANDB_BIN="${WANDB_BIN:-/home/yichuan/torchtitan/.venv/bin/wandb}"
BUCKET_WANDB="torchtrain_datasets/tree/yichuan/mast_runs/${RUN_NAME}/wandb"
TMP="${TMP_DIR:-/tmp/wandb_sync_${RUN_NAME}}"

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://meta.wandb.io}"
export https_proxy="${https_proxy:-http://fwdproxy:8080}"
export http_proxy="${http_proxy:-http://fwdproxy:8080}"
export NO_PROXY="127.0.0.1,localhost"

echo "autosync: run='${RUN_NAME}' every ${INTERVAL}s -> ${WANDB_BASE_URL}"

clean() { grep -ivE 'ClientExecutor|Gojira|ThreadPool|^I[0-9]{4}|HostResource|^W[0-9]{4}'; }

while true; do
    OFF=$(manifold ls "${BUCKET_WANDB}/" 2>/dev/null | clean | awk '/offline-run-/{print $NF}' | sort | tail -1)
    if [[ -n "${OFF}" ]]; then
        rm -rf "${TMP}" && mkdir -p "${TMP}"
        # getr writes to <dst>/<basename(src)> -> point at parent.
        if manifold getr "${BUCKET_WANDB}/${OFF}" "${TMP}" >/dev/null 2>&1; then
            echo "$(date +%H:%M:%S) syncing ${OFF}"
            "${WANDB_BIN}" sync -e yichuan -p torchtitan "${TMP}/${OFF}" 2>&1 | grep -iE 'syncing|done|view run|view project|error' || true
        else
            echo "$(date +%H:%M:%S) getr failed for ${OFF}"
        fi
    else
        echo "$(date +%H:%M:%S) no offline-run yet under ${BUCKET_WANDB}"
    fi
    sleep "${INTERVAL}"
done
