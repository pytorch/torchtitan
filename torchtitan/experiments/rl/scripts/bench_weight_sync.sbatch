#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 2-node launcher for the weight-sync transport benchmark (bench_weight_sync.py
# --mode transport). Rank 0 on node A PUTs a synthetic state dict into its local
# storage volume; rank 0 of node B GETs it -- a remote volume, so the fetch really
# crosses the fabric.
#
# The data-plane transport is selectable via DATA_TRANSPORT (auto | gloo |
# monarch_rdma | monarch_rpc | torchcomms), passed through to --data-transport; it
# also derives the torchcomms env below, so one row of a transport matrix is one
# submission. Under the default "auto" and the RDMA env below, cross-host transfers
# resolve to MonarchRDMA and same-host to SHM; any other value pins that transport
# for every transfer, same-host included. Note the MonarchRDMA path is expected to
# be unreliable on this fabric (see the NIC-pin note below) -- gloo is what works.
#
# Submit:  sbatch --partition=g3 torchtitan/experiments/rl/scripts/bench_weight_sync.sbatch
#          DATA_TRANSPORT=torchcomms sbatch --partition=g3 torchtitan/experiments/rl/scripts/bench_weight_sync.sbatch
#
# One task (rank) per node keeps the PUT volume and the GET client on distinct
# hosts; bump --ntasks-per-node only if you want more concurrent volumes.

#SBATCH --job-name=bench_weight_sync
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=slurm_%j_bench_weight_sync.out
#SBATCH --error=slurm_%j_bench_weight_sync.err

set -u

# Submit from the repo root and both defaults below are already right: sbatch sets
# SLURM_SUBMIT_DIR to the submitting directory and exports the submitter's
# environment, so an activated venv arrives as VIRTUAL_ENV. Override either to run a
# worktree or a venv you did not activate.
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
VENV=${VENV:-${VIRTUAL_ENV:-}}
if [ -z "$VENV" ]; then
    echo "set VENV=<path to the venv with torch/torchstore/monarch>, or submit with" \
         "that venv activated" >&2
    exit 1
fi
CU13=$VENV/lib/python3.12/site-packages/nvidia/cu13
# Optional: a directory of *.safetensors shards to extrapolate the fit onto. Without
# it the bench still runs and reports the fit, just with no real-model target.
MODEL_DIR=${MODEL_DIR:-}
WORLD_SIZE_EXTRAPOLATE=${WORLD_SIZE_EXTRAPOLATE:-64}
DATA_TRANSPORT=${DATA_TRANSPORT:-auto}
# PROBE_DIRECT_GPU=1 appends --probe-direct-gpu, which retries the GET straight
# into a CUDA dst after the staged sweep. That is the pre-staging path, and the
# one expected to fail here, so it runs last and off by default -- see the flag's
# help for why (it can take the shared StorageVolume actor down with it).
PROBE_DIRECT_GPU=${PROBE_DIRECT_GPU:-0}
# MONARCH_RDMA_IBVERBS_TARGET needs no handling here: sbatch exports the
# submitter's environment (--export=ALL), so a pin set on the submit line reaches
# the srun'd ranks and is echoed below. See the plane note further down for why
# pinning changes the outcome rather than just the bandwidth.

# --- TorchStore data-plane transport: DATA_TRANSPORT + the env the "auto" row needs ---
# (the RDMA half is identical to the env the confirmed 2-node RL run used)
#
# torchcomms stays off by default (its CtranIb path is broken on the MLNX stack here),
# which is what keeps the auto cascade on MonarchRDMA. But USE_TORCHCOMMS /
# USE_TORCHCOMMS_RDMA also gate torchcomms_uniflow_available() and
# torchcomms_rdma_available(), and a "torchcomms" PIN consults exactly those: pinning
# torchcomms with USE_TORCHCOMMS=0 yields a hard RuntimeError, not a torchcomms run --
# the code under test is never entered. So derive them from DATA_TRANSPORT.
case "$DATA_TRANSPORT" in
    torchcomms) USE_TORCHCOMMS=1 ;;
    *)          USE_TORCHCOMMS=0 ;;
esac
export USE_TORCHCOMMS
export USE_TORCHCOMMS_RDMA=$USE_TORCHCOMMS   # torchcomms needs BOTH set to 1
#
# A gloo / monarch_rdma / monarch_rpc pin, by contrast, ignores its TORCHSTORE_*_ENABLED
# gate entirely -- only the auto cascade (and the torchcomms check above) reads those.
# So the RDMA env below shapes the auto row and torchcomms availability, not those pins.
#
# MONARCH_RDMA_IBVERBS_TARGET is deliberately NOT set, so expect the MonarchRDMA
# rows to be unreliable -- that is the honest state of monarch RDMA here, and part
# of what this bench measures. This fabric has 4 disjoint PLANES
# (ibp<rail>p<plane>) and a port reaches only ports in its own plane. monarch picks
# a host-memory NIC by hashing each region's own address, which is plane-blind, so
# the two ends choose independently and mismatch most of the time -- surfacing as
# IBV_WC_RETRY_EXC_ERR at completion polling, not at connect. Pinning every
# endpoint to one device would collapse them into one plane, but it caps the
# transfer at a single port (~24 GB/s vs ~90 GB/s across the 4 cards) and it made
# GPU memory-region registration fail in its own way, so it is not worth setting.
# Note it is the PLANE that constrains reachability, not the rail: cross-rail pairs
# route fine as long as the plane matches. Revisit if monarch gains plane-aware
# host-memory NIC selection.
export TORCHSTORE_RDMA_ENABLED=1                # cross-host -> MonarchRDMA

# --- venv / CUDA ---
export CUDA_HOME=$CU13
export PATH=$VENV/bin:$CU13/bin:$PATH
export PYTHONPATH=$REPO
export TRITON_CACHE_DIR=/scratch/$USER/triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/$USER/inductor

# --- torchrun-style rendezvous for torchstore.spmd (derived from SLURM below by
#     the script's populate_env_from_slurm; MASTER_ADDR/PORT set here for clarity)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
export MASTER_PORT=29610

echo "nodes=$SLURM_NNODES nodelist=$SLURM_NODELIST master=$MASTER_ADDR:$MASTER_PORT"
echo "data_transport=$DATA_TRANSPORT USE_TORCHCOMMS=$USE_TORCHCOMMS/$USE_TORCHCOMMS_RDMA"
echo "RDMA: TORCHSTORE_RDMA_ENABLED=$TORCHSTORE_RDMA_ENABLED nic=${MONARCH_RDMA_IBVERBS_TARGET:-<unpinned>}"
echo "probe_direct_gpu=$PROBE_DIRECT_GPU"

EXTRA_ARGS=()
if [ "$PROBE_DIRECT_GPU" = 1 ]; then
    EXTRA_ARGS+=(--probe-direct-gpu)
fi
if [ -n "$MODEL_DIR" ]; then
    EXTRA_ARGS+=(--model-dir "$MODEL_DIR")
fi

# transport=tcp -> monarch TcpWithHostname (the proven GB300 control-plane transport;
# metatls* needs Meta TLS infra absent here). The worker bind and controller dial
# addresses come from monarch itself, which resolves both through one helper
# (get_consistent_sockaddr) so they agree byte for byte; installs predating
# https://github.com/meta-pytorch/monarch/pull/4619 interpolate an unresolved
# hostname and die at MESH_ATTACH_CONFIG_TIMEOUT instead.
srun --kill-on-bad-exit=1 $VENV/bin/python -m \
    torchtitan.experiments.rl.scripts.bench_weight_sync \
    --mode transport \
    --transport tcp \
    --data-transport "$DATA_TRANSPORT" \
    --src-device cuda \
    --world-size-extrapolate "$WORLD_SIZE_EXTRAPOLATE" \
    "${EXTRA_ARGS[@]}"
RC=$?

# A transport failure inside the StorageVolume actor reaches the client only as
# "Actor call StorageVolumes_<id>.put failed" -- the ibverbs status/vendor_err that
# says WHY stays in monarch's own log, which it writes under node-local /tmp and
# which dies with the allocation. Copy it out per node (best effort, never fails
# the job) when MONARCH_LOG_COLLECT_DIR is set.
if [ -n "${MONARCH_LOG_COLLECT_DIR:-}" ]; then
    mkdir -p "$MONARCH_LOG_COLLECT_DIR"
    srun --ntasks-per-node=1 bash -c \
        "cp -f /tmp/$USER/monarch_log.log \
         '$MONARCH_LOG_COLLECT_DIR'/monarch_${SLURM_JOB_ID}_\$(hostname).log" \
        2>/dev/null || echo "monarch log collection: nothing to copy"
fi
exit $RC
