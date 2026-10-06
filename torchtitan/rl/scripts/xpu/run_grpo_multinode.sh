#!/bin/bash
#PBS -l select=8
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug-scaling
#PBS -N grpo_full_mn
#PBS -l filesystems=flare:home
#PBS -A Intel-Aurora
#PBS -j oe
#
# Multi-node full-parameter GRPO on Intel XPU (Aurora / PBS). Default: 8 nodes.
#
# Submit:
#   qsub torchtitan/rl/scripts/xpu/run_grpo_multinode.sh
# Fewer/more nodes (the -l select on the command line wins over the header):
#   qsub -l select=4 torchtitan/rl/scripts/xpu/run_grpo_multinode.sh
# Interactive:
#   qsub -I -l select=8 -l walltime=01:00:00 -A <account> -q debug-scaling
#   bash torchtitan/rl/scripts/xpu/run_grpo_multinode.sh
#
# Overridable via env: NUM_NODES (use only the first N of the allocation), PPN
# (GPUs per node), TP, DP_REPLICATE, CONFIG, NUM_STEPS, VAL_SAMPLES,
# HF_ASSETS_PATH, DUMP_FOLDER, MPIEXEC, EXTRA_ARGS. `qsub -v` is
# needed to forward them into a batch job. Extra args go to the training command
# verbatim -- but only when run with bash: qsub accepts no trailing script args,
# so under PBS use EXTRA_ARGS instead.
#   TP=2 NUM_STEPS=20 qsub -l select=4 -v TP,NUM_STEPS run_grpo_multinode.sh
#   NUM_NODES=2 bash run_grpo_multinode.sh --override my_pkg.mod.fn
#   EXTRA_ARGS='--override my_pkg.mod.fn' qsub -v EXTRA_ARGS ...
set +e

# torch 2.14.0+xpu stack. TORCHTITAN_DIR, ONEAPI_ENV_SCRIPT, CONDA_PREFIX_BASE and
# CONDA_ENV are overridable; see torchtitan/rl/docs/xpu_torch214.md.
# Absolute, not relative to this file: PBS copies the submitted script into a
# spool dir, so BASH_SOURCE does not resolve back into the repo.
TORCHTITAN_DIR=${TORCHTITAN_DIR:-$HOME/git/torchtitan}
source ${ONEAPI_ENV_SCRIPT:-$TORCHTITAN_DIR/torchtitan/rl/scripts/xpu/env_torch214.sh}
eval "$(${CONDA_PREFIX_BASE:-$HOME/miniforge3}/bin/conda shell.bash hook)"
conda activate ${CONDA_ENV:-monarch214}

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1,2,3}
# Cray libfabric supplies the CXI provider oneCCL needs. Aurora system
# refreshes rotate the version directory (1.22.0 disappeared in the 2026-09
# refresh), so resolve the newest one instead of pinning a version -- a stale
# path surfaces only as an opaque oneCCL "failed to initialize ATL" cascade.
LIBFABRIC_ROOT=$(ls -d /opt/cray/libfabric/*/ 2>/dev/null | sort -V | tail -1)
LIBFABRIC_ROOT=${LIBFABRIC_ROOT%/}
if [ ! -e "${LIBFABRIC_ROOT}/lib64/libfabric.so.1" ]; then
    echo "ERROR: no Cray libfabric found under /opt/cray/libfabric; FI_PROVIDER=cxi cannot work"
    exit 1
fi
export CCL_OFI_LIBRARY_PATH=${LIBFABRIC_ROOT}/lib64/libfabric.so.1
export FI_PROVIDER=cxi
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_OFI_PROVIDER=cxi

export LD_LIBRARY_PATH=${LIBFABRIC_ROOT}/lib64:${LD_LIBRARY_PATH}
export FI_PROVIDER_PATH=${LIBFABRIC_ROOT}/lib64/libfabric

# CXI/oneCCL scale-out tuning. Required for correct AND working cross-node runs:
# without it, tp=8 dies on `atl_ofi.cpp:1071 fi_cq_readerr err 5` and tp4/rep2
# trains with a corrupted grad_norm (88-2464 vs a healthy 0.05-0.15).
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OVFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20
export CCL_ALLREDUCE_SCALEOUT=direct
export CCL_BCAST=double_tree
export CCL_SYCL_SCALEOUT_HOST_BUF_SIZE=$((2 * 1024 * 1024 * 1024))
# NOT CCL_OP_SYNC=1: on oneCCL 2022 it turns every cross-node all_reduce above
# 4 MiB into a silent no-op (each rank keeps its own input; FSDP2 HSDP replicas
# then train independently) and makes the gradient sync ~6x slower. Correct
# without it. uxlfoundation/oneCCL#223.
unset CCL_OP_SYNC
# Progress threads off the compute cores. Longer than the rank count on purpose;
# oneCCL takes what it needs. NOT CCL_WORKER_COUNT, which measured 29% SLOWER.
# oneCCL 2022.1 (monarch214) truncates an explicit list to local_size*worker_count
# at parse time (local_size=1 under PALS -ppn 1), then aborts init_comm with
# "unexpected worker affinity length" once a comm has >1 local rank. Its default
# (auto) already pins workers to the last cores, so leave it unset here.
unset CCL_WORKER_AFFINITY
# oneCCL 2022 stack: the v1 Level Zero UR adapter intermittently aborts with
# "ur_die: urEventWait must not be called for an internal event" (~1 per 100-150
# steps, trainer and generator alike); the v2 adapter ran 200/200 steps clean
# (8 nodes, full and LoRA). Not root-caused; drop it once the v1 adapter is fixed.
export SYCL_UR_USE_LEVEL_ZERO_V2=1
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export TORCH_LLM_ALLREDUCE=1

# Cray PALS by ABSOLUTE PATH: a oneAPI env can put Intel MPI's Hydra mpiexec ahead of
# PALS on PATH, and under Hydra a cross-node tp=8 run dies on a oneCCL SEND fault.
MPIEXEC=${MPIEXEC:-/opt/cray/pals/1.8/bin/mpiexec}

# PALS forwards the head node's env verbatim (--envall), so these two must be
# fixed up or the other nodes inherit values that are wrong for them:
#   TMPDIR - PBS sets /var/tmp/pbs.<jobid>, which exists only on the head node;
#            Monarch's worker bootstrap dies on the missing directory.
#   HOSTNAME - torchstore reads it for shared-memory locality (get_local_hostname),
#            so every rank thinks it is local to the storage volume and a
#            cross-node pull dies on "Shared memory storage not found".
export TMPDIR=/tmp
unset HOSTNAME

export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-~/.cache/torchinductor_xpu}
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MONARCH_ACTOR_QUEUE_DISPATCH=${MONARCH_ACTOR_QUEUE_DISPATCH:-1}

# No RMSNorm interposer: on torch 2.14.0+xpu the stock XPU kernel passes the
# 65536-row weight-grad backward that needed it on 2.12 (intel/torch-xpu-ops#4790).

HF_ASSETS_PATH=${HF_ASSETS_PATH:-/flare/Aurora_deployment/intel/models/Qwen3-0.6B}
PPN=${PPN:-4}
NUM_STEPS=${NUM_STEPS:-10}
VAL_SAMPLES=${VAL_SAMPLES:-0}
CONFIG=${CONFIG:-rl_grpo_full_qwen3_0_6b_flex}
DUMP_FOLDER=${DUMP_FOLDER:-outputs/rl_full_multinode}
# Nodes split half trainer / half generator by multinode_launcher. Trainer
# dp_shard = trainer_gpus/(TP*DP_REPLICATE), capped at 4 without LoRA, with the
# excess spilling onto dp_replicate. Keep TP inside a node: cross-node TP is
# correct but ~10x slower.
TP=${TP:-1}
DP_REPLICATE=${DP_REPLICATE:-1}
# qsub takes no trailing script arguments (unlike sbatch), so "$@" is always
# empty under PBS. EXTRA_ARGS is the only way to reach the launcher's flags from
# a qsub -v. Word-split on purpose so several flags can be passed.
EXTRA_ARGS=${EXTRA_ARGS:-}

# The config CLI takes no scalar field flags (--override selects a Configurable
# implementation, it does not set values), so these reach the recipe through the
# environment; mpiexec --envall forwards them to every node. See
# alphabet_sort_xpu.py, which also pins load_only so the DCP save that OOMs at
# dp_shard=16 never runs. Never turn the checkpointer off instead: `enable` also
# gates the initial HF weight load, so disabling it silently trains from random
# init and the first step never completes.
export TITAN_XPU_HF_ASSETS_PATH="$HF_ASSETS_PATH"
export TITAN_XPU_NUM_STEPS="$NUM_STEPS"
export TITAN_XPU_VAL_SAMPLES="$VAL_SAMPLES"

# A stale checkpoint from a different mesh shape fails to load, so start clean.
rm -rf "$TORCHTITAN_DIR/$DUMP_FOLDER"/checkpoint/ 2>/dev/null
rm -rf "$TORCHINDUCTOR_CACHE_DIR"/triton 2>/dev/null

# Bare (mgmt) names for both mpiexec --hosts and Monarch --all_nodes. The data
# plane still rides CXI/hsn0 via oneCCL. Do not use .hsn names: they are
# multi-rail and cause MESH_ATTACH_CONFIG_TIMEOUT.
if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE}" ]]; then
    mapfile -t NODE_LIST < <(sort -u "$PBS_NODEFILE" | sed 's/\..*//')
else
    NODE_LIST=("$(hostname -s)")
fi
AVAILABLE_NODES=${#NODE_LIST[@]}
NUM_NODES=${NUM_NODES:-$AVAILABLE_NODES}
if [ "$NUM_NODES" -gt "$AVAILABLE_NODES" ]; then
    echo "ERROR: NUM_NODES=$NUM_NODES but only $AVAILABLE_NODES allocated."
    echo "       Submit with: qsub -l select=$NUM_NODES $0"
    exit 1
fi
ALL_NODES=$(IFS=,; echo "${NODE_LIST[*]:0:$NUM_NODES}")
# MUST come after NUM_NODES is resolved above: assigned any earlier, ${NUM_NODES}
# expands to nothing under qsub (which forwards no env unless -v names it) and
# every run at every node count writes the SAME train_full_n.log, silently
# truncating the previous run's evidence. Even correctly named, the per-node-count
# default still collides when two full-param runs of the same size overlap (e.g. a
# 10-step smoke test and a 200-step run), so override LOG per run to keep them apart.
LOG=${LOG:-$DUMP_FOLDER/train_full_${NUM_NODES}n.log}

echo "=== Multi-node full GRPO: ${NUM_NODES}/${AVAILABLE_NODES} nodes, ${PPN} GPUs/node ==="
echo "Nodes:      ${ALL_NODES}"
echo "Config:     ${CONFIG}  TP=${TP}  DP_REPLICATE=${DP_REPLICATE}"

cd "$TORCHTITAN_DIR" || { echo "ERROR: TORCHTITAN_DIR=$TORCHTITAN_DIR not found"; exit 1; }
mkdir -p "$DUMP_FOLDER"

# Confirm the 65536-row RMSNorm weight-grad backward (the shape full GRPO hits)
# works on this torch before spending the allocation.
python3 -c "
import torch
x = torch.randn(2, 2048, 16, 128, device='xpu', dtype=torch.bfloat16, requires_grad=True)
torch.nn.RMSNorm(128, eps=1e-6, dtype=torch.bfloat16).to('xpu')(x).sum().backward()
torch.xpu.synchronize()
print('stock torch; 65536-row RMSNorm weight-grad backward: OK')
"
if [ $? -ne 0 ]; then
    echo "FATAL: RMSNorm weight-grad precheck failed, not starting the pipeline."
    exit 1
fi

# --cpu-bind none is REQUIRED. PALS binds each rank to its own core slice, and at
# -ppn 1 that slice is a SINGLE core; every Monarch actor and vLLM worker forked
# from the launcher inherits the mask, so ~126 threads share 1 core of 204. Costs
# 2-4x end-to-end (generator ITL 55 ms -> 215-260 ms).
#
# ulimit -c 0 runs inside each node's command: every RL teardown aborts and dumps
# ~10 GB of core per process (~150 GB per 8-node run) otherwise.
"$MPIEXEC" -n "$NUM_NODES" -ppn 1 --hosts "$ALL_NODES" --cpu-bind none --envall \
    bash -c 'ulimit -c 0; exec "$@"' _ python3 -m torchtitan.rl.multinode_launcher \
    --num_nodes="$NUM_NODES" \
    --gpus_per_node="$PPN" \
    --all_nodes="$ALL_NODES" \
    --tensor_parallel_degree="$TP" \
    --data_parallel_replicate_degree="$DP_REPLICATE" \
    --module torchtitan_recipes.rl.alphabet_sort_xpu --config "$CONFIG" \
    --output-dir="$DUMP_FOLDER" \
    $EXTRA_ARGS \
    "$@" \
    > "$LOG" 2>&1

echo "Exit code: $?"
echo "=== rmsnorm crashes (must be 0) ==="
grep -c "tensor does not have a device" "$LOG"
echo "=== steps completed ==="
grep -E "Train \| Step" "$LOG" | tail -5
