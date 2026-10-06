#!/bin/bash
#PBS -l select=8
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug-scaling
#PBS -N grpo_lora_mn
#PBS -l filesystems=flare:home
#PBS -A Intel-Aurora
#PBS -j oe
#
# Multi-node GRPO+LoRA on Intel XPU (Aurora / PBS). Serves any node count: nodes
# come from PBS_NODEFILE and multinode_launcher auto-scales the mesh.
#
# Submit:
#   qsub -l select=4 torchtitan/rl/scripts/xpu/run_grpo_lora_multinode.sh
# Interactive:
#   qsub -I -l select=4 -l walltime=01:00:00 -A <account> -q debug-scaling
#   bash torchtitan/rl/scripts/xpu/run_grpo_lora_multinode.sh
#
# Overridable via env: NUM_NODES (use only the first N of the allocation), PPN,
# TP, DP_REPLICATE, CONFIG, NUM_STEPS, VAL_SAMPLES, HF_ASSETS_PATH, DUMP_FOLDER,
# MPIEXEC, EXTRA_ARGS. A batch job needs `qsub -v` to forward them. Trailing args
# go to the training command verbatim, but qsub accepts none, so under PBS reach
# the launcher's flags through EXTRA_ARGS instead.
#   NUM_STEPS=20 qsub -l select=4 -v NUM_STEPS run_grpo_lora_multinode.sh
#   NUM_NODES=2 bash run_grpo_lora_multinode.sh --override my_pkg.mod.fn
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

# Fabric. Every export below is load-bearing; see GRPO_XPU.md for the bug each
# one closes. oneAPI 2025.3 pointed FI_PROVIDER_PATH at a libfabric with no cxi
# provider, so only oneCCL (via CCL_OFI_LIBRARY_PATH) would reach CXI/Slingshot
# and every other consumer would fall back to tcp; force the Cray libfabric for
# all of them. The FI_CXI_*/CCL_* tuning is what makes cross-node runs both
# correct and alive: without it tp=8 dies on `atl_ofi.cpp:1071 fi_cq_readerr
# err 5` and tp4/rep2 trains with grad_norm 88-2464 vs a healthy 0.05-0.15.
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
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export TORCH_LLM_ALLREDUCE=1
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

# Cray PALS by ABSOLUTE PATH: a oneAPI env can put Intel MPI's Hydra mpiexec ahead of
# PALS on PATH, and under Hydra a cross-node tp=8 run dies on a oneCCL SEND fault.
MPIEXEC=${MPIEXEC:-/opt/cray/pals/1.8/bin/mpiexec}

# PALS --envall forwards the head node's env verbatim, so anything read from env
# rather than from a syscall must be fixed up first. PBS sets
# TMPDIR=/var/tmp/pbs.<jobid>, which exists only on the head node, and Monarch's
# worker bootstrap dies on the missing directory. torchstore reads HOSTNAME for
# shared-memory locality, so every rank would think it is local to the storage
# volume and a cross-node pull dies on "Shared memory storage not found". Intel
# Hydra masked both by re-running a login shell per node.
export TMPDIR=/tmp
unset HOSTNAME

export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-~/.cache/torchinductor_xpu}
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MONARCH_ACTOR_QUEUE_DISPATCH=${MONARCH_ACTOR_QUEUE_DISPATCH:-1}

# No LD_PRELOAD interposer here: the patched RMSNorm backward kernel is needed
# only by FULL-parameter GRPO, which trains the qk_norm weights. LoRA freezes
# them, so its backward never asks for that weight gradient. See run_grpo_sn.sh
# / run_grpo_multinode.sh for the full-parameter arm.

HF_ASSETS_PATH=${HF_ASSETS_PATH:-/flare/Aurora_deployment/intel/models/Qwen3-0.6B}
CONFIG=${CONFIG:-rl_grpo_lora_qwen3_0_6b}
NUM_STEPS=${NUM_STEPS:-150}
VAL_SAMPLES=${VAL_SAMPLES:-0}
PPN=${PPN:-4}
DUMP_FOLDER=${DUMP_FOLDER:-outputs/rl_lora_multinode}
# multinode_launcher splits the nodes half trainer / half generator, so
# dp_shard = (NUM_NODES/2 * PPN) / (TP * DP_REPLICATE), capped at the LoRA rank
# with the excess spilling onto dp_replicate. TP=1/DP_REPLICATE=1 (pure dp_shard)
# is the fastest measured shape at every node count tried: at 2 nodes 13-17k
# tok/s vs 8-10k for TP=2 x rep2 and 5.0-5.8k for TP=4. Keep TP inside a node --
# cross-node TP is correct but ~10x slower (tp=8 measured ~500 tok/s).
TP=${TP:-1}
DP_REPLICATE=${DP_REPLICATE:-1}
# qsub takes no trailing script arguments (unlike sbatch), so "$@" is always
# empty under PBS. EXTRA_ARGS is the only way to reach the launcher's flags from
# a qsub -v. Word-split on purpose so several flags can be passed.
EXTRA_ARGS=${EXTRA_ARGS:-}

# The config CLI takes no scalar field flags (--override selects a Configurable
# implementation, it does not set values), so these reach the recipe through the
# environment; mpiexec --envall forwards them to every node. See
# alphabet_sort_xpu.py, which also pins load_only so the DCP save that OOMed at
# dp_shard=16 never runs. Never turn the checkpointer off instead: `enable` also
# gates the initial HF weight load, so disabling it silently trains from random
# init and the first step never completes.
export TITAN_XPU_HF_ASSETS_PATH="$HF_ASSETS_PATH"
export TITAN_XPU_NUM_STEPS="$NUM_STEPS"
export TITAN_XPU_VAL_SAMPLES="$VAL_SAMPLES"

# A stale checkpoint from a different mesh shape fails to load, so start clean.
rm -rf "$TORCHTITAN_DIR/$DUMP_FOLDER"/checkpoint/ 2>/dev/null
rm -rf "$TORCHINDUCTOR_CACHE_DIR"/triton 2>/dev/null

cd "$TORCHTITAN_DIR" || { echo "ERROR: TORCHTITAN_DIR=$TORCHTITAN_DIR not found"; exit 1; }
mkdir -p "$DUMP_FOLDER"
if [ ! -f "$HF_ASSETS_PATH/config.json" ]; then
    echo "ERROR: no model at $HF_ASSETS_PATH (config.json missing)."
    exit 1
fi

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
LOG=$DUMP_FOLDER/train_lora_${NUM_NODES}n.log

echo "=== GRPO+LoRA: ${NUM_NODES}/${AVAILABLE_NODES} nodes x ${PPN} GPUs, ${CONFIG}," \
     "TP=${TP} DP_REPLICATE=${DP_REPLICATE}, ${NUM_STEPS} steps ==="
echo "Nodes: ${ALL_NODES}"
echo "Log:   ${LOG}"

# --cpu-bind none is REQUIRED. PALS binds each rank to its own core slice, and at
# -ppn 1 that slice is a SINGLE core; every Monarch actor and vLLM worker forked
# from the launcher inherits the mask, so ~126 threads share 1 core of 204. Costs
# 2-4x end-to-end (generator ITL 55 ms -> 215-260 ms). Intel Hydra hid this by
# re-execing a login shell per node, which reset the mask.
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
echo "=== steps completed ==="
# Step lines are logged twice, so dedupe; sort NUMERICALLY or "9" outranks "50".
grep -oE 'Train \| Step: *[0-9]+' "$LOG" | grep -oE '[0-9]+' | sort -un | tail -5
