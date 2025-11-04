#!/bin/bash
printenv
ulimit -n 32000
# if NODEID == 0...
if [[ "$SLURM_NODEID" -eq 0 ]]; then
    # Create the trajectory handler stuff
    echo "Starting job at $(date)"
    source ${API_ENV}/bin/activate
    # Start trajectory handler
    echo "Starting trajectory handler..."
    run-api > ${LOGDIR}/api.log 2>&1 &
    python $PYTHON_SCRIPT serve --slurm=True $PYTHON_ARGS > ${LOGDIR}/env_server.log 2>&1 &
    deactivate
    echo "Started trajectory handler..."
fi
echo $SLURM_NODEID ", " $NUM_TRAINING_NODES
# now, if we're within the number of nodes allocated to training...
if [[ "$SLURM_NODEID" -lt "$NUM_TRAINING_NODES" ]]; then
    source ${TRAIN_ENV}/bin/activate
    cd $TRAIN_PATH
    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}

    echo Node IP: $head_node_ip
    export LOGLEVEL=INFO
    # Enable for A100
#    export FI_PROVIDER="efa"
    # Ensure that P2P is available
    # export NCCL_P2P_DISABLE=1
#    export NCCL_IB_DISABLE=1

    # debugging flags (optional)
    export NCCL_DEBUG=WARN
    export PYTHONFAULTHANDLER=1
    # optional debug settings
    # export NCCL_DEBUG=INFO
    # NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

#    export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
    export CUDA_LAUNCH_BLOCKING=0
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

    # on your cluster you might need these:
    # set the network interface
#    export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
#    export NCCL_BUFFSIZE=2097152
#    export TORCH_DIST_INIT_BARRIER=1
#    export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

#    dcgmi profile --pause
    # adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
    # to your specific node count, and update target launch file.
    torchrun --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint="$head_node_ip:29500"  --role rank --tee 3 \
-m torchtitan.grpo_train --job.config_file ${CONFIG_FILE}  --grpo.sglang_slurm_num_nodes ${NUM_INFERENCE_NODES} ${TRAINING_ARGS}
    scancel $SLURM_JOBID
#    dcgmi profile --resume
# else we're inferencing...
else

    # Setup 8 sglang instances with model in sglang venv
    echo "Starting sglang instances..."

    # Startup wandb monitoring...
    source ${API_ENV}/bin/activate
    API_ADDR="http://${head_node_ip}:8000"
    inference-node-wandb-watcher --api_addr ${API_ADDR} --tp 1 --node_num ${SLURM_NODEID} > ${LOGDIR}/wandb_${SLURM_NODEID}.log 2>&1  &

    source ${SGLANG_ENV}/bin/activate

    PORT_BASE=9000
    mkdir -p ${LOGDIR}/cache

    # Start 8 sglang instances on GPUs 0-3
    # this assumes you can run it with tp=1
    # if not, well, good luck with single node training, I'll pray for you
    LOG_OFFSET=$((SLURM_NODEID * 8))
    for i in {0..6}; do
        GPU_ID=$i
        LOG_ID=$((GPU_ID + LOG_OFFSET))
        PORT=$((PORT_BASE + i))
        OUTLINES_CACHE_DIR_CALC=${LOGDIR}/cache/sglang_${LOG_ID}
        mkdir -p $OUTLINES_CACHE_DIR_CALC
        echo "Starting sglang instance on GPU $GPU_ID, logdir $LOG_ID, port $PORT"
        OUTLINES_CACHE_DIR=$OUTLINES_CACHE_DIR_CALC CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m sglang.launch_server \
          --model-path $MODEL_NAME \
          --host 0.0.0.0 \
          --mem-fraction-static 0.80 \
          --log-level="error" \
          --attention-backend triton \
          --dtype="bfloat16" \
          --port $PORT > ${LOGDIR}/sglang_${LOG_ID}.log 2>&1 &
        sleep 3  # wait so sglang can find ports without conflicts :)
    done
    GPU_ID=7
    LOG_ID=$((GPU_ID + LOG_OFFSET))
    PORT=$((PORT_BASE + 7))
    OUTLINES_CACHE_DIR_CALC=${LOGDIR}/cache/sglang_${LOG_ID}
    mkdir -p $OUTLINES_CACHE_DIR_CALC
    echo "Starting sglang instance on GPU 7, port 9007"
    OUTLINES_CACHE_DIR=$OUTLINES_CACHE_DIR_CALC CUDA_VISIBLE_DEVICES=7 nohup python -m sglang.launch_server \
      --model-path $MODEL_NAME \
      --host 0.0.0.0 \
      --mem-fraction-static 0.80 \
      --log-level="error" \
      --attention-backend triton \
      --dtype="bfloat16" \
      --port 9007 > ${LOGDIR}/sglang_${LOG_ID}.log 2>&1
fi
