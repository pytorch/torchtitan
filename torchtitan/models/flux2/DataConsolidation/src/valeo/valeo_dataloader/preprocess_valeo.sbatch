#!/bin/bash
#SBATCH --account=nxtaim-1
#SBATCH --nodes=1                       # Increase for long run, e.g. 64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48              # This is the "pool" of CPUs for your script
#SBATCH --time=01:50:00                 # Increase for long run, MAX is 24h on booster 2h on develbooster
#SBATCH --gres=gpu:1                    # Keep 1 GPU for faster tensor operations
#SBATCH --partition=develbooster        # Changed to booster for long runs (develbooster is for development)
#SBATCH --output=logs/preproc_%j.out

# Without this, srun does not inherit cpus-per-task from sbatch. #128
echo "START SETTING UP SBATCH"
echo "----------------------------------"

# Add the directory containing 'valeo_dataset.py' to the PYTHONPATH.
# This tells the Python interpreter where to find your custom module.
# The ':$PYTHONPATH' part is important to preserve any existing paths.
export PYTHONPATH="/p/project1/nxtaim-1/alagar1/valeo_dataloader/:/p/project1/nxtaim-1/alagar1/LiDM-valeo/:$PYTHONPATH"
export POLARS_MAX_THREADS=1

# activating the virtual environment. TODO! needs your local base path
ENV_PATH="/p/project1/nxtaim-1/alagar1/CleanStableDiffusion/"
source $ENV_PATH/JSC_training/activate.sh

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
# so processes know who to talk to
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

echo "Job id: $SLURM_JOB_ID"

echo ""
echo "START EXECUTING MAIN"

# --- START TIMER HERE ---
SECONDS=0

srun python /p/project1/nxtaim-1/alagar1/valeo_dataloader/preprocess_valeo.py \
    --dataset_root_path /p/data1/nxtaim/proprietary/valeo/my_test_space/
# For development (carfull) start a 4 GPU node and acces it from the comand line
#srun --time=2:00:00 --cpu_bind=none --nodes=1 --ntasks=1 --partition=develbooster --gres=gpu:4 -A nxtaim-1 --pty /bin/bash
# --- END CALCULATIONS HERE ---
DURATION=$SECONDS
echo "----------------------------------"
echo "TOTAL EXECUTION TIME: $((DURATION / 3600))h $(((DURATION % 3600) / 60))m $((DURATION % 60))s"
echo "----------------------------------"