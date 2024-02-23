# torchtrain

Note: This repository is currently under heavy development.

torchtrain contains PyTorch native parallelisms, tools and utilities to train large models.

## Design Principles

TorchTrain is a native PyTorch library with various training techniques. While it utilizes the PyTorch ecosystem for things like data loading (i.e. HuggingFace datasets), the core functionality is written in PyTorch.

* Designed to be easy to understand, use and extend for different training purposes.
* Minimal changes to the model code, when applying 1D/2D or 3D Parallelisms.
* Modular components instead of monolithic codebase

# Installation

Install PyTorch from source or install the latest pytorch nightly, then install requirements by

```python
pip install -r requirements.txt
```

Install additional dev requirements if you want to contribute to the repo:
```
pip install -r dev-requirements.txt
```

run the llama debug model locally to verify the setup is correct:

```
./run_llama_train.sh
```

# TensorBoard

To visualize training metrics on TensorBoard:

1. (by default) set `enable_tensorboard = true` in `torchtrain/train_configs/train_config.toml`

2. set up SSH tunneling
```
ssh -L 6006:127.0.0.1:6006 [username]@[hostname]
```

3. then in the torchtrain repo
```
tensorboard --logdir=./torchtrain/outputs/tb
```

4. go to the URL it provides OR to http://localhost:6006/

## Multi-Node Training
For training on ParallelCluster/Slurm type configurations, you can use the multinode_trainer.slurm file to submit your sbatch job.</br>
Note that you will need to adjust the number of nodes and gpu count to your cluster configs.</br>
<b>To adjust total nodes:</b>
```
#SBATCH --ntasks=2
#SBATCH --nodes=2
```
should both be set to your total node count.
Then update the srun launch parameters to match:
```
srun torchrun --nnodes 2
```
where nnodes is your total node count, matching the sbatch node count above.

<b>To adjust gpu count per node:</b>

If your gpu count per node is not 8, adjust:

```--nproc_per_node```

 in the torchrun command and

```#SBATCH --gpus-per-task```

in the SBATCH command section.
